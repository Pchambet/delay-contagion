"""Where to add turnaround buffer: a scenario-based LP against simple rules.

Delay recursion along each aircraft chain (leg j, predecessor i in the chain):

    z_j = max(0, A_i - (s_j + b_c(j)) + tau)     propagated part, capped by the turn slack
    D_j = p_j + beta * z_j                       departure delay
    A_j = D_j + e_j                              arrival delay

`s_j` is the scheduled turn, `b_c` the extra buffer given to every turn in decision cell
`c` (station x local departure hour, a lever a schedule planner can actually pull),
(tau, beta) the carrier's fitted hinge parameters. Primary delays `p_j` and en-route
changes `e_j` are backed out of the observed data, so with b = 0 the recursion reproduces
every observed delay exactly; buffers then answer "what would that day have looked like".

Decision (sample average approximation over S training days):

    min  1/S * sum_s sum_j max(0, A_j)
    s.t. sum_c nbar_c * b_c <= B,   0 <= b_c <= b_max

where nbar_c is the mean number of daily turns in cell c, so B is in buffer minutes per
day. The max() terms are linearised with z_j >= 0, a_j >= 0 and one inequality each;
because every coefficient pushing them up is non-negative (beta >= 0), the LP optimum sits
exactly on the recursion. Solved with HiGHS; budgets are swept with warm starts.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import highspy
import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class Scenarios:
    """Legs sorted by (day, chain, leg_seq); `pred` is -1 for the first leg of a chain."""

    day: np.ndarray  # day index (0..n_days-1)
    pred: np.ndarray
    seq: np.ndarray
    s: np.ndarray  # scheduled turn (NaN for chain starts)
    p: np.ndarray  # primary departure delay
    e: np.ndarray  # en-route change: arrival delay - departure delay
    cell: np.ndarray  # decision cell of the turn (-1 for chain starts)
    obs_dep: np.ndarray
    obs_arr: np.ndarray
    n_days: int

    @property
    def linked(self) -> np.ndarray:
        return self.pred >= 0


def from_legs(legs: pd.DataFrame, cell_index: dict, tau: float, beta: float) -> Scenarios:
    """Build scenarios from observed legs (one row per leg, see `load_legs`)."""
    legs = legs.sort_values(["flight_date", "chain_id", "leg_seq"]).reset_index(drop=True)
    # A leg starts a chain if it opens a new chain_id or follows a gap (a dropped leg).
    first = (
        legs["chain_id"].ne(legs["chain_id"].shift())
        | legs["leg_seq"].ne(legs["leg_seq"].shift() + 1)
    ).to_numpy()
    idx = np.arange(len(legs))
    pred = np.where(first, -1, idx - 1)
    dep = legs["dep_delay"].to_numpy(float)
    arr = legs["arr_delay"].to_numpy(float)
    s = np.where(first, np.nan, legs["sched_turn_min"].to_numpy(float))
    inbound = np.where(first, 0.0, arr[np.maximum(pred, 0)])
    z_obs = np.where(first, 0.0, np.maximum(0.0, inbound - s + tau))
    keys = list(zip(legs["origin"], legs["dep_hour_local"], strict=True))
    cell = np.array([-1 if f else cell_index.get(k, -1) for f, k in zip(first, keys, strict=True)])
    days = legs["flight_date"].map(
        {d: i for i, d in enumerate(sorted(legs["flight_date"].unique()))}
    )
    return Scenarios(
        day=days.to_numpy(),
        pred=pred,
        seq=legs["leg_seq"].to_numpy(),
        s=s,
        p=dep - beta * z_obs,
        e=arr - dep,
        cell=cell,
        obs_dep=dep,
        obs_arr=arr,
        n_days=int(days.max()) + 1,
    )


def load_legs(con: duckdb.DuckDBPyConnection, carrier: str, days: list) -> pd.DataFrame:
    """Whole chains of `carrier` that start on `days`, excluding diverted legs (always the
    last leg of a chain, so chains stay intact)."""
    con.register("lp_days", pd.DataFrame({"d": days}))
    legs = con.execute(
        """
        select chain_id, leg_seq, chain_date as flight_date, origin, dep_hour_local,
               sched_turn_min, dep_delay, arr_delay
        from marts.fct_legs
        where carrier = ? and chain_date in (select d from lp_days) and not is_diverted
          and arr_delay is not null
        """,
        [carrier],
    ).df()
    con.unregister("lp_days")
    return legs


def simulate(sc: Scenarios, b_cell: np.ndarray, tau: float, beta: float) -> np.ndarray:
    """Arrival delay of every leg under per-cell buffers (vectorised by chain position)."""
    b = np.where(sc.cell >= 0, b_cell[np.maximum(sc.cell, 0)], 0.0)
    arr = np.empty_like(sc.p)
    start = ~sc.linked
    arr[start] = sc.p[start] + sc.e[start]
    for k in range(2, int(sc.seq.max()) + 1):
        j = np.flatnonzero((sc.seq == k) & sc.linked)
        z = np.maximum(0.0, arr[sc.pred[j]] - sc.s[j] - b[j] + tau)
        arr[j] = sc.p[j] + beta * z + sc.e[j]
    return arr


def daily_delay(sc: Scenarios, arr: np.ndarray) -> np.ndarray:
    """Total positive arrival delay minutes per scenario day."""
    return np.bincount(sc.day, weights=np.maximum(arr, 0.0), minlength=sc.n_days)


def daily_buffer(sc: Scenarios, b_cell: np.ndarray) -> np.ndarray:
    """Buffer minutes actually scheduled per day (turns in padded cells x buffer)."""
    b = np.where(sc.cell >= 0, b_cell[np.maximum(sc.cell, 0)], 0.0)
    return np.bincount(sc.day, weights=b, minlength=sc.n_days)


def cell_turns_per_day(sc: Scenarios, n_cells: int) -> np.ndarray:
    turns = np.bincount(sc.cell[sc.cell >= 0], minlength=n_cells)
    return turns / sc.n_days


class BufferLP:
    """The SAA linear program, built once and re-solved for each budget."""

    def __init__(self, sc: Scenarios, n_cells: int, tau: float, beta: float, b_max: float):
        if (sc.cell[sc.linked] < 0).any():
            raise ValueError("every training turn must belong to a decision cell")
        self.sc, self.n_cells = sc, n_cells
        self.nbar = cell_turns_per_day(sc, n_cells)
        n_legs = len(sc.p)
        linked = np.flatnonzero(sc.linked)
        zpos = np.full(n_legs, -1)
        zpos[linked] = n_cells + np.arange(len(linked))
        a0 = n_cells + len(linked)
        n_var = a0 + n_legs

        rows, cols, vals, lower = [], [], [], []
        r = 0
        # z_j - beta z_i + b_c >= p_i + e_i - s_j + tau      (one row per linked leg)
        i = sc.pred[linked]
        k = len(linked)
        rid = r + np.arange(k)
        rows += [rid, rid]
        cols += [zpos[linked], sc.cell[linked]]
        vals += [np.ones(k), np.ones(k)]
        has_z = zpos[i] >= 0
        rows.append(rid[has_z])
        cols.append(zpos[i][has_z])
        vals.append(np.full(has_z.sum(), -beta))
        lower.append(sc.p[i] + sc.e[i] - sc.s[linked] + tau)
        r += k
        # a_j - beta z_j >= p_j + e_j                           (one row per leg)
        rid = r + np.arange(n_legs)
        rows.append(rid)
        cols.append(a0 + np.arange(n_legs))
        vals.append(np.ones(n_legs))
        rows.append(rid[linked])
        cols.append(zpos[linked])
        vals.append(np.full(k, -beta))
        lower.append(sc.p + sc.e)
        r += n_legs
        # sum_c nbar_c b_c <= B                                 (budget row, last)
        self.budget_row = r
        rows.append(np.full(n_cells, r))
        cols.append(np.arange(n_cells))
        vals.append(self.nbar)
        r += 1

        a = sp.csc_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(r, n_var),
        )
        inf = highspy.kHighsInf
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = n_var, r
        cost = np.zeros(n_var)
        cost[a0:] = 1.0 / sc.n_days
        lp.col_cost_ = cost
        ub = np.full(n_var, inf)
        ub[:n_cells] = np.where(self.nbar > 0, b_max, 0.0)
        lp.col_lower_ = np.zeros(n_var)
        lp.col_upper_ = ub
        lp.row_lower_ = np.concatenate([*lower, [-inf]])
        lp.row_upper_ = np.concatenate([np.full(r - 1, inf), [0.0]])
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = a.indptr
        lp.a_matrix_.index_ = a.indices
        lp.a_matrix_.value_ = a.data
        self.h = highspy.Highs()
        self.h.setOptionValue("output_flag", False)
        self.h.setOptionValue("threads", 3)
        self.h.passModel(lp)
        self.shape = (r, n_var)

    def solve(self, budget: float) -> tuple[np.ndarray, float]:
        """Optimal per-cell buffers and objective (mean daily delay minutes)."""
        self.h.changeRowBounds(self.budget_row, -highspy.kHighsInf, float(budget))
        self.h.run()
        status = self.h.getModelStatus()
        if status != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(f"LP not optimal: {self.h.modelStatusToString(status)}")
        x = np.asarray(self.h.getSolution().col_value)
        return x[: self.n_cells].copy(), self.h.getInfo().objective_function_value


def uniform_policy(nbar: np.ndarray, budget: float, b_max: float) -> np.ndarray:
    """Same extra minutes on every turn of the network."""
    per_turn = min(b_max, budget / nbar.sum()) if nbar.sum() > 0 else 0.0
    return np.where(nbar > 0, per_turn, 0.0)


def greedy_policy(score: np.ndarray, nbar: np.ndarray, budget: float, b_max: float) -> np.ndarray:
    """Pad the worst cells first: b_max to cells in decreasing `score` until the budget is
    spent (the last cell gets the remainder)."""
    b = np.zeros_like(nbar, dtype=float)
    left = budget
    for c in np.argsort(-score, kind="stable"):
        if left <= 0 or nbar[c] <= 0 or score[c] <= 0:
            continue
        b[c] = min(b_max, left / nbar[c])
        left -= b[c] * nbar[c]
    return b


def propagated_delay_score(sc: Scenarios, n_cells: int, tau: float, beta: float) -> np.ndarray:
    """Mean observed propagated delay (beta * z) per turn in each cell: where the reactionary
    minutes show up in the training data."""
    linked = sc.linked
    inbound = sc.obs_arr[np.maximum(sc.pred, 0)]
    z = np.where(linked, np.maximum(0.0, inbound - sc.s + tau), 0.0)
    total = np.bincount(sc.cell[linked], weights=beta * z[linked], minlength=n_cells)
    count = np.bincount(sc.cell[linked], minlength=n_cells)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(count > 0, total / count, 0.0)
