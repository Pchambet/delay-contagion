"""Hinge model of delay propagation through a turn.

    outbound_dep_delay = a + beta * max(0, inbound_arr_delay - (sched_turn - tau)) + noise

`tau` is the *effective minimum turn time*: as long as the late inbound leaves at least
`tau` minutes on the ground, the outbound can still leave on time. Below that, every
extra minute of inbound delay passes through at rate `beta`. `a` absorbs the outbound's
own (primary) delay.

For a fixed tau the model is linear, so we profile tau over a 1-minute grid. All fits run
on per-(group, day, tau) sufficient statistics computed in DuckDB: one SQL pass over the
turns, after which the point fit and a day-clustered bootstrap (resampling whole days,
because turns on the same day share weather and ATC conditions) are tiny array operations.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

TAU_GRID = np.arange(0, 121)  # minutes
# Analysis window on delays: drops data errors and the extreme tail (> 10 h) that would
# dominate least squares without telling us anything about turn dynamics.
DELAY_FILTER = """
    inbound_arr_delay between -90 and 600
    and outbound_dep_delay between -60 and 600
"""
# Stats layout along the last axis.
N, SX, SXX, SY, SXY, SYY = range(6)


@dataclass(frozen=True)
class HingeFit:
    a: float
    beta: float
    tau: float
    n: int
    sse: float


def hinge_x(inbound: np.ndarray, slack: np.ndarray, tau: float | np.ndarray) -> np.ndarray:
    """Delay that cannot be absorbed on the ground: max(0, inbound - (slack - tau))."""
    return np.maximum(0.0, inbound - (slack - tau))


def stats_from_arrays(
    inbound: np.ndarray, slack: np.ndarray, out: np.ndarray, tau_grid: np.ndarray = TAU_GRID
) -> np.ndarray:
    """Sufficient statistics, shape (len(tau_grid), 6), for in-memory data."""
    x = hinge_x(inbound[None, :], slack[None, :], tau_grid[:, None])
    y = out[None, :]
    n = np.full(len(tau_grid), len(out), dtype=float)
    return np.stack(
        [
            n,
            x.sum(1),
            (x * x).sum(1),
            np.repeat(y.sum(), len(tau_grid)),
            (x * y).sum(1),
            np.repeat((y * y).sum(), len(tau_grid)),
        ],
        axis=-1,
    )


def solve(stats: np.ndarray, tau_grid: np.ndarray = TAU_GRID) -> HingeFit:
    """Least-squares (a, beta) for every tau, then the tau with the smallest SSE."""
    n, sx, sxx, sy, sxy, syy = (stats[:, k] for k in range(6))
    cxx = sxx - sx * sx / n
    cxy = sxy - sx * sy / n
    cyy = syy - sy * sy / n
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.where(cxx > 1e-9, cxy / cxx, 0.0)
    sse = cyy - beta * cxy
    k = int(np.argmin(sse))
    a = (sy[k] - beta[k] * sx[k]) / n[k]
    return HingeFit(
        a=float(a), beta=float(beta[k]), tau=float(tau_grid[k]), n=int(n[k]), sse=float(sse[k])
    )


def fit_arrays(inbound: np.ndarray, slack: np.ndarray, out: np.ndarray) -> HingeFit:
    return solve(stats_from_arrays(inbound, slack, out))


def daily_stats(
    con: duckdb.DuckDBPyConnection, group_col: str, months: list[str], groups: list[str] | None
) -> tuple[list[str], list, np.ndarray]:
    """Per (group, day, tau) sufficient statistics from `marts.fct_turns`.

    Returns (group keys, days, array of shape (groups, days, taus, 6)).
    """
    month_list = ", ".join(f"'{m}'" for m in months)
    group_filter = f"and {group_col} in ({', '.join(f"'{g}'" for g in groups)})" if groups else ""
    df = con.execute(
        f"""
        with t as (
            select {group_col} as g, flight_date as d, inbound_arr_delay as i,
                   sched_turn_min as s, outbound_dep_delay as y
            from marts.fct_turns
            where flight_month in ({month_list}) {group_filter} and {DELAY_FILTER}
        ),
        x as (
            select g, d, tau, y, greatest(0, i - (s - tau)) as x
            from t, range({TAU_GRID[0]}, {TAU_GRID[-1] + 1}) r(tau)
        )
        select g, d, tau, count(*) n, sum(x) sx, sum(x * x) sxx, sum(y) sy, sum(x * y) sxy,
               sum(y * y) syy
        from x group by all
        """
    ).df()
    keys = sorted(df["g"].unique())
    days = sorted(df["d"].unique())
    arr = np.zeros((len(keys), len(days), len(TAU_GRID), 6))
    gi = df["g"].map({k: i for i, k in enumerate(keys)}).to_numpy()
    di = df["d"].map({d: i for i, d in enumerate(days)}).to_numpy()
    ti = df["tau"].to_numpy() - TAU_GRID[0]
    arr[gi, di, ti] = df[["n", "sx", "sxx", "sy", "sxy", "syy"]].to_numpy()
    return keys, days, arr


def fit_with_bootstrap(
    stats: np.ndarray, n_boot: int = 300, seed: int = 0, level: float = 0.95
) -> dict[str, float]:
    """Point fit plus day-clustered percentile bootstrap CIs.

    `stats` has shape (days, taus, 6) for one group.
    """
    point = solve(stats.sum(0))
    rng = np.random.default_rng(seed)
    n_days = stats.shape[0]
    draws = np.empty((n_boot, 3))
    for b in range(n_boot):
        w = rng.multinomial(n_days, np.full(n_days, 1.0 / n_days))
        fit = solve(np.tensordot(w, stats, axes=1))
        draws[b] = (fit.a, fit.beta, fit.tau)
    lo, hi = np.quantile(draws, [(1 - level) / 2, (1 + level) / 2], axis=0)
    return {
        "n_turns": point.n,
        "a": point.a, "a_lo": lo[0], "a_hi": hi[0],
        "beta": point.beta, "beta_lo": lo[1], "beta_hi": hi[1],
        "tau": point.tau, "tau_lo": lo[2], "tau_hi": hi[2],
    }  # fmt: skip


def fit_groups(
    con: duckdb.DuckDBPyConnection,
    group_col: str,
    months: list[str],
    groups: list[str] | None = None,
    n_boot: int = 300,
) -> pd.DataFrame:
    keys, _, arr = daily_stats(con, group_col, months, groups)
    rows = [{"group": k, **fit_with_bootstrap(arr[i], n_boot=n_boot)} for i, k in enumerate(keys)]
    return pd.DataFrame(rows)


def holdout_scores(
    con: duckdb.DuckDBPyConnection,
    params: pd.DataFrame,
    group_col: str,
    train_months: list[str],
    test_months: list[str],
) -> pd.DataFrame:
    """Out-of-sample MAE/RMSE of the hinge model against two baselines fitted on train:
    a constant (group mean) and a linear-in-inbound-delay model ignoring slack."""
    con.register("hinge_params", params[["group", "a", "beta", "tau"]])
    train = ", ".join(f"'{m}'" for m in train_months)
    test = ", ".join(f"'{m}'" for m in test_months)
    out = con.execute(
        f"""
        with base as (
            select {group_col} as g, flight_month, inbound_arr_delay as i, sched_turn_min as s,
                   outbound_dep_delay as y
            from marts.fct_turns where {DELAY_FILTER}
        ),
        lin as (
            select g, avg(y) as mean_y, regr_slope(y, i) as b1, regr_intercept(y, i) as b0
            from base where flight_month in ({train}) group by g
        ),
        pred as (
            select b.g, b.y,
                   l.mean_y                                          as p_const,
                   l.b0 + l.b1 * b.i                                 as p_linear,
                   h.a + h.beta * greatest(0, b.i - (b.s - h.tau))   as p_hinge
            from base b
            join lin l using (g)
            join hinge_params h on h."group" = b.g
            where b.flight_month in ({test})
        )
        select g as "group", count(*) as n_test,
               avg(abs(y - p_const))  as mae_constant,
               avg(abs(y - p_linear)) as mae_linear,
               avg(abs(y - p_hinge))  as mae_hinge,
               sqrt(avg((y - p_const)  ^ 2)) as rmse_constant,
               sqrt(avg((y - p_linear) ^ 2)) as rmse_linear,
               sqrt(avg((y - p_hinge)  ^ 2)) as rmse_hinge,
               1 - avg((y - p_hinge) ^ 2) / avg((y - p_const) ^ 2) as r2_hinge,
               1 - avg((y - p_linear) ^ 2) / avg((y - p_const) ^ 2) as r2_linear
        from pred group by g order by g
        """
    ).df()
    con.unregister("hinge_params")
    return out
