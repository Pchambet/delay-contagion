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

from delay_contagion.warehouse import fetch_one

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
    # Same model with tau pinned at 0 (delay passes once the scheduled turn is used up): a
    # baseline that shows what the estimated minimum turn time adds.
    zero = solve(stats.sum(0)[:1], TAU_GRID[:1])
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
        "a_tau0": zero.a, "beta_tau0": zero.beta,
    }  # fmt: skip


def fit_groups(
    con: duckdb.DuckDBPyConnection,
    group_col: str,
    months: list[str],
    groups: list[str] | None = None,
    n_boot: int = 300,
    seed: int = 0,
) -> pd.DataFrame:
    keys, _, arr = daily_stats(con, group_col, months, groups)
    rows = [
        {"group": k, **fit_with_bootstrap(arr[i], n_boot=n_boot, seed=seed + i)}
        for i, k in enumerate(keys)
    ]
    return pd.DataFrame(rows)


def load_turns(con: duckdb.DuckDBPyConnection, months: list[str]) -> pd.DataFrame:
    """Turns of `months` inside the analysis window, with the features the baselines use."""
    month_list = ", ".join(f"'{m}'" for m in months)
    return con.execute(
        f"""
        select carrier, station, flight_month, dep_hour_local as hour,
               inbound_arr_delay::double as i, sched_turn_min::double as s,
               outbound_dep_delay::double as y
        from marts.fct_turns
        where flight_month in ({month_list}) and {DELAY_FILTER}
        """
    ).df()


def filter_excluded_share(con: duckdb.DuckDBPyConnection, months: list[str]) -> float:
    """Share of turns (with both delays reported) that the analysis window drops."""
    month_list = ", ".join(f"'{m}'" for m in months)
    return float(
        fetch_one(
            con,
            f"""
            select 1 - count(*) filter (where {DELAY_FILTER}) / count(*)
            from marts.fct_turns
            where flight_month in ({month_list})
              and inbound_arr_delay is not null and outbound_dep_delay is not null
            """,
        )[0]
    )


GBM_FEATURES = ["i", "s", "carrier", "hour"]


def gbm_predictions(train: pd.DataFrame, test: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """A pooled gradient-boosted model on (inbound delay, scheduled turn, carrier, hour).

    The flexible benchmark: if it beats the hinge, the hinge is kept for what it gives the
    decision layer (two interpretable parameters and a recursion that stays linear).
    """
    import lightgbm as lgb

    def features(df: pd.DataFrame) -> pd.DataFrame:
        x = df[GBM_FEATURES].copy()
        x["carrier"] = pd.Categorical(x["carrier"], categories=sorted(train["carrier"].unique()))
        return x

    params = {
        "objective": "regression",
        "learning_rate": 0.1,
        "num_leaves": 63,
        "min_data_in_leaf": 200,
        "bagging_fraction": 0.5,
        "bagging_freq": 1,
        "num_threads": 3,
        "seed": seed,
        "deterministic": True,
        "verbose": -1,
    }
    data = lgb.Dataset(features(train), train["y"], categorical_feature=["carrier"])
    model = lgb.train(params, data, num_boost_round=300)
    return np.asarray(model.predict(features(test)))


def _ols(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    x = np.column_stack([np.ones(len(df)), *(df[c].to_numpy() for c in cols)])
    return np.linalg.lstsq(x, df["y"].to_numpy(), rcond=None)[0]


def holdout_scores(
    params: pd.DataFrame, group_col: str, train: pd.DataFrame, test: pd.DataFrame
) -> pd.DataFrame:
    """Out-of-sample MAE/RMSE of the hinge against baselines fitted on the same train turns.

    constant: group mean. linear: in inbound delay, ignoring slack. linear_slack: in inbound
    delay and scheduled turn. hinge_tau0: the hinge with tau pinned at 0. gbm: the pooled
    boosted model (column `p_gbm` of `test`). hinge: the fitted model.
    """
    rows = []
    for g, p in params.set_index("group").iterrows():
        tr, te = train[train[group_col] == g], test[test[group_col] == g]
        if te.empty:  # e.g. a carrier that stopped reporting before the test months
            continue
        b_lin, b_ls = _ols(tr, ["i"]), _ols(tr, ["i", "s"])
        i, s, y = te["i"].to_numpy(), te["s"].to_numpy(), te["y"].to_numpy()
        preds = {
            "constant": np.full(len(te), tr["y"].mean()),
            "linear": b_lin[0] + b_lin[1] * i,
            "linear_slack": b_ls[0] + b_ls[1] * i + b_ls[2] * s,
            "hinge_tau0": p["a_tau0"] + p["beta_tau0"] * hinge_x(i, s, 0.0),
            "gbm": te["p_gbm"].to_numpy(),
            "hinge": p["a"] + p["beta"] * hinge_x(i, s, p["tau"]),
        }
        row: dict[str, object] = {"group": g, "n_test": len(te)}
        for m, pred in preds.items():
            row[f"mae_{m}"] = float(np.abs(y - pred).mean())
            row[f"rmse_{m}"] = float(np.sqrt(((y - pred) ** 2).mean()))
        rows.append(row)
    return pd.DataFrame(rows)
