"""Contagion multiplier: how many downstream delay minutes one primary minute generates.

Event. A leg where the aircraft was ready (first leg of its chain, or inbound arrived on
time) but the flight still left `x` minutes late: the `x` minutes are primary by
construction. Its *downstream delay* is the sum of positive arrival delays on every later
leg of the same aircraft chain.

Counterfactual. Later legs also pick up delay of their own, unrelated to the event. We
compare each event with *controls* - clean starts that left on time (x <= 0) - from the
same carrier, the same calendar day, the same part of the day, and the same number of
legs left to fly. Matching on the day absorbs weather and ATC conditions; matching on legs
remaining absorbs exposure. Excess downstream delay = downstream - control mean; the
multiplier is sum(excess) / sum(x).

Uncertainty. Events on the same day are correlated, so CIs come from a day-clustered
bootstrap that resamples whole days, shared across groups.
"""

from __future__ import annotations

from itertools import pairwise

import duckdb
import numpy as np
import pandas as pd

X_BINS = [1, 15, 30, 60, 120, 240, 601]  # primary delay bins (minutes), right-open
MIN_CONTROLS = 5
PRIMARY_MIN = 15  # "a delayed departure" for the headline multipliers (DOT definition)


def build_events(con: duckdb.DuckDBPyConnection, months: list[str]) -> int:
    """Materialise the matched event table `contagion_events` and return its size."""
    month_list = ", ".join(f"'{m}'" for m in months)
    con.execute(
        f"""
        create or replace temp table contagion_events as
        with legs as (
            select *,
                sum(greatest(coalesce(arr_delay, 0), 0)) over (
                    partition by chain_id order by leg_seq
                    rows between 1 following and unbounded following
                ) as downstream_min
            from marts.fct_legs
            where flight_month in ({month_list})
        ),
        starts as (
            select flight_id, carrier, origin, flight_date, dep_hour_local, dep_period,
                   least(legs_remaining, 4) as rem, dep_delay as x, downstream_min as s
            from legs
            where legs_remaining >= 1
              and (leg_seq = 1 or inbound_arr_delay <= 0)
              and dep_delay between -60 and 600
        ),
        controls as (
            select carrier, flight_date, dep_period, rem, avg(s) as baseline, count(*) as n_ctrl
            from starts where x <= 0 group by all
        )
        select st.*, c.baseline, c.n_ctrl, st.s - c.baseline as excess
        from starts st join controls c using (carrier, flight_date, dep_period, rem)
        where c.n_ctrl >= {MIN_CONTROLS}
        """
    )
    return con.execute("select count(*) from contagion_events").fetchone()[0]


def day_cluster_ratio(
    frame: pd.DataFrame, key: str, n_boot: int = 500, seed: int = 0, level: float = 0.95
) -> pd.DataFrame:
    """sum(excess) / sum(x) per key with a day-clustered bootstrap CI.

    `frame` has one row per (key, flight_date) with columns excess, x, n.
    """
    keys = sorted(frame[key].unique())
    days = sorted(frame["flight_date"].unique())
    ki = frame[key].map({k: i for i, k in enumerate(keys)}).to_numpy()
    di = frame["flight_date"].map({d: i for i, d in enumerate(days)}).to_numpy()
    shape = (len(days), len(keys))
    excess, x, n = (np.zeros(shape) for _ in range(3))
    np.add.at(excess, (di, ki), frame["excess"].to_numpy())
    np.add.at(x, (di, ki), frame["x"].to_numpy())
    np.add.at(n, (di, ki), frame["n"].to_numpy())

    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(days), np.full(len(days), 1 / len(days)), size=n_boot)
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = (w @ excess) / (w @ x)
    lo, hi = np.nanquantile(draws, [(1 - level) / 2, (1 + level) / 2], axis=0)
    return pd.DataFrame(
        {
            key: keys,
            "n_events": n.sum(0).astype(int),
            "primary_min": x.sum(0),
            "excess_min": excess.sum(0),
            "multiplier": excess.sum(0) / x.sum(0),
            "multiplier_lo": lo,
            "multiplier_hi": hi,
        }
    )


def _by_day(con: duckdb.DuckDBPyConnection, key_sql: str, where: str) -> pd.DataFrame:
    return con.execute(
        f"""
        select {key_sql} as k, flight_date, sum(excess) as excess, sum(x) as x, count(*) as n
        from contagion_events where {where} group by all
        """
    ).df()


def multiplier_curve(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Excess downstream minutes by primary-delay bin (the contagion dose-response)."""
    edges = X_BINS
    case = " ".join(
        f"when x >= {lo} and x < {hi} then '{lo}-{hi - 1}'" for lo, hi in pairwise(edges)
    )
    df = _by_day(con, f"case {case} end", f"x >= {edges[0]}")
    out = day_cluster_ratio(df, "k").rename(columns={"k": "x_bin"})
    means = con.execute(
        f"select case {case} end as x_bin, avg(x) as mean_x, avg(excess) as mean_excess, "
        f"avg(s) as mean_downstream, avg(baseline) as mean_baseline "
        f"from contagion_events where x >= {edges[0]} group by 1"
    ).df()
    out = out.merge(means, on="x_bin")
    return out.sort_values("mean_x").reset_index(drop=True)


def multiplier_by(con: duckdb.DuckDBPyConnection, key: str, min_events: int = 0) -> pd.DataFrame:
    df = _by_day(con, key, f"x >= {PRIMARY_MIN}")
    out = day_cluster_ratio(df, "k").rename(columns={"k": key})
    return out[out["n_events"] >= min_events].reset_index(drop=True)


def overall(con: duckdb.DuckDBPyConnection) -> dict[str, float]:
    df = _by_day(con, "'all'", f"x >= {PRIMARY_MIN}")
    row = day_cluster_ratio(df, "k").iloc[0]
    coverage = con.execute(
        f"select count(*) filter (where x >= {PRIMARY_MIN}) from contagion_events"
    ).fetchone()[0]
    return {
        "multiplier": float(row["multiplier"]),
        "multiplier_lo": float(row["multiplier_lo"]),
        "multiplier_hi": float(row["multiplier_hi"]),
        "n_events": int(coverage),
        "primary_min": float(row["primary_min"]),
        "excess_min": float(row["excess_min"]),
    }
