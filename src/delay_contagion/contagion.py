"""Contagion multiplier: how many downstream delay minutes one primary minute generates.

Event. A leg where the aircraft was ready (first leg of its chain, or inbound arrived on
time) but the flight still left `x` minutes late. The `x` minutes are treated as primary;
that is an approximation (a chain start can follow a tail swap or a broken chain, and some
of these legs carry a late-aircraft cause code), which `robustness` quantifies. Its
*downstream delay* is the sum of positive arrival delays on every later leg of the same
aircraft chain.

Counterfactual. Later legs also pick up delay of their own, unrelated to the event. We
compare each event with *controls* - clean starts that left on time (x <= 0) - from the
same carrier, the same calendar day, the same part of the day, and the same number of
legs left to fly (capped at 4+). Matching on the day absorbs carrier-wide, day-level
conditions (not local weather at one airport); matching on legs remaining absorbs
exposure. Excess downstream delay = downstream - control mean; the multiplier is
sum(excess) / sum(x).

Uncertainty. Events on the same day are correlated, so CIs come from a day-clustered
bootstrap that resamples whole days, shared across groups.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from delay_contagion.warehouse import fetch_one

X_BINS = [1, 15, 30, 60, 120, 240, 601]  # primary delay bins (minutes), right-open
MIN_CONTROLS = 5
PRIMARY_MIN = 15  # "a delayed departure" for the headline multipliers (DOT definition)
MATCH = ("carrier", "flight_date", "dep_period", "rem")


def build_events(
    con: duckdb.DuckDBPyConnection,
    months: list[str],
    match: tuple[str, ...] = MATCH,
    rem_cap: int | None = 4,
    exclude_late_aircraft_coded: bool = False,
) -> int:
    """Materialise the matched event table `contagion_events` and return its size.

    The keyword arguments define the robustness variants: extra match keys (e.g. origin),
    exact legs remaining (`rem_cap=None`), or dropping legs that carry a late-aircraft code.
    """
    month_list = ", ".join(f"'{m}'" for m in months)
    rem = "legs_remaining" if rem_cap is None else f"least(legs_remaining, {rem_cap})"
    keys = ", ".join(match)
    coded = "and coalesce(late_aircraft_delay, 0) = 0" if exclude_late_aircraft_coded else ""
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
                   {rem} as rem, dep_delay as x, downstream_min as s, late_aircraft_delay
            from legs
            where legs_remaining >= 1
              and (leg_seq = 1 or inbound_arr_delay <= 0)
              and dep_delay between -60 and 600 {coded}
        ),
        controls as (
            select {keys}, avg(s) as baseline, count(*) as n_ctrl
            from starts where x <= 0 group by all
        )
        select st.*, c.baseline, c.n_ctrl, st.s - c.baseline as excess
        from starts st join controls c using ({keys})
        where c.n_ctrl >= {MIN_CONTROLS}
        """
    )
    return int(fetch_one(con, "select count(*) from contagion_events")[0])


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


def multiplier_curve(con: duckdb.DuckDBPyConnection, seed: int = 0) -> pd.DataFrame:
    """Excess downstream minutes by primary-delay bin (the contagion dose-response)."""
    edges = X_BINS
    case = " ".join(
        f"when x >= {lo} and x < {hi} then '{lo}-{hi - 1}'" for lo, hi in pairwise(edges)
    )
    df = _by_day(con, f"case {case} end", f"x >= {edges[0]}")
    out = day_cluster_ratio(df, "k", seed=seed).rename(columns={"k": "x_bin"})
    means = con.execute(
        f"select case {case} end as x_bin, avg(x) as mean_x, avg(excess) as mean_excess, "
        f"avg(s) as mean_downstream, avg(baseline) as mean_baseline "
        f"from contagion_events where x >= {edges[0]} group by 1"
    ).df()
    out = out.merge(means, on="x_bin")
    return out.sort_values("mean_x").reset_index(drop=True)


def multiplier_by(
    con: duckdb.DuckDBPyConnection, key: str, min_events: int = 0, seed: int = 0
) -> pd.DataFrame:
    df = _by_day(con, key, f"x >= {PRIMARY_MIN}")
    out = day_cluster_ratio(df, "k", seed=seed).rename(columns={"k": key})
    return out[out["n_events"] >= min_events].reset_index(drop=True)


def carrier_adjusted_by(
    con: duckdb.DuckDBPyConnection,
    key: str,
    carrier_multiplier: pd.DataFrame,
    min_events: int = 0,
    seed: int = 0,
) -> pd.DataFrame:
    """Multiplier of `key` in excess of what its carrier mix predicts.

    Each event's excess is compared with x times its carrier's network multiplier, so an
    airport dominated by a high-multiplier airline is not ranked high for that alone:
    excess_over_mix = sum(excess - m_carrier * x) / sum(x), with a day-clustered CI.
    `carrier_mix` = sum(m_carrier * x) / sum(x) is the part explained by the carrier mix.
    """
    con.register("carrier_m", carrier_multiplier[["carrier", "multiplier"]])
    df = con.execute(
        f"""
        select {key} as k, flight_date, sum(excess - m.multiplier * x) as excess, sum(x) as x,
               count(*) as n, sum(m.multiplier * x) as mix
        from contagion_events e join carrier_m m using (carrier)
        where x >= {PRIMARY_MIN} group by all
        """
    ).df()
    con.unregister("carrier_m")
    out = day_cluster_ratio(df, "k", seed=seed).rename(
        columns={
            "k": key,
            "multiplier": "excess_over_mix",
            "multiplier_lo": "excess_over_mix_lo",
            "multiplier_hi": "excess_over_mix_hi",
        }
    )
    mix = df.groupby("k")[["mix", "x"]].sum()
    out["carrier_mix"] = out[key].map(mix["mix"] / mix["x"])
    out = out.drop(columns=["excess_min"])
    return out[out["n_events"] >= min_events].reset_index(drop=True)


def overall(con: duckdb.DuckDBPyConnection, seed: int = 0) -> dict[str, float]:
    df = _by_day(con, "'all'", f"x >= {PRIMARY_MIN}")
    row = day_cluster_ratio(df, "k", seed=seed).iloc[0]
    coverage = fetch_one(
        con, f"select count(*) filter (where x >= {PRIMARY_MIN}) from contagion_events"
    )[0]
    return {
        "multiplier": float(row["multiplier"]),
        "multiplier_lo": float(row["multiplier_lo"]),
        "multiplier_hi": float(row["multiplier_hi"]),
        "n_events": int(coverage),
        "primary_min": float(row["primary_min"]),
        "excess_min": float(row["excess_min"]),
    }


def late_aircraft_coded(con: duckdb.DuckDBPyConnection) -> dict[str, float]:
    """How many headline events carry a late-aircraft cause code anyway, and their weight."""
    n, share_n, share_min = fetch_one(
        con,
        f"""
        select count(*),
               avg((coalesce(late_aircraft_delay, 0) > 0)::int),
               sum(x) filter (where coalesce(late_aircraft_delay, 0) > 0) / sum(x)
        from contagion_events where x >= {PRIMARY_MIN}
        """,
    )
    return {"events": int(n), "share_events": float(share_n), "share_primary_min": float(share_min)}


ROBUSTNESS: dict[str, dict[str, Any]] = {
    "Headline: carrier x day x part of day x legs left (4+)": {},
    "+ origin airport in the match": {"match": (*MATCH, "origin")},
    "Exact legs left (no 4+ cap)": {"rem_cap": None},
    "Without legs that carry a late-aircraft code": {"exclude_late_aircraft_coded": True},
}


def robustness(con: duckdb.DuckDBPyConnection, months: list[str], seed: int = 0) -> pd.DataFrame:
    """Headline and small-slip multipliers under alternative matching designs.

    Rebuilds `contagion_events` for each variant, so call it last (it leaves the table in
    the state of the final variant).
    """
    rows = []
    for name, kwargs in ROBUSTNESS.items():
        build_events(con, months, **kwargs)
        o = overall(con, seed=seed)
        small = day_cluster_ratio(_by_day(con, "'small'", f"x >= 1 and x < {PRIMARY_MIN}"), "k",
                                  seed=seed).iloc[0]  # fmt: skip
        rows.append(
            {
                "variant": name,
                "n_events": o["n_events"],
                "multiplier": o["multiplier"],
                "multiplier_lo": o["multiplier_lo"],
                "multiplier_hi": o["multiplier_hi"],
                "small_slip_multiplier": small["multiplier"],
                "small_slip_lo": small["multiplier_lo"],
                "small_slip_hi": small["multiplier_hi"],
            }
        )
    return pd.DataFrame(rows)
