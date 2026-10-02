"""Descriptive layer: how much delay is reactionary, for whom, and when in the day.

BTS attributes arrival delays of 15+ minutes to five causes. `LateAircraftDelay` is the
reactionary part: the aircraft arrived late from its previous leg. The other four
(carrier, weather, NAS, security) are primary.
"""

from __future__ import annotations

import duckdb
import pandas as pd

CAUSES = ["carrier_min", "weather_min", "nas_min", "security_min", "late_aircraft_min"]
CARRIER_NAMES = {
    "AA": "American", "AS": "Alaska", "B6": "JetBlue", "DL": "Delta", "F9": "Frontier",
    "G4": "Allegiant", "HA": "Hawaiian", "MQ": "Envoy", "NK": "Spirit", "OH": "PSA",
    "OO": "SkyWest", "UA": "United", "WN": "Southwest", "YX": "Republic", "9E": "Endeavor",
    "QX": "Horizon", "ZW": "Air Wisconsin", "PT": "Piedmont", "C5": "CommutAir",
}  # fmt: skip


def cause_shares_by_carrier(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute(
        f"""
        select carrier, sum(flights) as flights, {", ".join(f"sum({c}) as {c}" for c in CAUSES)}
        from marts.agg_carrier_month group by carrier
        """
    ).df()
    total = df[CAUSES].sum(axis=1)
    df["reactionary_share"] = df["late_aircraft_min"] / total
    df["delay_min_per_flight"] = total / df["flights"]
    df["name"] = df["carrier"].map(CARRIER_NAMES).fillna(df["carrier"])
    return df.sort_values("reactionary_share", ascending=False).reset_index(drop=True)


def reactionary_by_hour(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Cause minutes per departing flight by local scheduled departure hour (all carriers)."""
    df = con.execute(
        f"""
        select dep_hour_local, sum(flights) as flights,
               {", ".join(f"sum({c}) as {c}" for c in CAUSES)}
        from marts.agg_delay_causes group by 1 order by 1
        """
    ).df()
    primary = df[["carrier_min", "weather_min", "nas_min", "security_min"]].sum(axis=1)
    df["primary_per_flight"] = primary / df["flights"]
    df["reactionary_per_flight"] = df["late_aircraft_min"] / df["flights"]
    df["reactionary_share"] = df["late_aircraft_min"] / (primary + df["late_aircraft_min"])
    return df


def network_summary(con: duckdb.DuckDBPyConnection) -> dict[str, float]:
    row = con.execute(
        f"""
        select sum(flights), {", ".join(f"sum({c})" for c in CAUSES)},
               min(flight_month), max(flight_month)
        from marts.agg_carrier_month
        """
    ).fetchone()
    flights, *mins, first, last = row
    turns, chains = con.execute(
        "select count(*), count(distinct chain_id) from marts.fct_turns"
    ).fetchone()
    legs = con.execute("select count(*) from marts.fct_legs").fetchone()[0]
    return {
        "flights": int(flights),
        "operated_legs_in_chains": int(legs),
        "turns": int(turns),
        "chains_with_turns": int(chains),
        "first_month": first,
        "last_month": last,
        "reactionary_share": mins[-1] / sum(mins),
        "total_cause_minutes": float(sum(mins)),
    }
