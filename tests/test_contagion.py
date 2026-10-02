"""Contagion multiplier: recovers a known pass-through on synthetic chains with day shocks."""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from delay_contagion.contagion import (
    build_events,
    carrier_adjusted_by,
    multiplier_by,
    overall,
)


def synthetic_legs(
    true_multiplier: float, seed: int = 3, carriers: dict[str, float] | None = None
) -> pd.DataFrame:
    """Three-leg chains. Leg 1 is a clean start that leaves x late (x = 0 for half the
    aircraft). Legs 2 and 3 each arrive day_shock + m/2 * x late, so every primary minute
    adds m downstream minutes on top of a day-level shock shared by all aircraft (bad
    weather days), which a naive comparison across days would wrongly attribute to x.

    With `carriers` ({code: m}, two carriers), HUB is flown 80% by the first carrier and OUT
    80% by the second, so the airports differ only through their carrier mix."""
    rng = np.random.default_rng(seed)
    codes = list(carriers or {"ZZ": true_multiplier})
    rows = []
    for d in range(40):
        date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=d)
        shock = rng.gamma(2, 15)
        for a in range(60):
            x = 0.0 if a % 2 else rng.choice([20.0, 45.0, 90.0])
            first = (a < 24) or (30 <= a < 36)
            carrier = codes[0] if first or len(codes) == 1 else codes[1]
            m = carriers[carrier] if carriers else true_multiplier
            chain = f"c{d}-{a}"
            for seq in (1, 2, 3):
                late = x if seq == 1 else shock + m / 2 * x + rng.normal(0, 3)
                rows.append(
                    {
                        "flight_id": f"{chain}-{seq}", "chain_id": chain, "leg_seq": seq,
                        "legs_remaining": 3 - seq, "carrier": carrier,
                        "origin": "HUB" if a < 30 else "OUT", "flight_date": date.date(),
                        "flight_month": "2026-01", "dep_hour_local": 8 + 2 * seq,
                        "dep_period": "morning", "inbound_arr_delay": None if seq == 1 else late,
                        "dep_delay": late, "arr_delay": late, "late_aircraft_delay": None,
                    }
                )  # fmt: skip
    return pd.DataFrame(rows)


@pytest.fixture()
def con():
    c = duckdb.connect()
    c.execute("create schema marts")
    c.register("legs", synthetic_legs(0.8))
    c.execute("create table marts.fct_legs as select * from legs")
    yield c
    c.close()


def test_matched_multiplier_recovers_truth(con):
    assert build_events(con, ["2026-01"]) > 0
    res = overall(con)
    assert res["multiplier"] == pytest.approx(0.8, abs=0.03)
    assert res["multiplier_lo"] < 0.8 < res["multiplier_hi"]


def test_multiplier_by_airport_has_one_row_per_station(con):
    build_events(con, ["2026-01"])
    by_airport = multiplier_by(con, "origin")
    assert sorted(by_airport["origin"]) == ["HUB", "OUT"]
    assert by_airport["multiplier"].to_numpy() == pytest.approx([0.8, 0.8], abs=0.06)


def test_carrier_adjustment_removes_carrier_mix():
    # Carrier multipliers 0.5 and 1.5; HUB is 80% the first, OUT 80% the second. Raw airport
    # multipliers differ (about 0.7 vs 1.3), but neither airport adds anything of its own.
    c = duckdb.connect()
    c.execute("create schema marts")
    c.register("legs", synthetic_legs(0.0, carriers={"AA": 0.5, "BB": 1.5}))
    c.execute("create table marts.fct_legs as select * from legs")
    build_events(c, ["2026-01"])
    raw = multiplier_by(c, "origin").set_index("origin")["multiplier"]
    assert raw["HUB"] == pytest.approx(0.7, abs=0.1)
    assert raw["OUT"] == pytest.approx(1.3, abs=0.1)
    adj = carrier_adjusted_by(c, "origin", multiplier_by(c, "carrier")).set_index("origin")
    assert adj["excess_over_mix"].abs().max() < 0.06
    assert adj.loc["HUB", "carrier_mix"] == pytest.approx(raw["HUB"], abs=0.06)
    c.close()
