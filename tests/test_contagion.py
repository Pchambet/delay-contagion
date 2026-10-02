"""Contagion multiplier: recovers a known pass-through on synthetic chains with day shocks."""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from delay_contagion.contagion import build_events, multiplier_by, overall


def synthetic_legs(true_multiplier: float, seed: int = 3) -> pd.DataFrame:
    """Three-leg chains. Leg 1 is a clean start that leaves x late (x = 0 for half the
    aircraft). Legs 2 and 3 each arrive day_shock + m/2 * x late, so every primary minute
    adds m downstream minutes on top of a day-level shock shared by all aircraft (bad
    weather days), which a naive comparison across days would wrongly attribute to x."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(40):
        date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=d)
        shock = rng.gamma(2, 15)
        for a in range(60):
            x = 0.0 if a % 2 else rng.choice([20.0, 45.0, 90.0])
            chain = f"c{d}-{a}"
            for seq in (1, 2, 3):
                late = x if seq == 1 else shock + true_multiplier / 2 * x + rng.normal(0, 3)
                rows.append(
                    {
                        "flight_id": f"{chain}-{seq}", "chain_id": chain, "leg_seq": seq,
                        "legs_remaining": 3 - seq, "carrier": "ZZ",
                        "origin": "HUB" if a < 30 else "OUT", "flight_date": date.date(),
                        "flight_month": "2026-01", "dep_hour_local": 8 + 2 * seq,
                        "dep_period": "morning", "inbound_arr_delay": None if seq == 1 else late,
                        "dep_delay": late, "arr_delay": late,
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
