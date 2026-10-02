"""dbt models on a hand-made schedule: UTC conversion (DST, overnight, date line) and
rotation chaining. Expected values are worked out by hand in the comments."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from delay_contagion.warehouse import connect, run_dbt

TOY = Path(__file__).parent / "fixtures" / "toy_schedule.csv"


@pytest.fixture(scope="module")
def con(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dbt")
    warehouse = tmp / "toy.duckdb"
    run_dbt(["build"], flights=TOY, warehouse=warehouse, artifacts_dir=tmp)
    c = connect(warehouse)
    yield c
    c.close()


def _utc(con, flight_number: int) -> tuple[datetime, datetime]:
    return con.execute(
        "select crs_dep_utc, crs_arr_utc from staging.stg_flights where flight_number = ?",
        [flight_number],
    ).fetchone()


@pytest.mark.parametrize(
    ("flight", "dep", "arr"),
    [
        # Red-eye LAX 23:30 PDT -> JFK 07:55 EDT next morning.
        (1, datetime(2026, 7, 2, 6, 30), datetime(2026, 7, 2, 11, 55)),
        # Night of spring-forward: JFK 01:30 EST, MIA 04:30 EDT; the clock gap is 3 h, the
        # block time 2 h.
        (2, datetime(2026, 3, 8, 6, 30), datetime(2026, 3, 8, 8, 30)),
        # Night of fall-back: ORD 00:30 CDT, LGA 02:45 EST (after the 02:00 EDT rollback).
        (3, datetime(2025, 11, 2, 5, 30), datetime(2025, 11, 2, 7, 45)),
        # HNL 22:00 HST -> LAX 06:00 PDT next day.
        (4, datetime(2026, 7, 2, 8, 0), datetime(2026, 7, 2, 13, 0)),
        # Date line: GUM 23:00 ChST -> HNL 11:00 HST the *same* calendar day, earlier on the
        # clock than departure. A naive "arrival < departure => next day" rule gets this wrong.
        (5, datetime(2026, 7, 1, 13, 0), datetime(2026, 7, 1, 21, 0)),
        # "2400" departure = midnight ending the day; Phoenix has no DST (MST all year).
        (6, datetime(2026, 7, 2, 7, 0), datetime(2026, 7, 2, 10, 30)),
        # 02:30 does not exist in Atlanta on 2026-03-08: read with the pre-transition offset.
        (7, datetime(2026, 3, 8, 7, 30), datetime(2026, 3, 8, 8, 0)),
    ],
)
def test_local_times_convert_to_utc(con, flight, dep, arr):
    assert _utc(con, flight) == (dep, arr)


def test_block_time_reproduced_exactly(con):
    worst = con.execute(
        "select max(abs(block_time_residual_min)) from staging.stg_flights"
    ).fetchone()[0]
    assert worst == 0


def test_cancelled_flag_kept_in_staging_and_dropped_from_sequences(con):
    assert con.execute(
        "select is_cancelled from staging.stg_flights where flight_number = 103"
    ).fetchone() == (True,)
    assert con.execute(
        "select count(*) from marts.fct_legs where flight_id like 'WN-103-%'"
    ).fetchone() == (0,)


def test_rotation_chain_for_tail_n100(con):
    rows = con.execute(
        """
        select split_part(flight_id, '-', 2)::int, leg_seq, chain_legs, legs_remaining,
               sched_turn_min, inbound_arr_delay
        from marts.fct_legs where tail_number = 'N100' order by crs_dep_utc
        """
    ).fetchall()
    assert rows == [
        # MDW-DEN starts the chain.
        (101, 1, 3, 2, None, None),
        # DEN-PHX: 14:00Z - 13:10Z = 50 min scheduled turn, inbound 25 min late.
        (102, 2, 3, 1, 50, 25.0),
        # 103 is cancelled; PHX-LAS turns off 102: 16:30Z - 15:20Z = 70 min.
        (104, 3, 3, 0, 70, 28.0),
        # LAS-MDW leaves 565 min after 104 lands (> 300): a new chain.
        (105, 1, 1, 0, None, None),
    ]


def test_only_physical_turns_are_linked(con):
    turns = con.execute(
        "select in_flight_id, out_flight_id from marts.fct_turns order by out_flight_id"
    ).fetchall()
    # Not linked: N200 (actual ground time 30 + 0 - 90 < 0, a tail swap), N300 (BWI != DCA),
    # N400 (inbound diverted). Every time-zone test flight is a lone leg.
    assert [(i.split("-")[1], o.split("-")[1]) for i, o in turns] == [("101", "102"), ("102", "104")]


def test_turn_actual_ground_time(con):
    # 102 -> 104: 70 scheduled + 10 outbound delay - 28 inbound delay = 52 minutes.
    assert con.execute(
        "select actual_turn_min, available_turn_min from marts.fct_turns "
        "where out_flight_id like 'WN-104-%'"
    ).fetchone() == (52, 42)
