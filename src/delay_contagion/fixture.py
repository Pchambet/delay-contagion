"""Small real-data fixture for CI: complete aircraft days around the March DST switch.

Whole tails are sampled (every leg they flew on the chosen days, cancellations included) so
that rotation chaining is exercised exactly as on the full data. The sample is
deterministic (md5 of the tail number, stable across DuckDB versions, unlike `hash()`), so
regenerating it yields the same file.
"""

from __future__ import annotations

import duckdb

from delay_contagion.ingest import COLUMNS
from delay_contagion.paths import FIXTURE_FLIGHTS, FLIGHTS_DIR

DAYS = ("2026-03-07", "2026-03-08")  # US clocks spring forward on 2026-03-08
TAILS_PER_CARRIER = 25


def write() -> int:
    con = duckdb.connect()
    cols = ", ".join(f'"{c}"' for c in COLUMNS)
    days = ", ".join(f"DATE '{d}'" for d in DAYS)
    con.execute(
        f"""
        create temp table f as
        select {cols} from read_parquet('{FLIGHTS_DIR}/*.parquet') where FlightDate in ({days})
        """
    )
    FIXTURE_FLIGHTS.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""
        copy (
            with tails as (
                select Reporting_Airline, Tail_Number
                from (select distinct Reporting_Airline, Tail_Number from f
                      where Tail_Number is not null)
                qualify row_number() over (
                    partition by Reporting_Airline order by md5(Tail_Number)
                ) <= {TAILS_PER_CARRIER}
            )
            select f.* from f join tails using (Reporting_Airline, Tail_Number)
            order by Reporting_Airline, Tail_Number, FlightDate, CRSDepTime
        ) to '{FIXTURE_FLIGHTS}' (header, delimiter ',')
        """
    )
    n = con.execute(f"select count(*) from read_csv('{FIXTURE_FLIGHTS}')").fetchone()[0]
    con.close()
    return n
