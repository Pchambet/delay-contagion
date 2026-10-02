"""Download the BTS Reporting Carrier On-Time Performance files and cache them as Parquet.

Source: US DOT Bureau of Transportation Statistics, TranStats "Reporting Carrier On-Time
Performance (1987-present)", public domain US government data.
https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ

The window is the latest N *published* months, discovered with HEAD requests rather than
hard-coded, so re-running the pipeline later moves the window forward. Every step is
idempotent: a month already downloaded (zip) or converted (parquet) is skipped.
"""

from __future__ import annotations

import datetime as dt
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import duckdb

from delay_contagion.paths import DUCKDB_MEMORY_LIMIT, DUCKDB_THREADS, FLIGHTS_DIR, RAW

URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)

# Only the columns the analysis needs; the raw files carry ~110.
COLUMNS = [
    "FlightDate",
    "Reporting_Airline",
    "Tail_Number",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "DepTime",
    "DepDelay",
    "CRSArrTime",
    "ArrTime",
    "ArrDelay",
    "Cancelled",
    "Diverted",
    "CRSElapsedTime",
    "Distance",
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",
]
# hhmm fields keep their leading zeros; everything else is typed by DuckDB.
_VARCHAR = ["Tail_Number", "CRSDepTime", "DepTime", "CRSArrTime", "ArrTime"]

Month = tuple[int, int]


def _url(month: Month) -> str:
    return URL.format(year=month[0], month=month[1])


def previous(month: Month) -> Month:
    year, m = month
    return (year - 1, 12) if m == 1 else (year, m - 1)


def _exists(month: Month, timeout: float = 30.0) -> bool:
    request = urllib.request.Request(_url(month), method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status == 200
    except urllib.error.HTTPError:
        return False


def latest_months(n: int = 12, today: dt.date | None = None, max_lookback: int = 12) -> list[Month]:
    """Return the latest `n` published months, oldest first.

    BTS publishes with a lag of two to three months; we probe backwards from the current
    month until the first file that exists.
    """
    today = today or dt.date.today()
    month: Month = (today.year, today.month)
    for _ in range(max_lookback):
        if _exists(month):
            window = [month]
            while len(window) < n:
                window.append(previous(window[-1]))
            return window[::-1]
        month = previous(month)
    raise RuntimeError("No BTS on-time file found in the last year; is transtats.bts.gov up?")


def zip_path(month: Month) -> Path:
    return RAW / Path(_url(month)).name


def parquet_path(month: Month) -> Path:
    return FLIGHTS_DIR / f"flights_{month[0]}_{month[1]:02d}.parquet"


def download(month: Month) -> Path:
    target = zip_path(month)
    if target.exists() and zipfile.is_zipfile(target):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    with urllib.request.urlopen(_url(month), timeout=600) as response, partial.open("wb") as f:
        shutil.copyfileobj(response, f)
    partial.rename(target)
    return target


def to_parquet(month: Month) -> Path:
    """Extract the CSV from the zip and keep the needed columns as typed Parquet."""
    target = parquet_path(month)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path(month)) as archive, tempfile.TemporaryDirectory() as tmp:
        member = next(n for n in archive.namelist() if n.endswith(".csv"))
        csv_path = Path(archive.extract(member, tmp))
        con = duckdb.connect()
        con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'; SET threads={DUCKDB_THREADS}")
        types = ", ".join(f"'{c}': 'VARCHAR'" for c in _VARCHAR)
        cols = ", ".join(f'"{c}"' for c in COLUMNS)
        con.execute(
            f"""
            COPY (
              SELECT {cols}
              FROM read_csv('{csv_path}', header=true, null_padding=true,
                            types={{{types}}})
            ) TO '{target}' (FORMAT parquet, COMPRESSION zstd)
            """
        )
        con.close()
    return target


def run(n_months: int = 12) -> list[Month]:
    months = latest_months(n_months)
    for month in months:
        download(month)
        to_parquet(month)
        print(f"  {month[0]}-{month[1]:02d}  {parquet_path(month).name}")
    return months
