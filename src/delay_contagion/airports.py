"""Airport reference table (IANA time zone and coordinates) for the dbt seed.

BTS reports scheduled and actual times in *local* time at each airport. Chaining an
aircraft's legs requires a common clock, so every airport needs its IANA zone; the
`airportsdata` package (MIT, derived from OurAirports and FAA data) provides it.
The seed is committed so CI and the dbt build never need the network.
"""

from __future__ import annotations

import csv
from pathlib import Path

import airportsdata

from delay_contagion.paths import AIRPORT_SEED

# BTS covers the 50 states plus territories served by US carriers.
COUNTRIES = {"US", "PR", "VI", "GU", "MP", "AS"}
FIELDS = ["iata", "name", "city", "state", "tz", "lat", "lon"]


def build_seed(path: Path = AIRPORT_SEED) -> int:
    airports = airportsdata.load("IATA")
    rows = sorted(
        (
            {
                "iata": a["iata"],
                "name": a["name"],
                "city": a["city"],
                "state": a["subd"],
                "tz": a["tz"],
                "lat": round(a["lat"], 5),
                "lon": round(a["lon"], 5),
            }
            for a in airports.values()
            if a["country"] in COUNTRIES and a["iata"] and a["tz"]
        ),
        key=lambda r: str(r["iata"]),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
