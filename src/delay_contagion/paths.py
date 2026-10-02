"""Filesystem layout and shared run settings.

Everything is resolved from the repository root so the CLI behaves the same whether it is
called from the root, from `dbt/`, or from a test.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("DC_ROOT", Path(__file__).resolve().parents[2]))

DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
FLIGHTS_DIR = INTERIM / "flights"
WAREHOUSE = Path(os.environ.get("DC_WAREHOUSE", DATA / "warehouse.duckdb"))

DBT_DIR = ROOT / "dbt"
FIXTURE_FLIGHTS = DBT_DIR / "fixtures" / "flights_fixture.csv"
AIRPORT_SEED = DBT_DIR / "seeds" / "airports.csv"

RESULTS = ROOT / "results"
FIGURES = ROOT / "docs" / "figures"
SITE = ROOT / "site"

# Resource caps: the pipeline must coexist with other work on a laptop.
DUCKDB_MEMORY_LIMIT = "3GB"
DUCKDB_THREADS = 3
