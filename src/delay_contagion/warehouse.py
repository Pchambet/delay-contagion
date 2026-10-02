"""Run the dbt project and open the resulting DuckDB warehouse.

dbt owns every transformation from raw flights to marts; Python only reads the marts.
dbt runs in a subprocess (the venv's `dbt` executable) so its DuckDB connection is closed
when it exits; the CLI, the Makefile and the tests share this one code path.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import duckdb

from delay_contagion.paths import (
    DBT_DIR,
    DUCKDB_MEMORY_LIMIT,
    DUCKDB_THREADS,
    FLIGHTS_DIR,
    WAREHOUSE,
)


def run_dbt(
    command: Sequence[str] = ("build",),
    flights: str | Path = FLIGHTS_DIR / "*.parquet",
    warehouse: Path = WAREHOUSE,
    artifacts_dir: Path | None = None,
) -> None:
    """Invoke dbt with the flights source and warehouse path passed through env vars."""
    warehouse.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "DC_FLIGHTS": str(flights), "DC_WAREHOUSE": str(warehouse)}
    dbt = Path(sys.executable).with_name("dbt")
    args = [str(dbt), *command, "--project-dir", str(DBT_DIR), "--profiles-dir", str(DBT_DIR)]
    if artifacts_dir is not None:
        args += ["--target-path", str(artifacts_dir / "target"), "--log-path", str(artifacts_dir)]
    result = subprocess.run(args, env=env, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"dbt {' '.join(command)} failed:\n{result.stdout[-4000:]}")


def connect(warehouse: Path = WAREHOUSE, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(warehouse), read_only=read_only)
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'; SET threads={DUCKDB_THREADS}")
    return con
