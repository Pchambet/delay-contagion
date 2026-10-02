"""Command line: `uv run delay-contagion <step>`.

Steps mirror the Makefile: data -> build -> analyze -> figures -> report.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from delay_contagion import paths


def cmd_data(args: argparse.Namespace) -> None:
    from delay_contagion import airports, ingest

    print(f"airport seed: {airports.build_seed()} airports")
    months = ingest.run(args.months)
    print(f"cached {len(months)} months in {paths.FLIGHTS_DIR}")


def cmd_build(args: argparse.Namespace) -> None:
    from delay_contagion.warehouse import run_dbt

    flights = paths.FIXTURE_FLIGHTS if args.fixture else paths.FLIGHTS_DIR / "*.parquet"
    warehouse = (
        paths.DATA / "fixture.duckdb" if args.fixture and args.warehouse is None else args.warehouse
    )
    t0 = time.perf_counter()
    run_dbt(["build"], flights=flights, warehouse=warehouse or paths.WAREHOUSE)
    print(f"dbt build ok in {time.perf_counter() - t0:.0f}s ({flights})")


def cmd_analyze(args: argparse.Namespace) -> None:
    from delay_contagion import analysis
    from delay_contagion.warehouse import connect

    con = connect(paths.WAREHOUSE)
    try:
        analysis.run(con)
    finally:
        con.close()


def cmd_figures(args: argparse.Namespace) -> None:
    from delay_contagion import figures

    for path in figures.render_all():
        print(f"  {path.relative_to(paths.ROOT)}")


def cmd_report(args: argparse.Namespace) -> None:
    from delay_contagion import report

    print(f"  {report.render().relative_to(paths.ROOT)}")
    print(f"  {report.render_readme().relative_to(paths.ROOT)}")


def cmd_docs(args: argparse.Namespace) -> None:
    from delay_contagion.warehouse import run_dbt

    run_dbt(
        ["docs", "generate"], flights=paths.FIXTURE_FLIGHTS, warehouse=paths.DATA / "fixture.duckdb"
    )
    print(f"dbt docs in {paths.DBT_DIR / 'target'}; browse them with:")
    print("  uv run dbt docs serve --project-dir dbt --profiles-dir dbt")


def cmd_fixture(args: argparse.Namespace) -> None:
    from delay_contagion import fixture

    print(f"fixture: {fixture.write()} flights -> {paths.FIXTURE_FLIGHTS.relative_to(paths.ROOT)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="delay-contagion", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("data", help="download and cache the latest BTS months")
    p.add_argument("--months", type=int, default=12)
    p.set_defaults(func=cmd_data)
    p = sub.add_parser("build", help="dbt build (staging -> intermediate -> marts + tests)")
    p.add_argument("--fixture", action="store_true", help="build on the committed CI fixture")
    p.add_argument("--warehouse", type=Path, default=None)
    p.set_defaults(func=cmd_build)
    sub.add_parser("analyze", help="fit models, run the LP, write results/").set_defaults(
        func=cmd_analyze
    )
    sub.add_parser("figures", help="render docs/figures from results/").set_defaults(
        func=cmd_figures
    )
    sub.add_parser(
        "report", help="render site/index.html and README.md from results/"
    ).set_defaults(func=cmd_report)
    sub.add_parser("docs", help="dbt docs (lineage) on the fixture warehouse").set_defaults(
        func=cmd_docs
    )
    sub.add_parser("fixture", help="regenerate the CI fixture from the warehouse").set_defaults(
        func=cmd_fixture
    )
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
