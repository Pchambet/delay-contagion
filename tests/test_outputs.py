"""Published outputs stay in sync with the committed result tables."""

from __future__ import annotations

import re

from delay_contagion import report
from delay_contagion.paths import ROOT


def test_readme_is_rendered_from_current_results():
    # Every number in the README comes from `readme_template.md` + `results/`; a stale or
    # hand-edited number fails here. Fix with `make report`.
    assert (ROOT / "README.md").read_text() == report.readme_text()


def test_report_renders_from_committed_results(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "SITE", tmp_path)
    page = report.render().read_text()
    assert not re.search(r"\$[a-z_]+", page), "unfilled template placeholder"
    charts = ["chart-hour", "chart-hinge", "chart-contagion", "chart-contagion-hour",
              "chart-map", "chart-frontier", "chart-sizes"]  # fmt: skip
    for chart in charts:
        assert f'id="{chart}"' in page
        assert f'"{chart}":' in page
    assert len(page) < 1_000_000
