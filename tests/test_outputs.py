"""Published outputs stay in sync with the committed result tables."""

from __future__ import annotations

import re

from delay_contagion import headlines, report
from delay_contagion.paths import ROOT

# Headline numbers the README quotes; each must match results/ exactly.
README_KEYS = [
    "flights",
    "turns",
    "reactionary_share",
    "reactionary_6am",
    "reactionary_peak",
    "wn_tau",
    "wn_tau_ci",
    "wn_beta",
    "tau_range",
    "mae_hinge",
    "mae_linear",
    "mae_constant",
    "hinge_wins",
    "multiplier",
    "multiplier_ci",
    "events",
    "top_spreader_multiplier",
    "carrier_mult_low",
    "carrier_mult_high",
    "wn_multiplier",
    "budget",
    "lp_avoided",
    "lp_avoided_ci",
    "uniform_avoided",
    "greedy_avoided",
    "lp_vs_uniform",
    "lp_gain_ci",
    "lp_per_min",
    "uniform_per_min",
    "greedy_per_min",
    "scenario_days",
    "test_days",
    "size_train_small",
    "size_train_large",
    "size_test_range",
]


def test_readme_quotes_current_results():
    readme = (ROOT / "README.md").read_text()
    h = headlines.compute()
    missing = {k: h[k] for k in README_KEYS if h[k] not in readme}
    assert not missing, f"README numbers out of date: {missing}"


def test_report_renders_from_committed_results(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "SITE", tmp_path)
    page = report.render().read_text()
    assert not re.search(r"\$[a-z_]+", page), "unfilled template placeholder"
    charts = ["chart-hour", "chart-hinge", "chart-contagion", "chart-map", "chart-frontier",
              "chart-sizes"]  # fmt: skip
    for chart in charts:
        assert f'id="{chart}"' in page
        assert f'"{chart}":' in page
    assert len(page) < 1_000_000
