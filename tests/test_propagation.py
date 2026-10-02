"""Hinge propagation model: hand-checkable cases and recovery of known parameters."""

from __future__ import annotations

import numpy as np
import pytest

from delay_contagion.propagation import fit_arrays, fit_with_bootstrap, hinge_x, stats_from_arrays


def test_hinge_feature_by_hand():
    # 60 min late into a 45-min turn with a 30-min minimum turn: 60 - (45 - 30) = 45 min
    # cannot be absorbed; 10 min late into the same turn is fully absorbed.
    x = hinge_x(np.array([60.0, 10.0]), np.array([45.0, 45.0]), 30.0)
    assert x.tolist() == [45.0, 0.0]


def test_recovers_known_parameters():
    rng = np.random.default_rng(1)
    n = 40_000
    inbound = rng.exponential(25, n) - 10
    slack = rng.uniform(25, 120, n)
    a, beta, tau = 4.0, 0.85, 37
    out = a + beta * hinge_x(inbound, slack, tau) + rng.normal(0, 8, n)
    fit = fit_arrays(inbound, slack, out)
    assert fit.tau == pytest.approx(tau, abs=2)
    assert fit.beta == pytest.approx(beta, abs=0.03)
    assert fit.a == pytest.approx(a, abs=0.5)


def test_bootstrap_interval_covers_truth():
    rng = np.random.default_rng(7)
    days = []
    for _ in range(60):
        inbound = rng.exponential(30, 500) - 10
        slack = rng.uniform(25, 120, 500)
        out = 2.0 + 0.9 * hinge_x(inbound, slack, 30) + rng.normal(0, 10, 500)
        days.append(stats_from_arrays(inbound, slack, out))
    res = fit_with_bootstrap(np.stack(days), n_boot=200)
    assert res["beta_lo"] <= 0.9 <= res["beta_hi"]
    assert res["tau_lo"] <= 30 <= res["tau_hi"]
    assert res["beta_hi"] - res["beta_lo"] < 0.1
