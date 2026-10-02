"""Held-out uncertainty: whole weeks are resampled together."""

from __future__ import annotations

import datetime as dt

import numpy as np

from delay_contagion.analysis import week_bootstrap, week_index


def test_week_bootstrap_moves_days_in_blocks():
    days = [dt.date(2026, 5, 1) + dt.timedelta(days=i) for i in range(17)]
    week = week_index(days)
    assert week.tolist() == [0] * 7 + [1] * 7 + [2] * 3
    w = week_bootstrap(days, n_boot=200, seed=0)
    assert w.shape == (200, 17)
    for k in range(3):
        block = w[:, week == k]
        assert (block == block[:, :1]).all()  # one weight per week
    # Each draw resamples 3 weeks, so the week counts sum to 3.
    assert (w[:, [0, 7, 14]].sum(1) == 3).all()
    assert np.unique(w).size > 1
