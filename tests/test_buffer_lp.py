"""Buffer LP: hand-computed toy optimum, exact replay of observed delays, policy rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from delay_contagion.buffer_lp import (
    BufferLP,
    daily_delay,
    from_legs,
    greedy_policy,
    marginal_value_score,
    round_to_step,
    schedule_shift,
    simulate,
    uniform_policy,
    uniform_step_policy,
)

TAU, BETA = 30.0, 1.0


def toy_chain() -> pd.DataFrame:
    """One aircraft, three legs, one day.

    Leg 1 leaves 60 min late and arrives 60 late. Turn 1->2 has 40 min of slack, so
    z2 = 60 - 40 + 30 = 50 min propagate. Turn 2->3 has 45 min: z3 = 50 - 45 + 30 = 35.
    Total arrival delay with no buffer: 60 + 50 + 35 = 145.
    """
    return pd.DataFrame(
        {
            "chain_id": ["c"] * 3,
            "leg_seq": [1, 2, 3],
            "flight_date": ["2026-01-01"] * 3,
            "origin": ["AAA", "BBB", "CCC"],
            "dep_hour_local": [8, 10, 12],
            "sched_turn_min": [np.nan, 40, 45],
            "dep_delay": [60.0, 50.0, 35.0],
            "arr_delay": [60.0, 50.0, 35.0],
        }
    )


CELLS = {("BBB", 10): 0, ("CCC", 12): 1}


def test_replay_reproduces_observed_delays():
    sc = from_legs(toy_chain(), CELLS, TAU, BETA)
    assert simulate(sc, np.zeros(2), TAU, BETA) == pytest.approx(sc.obs_arr)
    assert daily_delay(sc, sc.obs_arr)[0] == pytest.approx(145)


def test_lp_matches_hand_computed_optimum():
    # A minute on turn 2 removes delay on legs 2 and 3 (worth 2); on turn 3 only on leg 3
    # (worth 1). With B = 20 and b_max = 15: b = (15, 5), delays 60 + 35 + 15 = 110.
    sc = from_legs(toy_chain(), CELLS, TAU, BETA)
    lp = BufferLP(sc, n_cells=2, tau=TAU, beta=BETA, b_max=15)
    b, objective = lp.solve(20)
    assert b == pytest.approx([15, 5], abs=1e-6)
    assert objective == pytest.approx(110)
    assert daily_delay(sc, simulate(sc, b, TAU, BETA))[0] == pytest.approx(110)
    # Warm re-solve with a larger budget: turn 3 also saturates -> 60 + 35 + 5 = 100.
    b, objective = lp.solve(100)
    assert b == pytest.approx([15, 15], abs=1e-6)
    assert objective == pytest.approx(100)


@pytest.mark.parametrize("solver", ["simplex", "ipm"])
def test_lp_solvers_agree(solver):
    sc = from_legs(toy_chain(), CELLS, TAU, BETA)
    b, objective = BufferLP(sc, 2, TAU, BETA, b_max=15, solver=solver).solve(20)
    assert b == pytest.approx([15, 5], abs=1e-6)
    assert objective == pytest.approx(110)


def test_partial_propagation_slope():
    # With beta = 0.5 only half of the excess crosses each turn.
    sc = from_legs(toy_chain(), CELLS, TAU, 0.5)
    b, _ = BufferLP(sc, 2, TAU, 0.5, b_max=15).solve(0)
    assert b == pytest.approx([0, 0])
    assert simulate(sc, b, TAU, 0.5) == pytest.approx(sc.obs_arr)


def test_uniform_and_greedy_rules():
    nbar = np.array([10.0, 5.0, 0.0])
    assert uniform_policy(nbar, 30, b_max=15).tolist() == [2.0, 2.0, 0.0]
    # Greedy fills the worst cell (index 1) to b_max (75 min), then 25 min over 10 turns.
    b = greedy_policy(np.array([1.0, 3.0, 9.0]), nbar, 100, b_max=15)
    assert b.tolist() == [2.5, 15.0, 0.0]


def test_marginal_value_by_hand():
    # b_max = 15 on turn 2 alone: z2 = 35, z3 = 20, total 115, so 30 min avoided for 15 buffer
    # minutes (2 per minute). On turn 3 alone: z3 = 20, 15 min avoided (1 per minute).
    sc = from_legs(toy_chain(), CELLS, TAU, BETA)
    score = marginal_value_score(sc, np.array([1.0, 1.0]), TAU, BETA, b_max=15)
    assert score == pytest.approx([2.0, 1.0])


def test_schedule_shift_and_rounding():
    sc = from_legs(toy_chain(), CELLS, TAU, BETA)
    # A buffer moves its own departure and every later leg of the chain.
    assert schedule_shift(sc, np.array([15.0, 5.0])).tolist() == [0.0, 15.0, 20.0]
    assert round_to_step(np.array([17.57, 2.4, 12.5])).tolist() == [20.0, 0.0, 10.0]


def test_uniform_step_rule_spends_whole_steps():
    nbar = np.array([2.0, 1.0, 4.0, 0.0, 3.0])
    b = uniform_step_policy(nbar, 30, step=5, seed=1)
    assert set(b.tolist()) <= {0.0, 5.0}
    assert b[3] == 0  # a cell with no turns is never padded
    spent = float(b @ nbar)
    assert spent <= 30
    # Every unpadded cell with turns would overshoot what is left of the budget.
    left = 30 - spent
    assert all(5 * nbar[c] > left for c in np.flatnonzero((b == 0) & (nbar > 0)))
