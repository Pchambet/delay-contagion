"""The key numbers, formatted once.

The README, the report page and a test all read these strings, so a published number can
only come from `results/`.
"""

from __future__ import annotations

import json

import pandas as pd

from delay_contagion.descriptive import CARRIER_NAMES
from delay_contagion.paths import FIXTURE_FLIGHTS, RESULTS

# Carriers shown in the README hinge table (largest networks first).
TABLE_CARRIERS = ["WN", "DL", "OO", "AA", "UA", "YX"]


def _month(m: str) -> str:
    return pd.Timestamp(f"{m}-01").strftime("%b %Y")


def _name(code: str) -> str:
    return CARRIER_NAMES.get(code, code)


def _range_with_names(values: pd.Series, fmt: str) -> str:
    """'47 (ORD) to 57 (ATL, SEA)': extremes with every group that attains them."""
    lo, hi = values.min(), values.max()

    def at(v: float) -> str:
        return ", ".join(sorted(values.index[values == v]))

    return f"{fmt.format(lo)} ({at(lo)}) to {fmt.format(hi)} ({at(hi)})"


def carrier_table_md() -> str:
    """The README hinge table, rendered from `results/hinge_carriers.csv`."""
    h = pd.read_csv(RESULTS / "hinge_carriers.csv").set_index("group")
    lines = [
        (
            "| Carrier | τ (min, 95% CI) | β (95% CI) | Held-out MAE: constant / linear / "
            "linear + slack / boosted trees / hinge |"
        ),
        "|---|---|---|---|",
    ]
    for c in TABLE_CARRIERS:
        r = h.loc[c]
        lines.append(
            f"| {_name(c)} | {r.tau:.0f} [{r.tau_lo:.0f}, {r.tau_hi:.0f}] | "
            f"{r.beta:.2f} [{r.beta_lo:.2f}, {r.beta_hi:.2f}] | "
            f"{r.mae_constant:.1f} / {r.mae_linear:.1f} / {r.mae_linear_slack:.1f} / "
            f"{r.mae_gbm:.1f} / **{r.mae_hinge:.1f}** |"
        )
    return "\n".join(lines)


def _pct(x: float) -> str:
    return f"{x:.1%}"


POLICY_LABELS = {
    "LP-optimised": "LP-optimised",
    "LP, 5-min steps": "LP rounded to 5-min steps",
    "Marginal greedy": "Marginal-value greedy",
    "Greedy": "Greedy: most-delayed turns",
    "Uniform": "Uniform padding (fractional)",
    "Uniform, 5-min steps": "Uniform, 5 min on random cells",
}


def policy_table_md() -> str:
    """The README policy table at the reference budget, from `results/`."""
    budget = json.loads((RESULTS / "summary.json").read_text())["buffer"]["reference_budget"]
    f = pd.read_csv(RESULTS / "buffer_frontier.csv")
    f = f[f.budget == budget].set_index("policy")
    sched = pd.read_csv(RESULTS / "buffer_schedule_metrics.csv").set_index("policy")
    lines = [
        (
            "| Policy | Buffer used (min/day) | Delay avoided (min/day, 95% CI) "
            "| Per buffer minute | On time (A15) | Clock lateness (min/day) |"
        ),
        "|---|---|---|---|---|---|",
        f"| No buffer | 0 | 0 | | {sched.loc['No buffer', 'ontime15']:.1%} | 0 |",
    ]
    for p, label in POLICY_LABELS.items():
        r = f.loc[p]
        lines.append(
            f"| {label} | {r.buffer_test:,.0f} | {r.avoided_test:,.0f} "
            f"[{r.avoided_test_lo:,.0f}, {r.avoided_test_hi:,.0f}] | "
            f"**{r.avoided_per_buffer_min:.2f}** | {sched.loc[p, 'ontime15']:.1%} | "
            f"{sched.loc[p, 'clock_late_change_per_day']:+,.0f} |"
        )
    return "\n".join(lines)


def robustness_table_md() -> str:
    """The README contagion robustness table, from `results/contagion_robustness.csv`."""
    df = pd.read_csv(RESULTS / "contagion_robustness.csv")
    lines = [
        "| Matching design | Events (15+ min) | Multiplier (95% CI) | Slips of 1-14 min |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {r.variant} | {r.n_events:,} | {r.multiplier:.2f} [{r.multiplier_lo:.2f}, "
        f"{r.multiplier_hi:.2f}] | {r.small_slip_multiplier:.2f} |"
        for r in df.itertuples()
    ]
    return "\n".join(lines)


def sensitivity_table_md() -> str:
    """The README evaluation-sensitivity table, from `results/buffer_sensitivity.csv`."""
    df = pd.read_csv(RESULTS / "buffer_sensitivity.csv")
    lp = df[df.policy == "LP-optimised"].set_index(["tau_eval", "beta_eval"])
    uni = df[df.policy == "Uniform"].set_index(["tau_eval", "beta_eval"])
    lines = [
        (
            "| Replay τ (min) | Replay β | LP avoided (min/day) | Uniform avoided (min/day) "
            "| LP / uniform per buffer minute |"
        ),
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {t:.0f} | {b:.2f} | {lp.loc[(t, b), 'avoided_test']:,.0f} | "
        f"{uni.loc[(t, b), 'avoided_test']:,.0f} | {lp.loc[(t, b), 'per_min_vs_uniform']:.2f}× |"
        for t, b in lp.index
    ]
    return "\n".join(lines)


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def compute() -> dict[str, str]:
    s = json.loads((RESULTS / "summary.json").read_text())
    net, prop, con, buf = s["network"], s["propagation"], s["contagion"], s["buffer"]
    carrier = buf["carrier"]
    hinge = pd.read_csv(RESULTS / "hinge_carriers.csv").set_index("group")
    hubs = pd.read_csv(RESULTS / "hinge_airports.csv").set_index("group")
    lpc = hinge.loc[carrier]
    ref = buf["reference"]
    lp, uni, greedy = ref["LP-optimised"], ref["Uniform"], ref["Greedy"]
    marg, rounded = ref["Marginal greedy"], ref["LP, 5-min steps"]
    uni_step = ref["Uniform, 5-min steps"]
    spreaders = pd.read_csv(RESULTS / "superspreaders.csv")
    top = spreaders.iloc[0]
    top3 = spreaders.head(3)
    # Is the leader's interval clear of the runner-up's? If not, the README names a group.
    top_separated = top.excess_over_mix_lo > spreaders.iloc[1].excess_over_mix_hi
    top_raw = spreaders.sort_values("multiplier", ascending=False).iloc[0]
    by_hour = pd.read_csv(RESULTS / "reactionary_by_hour.csv").set_index("dep_hour_local")
    peak_hour = int(by_hour.loc[5:23, "reactionary_per_flight"].idxmax())
    mult_hour = pd.read_csv(RESULTS / "contagion_by_hour.csv").set_index("dep_hour_local")
    mult_hour = mult_hour.loc[5:21, "multiplier"]
    sizes = pd.read_csv(RESULTS / "buffer_scenario_sizes.csv")
    size_mean = sizes.groupby("scenario_days")[["gain_train", "gain_test"]].mean()
    by_carrier = pd.read_csv(RESULTS / "contagion_by_carrier.csv").sort_values("multiplier")
    lo_c, hi_c = by_carrier.iloc[0], by_carrier.iloc[-1]
    lpc_mult = by_carrier.set_index("carrier").loc[carrier]
    causes = pd.read_csv(RESULTS / "cause_shares_by_carrier.csv").set_index("carrier")
    shares = causes["reactionary_share"]
    rot = pd.read_csv(RESULTS / "carrier_rotations.csv").set_index("carrier")
    legs_rank = int(rot["legs_per_aircraft_day"].rank(ascending=False, method="min")[carrier])
    shortest_turn = rot["median_sched_turn_min"].min()
    shortest_turn_by = _and(
        sorted(_name(c) for c in rot.index[rot["median_sched_turn_min"] == shortest_turn])
    )
    sched = pd.read_csv(RESULTS / "buffer_schedule_metrics.csv").set_index("policy")
    sens = pd.read_csv(RESULTS / "buffer_sensitivity.csv")
    sens_lp = sens[sens.policy == "LP-optimised"]
    robust = pd.read_csv(RESULTS / "contagion_robustness.csv").set_index("variant")
    headline_v, origin_v, exact_v, coded_v = robust.index
    curve = pd.read_csv(RESULTS / "contagion_curve.csv").set_index("x_bin")
    coded = con["late_aircraft_coded"]
    mae = prop["mae"]
    o = con["overall"]
    fixture_flights = sum(1 for _ in FIXTURE_FLIGHTS.open()) - 1
    return {
        "window": f"{_month(net['first_month'])} to {_month(net['last_month'])}",
        "train_window": f"{_month(prop['train_months'][0])} to {_month(prop['train_months'][-1])}",
        "test_window": f"{_month(prop['test_months'][0])} to {_month(prop['test_months'][-1])}",
        "flights": f"{net['flights'] / 1e6:.2f} million",
        "turns": f"{net['turns'] / 1e6:.2f} million",
        "fixture_flights": f"{fixture_flights:,}",
        # Descriptive
        "reactionary_share": f"{net['reactionary_share']:.0%}",
        "reactionary_6am": f"{by_hour.loc[6, 'reactionary_per_flight']:.1f}",
        "reactionary_peak": f"{by_hour.loc[peak_hour, 'reactionary_per_flight']:.1f}",
        "reactionary_peak_hour": f"{peak_hour}:00",
        "cause_share_range": (
            f"{shares.min():.0%} ({_name(shares.idxmin())}) to "
            f"{shares.max():.0%} ({_name(shares.idxmax())})"
        ),
        "lp_carrier_legs_per_day": f"{rot.loc[carrier, 'legs_per_aircraft_day']:.1f}",
        "lp_carrier_legs_rank": "the most" if legs_rank == 1 else f"rank {legs_rank} in",
        "lp_carrier_median_turn": f"{rot.loc[carrier, 'median_sched_turn_min']:.0f}",
        "shortest_median_turn": f"{shortest_turn:.0f} min for {shortest_turn_by}",
        # Propagation
        "lp_carrier_name": _name(carrier),
        "lp_carrier_tau": f"{lpc.tau:.0f}",
        "lp_carrier_beta": f"{lpc.beta:.2f}",
        "tau_range": f"{prop['tau_range'][0]:.0f} to {prop['tau_range'][1]:.0f}",
        "beta_range": f"{prop['beta_range'][0]:.2f} to {prop['beta_range'][1]:.2f}",
        "hub_tau_range": _range_with_names(hubs["tau"], "{:.0f}"),
        "n_hubs": f"{len(hubs)}",
        "mae_hinge": f"{mae['hinge']:.2f}",
        "mae_linear": f"{mae['linear']:.2f}",
        "mae_linear_slack": f"{mae['linear_slack']:.2f}",
        "mae_hinge_tau0": f"{mae['hinge_tau0']:.2f}",
        "mae_gbm": f"{mae['gbm']:.2f}",
        "mae_constant": f"{mae['constant']:.2f}",
        "n_carriers_test": f"{prop['n_carriers']}",
        "carriers_without_test": _and([_name(c) for c in prop["carriers_without_test"]]),
        "gbm_wins": (
            f"{prop['n_carriers'] - prop['carriers_hinge_beats_gbm_mae']} of {prop['n_carriers']}"
        ),
        "hinge_share_of_gbm": (
            f"{(mae['linear_slack'] - mae['hinge']) / (mae['linear_slack'] - mae['gbm']):.0%}"
        ),
        "filter_excluded": _pct(prop["test_filter_excluded_share"]),
        "spirit_test_turns": f"{int(hinge.loc['NK', 'n_test']):,}" if "NK" in hinge.index else "0",
        # Contagion
        "multiplier": f"{o['multiplier']:.2f}",
        "multiplier_ci": f"[{o['multiplier_lo']:.2f}, {o['multiplier_hi']:.2f}]",
        "events": f"{o['n_events']:,}",
        "coded_share_events": f"{coded['share_events']:.0%}",
        "coded_share_min": f"{coded['share_primary_min']:.0%}",
        "mult_origin": f"{robust.loc[origin_v, 'multiplier']:.2f}",
        "mult_origin_events": f"{robust.loc[origin_v, 'n_events']:,}",
        "mult_exact_legs": f"{robust.loc[exact_v, 'multiplier']:.2f}",
        "mult_no_coded": f"{robust.loc[coded_v, 'multiplier']:.2f}",
        "small_slip_mult": f"{robust.loc[headline_v, 'small_slip_multiplier']:.2f}",
        "small_slip_origin": f"{robust.loc[origin_v, 'small_slip_multiplier']:.2f}",
        "small_bin_mult": f"{curve.loc['1-14', 'multiplier']:.2f}",
        "mult_hour_5": f"{mult_hour.loc[5]:.2f}",
        "mult_hour_peak": f"{mult_hour.max():.2f}",
        "mult_hour_peak_at": f"{int(mult_hour.idxmax())}:00",
        "mult_hour_21": f"{mult_hour.loc[21]:.2f}",
        "top_spreaders": _and([f"{r.origin} ({r.city})" for r in top3.itertuples()]),
        "top_spreaders_range": (
            f"{top3.excess_over_mix.min():+.2f} to {top3.excess_over_mix.max():+.2f}"
        ),
        "top_spreaders_ci_note": (
            "the leader's 95% interval is clear of the runner-up's"
            if top_separated
            else "their 95% intervals overlap, so no single airport is singled out"
        ),
        "top_raw_spreader": f"{top_raw.origin} ({top_raw.city})",
        "top_raw_multiplier": f"{top_raw.multiplier:.2f}",
        "top_raw_mix": f"{top_raw.carrier_mix:.2f}",
        "carrier_mult_low": f"{lo_c.multiplier:.2f} ({_name(lo_c.carrier)})",
        "carrier_mult_high": f"{hi_c.multiplier:.2f} ({_name(hi_c.carrier)})",
        "lp_carrier_multiplier": f"{lpc_mult.multiplier:.2f}",
        "min_events": f"{s['config']['min_events_airport']:,}",
        # Buffer decision
        "budget": f"{buf['reference_budget']:,.0f}",
        "turns_per_day": f"{buf['turns_per_day_train']:,.0f}",
        "turns_per_day_test": f"{buf['turns_per_day_test']:,.0f}",
        "uniform_seconds": f"{buf['reference_budget'] / buf['turns_per_day_train'] * 60:.0f}",
        "lp_spend": f"{lp['buffer_test']:,.0f}",
        "uniform_spend": f"{uni['buffer_test']:,.0f}",
        "rounded_spend": f"{rounded['buffer_test']:,.0f}",
        "lp_avoided": f"{lp['avoided_test']:,.0f}",
        "lp_avoided_ci": f"[{lp['avoided_test_lo']:,.0f}, {lp['avoided_test_hi']:,.0f}]",
        "uniform_at_lp_spend": f"{buf['uniform_at_lp_spend']:,.0f}",
        "lp_vs_uniform": (f"{lp['avoided_per_buffer_min'] / uni['avoided_per_buffer_min']:.1f}×"),
        "lp_vs_marginal_pct": (
            f"{lp['avoided_per_buffer_min'] / marg['avoided_per_buffer_min'] - 1:.0%}"
        ),
        "lp_gain": f"{lp['gain_vs_uniform']:,.0f}",
        "lp_gain_ci": f"[{lp['gain_vs_uniform_lo']:,.0f}, {lp['gain_vs_uniform_hi']:,.0f}]",
        "greedy_gain": f"{greedy['gain_vs_uniform']:+,.0f}",
        "greedy_gain_ci": (
            f"[{greedy['gain_vs_uniform_lo']:+,.0f}, {greedy['gain_vs_uniform_hi']:+,.0f}]"
        ),
        "lp_per_min": f"{lp['avoided_per_buffer_min']:.2f}",
        "uniform_per_min": f"{uni['avoided_per_buffer_min']:.2f}",
        "greedy_per_min": f"{greedy['avoided_per_buffer_min']:.2f}",
        "marginal_per_min": f"{marg['avoided_per_buffer_min']:.2f}",
        "rounded_per_min": f"{rounded['avoided_per_buffer_min']:.2f}",
        "uniform_step_per_min": f"{uni_step['avoided_per_buffer_min']:.2f}",
        "lp_share_of_baseline": f"{lp['avoided_test'] / buf['baseline_delay_per_day_test']:.1%}",
        "lp_cells": f"{lp['cells_padded']:.0f}",
        "cells": f"{buf['cells']:,}",
        "scenario_days": f"{buf['scenario_days']}",
        "test_days": f"{buf['test_days']}",
        "test_weeks": f"{buf['test_weeks']}",
        "lp_size": f"{buf['lp_rows']:,} constraints, {buf['lp_cols']:,} variables",
        "baseline_delay": f"{buf['baseline_delay_per_day_test']:,.0f}",
        # On the clock and on the DOT on-time measure
        "ontime_base": _pct(sched.loc["No buffer", "ontime15"]),
        "ontime_lp": _pct(sched.loc["LP-optimised", "ontime15"]),
        "ontime_uniform": _pct(sched.loc["Uniform", "ontime15"]),
        "ontime_marginal": _pct(sched.loc["Marginal greedy", "ontime15"]),
        "ontime_uniform_step": _pct(sched.loc["Uniform, 5-min steps", "ontime15"]),
        "ontime_rounded": _pct(sched.loc["LP, 5-min steps", "ontime15"]),
        "clock_base": f"{sched.loc['No buffer', 'clock_late_per_day']:,.0f}",
        "clock_lp": f"{sched.loc['LP-optimised', 'clock_late_change_per_day']:+,.0f}",
        "clock_uniform": f"{sched.loc['Uniform', 'clock_late_change_per_day']:+,.0f}",
        # Evaluation-model sensitivity (same buffers, other tau/beta in the replay)
        "sens_ratio_range": (
            f"{sens_lp.per_min_vs_uniform.min():.1f}× to {sens_lp.per_min_vs_uniform.max():.1f}×"
        ),
        "sens_lp_range": f"{sens_lp.avoided_test.min():,.0f} to {sens_lp.avoided_test.max():,.0f}",
        # Scenario-size study
        "size_small": f"{size_mean.index[0]}",
        "size_full": f"{size_mean.index[-1]}",
        "size_train_small": f"{size_mean.gain_train.iloc[0]:.0%}",
        "size_train_large": f"{size_mean.gain_train.iloc[-1]:.0%}",
        "size_test_small": f"{size_mean.gain_test.iloc[0]:.0%}",
        "size_test_90": f"{size_mean.gain_test.loc[90]:.0%}",
        "size_test_prev": f"{size_mean.gain_test.iloc[-2]:.0%}",
        "size_prev": f"{size_mean.index[-2]}",
        "size_test_large": f"{size_mean.gain_test.iloc[-1]:.0%}",
    }
