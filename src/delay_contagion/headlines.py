"""The key numbers, formatted once.

The README, the report page and a test all read these strings, so a published number can
only come from `results/summary.json`.
"""

from __future__ import annotations

import json

import pandas as pd

from delay_contagion.paths import RESULTS


def _month(m: str) -> str:
    return pd.Timestamp(f"{m}-01").strftime("%b %Y")


def compute() -> dict[str, str]:
    s = json.loads((RESULTS / "summary.json").read_text())
    net, prop, con, buf = s["network"], s["propagation"], s["contagion"], s["buffer"]
    hinge = pd.read_csv(RESULTS / "hinge_carriers.csv").set_index("group")
    wn = hinge.loc[buf["carrier"]]
    ref = buf["reference"]
    lp, uni, greedy = ref["LP-optimised"], ref["Uniform"], ref["Greedy"]
    top = pd.read_csv(RESULTS / "superspreaders.csv").iloc[0]
    by_hour = pd.read_csv(RESULTS / "reactionary_by_hour.csv").set_index("dep_hour_local")
    peak_hour = int(by_hour.loc[5:23, "reactionary_per_flight"].idxmax())
    sizes = pd.read_csv(RESULTS / "buffer_scenario_sizes.csv")
    size_mean = sizes.groupby("scenario_days")[["gain_train", "gain_test"]].mean()
    mae = prop["mae"]
    o = con["overall"]
    return {
        "window": f"{_month(net['first_month'])} to {_month(net['last_month'])}",
        "train_window": f"{_month(prop['train_months'][0])} to {_month(prop['train_months'][-1])}",
        "test_window": f"{_month(prop['test_months'][0])} to {_month(prop['test_months'][-1])}",
        "flights": f"{net['flights'] / 1e6:.2f} million",
        "turns": f"{net['turns'] / 1e6:.2f} million",
        "reactionary_share": f"{net['reactionary_share']:.0%}",
        "reactionary_6am": f"{by_hour.loc[6, 'reactionary_per_flight']:.1f}",
        "reactionary_peak": f"{by_hour.loc[peak_hour, 'reactionary_per_flight']:.1f}",
        "reactionary_peak_hour": f"{peak_hour}:00",
        "wn_tau": f"{wn.tau:.0f}",
        "wn_tau_ci": f"[{wn.tau_lo:.0f}, {wn.tau_hi:.0f}]",
        "wn_beta": f"{wn.beta:.2f}",
        "wn_beta_ci": f"[{wn.beta_lo:.2f}, {wn.beta_hi:.2f}]",
        "tau_range": f"{prop['tau_range'][0]:.0f} to {prop['tau_range'][1]:.0f}",
        "beta_range": f"{prop['beta_range'][0]:.2f} to {prop['beta_range'][1]:.2f}",
        "mae_hinge": f"{mae['hinge']:.2f}",
        "mae_linear": f"{mae['linear']:.2f}",
        "mae_constant": f"{mae['constant']:.2f}",
        "hinge_wins": f"{prop['carriers_hinge_beats_linear_mae']} of {prop['n_carriers']}",
        "multiplier": f"{o['multiplier']:.2f}",
        "multiplier_ci": f"[{o['multiplier_lo']:.2f}, {o['multiplier_hi']:.2f}]",
        "events": f"{o['n_events']:,}",
        "top_spreader": f"{top.origin} ({top.city})",
        "min_events": f"{s['config']['min_events_airport']:,}",
        "top_spreader_multiplier": f"{top.multiplier:.2f}",
        "budget": f"{buf['reference_budget']:,.0f}",
        "lp_avoided": f"{lp['avoided_test']:,.0f}",
        "lp_avoided_ci": f"[{lp['avoided_test_lo']:,.0f}, {lp['avoided_test_hi']:,.0f}]",
        "uniform_avoided": f"{uni['avoided_test']:,.0f}",
        "greedy_avoided": f"{greedy['avoided_test']:,.0f}",
        "lp_vs_uniform": f"{lp['avoided_test'] / uni['avoided_test']:.1f}×",
        "lp_gain": f"{lp['gain_vs_uniform']:,.0f}",
        "lp_gain_ci": f"[{lp['gain_vs_uniform_lo']:,.0f}, {lp['gain_vs_uniform_hi']:,.0f}]",
        "lp_per_min": f"{lp['avoided_per_buffer_min']:.2f}",
        "uniform_per_min": f"{uni['avoided_per_buffer_min']:.2f}",
        "greedy_per_min": f"{greedy['avoided_per_buffer_min']:.2f}",
        "lp_cells": f"{lp['cells_padded']:.0f}",
        "cells": f"{buf['cells']:,}",
        "scenario_days": f"{buf['scenario_days']}",
        "test_days": f"{buf['test_days']}",
        "lp_size": f"{buf['lp_rows']:,} constraints, {buf['lp_cols']:,} variables",
        "turns_per_day": f"{buf['turns_per_day_train']:,.0f}",
        "size_train_small": f"{size_mean.gain_train.iloc[0]:.0%}",
        "size_train_large": f"{size_mean.gain_train.iloc[-1]:.0%}",
        "size_test_range": f"{size_mean.gain_test.min():.0%} and {size_mean.gain_test.max():.0%}",
        "size_test_large": f"{size_mean.gain_test.iloc[-1]:.0%}",
        "baseline_delay": f"{buf['baseline_delay_per_day_test']:,.0f}",
    }
