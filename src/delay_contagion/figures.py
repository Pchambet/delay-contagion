"""Static figures (PNG) for the README, rendered from `results/` only.

Titles state the finding and are computed from the result tables, so a re-run on new
months can never leave a stale number in a title.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from delay_contagion.paths import FIGURES, RESULTS

INK, TEAL, AMBER, SLATE, GRID = "#0f172a", "#0d9488", "#d97706", "#64748b", "#e2e8f0"
POLICY_COLORS = {"LP-optimised": TEAL, "Greedy": AMBER, "Uniform": SLATE}
POLICY_LABELS = {
    "LP-optimised": "LP-optimised",
    "Greedy": "Greedy: pad the most-delayed turns",
    "Uniform": "Uniform padding",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": SLATE,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 22,
            "legend.frameon": False,
        }
    )


def _subtitle(ax: plt.Axes, text: str) -> None:
    ax.text(0, 1.015, text, transform=ax.transAxes, fontsize=9, color=SLATE, va="bottom")


def _save(fig: plt.Figure, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _summary() -> dict:
    return json.loads((RESULTS / "summary.json").read_text())


def _month_label(m: str) -> str:
    return pd.Timestamp(f"{m}-01").strftime("%b %Y")


def hero_frontier() -> Path:
    s = _summary()
    buf = s["buffer"]
    ref = buf["reference"]
    lp, uni = ref["LP-optimised"], ref["Uniform"]
    test = s["propagation"]["test_months"]
    df = pd.read_csv(RESULTS / "buffer_frontier.csv")

    fig, ax = plt.subplots(figsize=(9, 5.4))
    for policy in ["Uniform", "Greedy", "LP-optimised"]:
        d = df[df.policy == policy].sort_values("budget")
        c = POLICY_COLORS[policy]
        ax.fill_between(d.buffer_test, d.avoided_test_lo, d.avoided_test_hi, color=c, alpha=0.15,
                        lw=0)  # fmt: skip
        ax.plot(d.buffer_test, d.avoided_test, color=c, lw=2.4 if policy == "LP-optimised" else 1.8,
                marker="o", ms=3.5)  # fmt: skip
        last = d.iloc[-1]
        ax.annotate(POLICY_LABELS[policy], (last.buffer_test, last.avoided_test), xytext=(6, 0),
                    textcoords="offset points", color=c, fontsize=9.5, va="center",
                    fontweight="bold" if policy == "LP-optimised" else "normal")  # fmt: skip
    ratio = lp["avoided_test"] / uni["avoided_test"]
    ax.scatter([lp["buffer_test"]], [lp["avoided_test"]], s=90, facecolor="none",
               edgecolor=INK, lw=1.2, zorder=4)  # fmt: skip
    ax.annotate(
        f"{lp['avoided_test']:,.0f} min/day avoided\n"
        f"vs {uni['avoided_test']:,.0f} for uniform ({ratio:.1f}×)\n"
        f"at a {buf['reference_budget']:,.0f} min/day budget",
        (lp["buffer_test"], lp["avoided_test"]),
        xytext=(df.buffer_test.max() * 0.03, df.avoided_test.max() * 0.78),
        textcoords="data",
        fontsize=9,
        color=INK,
        arrowprops={"arrowstyle": "-", "color": SLATE, "lw": 0.8},
    )
    ax.set_xlabel("Extra scheduled turn time on held-out days (minutes per day, network-wide)")
    ax.set_ylabel("Arrival delay avoided (minutes per day)")
    ax.set_xlim(0, df.buffer_test.max() * 1.32)
    ax.set_ylim(0, None)
    ax.set_title(f"Placed by the LP, buffer removes {ratio:.1f}× more delay than uniform padding")
    _subtitle(
        ax,
        f"Southwest Airlines, {buf['test_days']} held-out days "
        f"({_month_label(test[0])} to {_month_label(test[-1])}); buffers chosen on "
        f"{buf['scenario_days']} training days. Bands: 95% day-bootstrap CI.",
    )
    return _save(fig, "hero_frontier.png")


def reactionary_by_hour() -> Path:
    df = pd.read_csv(RESULTS / "reactionary_by_hour.csv")
    df = df[df.dep_hour_local.between(5, 23)]
    share = _summary()["network"]["reactionary_share"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(df.dep_hour_local, df.primary_per_flight, color=SLATE, lw=2)
    ax.plot(df.dep_hour_local, df.reactionary_per_flight, color=AMBER, lw=2.4)
    end = df.iloc[-1]
    ax.annotate("Primary causes\n(carrier, weather, NAS, security)", (end.dep_hour_local,
                end.primary_per_flight), xytext=(6, 4), textcoords="offset points", color=SLATE,
                fontsize=9)  # fmt: skip
    ax.annotate("Reactionary\n(late-arriving aircraft)", (end.dep_hour_local,
                end.reactionary_per_flight), xytext=(6, -14), textcoords="offset points",
                color=AMBER, fontsize=9, fontweight="bold")  # fmt: skip
    first = df.loc[df.dep_hour_local == 6].iloc[0]
    peak = df.loc[df.reactionary_per_flight.idxmax()]
    ax.annotate(f"{peak.reactionary_per_flight:.1f} min", (peak.dep_hour_local,
                peak.reactionary_per_flight), xytext=(0, 8), textcoords="offset points",
                ha="center", color=AMBER, fontsize=9)  # fmt: skip
    ax.annotate(f"{first.reactionary_per_flight:.1f} min", (6, first.reactionary_per_flight),
                xytext=(0, -16), textcoords="offset points", ha="center", color=AMBER,
                fontsize=9)  # fmt: skip
    ax.set_xticks(range(5, 24, 2))
    ax.set_xlim(4.5, 26.5)
    ax.set_ylim(0, None)
    ax.set_xlabel("Scheduled departure hour (local)")
    ax.set_ylabel("Delay minutes per departing flight")
    ax.set_title(
        f"Late-arriving aircraft: {first.reactionary_per_flight:.1f} min per flight at 6:00, "
        f"{peak.reactionary_per_flight:.1f} by {int(peak.dep_hour_local)}:00"
    )
    _subtitle(ax, f"All US reporting carriers, 12 months. Reactionary = {share:.0%} of all "
                  "cause-coded delay minutes.")  # fmt: skip
    return _save(fig, "reactionary_by_hour.png")


def hinge_fit(carriers: tuple[str, ...] = ("WN", "DL", "OO")) -> Path:
    params = pd.read_csv(RESULTS / "hinge_carriers.csv").set_index("group")
    binned = pd.read_csv(RESULTS / "hinge_binned_test.csv")
    names = {"WN": "Southwest", "DL": "Delta", "OO": "SkyWest", "AA": "American", "UA": "United"}
    colors = [TEAL, AMBER, SLATE]
    fig, ax = plt.subplots(figsize=(9, 5))
    g = np.linspace(-60, 150, 400)
    for c, color in zip(carriers, colors, strict=False):
        p = params.loc[c]
        d = binned[(binned.carrier == c) & binned.ground_left.between(-60, 150)]
        ax.scatter(d.ground_left, d.mean_out, s=12, color=color, alpha=0.75, lw=0)
        ax.plot(g, p.a + p.beta * np.maximum(0, p.tau - g), color=color, lw=1.8)
        ax.text(
            0.98,
            0.95 - 0.075 * list(carriers).index(c),
            f"{names.get(c, c)}:  τ = {p.tau:.0f} min [{p.tau_lo:.0f}, {p.tau_hi:.0f}],  "
            f"β = {p.beta:.2f}",
            transform=ax.transAxes,
            ha="right",
            color=color,
            fontsize=9.5,
            fontweight="bold",
        )
    ax.set_xlabel("Ground time left = scheduled turn - inbound arrival delay (min)")
    ax.set_ylabel("Mean outbound departure delay (min)")
    ax.set_ylim(-10, None)
    wn = params.loc[carriers[0]]
    ax.set_title(
        f"Below ~{wn.tau:.0f} min of ground time left, inbound delay passes through "
        f"one-for-one (β = {wn.beta:.2f})"
    )
    _subtitle(ax, "Dots: held-out (summer) months, 5-min bins. Lines: hinge fitted on training "
                  "months. τ = effective minimum turn time, 95% CI in brackets.")  # fmt: skip
    return _save(fig, "hinge_fit.png")


def contagion_curve() -> Path:
    curve = pd.read_csv(RESULTS / "contagion_curve.csv")
    curve = curve[curve.mean_x >= 15]  # the headline definition of a delay
    hour = pd.read_csv(RESULTS / "contagion_by_hour.csv")
    o = _summary()["contagion"]["overall"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"wspace": 0.28})
    lo = curve.mean_x * curve.multiplier_lo
    hi = curve.mean_x * curve.multiplier_hi
    ax1.errorbar(curve.mean_x, curve.mean_excess, yerr=[curve.mean_excess - lo,
                 hi - curve.mean_excess], fmt="o-", color=TEAL, lw=2, ms=5, capsize=3)  # fmt: skip
    top = curve.mean_x.max() * 1.05
    ax1.plot([0, top], [0, top], color=SLATE, lw=1, ls="--")
    ax1.annotate("1 : 1", (top * 0.6, top * 0.6), xytext=(12, -8), textcoords="offset points",
                 color=SLATE, fontsize=8.5)  # fmt: skip
    for _, r in curve.iterrows():
        ax1.annotate(f"×{r.multiplier:.2f}", (r.mean_x, r.mean_excess), xytext=(8, -12),
                     textcoords="offset points", fontsize=8.5, color=TEAL)  # fmt: skip
    ax1.set_xlabel("Primary departure delay of the leg (min)")
    ax1.set_ylabel("Extra downstream arrival delay, same aircraft (min)")
    ax1.set_title("Dose-response", fontsize=11, pad=8)
    ax1.set_ylim(0, None)

    h = hour[(hour.dep_hour_local >= 5) & (hour.dep_hour_local <= 21)]
    ax2.fill_between(h.dep_hour_local, h.multiplier_lo, h.multiplier_hi, color=AMBER, alpha=0.2,
                     lw=0)  # fmt: skip
    ax2.plot(h.dep_hour_local, h.multiplier, color=AMBER, lw=2.2, marker="o", ms=3.5)
    ax2.set_xlabel("Departure hour of the delayed leg (local)")
    ax2.set_ylabel("Downstream minutes per primary minute")
    ax2.set_title("By time of day", fontsize=11, pad=8)
    ax2.set_ylim(0, None)
    ax2.set_xticks(range(5, 22, 2))
    fig.suptitle(
        f"Each primary minute of departure delay adds {o['multiplier']:.2f} "
        f"[{o['multiplier_lo']:.2f}, {o['multiplier_hi']:.2f}] minutes later in the aircraft's "
        "day",
        x=0.125, ha="left", fontsize=12.5, fontweight="bold", y=1.04,
    )  # fmt: skip
    fig.text(0.125, 0.965, f"{o['n_events']:,} delayed clean starts (15+ min), each matched to "
             "on-time departures of the same carrier, day, part of day and legs remaining. "
             "95% day-bootstrap CIs.", fontsize=9, color=SLATE)  # fmt: skip
    return _save(fig, "contagion.png")


def superspreaders(top: int = 15) -> Path:
    df = pd.read_csv(RESULTS / "superspreaders.csv").head(top).iloc[::-1]
    o = _summary()["contagion"]["overall"]
    fig, ax = plt.subplots(figsize=(8, 5.6))
    y = np.arange(len(df))
    ax.hlines(y, df.multiplier_lo, df.multiplier_hi, color=TEAL, lw=2, alpha=0.45)
    ax.scatter(df.multiplier, y, color=TEAL, s=30, zorder=3)
    ax.axvline(o["multiplier"], color=SLATE, lw=1, ls="--")
    ax.annotate(f"network {o['multiplier']:.2f}", (o["multiplier"], len(df) - 0.5),
                xytext=(4, 0), textcoords="offset points", color=SLATE, fontsize=8.5)  # fmt: skip
    ax.set_yticks(y, [f"{r.origin}  {r.city}" for r in df.itertuples()])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Downstream delay minutes per primary minute (95% CI)")
    first = df.iloc[-1]
    ax.set_title(f"Super-spreaders: each primary minute at {first.origin} adds "
                 f"{first.multiplier:.2f} minutes downstream (network: {o['multiplier']:.2f})")  # fmt: skip
    min_events = _summary()["config"]["min_events_airport"]
    _subtitle(ax, f"Airports with at least {min_events:,} delayed clean starts in 12 months.")
    return _save(fig, "superspreaders.png")


def scenario_sizes() -> Path:
    df = pd.read_csv(RESULTS / "buffer_scenario_sizes.csv")
    ref = _summary()["buffer"]["reference_budget"]
    mean = df.groupby("scenario_days")[["gain_train", "gain_test"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for col, color, label, dy in [
        ("gain_train", SLATE, "in-sample (the scenario days themselves)", 10),
        ("gain_test", TEAL, "out-of-sample (held-out days)", -18),
    ]:
        ax.scatter(df.scenario_days, df[col], color=color, s=14, alpha=0.45, lw=0)
        ax.plot(mean.scenario_days, mean[col], color=color, lw=2.2, marker="o", ms=4)
        ax.annotate(label, (mean.scenario_days.iloc[1], mean[col].iloc[1]), xytext=(6, dy),
                    textcoords="offset points", color=color, fontsize=9, ha="left",
                    fontweight="bold" if col == "gain_test" else "normal")  # fmt: skip
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, None)
    ax.set_xticks(mean.scenario_days)
    ax.set_xlabel("Training days used as LP scenarios")
    ax.set_ylabel("Extra delay avoided vs uniform padding")
    last = mean.iloc[-1]
    ax.set_title(
        f"More scenario days: in-sample optimism falls, the held-out edge rises to "
        f"{last.gain_test:.0%}"
    )
    n_rep = df.replicate.nunique()
    _subtitle(ax, f"Budget {ref:.0f} buffer minutes per day. Dots: {n_rep} independent draws of "
                  "training days per size; lines: mean.")  # fmt: skip
    return _save(fig, "scenario_sizes.png")


def render_all() -> list[Path]:
    _style()
    return [
        hero_frontier(),
        reactionary_by_hour(),
        hinge_fit(),
        contagion_curve(),
        superspreaders(),
        scenario_sizes(),
    ]
