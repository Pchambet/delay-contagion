"""Static report page `site/index.html` and the README, rendered from `results/` only.

One self-contained page: narrative, tables and Plotly charts (Plotly.js from jsDelivr),
light and dark themes, readable on a phone. Chart data are plain JSON traces built here,
so the page carries only the numbers it shows. The README is rendered from its own
template with the same headline strings, so no published number is typed by hand.
"""

from __future__ import annotations

import html
import itertools
import json
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd

from delay_contagion import headlines
from delay_contagion.descriptive import CARRIER_NAMES
from delay_contagion.paths import RESULTS, ROOT, SITE

TEAL, AMBER, SLATE, INDIGO = "#0d9488", "#d97706", "#64748b", "#4f46e5"
PLOTLY = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@4.1.1/plotly.min.js"
REPO = "https://github.com/Pchambet/delay-contagion"


def _r(x: pd.Series | np.ndarray, nd: int = 2) -> list[float | None]:
    return [None if pd.isna(v) else round(float(v), nd) for v in x]


# Charts carry `short` axis titles used below 600 px, where long titles get clipped.


def _hour_chart() -> dict:
    df = pd.read_csv(RESULTS / "reactionary_by_hour.csv")
    df = df[df.dep_hour_local.between(5, 23)]
    return {
        "data": [
            {"x": df.dep_hour_local.tolist(), "y": _r(df.primary_per_flight), "name": "Primary causes",
             "mode": "lines+markers", "line": {"color": SLATE, "width": 2.5}},
            {"x": df.dep_hour_local.tolist(), "y": _r(df.reactionary_per_flight),
             "name": "Late aircraft", "mode": "lines+markers",
             "line": {"color": AMBER, "width": 3}},
        ],
        "layout": {"showlegend": True, "margin": {"t": 48},
                   "legend": {"orientation": "h", "x": 0, "y": 1.02, "yanchor": "bottom"},
                   "xaxis": {"title": {"text": "Scheduled departure hour (local)"}, "dtick": 2},
                   "yaxis": {"title": {"text": "Delay minutes per flight"}, "rangemode": "tozero"}},
        "short": {"x": "Departure hour (local)", "y": "Min per flight"},
    }  # fmt: skip


def _hinge_chart() -> dict:
    params = pd.read_csv(RESULTS / "hinge_carriers.csv").set_index("group")
    binned = pd.read_csv(RESULTS / "hinge_binned_test.csv")
    shown = {"WN": TEAL, "DL": AMBER, "OO": SLATE}
    # Mid-luminance hues so every carrier stays legible on light and dark backgrounds.
    palette = ["#0891b2", "#b45309", "#7c3aed", "#15803d", "#be123c", "#a16207", "#0f766e",
               "#9333ea", "#c2410c", "#4d7c0f", "#db2777"]  # fmt: skip
    g = np.arange(-60, 151, 2)
    data = []
    others = itertools.cycle(palette)
    for c in params.index:
        p = params.loc[c]
        color = shown.get(c) or next(others)
        visible = True if c in shown else "legendonly"
        name = f"{CARRIER_NAMES.get(c, c)}  τ {p.tau:.0f}, β {p.beta:.2f}"
        d = binned[(binned.carrier == c) & binned.ground_left.between(-60, 150)]
        data.append({"x": d.ground_left.tolist(), "y": _r(d.mean_out, 1), "mode": "markers",
                     "marker": {"color": color, "size": 6, "opacity": 0.7}, "legendgroup": c,
                     "showlegend": False, "visible": visible, "name": name,
                     "hovertemplate": "ground left %{x} min<br>mean outbound %{y} min"
                                      "<extra>" + c + " held-out</extra>"})  # fmt: skip
        data.append({"x": g.tolist(), "y": _r(p.a + p.beta * np.maximum(0, p.tau - g), 1),
                     "mode": "lines", "line": {"color": color, "width": 2.5}, "legendgroup": c,
                     "name": name, "visible": visible,
                     "hovertemplate": "%{y} min<extra>" + c + " fit</extra>"})  # fmt: skip
    return {
        "data": data,
        # Desktop: a vertical legend right of the plot (13 carriers); phones: below it.
        "wide_legend": {"font": {"size": 11}, "orientation": "v", "x": 1.02, "xanchor": "left",
                        "y": 1, "yanchor": "top"},
        "layout": {"showlegend": True,
                   "xaxis": {"title": {"text": "Ground time left = scheduled turn - inbound delay (min)"}},
                   "yaxis": {"title": {"text": "Mean outbound departure delay (min)"}}},
        "short": {"x": "Ground time left (min)", "y": "Outbound delay (min)"},
    }  # fmt: skip


def _contagion_chart() -> dict:
    c = pd.read_csv(RESULTS / "contagion_curve.csv")
    c = c[c.mean_x >= 15]  # the headline definition of a delay
    top = round(float(c.mean_x.max()) * 1.05)
    lo, hi = c.mean_x * c.multiplier_lo, c.mean_x * c.multiplier_hi
    return {
        "data": [{"x": [0, top], "y": [0, top], "mode": "lines", "hoverinfo": "skip",
                  "line": {"color": SLATE, "width": 1, "dash": "dash"}},
                 {"x": _r(c.mean_x, 1), "y": _r(c.mean_excess, 1), "mode": "lines+markers",
                  "line": {"color": TEAL, "width": 3}, "marker": {"size": 8},
                  "error_y": {"type": "data", "symmetric": False, "array": _r(hi - c.mean_excess, 1),
                              "arrayminus": _r(c.mean_excess - lo, 1), "color": TEAL},
                  "customdata": [[b, round(m, 2), int(n)] for b, m, n in
                                 zip(c.x_bin, c.multiplier, c.n_events, strict=True)],
                  "hovertemplate": "primary %{customdata[0]} min<br>extra downstream %{y} min"
                                   "<br>multiplier %{customdata[1]}<br>%{customdata[2]:,} events"
                                   "<extra></extra>"}],
        "layout": {"xaxis": {"title": {"text": "Primary departure delay (min)"}},
                   "yaxis": {"title": {"text": "Extra downstream delay, same aircraft (min)"},
                             "rangemode": "tozero"}},
        "short": {"x": "Primary delay (min)", "y": "Extra downstream (min)"},
    }  # fmt: skip


def _contagion_hour_chart() -> dict:
    h = pd.read_csv(RESULTS / "contagion_by_hour.csv")
    h = h[h.dep_hour_local.between(5, 21)]
    x = h.dep_hour_local.tolist()
    return {
        "data": [{"x": x + x[::-1], "y": _r(h.multiplier_hi) + _r(h.multiplier_lo)[::-1],
                  "fill": "toself", "fillcolor": AMBER, "opacity": 0.2, "mode": "lines",
                  "line": {"width": 0}, "hoverinfo": "skip"},
                 {"x": x, "y": _r(h.multiplier), "mode": "lines+markers",
                  "line": {"color": AMBER, "width": 3}, "marker": {"size": 7},
                  "customdata": h.n_events.tolist(),
                  "hovertemplate": "%{x}:00 departures<br>multiplier %{y}<br>%{customdata:,} "
                                   "events<extra></extra>"}],
        "layout": {"xaxis": {"title": {"text": "Departure hour of the delayed leg (local)"},
                             "dtick": 2},
                   "yaxis": {"title": {"text": "Downstream minutes per primary minute"},
                             "rangemode": "tozero"}},
        "short": {"x": "Departure hour (local)", "y": "Multiplier"},
    }  # fmt: skip


def _map_chart() -> dict:
    df = pd.read_csv(RESULTS / "superspreaders.csv")
    size = 6 + 26 * np.sqrt(df.primary_min / df.primary_min.max())
    text = [
        f"<b>{r.origin}</b> {html.escape(str(r.city))}<br>above carrier mix "
        f"{r.excess_over_mix:+.2f} [{r.excess_over_mix_lo:+.2f}, {r.excess_over_mix_hi:+.2f}]"
        f"<br>raw multiplier {r.multiplier:.2f}, carrier mix {r.carrier_mix:.2f}"
        f"<br>{r.n_events:,} delayed starts"
        for r in df.itertuples()
    ]
    span = float(df.excess_over_mix.abs().max())
    return {
        "data": [{"type": "scattergeo", "locationmode": "USA-states", "lat": _r(df.lat, 3),
                  "lon": _r(df.lon, 3), "text": text, "hoverinfo": "text",
                  "marker": {"size": _r(size, 1), "color": _r(df.excess_over_mix, 3),
                             "colorscale": [[0, AMBER], [0.5, "#cbd5e1"], [1, TEAL]],
                             "cmin": -span, "cmax": span,
                             "line": {"width": 0.5, "color": "#0f172a"}, "opacity": 0.9,
                             "colorbar": {"title": {"text": "vs carrier mix"}, "thickness": 12,
                                          "len": 0.7}}}],
        "layout": {"geo": {"scope": "usa", "projection": {"type": "albers usa"},
                           "showland": True, "landcolor": "LAND", "subunitcolor": "SUB",
                           "bgcolor": "rgba(0,0,0,0)", "showlakes": False},
                   "margin": {"l": 0, "r": 0, "t": 0, "b": 0}},
    }  # fmt: skip


def _frontier_chart() -> dict:
    df = pd.read_csv(RESULTS / "buffer_frontier.csv")
    colors = {"Uniform": SLATE, "Greedy": AMBER, "Marginal greedy": INDIGO, "LP-optimised": TEAL}
    names = {"Uniform": "Uniform", "Greedy": "Greedy", "Marginal greedy": "Marginal greedy",
             "LP-optimised": "LP"}  # fmt: skip
    data = []
    for p in ["Uniform", "Greedy", "Marginal greedy", "LP-optimised"]:
        d = df[df.policy == p].sort_values("budget")
        x = _r(d.buffer_test, 0)
        data.append({"x": x + x[::-1], "y": _r(d.avoided_test_hi, 0) + _r(d.avoided_test_lo, 0)[::-1],
                     "fill": "toself", "fillcolor": colors[p], "opacity": 0.15, "mode": "lines",
                     "line": {"width": 0}, "hoverinfo": "skip", "showlegend": False})  # fmt: skip
        data.append({"x": x, "y": _r(d.avoided_test, 0), "name": names[p],
                     "mode": "lines+markers", "line": {"color": colors[p], "width": 3},
                     "customdata": _r(d.budget, 0),
                     "hovertemplate": "budget %{customdata} min/day<br>buffer used %{x} min/day"
                                      "<br>delay avoided %{y} min/day<extra>" + names[p]
                                      + "</extra>"})  # fmt: skip
    return {
        "data": data,
        "layout": {"showlegend": True,
                   "legend": {"orientation": "h", "x": 0, "y": 1.02, "yanchor": "bottom"},
                   "margin": {"t": 48},
                   "xaxis": {"title": {"text": "Buffer scheduled on held-out days (min/day)"}},
                   "yaxis": {"title": {"text": "Arrival delay avoided (min/day)"},
                             "rangemode": "tozero"}},
        "short": {"x": "Buffer used (min/day)", "y": "Delay avoided (min/day)"},
    }  # fmt: skip


def _sizes_chart() -> dict:
    df = pd.read_csv(RESULTS / "buffer_scenario_sizes.csv")
    mean = df.groupby("scenario_days")[["gain_train", "gain_test"]].mean().reset_index()
    data = []
    for col, color, name in [
        ("gain_train", SLATE, "In-sample"),
        ("gain_test", TEAL, "Held-out days"),
    ]:
        data.append(
            {
                "x": df.scenario_days.tolist(),
                "y": _r(df[col] * 100, 1),
                "mode": "markers",
                "marker": {"color": color, "size": 7, "opacity": 0.4},
                "showlegend": False,
                "hovertemplate": "%{y}%<extra>one draw</extra>",
            }
        )
        data.append(
            {
                "x": mean.scenario_days.tolist(),
                "y": _r(mean[col] * 100, 1),
                "name": name,
                "mode": "lines+markers",
                "line": {"color": color, "width": 3},
                "hovertemplate": "%{x} days: %{y}%<extra>" + name + "</extra>",
            }
        )
    return {
        "data": data,
        "layout": {"showlegend": True, "margin": {"t": 48},
                   "legend": {"orientation": "h", "x": 0, "y": 1.02, "yanchor": "bottom"},
                   "xaxis": {"title": {"text": "Training days used as LP scenarios"},
                             "tickvals": mean.scenario_days.tolist()},
                   "yaxis": {"title": {"text": "Extra delay avoided vs uniform (%)"},
                             "rangemode": "tozero", "ticksuffix": "%"}},
        "short": {"x": "Scenario days", "y": "Edge vs uniform"},
    }  # fmt: skip


def _carrier_table() -> str:
    df = pd.read_csv(RESULTS / "hinge_carriers.csv").sort_values("n_turns", ascending=False)
    rows = "".join(
        f"<tr><td>{CARRIER_NAMES.get(r.group, r.group)}</td><td class=opt>{r.n_turns:,}</td>"
        f"<td>{r.tau:.0f} <span class=ci>[{r.tau_lo:.0f}, {r.tau_hi:.0f}]</span></td>"
        f"<td>{r.beta:.2f} <span class=ci>[{r.beta_lo:.2f}, {r.beta_hi:.2f}]</span></td>"
        f"<td class=opt>{r.mae_constant:.1f}</td><td class=opt>{r.mae_linear:.1f}</td>"
        f"<td class=opt>{r.mae_linear_slack:.1f}</td><td>{r.mae_gbm:.1f}</td>"
        f"<td><b>{r.mae_hinge:.1f}</b></td></tr>"
        for r in df.itertuples()
    )
    return (
        "<table><thead><tr><th>Carrier</th><th class=opt>Train turns</th><th>τ (min)</th>"
        "<th>β</th><th class=opt>MAE const.</th><th class=opt>MAE linear</th>"
        "<th class=opt>MAE lin. + slack</th>"
        "<th>MAE boosted</th><th>MAE hinge</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _allocation_table(n: int = 12) -> str:
    df = pd.read_csv(RESULTS / "buffer_allocation.csv").head(n)
    rows = "".join(
        f"<tr><td>{r.station}</td><td>{r.dep_hour_local:02d}:00</td>"
        f"<td>{r.turns_per_day:.1f}</td><td>{r.buffer_min:.1f}</td>"
        f"<td><b>{r.buffer_min_per_day:.1f}</b></td>"
        f"<td class=opt>{r.propagated_delay_per_turn:.1f}</td>"
        f"<td class=opt>{r.marginal_value_per_min:.2f}</td></tr>"
        for r in df.itertuples()
    )
    return (
        "<table><thead><tr><th>Station</th><th>Departure hour</th><th>Turns/day</th>"
        "<th>Buffer per turn (min)</th><th>Buffer min/day</th>"
        "<th class=opt>Propagated delay per turn (min)</th>"
        "<th class=opt>Avoided per buffer min, alone</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


def _policy_table() -> str:
    """Every policy at the reference budget: spend, delay avoided and on-time share."""
    f = pd.read_csv(RESULTS / "buffer_frontier.csv")
    budget = json.loads((RESULTS / "summary.json").read_text())["buffer"]["reference_budget"]
    f = f[f.budget == budget].set_index("policy")
    sched = pd.read_csv(RESULTS / "buffer_schedule_metrics.csv").set_index("policy")
    labels = headlines.POLICY_LABELS
    rows = "".join(
        f"<tr><td>{labels[p]}</td><td class=opt>{f.loc[p, 'buffer_test']:,.0f}</td>"
        f"<td>{f.loc[p, 'avoided_test']:,.0f} <span class=ci>[{f.loc[p, 'avoided_test_lo']:,.0f}, "
        f"{f.loc[p, 'avoided_test_hi']:,.0f}]</span></td>"
        f"<td><b>{f.loc[p, 'avoided_per_buffer_min']:.2f}</b></td>"
        f"<td>{sched.loc[p, 'ontime15']:.1%}</td>"
        f"<td class=opt>{sched.loc[p, 'clock_late_change_per_day']:+,.0f}</td></tr>"
        for p in labels
    )
    return (
        "<table><thead><tr><th>Policy</th><th class=opt>Buffer used (min/day)</th>"
        "<th>Delay avoided (min/day)</th><th>Per buffer min</th><th>On time (A15)</th>"
        "<th class=opt>Clock lateness (min/day)</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _robustness_table() -> str:
    df = pd.read_csv(RESULTS / "contagion_robustness.csv")
    rows = "".join(
        f"<tr><td>{html.escape(r.variant)}</td><td class=opt>{r.n_events:,}</td>"
        f"<td><b>{r.multiplier:.2f}</b> <span class=ci>[{r.multiplier_lo:.2f}, "
        f"{r.multiplier_hi:.2f}]</span></td><td>{r.small_slip_multiplier:.2f}</td></tr>"
        for r in df.itertuples()
    )
    return (
        "<table><thead><tr><th>Matching design</th><th class=opt>Events (15+ min)</th>"
        "<th>Multiplier</th><th>Slips of 1-14 min</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _sensitivity_table() -> str:
    df = pd.read_csv(RESULTS / "buffer_sensitivity.csv")
    lp = df[df.policy == "LP-optimised"].set_index(["tau_eval", "beta_eval"])
    uni = df[df.policy == "Uniform"].set_index(["tau_eval", "beta_eval"])
    rows = "".join(
        f"<tr><td>{t:.0f}</td><td>{b:.2f}</td><td>{lp.loc[(t, b), 'avoided_test']:,.0f}</td>"
        f"<td>{uni.loc[(t, b), 'avoided_test']:,.0f}</td>"
        f"<td><b>{lp.loc[(t, b), 'per_min_vs_uniform']:.2f}×</b></td></tr>"
        for t, b in lp.index
    )
    return (
        "<table><thead><tr><th>Replay τ (min)</th><th>Replay β</th><th>LP avoided</th>"
        "<th>Uniform avoided</th><th>LP / uniform per buffer min</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def readme_text() -> str:
    """README.md as rendered from `readme_template.md` and the current `results/`."""
    template = Template((Path(__file__).parent / "readme_template.md").read_text())
    return template.substitute(
        **headlines.compute(),
        carrier_table_md=headlines.carrier_table_md(),
        policy_table_md=headlines.policy_table_md(),
        robustness_table_md=headlines.robustness_table_md(),
        sensitivity_table_md=headlines.sensitivity_table_md(),
    )


def render_readme() -> Path:
    out = ROOT / "README.md"
    out.write_text(readme_text())
    return out


def render() -> Path:
    h = headlines.compute()
    charts = {
        "chart-hour": _hour_chart(),
        "chart-hinge": _hinge_chart(),
        "chart-contagion": _contagion_chart(),
        "chart-contagion-hour": _contagion_hour_chart(),
        "chart-map": _map_chart(),
        "chart-frontier": _frontier_chart(),
        "chart-sizes": _sizes_chart(),
    }
    template = Template((Path(__file__).parent / "report_template.html").read_text())
    page = template.substitute(
        **h,
        plotly=PLOTLY,
        repo=REPO,
        charts=json.dumps(charts, separators=(",", ":")),
        carrier_table=_carrier_table(),
        allocation_table=_allocation_table(),
        policy_table=_policy_table(),
        robustness_table=_robustness_table(),
        sensitivity_table=_sensitivity_table(),
    )
    SITE.mkdir(parents=True, exist_ok=True)
    out = SITE / "index.html"
    out.write_text(page)
    return out
