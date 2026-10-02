"""Static report page `site/index.html`, rendered from `results/` only.

One self-contained file: narrative, tables and Plotly charts (Plotly.js from jsDelivr),
light and dark themes, readable on a phone. Chart data are plain JSON traces built here,
so the page carries only the numbers it shows.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd

from delay_contagion import headlines
from delay_contagion.descriptive import CARRIER_NAMES
from delay_contagion.paths import RESULTS, SITE

TEAL, AMBER, SLATE = "#0d9488", "#d97706", "#64748b"
PLOTLY = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@4.1.1/plotly.min.js"
REPO = "https://github.com/Pchambet/delay-contagion"


def _r(x, nd: int = 2):
    return [None if pd.isna(v) else round(float(v), nd) for v in x]


def _hour_chart() -> dict:
    df = pd.read_csv(RESULTS / "reactionary_by_hour.csv")
    df = df[df.dep_hour_local.between(5, 23)]
    return {
        "data": [
            {"x": df.dep_hour_local.tolist(), "y": _r(df.primary_per_flight), "name": "Primary",
             "mode": "lines+markers", "line": {"color": SLATE, "width": 2.5}},
            {"x": df.dep_hour_local.tolist(), "y": _r(df.reactionary_per_flight),
             "name": "Reactionary (late aircraft)", "mode": "lines+markers",
             "line": {"color": AMBER, "width": 3}},
        ],
        "layout": {"showlegend": True, "legend": {"orientation": "h", "y": 1.12},
                   "xaxis": {"title": {"text": "Scheduled departure hour (local)"}, "dtick": 2},
                   "yaxis": {"title": {"text": "Delay minutes per flight"}, "rangemode": "tozero"}},
    }  # fmt: skip


def _hinge_chart() -> dict:
    params = pd.read_csv(RESULTS / "hinge_carriers.csv").set_index("group")
    binned = pd.read_csv(RESULTS / "hinge_binned_test.csv")
    shown = {"WN": TEAL, "DL": AMBER, "OO": SLATE}
    palette = ["#0f766e", "#b45309", "#475569", "#0e7490", "#a16207", "#334155", "#115e59",
               "#92400e", "#1e293b", "#155e75", "#78350f"]  # fmt: skip
    g = np.arange(-60, 151, 2)
    data = []
    others = iter(palette)
    for c in params.index:
        p = params.loc[c]
        color = shown.get(c) or next(others)
        visible = True if c in shown else "legendonly"
        name = f"{CARRIER_NAMES.get(c, c)} (tau {p.tau:.0f}, beta {p.beta:.2f})"
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
        "layout": {"showlegend": True,
                   "legend": {"font": {"size": 11}, "orientation": "h", "y": -0.22},
                   "xaxis": {"title": {"text": "Ground time left = scheduled turn - inbound delay (min)"}},
                   "yaxis": {"title": {"text": "Mean outbound departure delay (min)"}}},
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
    }  # fmt: skip


def _map_chart() -> dict:
    df = pd.read_csv(RESULTS / "superspreaders.csv")
    size = 6 + 26 * np.sqrt(df.primary_min / df.primary_min.max())
    text = [
        f"<b>{r.origin}</b> {html.escape(str(r.city))}<br>multiplier {r.multiplier:.2f} "
        f"[{r.multiplier_lo:.2f}, {r.multiplier_hi:.2f}]<br>{r.n_events:,} delayed starts"
        for r in df.itertuples()
    ]
    return {
        "data": [{"type": "scattergeo", "locationmode": "USA-states", "lat": _r(df.lat, 3),
                  "lon": _r(df.lon, 3), "text": text, "hoverinfo": "text",
                  "marker": {"size": _r(size, 1), "color": _r(df.multiplier, 3),
                             "colorscale": [[0, "#ccfbf1"], [0.5, "#14b8a6"], [1, "#134e4a"]],
                             "line": {"width": 0.5, "color": "#0f172a"}, "opacity": 0.9,
                             "colorbar": {"title": {"text": "multiplier"}, "thickness": 12,
                                          "len": 0.7}}}],
        "layout": {"geo": {"scope": "usa", "projection": {"type": "albers usa"},
                           "showland": True, "landcolor": "LAND", "subunitcolor": "SUB",
                           "bgcolor": "rgba(0,0,0,0)", "showlakes": False},
                   "margin": {"l": 0, "r": 0, "t": 0, "b": 0}},
    }  # fmt: skip


def _frontier_chart() -> dict:
    df = pd.read_csv(RESULTS / "buffer_frontier.csv")
    colors = {"Uniform": SLATE, "Greedy": AMBER, "LP-optimised": TEAL}
    names = {"Uniform": "Uniform padding", "Greedy": "Greedy: pad most-delayed turns",
             "LP-optimised": "LP-optimised"}  # fmt: skip
    data = []
    for p in ["Uniform", "Greedy", "LP-optimised"]:
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
        "layout": {"showlegend": True, "legend": {"orientation": "h", "y": 1.12},
                   "xaxis": {"title": {"text": "Buffer scheduled on held-out days (min/day)"}},
                   "yaxis": {"title": {"text": "Arrival delay avoided (min/day)"},
                             "rangemode": "tozero"}},
    }  # fmt: skip


def _sizes_chart() -> dict:
    df = pd.read_csv(RESULTS / "buffer_scenario_sizes.csv")
    mean = df.groupby("scenario_days")[["gain_train", "gain_test"]].mean().reset_index()
    data = []
    for col, color, name in [
        ("gain_train", SLATE, "In-sample (scenario days)"),
        ("gain_test", TEAL, "Out-of-sample (held-out days)"),
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
        "layout": {"showlegend": True, "legend": {"orientation": "h", "y": 1.12},
                   "xaxis": {"title": {"text": "Training days used as LP scenarios"},
                             "tickvals": mean.scenario_days.tolist()},
                   "yaxis": {"title": {"text": "Extra delay avoided vs uniform (%)"},
                             "rangemode": "tozero", "ticksuffix": "%"}},
    }  # fmt: skip


def _carrier_table() -> str:
    df = pd.read_csv(RESULTS / "hinge_carriers.csv").sort_values("n_turns", ascending=False)
    rows = "".join(
        f"<tr><td>{CARRIER_NAMES.get(r.group, r.group)}</td><td>{r.n_turns:,}</td>"
        f"<td>{r.tau:.0f} <span class=ci>[{r.tau_lo:.0f}, {r.tau_hi:.0f}]</span></td>"
        f"<td>{r.beta:.2f} <span class=ci>[{r.beta_lo:.2f}, {r.beta_hi:.2f}]</span></td>"
        f"<td>{r.mae_constant:.1f}</td><td>{r.mae_linear:.1f}</td>"
        f"<td><b>{r.mae_hinge:.1f}</b></td></tr>"
        for r in df.itertuples()
    )
    return (
        "<table><thead><tr><th>Carrier</th><th>Train turns</th><th>τ (min)</th>"
        "<th>β</th><th>MAE const.</th><th>MAE linear</th><th>MAE hinge</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _allocation_table(n: int = 12) -> str:
    df = pd.read_csv(RESULTS / "buffer_allocation.csv").head(n)
    rows = "".join(
        f"<tr><td>{r.station}</td><td>{r.dep_hour_local:02d}:00</td>"
        f"<td>{r.turns_per_day:.1f}</td><td>{r.buffer_min:.1f}</td>"
        f"<td>{r.propagated_delay_per_turn:.1f}</td></tr>"
        for r in df.itertuples()
    )
    return (
        "<table><thead><tr><th>Station</th><th>Departure hour</th><th>Turns/day</th>"
        "<th>Buffer per turn (min)</th><th>Observed propagated delay per turn (min)</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


def render() -> Path:
    h = headlines.compute()
    charts = {
        "chart-hour": _hour_chart(),
        "chart-hinge": _hinge_chart(),
        "chart-contagion": _contagion_chart(),
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
    )
    SITE.mkdir(parents=True, exist_ok=True)
    out = SITE / "index.html"
    out.write_text(page)
    return out
