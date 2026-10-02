"""End-to-end analysis on the warehouse marts; writes the small result tables in `results/`.

Figures and the report are rendered from `results/` alone, so they can be rebuilt without
the ~350 MB of raw data.

Time split: the 12 months are cut chronologically into 9 training months and 3 held-out
months. Every fitted quantity (hinge parameters, LP buffers, heuristic rankings) sees only
training months; every reported performance number is computed on held-out months.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import duckdb
import numpy as np
import pandas as pd

from delay_contagion import buffer_lp, contagion, descriptive, propagation
from delay_contagion.paths import RESULTS


@dataclass
class Config:
    n_test_months: int = 3
    n_hub_airports: int = 15
    lp_carrier: str = "WN"
    lp_scenarios: int = 90  # training days sampled as SAA scenarios
    lp_scenario_sizes: list[int] = field(default_factory=lambda: [10, 30, 60, 90])
    lp_size_replicates: int = 3
    lp_b_max: float = 20.0  # max extra minutes on any one turn
    lp_budgets: list[float] = field(
        default_factory=lambda: [0, 150, 300, 600, 900, 1200, 1800, 2400]
    )
    lp_reference_budget: float = 600
    min_events_airport: int = 1500
    n_boot: int = 300
    seed: int = 2026


def months(con: duckdb.DuckDBPyConnection) -> list[str]:
    return [
        r[0]
        for r in con.execute(
            "select distinct flight_month from marts.fct_turns order by 1"
        ).fetchall()
    ]


def _log(msg: str, t0: float) -> None:
    print(f"[{time.perf_counter() - t0:6.1f}s] {msg}", flush=True)


def _round(obj, digits: int = 6):
    """Round floats in nested containers (last-digit float noise varies between runs)."""
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: _round(v, digits) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_round(v, digits) for v in obj]
    return obj


def _write(df: pd.DataFrame, name: str, decimals: int = 4) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.round(decimals).to_csv(RESULTS / name, index=False)


def run_descriptive(con, summary: dict) -> None:
    summary["network"] = descriptive.network_summary(con)
    _write(descriptive.cause_shares_by_carrier(con), "cause_shares_by_carrier.csv")
    _write(descriptive.reactionary_by_hour(con), "reactionary_by_hour.csv")


def run_propagation(con, cfg: Config, train: list[str], test: list[str], summary: dict) -> None:
    carriers = propagation.fit_groups(con, "carrier", train, n_boot=cfg.n_boot)
    hubs = [
        r[0]
        for r in con.execute(
            "select station from marts.fct_turns group by 1 order by count(*) desc limit ?",
            [cfg.n_hub_airports],
        ).fetchall()
    ]
    airports = propagation.fit_groups(con, "station", train, hubs, n_boot=cfg.n_boot)
    scores_c = propagation.holdout_scores(con, carriers, "carrier", train, test)
    scores_a = propagation.holdout_scores(con, airports, "station", train, test)
    carriers = carriers.merge(scores_c, on="group")
    airports = airports.merge(scores_a, on="group")
    _write(carriers, "hinge_carriers.csv")
    _write(airports, "hinge_airports.csv")

    # Network-level out-of-sample errors: test-turn-weighted over carriers.
    n = carriers["n_test"]
    summary["propagation"] = {
        "train_months": train,
        "test_months": test,
        "turns_train": int(carriers["n_turns"].sum()),
        "turns_test": int(n.sum()),
        "mae": {
            m: float((carriers[f"mae_{m}"] * n).sum() / n.sum())
            for m in ("constant", "linear", "hinge")
        },
        "rmse": {
            m: float(np.sqrt((carriers[f"rmse_{m}"] ** 2 * n).sum() / n.sum()))
            for m in ("constant", "linear", "hinge")
        },
        "carriers_hinge_beats_linear_mae": int(
            (carriers["mae_hinge"] < carriers["mae_linear"]).sum()
        ),
        "n_carriers": len(carriers),
        "tau_range": [float(carriers["tau"].min()), float(carriers["tau"].max())],
        "beta_range": [float(carriers["beta"].min()), float(carriers["beta"].max())],
    }

    # Out-of-sample shape check: mean outbound delay vs ground time left, test months.
    binned = con.execute(
        f"""
        select carrier, least(greatest(round((sched_turn_min - inbound_arr_delay) / 5) * 5,
                                       -120), 180) as ground_left,
               count(*) as n, avg(outbound_dep_delay) as mean_out
        from marts.fct_turns
        where flight_month in ({", ".join(f"'{m}'" for m in test)})
          and {propagation.DELAY_FILTER}
        group by all having count(*) >= 50 order by 1, 2
        """
    ).df()
    _write(binned, "hinge_binned_test.csv")


def run_contagion(con, cfg: Config, all_months: list[str], summary: dict) -> None:
    n = contagion.build_events(con, all_months)
    curve = contagion.multiplier_curve(con)
    by_hour = contagion.multiplier_by(con, "dep_hour_local", min_events=500)
    by_carrier = contagion.multiplier_by(con, "carrier")
    by_airport = contagion.multiplier_by(con, "origin", min_events=cfg.min_events_airport)
    coords = con.execute(
        "select iata as origin, name, city, state, lat, lon from main.airports"
    ).df()
    by_airport = by_airport.merge(coords, on="origin", how="left")
    by_airport = by_airport.sort_values("multiplier", ascending=False).reset_index(drop=True)
    _write(curve, "contagion_curve.csv")
    _write(by_hour, "contagion_by_hour.csv")
    _write(by_carrier, "contagion_by_carrier.csv")
    _write(by_airport, "superspreaders.csv")
    summary["contagion"] = {
        "matched_events": int(n),
        "overall": contagion.overall(con),
        "n_airports_ranked": len(by_airport),
    }


def _days(con, carrier: str, months_: list[str]) -> list:
    """Operating days (chain start dates) of `carrier` that fall in `months_`."""
    month_sql = ", ".join(f"'{m}'" for m in months_)
    return [
        r[0]
        for r in con.execute(
            f"select distinct chain_date from marts.fct_legs where carrier = ? "
            f"and strftime(chain_date, '%Y-%m') in ({month_sql}) order by 1",
            [carrier],
        ).fetchall()
    ]


def _cells(legs: pd.DataFrame) -> dict:
    """Decision cells = (station, local hour) of every turn seen in the training scenarios."""
    turns = legs[legs.leg_seq > 1]
    keys = sorted(set(zip(turns["origin"], turns["dep_hour_local"], strict=True)))
    return {k: i for i, k in enumerate(keys)}


def _avoided(sc, base: np.ndarray, b: np.ndarray, tau: float, beta: float) -> np.ndarray:
    """Delay minutes avoided per day under buffers `b`, relative to the observed day."""
    return base - buffer_lp.daily_delay(sc, buffer_lp.simulate(sc, b, tau, beta))


def _ci(boot: np.ndarray, per_day: np.ndarray) -> tuple[float, float]:
    draws = boot @ per_day / boot.sum(1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def scenario_size_study(
    con,
    cfg: Config,
    train_days: list,
    legs_te: pd.DataFrame,
    base_te: np.ndarray,
    tau: float,
    beta: float,
    t0: float,
) -> pd.DataFrame:
    """Out-of-sample value of the LP as the scenario set grows, over independent draws.

    In-sample gains are optimistic (the LP fits the days it sees); the held-out gain is what
    a planner would actually get. Each replicate is a fresh random order of training days,
    with nested sets of the first S days.
    """
    rows = []
    for rep in range(cfg.lp_size_replicates):
        order = np.random.default_rng(cfg.seed + 100 + rep).permutation(len(train_days))
        for n in cfg.lp_scenario_sizes:
            legs_n = buffer_lp.load_legs(con, cfg.lp_carrier, [train_days[i] for i in order[:n]])
            cells = _cells(legs_n)
            sc_n = buffer_lp.from_legs(legs_n, cells, tau, beta)
            sc_te = buffer_lp.from_legs(legs_te, cells, tau, beta)
            lp_n = buffer_lp.BufferLP(sc_n, len(cells), tau, beta, cfg.lp_b_max)
            b_lp, _ = lp_n.solve(cfg.lp_reference_budget)
            uni = buffer_lp.uniform_policy(lp_n.nbar, cfg.lp_reference_budget, cfg.lp_b_max)
            base_n = buffer_lp.daily_delay(sc_n, sc_n.obs_arr)
            rows.append(
                {
                    "replicate": rep,
                    "scenario_days": n,
                    "lp_avoided_train": _avoided(sc_n, base_n, b_lp, tau, beta).mean(),
                    "lp_avoided_test": _avoided(sc_te, base_te, b_lp, tau, beta).mean(),
                    "uniform_avoided_train": _avoided(sc_n, base_n, uni, tau, beta).mean(),
                    "uniform_avoided_test": _avoided(sc_te, base_te, uni, tau, beta).mean(),
                }
            )
            _log(f"sample-size study: replicate {rep}, {n} scenario days", t0)
    df = pd.DataFrame(rows)
    df["gain_train"] = df.lp_avoided_train / df.uniform_avoided_train - 1
    df["gain_test"] = df.lp_avoided_test / df.uniform_avoided_test - 1
    return df


def run_buffer(con, cfg: Config, train: list[str], test: list[str], summary: dict, t0) -> None:
    hinge = pd.read_csv(RESULTS / "hinge_carriers.csv").set_index("group").loc[cfg.lp_carrier]
    tau, beta = float(hinge["tau"]), float(hinge["beta"])
    rng = np.random.default_rng(cfg.seed)
    train_days = _days(con, cfg.lp_carrier, train)
    scen_days = sorted(rng.choice(train_days, size=cfg.lp_scenarios, replace=False).tolist())
    legs_tr = buffer_lp.load_legs(con, cfg.lp_carrier, scen_days)
    legs_te = buffer_lp.load_legs(con, cfg.lp_carrier, _days(con, cfg.lp_carrier, test))
    cells = _cells(legs_tr)
    n_cells = len(cells)
    sc_tr = buffer_lp.from_legs(legs_tr, cells, tau, beta)
    sc_te = buffer_lp.from_legs(legs_te, cells, tau, beta)
    zero = np.zeros(n_cells)
    base_tr = buffer_lp.daily_delay(sc_tr, buffer_lp.simulate(sc_tr, zero, tau, beta))
    base_te = buffer_lp.daily_delay(sc_te, buffer_lp.simulate(sc_te, zero, tau, beta))
    # Replay check: with no buffer the recursion reproduces the observed held-out delays.
    if not np.allclose(base_te, buffer_lp.daily_delay(sc_te, sc_te.obs_arr)):
        raise AssertionError("delay recursion does not replay the observed days")

    _log(f"LP: {len(sc_tr.p):,} legs, {sc_tr.n_days} scenario days, {n_cells} cells", t0)
    lp = buffer_lp.BufferLP(sc_tr, n_cells, tau, beta, cfg.lp_b_max)
    nbar = lp.nbar
    score = buffer_lp.propagated_delay_score(sc_tr, n_cells, tau, beta)
    boot = np.random.default_rng(cfg.seed + 1).multinomial(
        sc_te.n_days, np.full(sc_te.n_days, 1 / sc_te.n_days), size=2000
    )

    rows, allocations = [], {}
    for budget in cfg.lp_budgets:
        t = time.perf_counter()
        b_lp, _ = lp.solve(budget)
        solve_s = time.perf_counter() - t
        allocations[budget] = b_lp
        policies = {
            "LP-optimised": b_lp,
            "Uniform": buffer_lp.uniform_policy(nbar, budget, cfg.lp_b_max),
            "Greedy": buffer_lp.greedy_policy(score, nbar, budget, cfg.lp_b_max),
        }
        avoided = {p: _avoided(sc_te, base_te, b, tau, beta) for p, b in policies.items()}
        for name, b in policies.items():
            lo, hi = _ci(boot, avoided[name])
            dlo, dhi = _ci(boot, avoided[name] - avoided["Uniform"])
            rows.append(
                {
                    "policy": name,
                    "budget": budget,
                    "avoided_test": avoided[name].mean(),
                    "avoided_test_lo": lo,
                    "avoided_test_hi": hi,
                    "gain_vs_uniform": (avoided[name] - avoided["Uniform"]).mean(),
                    "gain_vs_uniform_lo": dlo,
                    "gain_vs_uniform_hi": dhi,
                    "avoided_train": _avoided(sc_tr, base_tr, b, tau, beta).mean(),
                    "buffer_test": buffer_lp.daily_buffer(sc_te, b).mean(),
                    "buffer_train": buffer_lp.daily_buffer(sc_tr, b).mean(),
                    "cells_padded": int((b > 1e-6).sum()),
                }
            )
        _log(f"budget {budget:>6.0f} min/day: LP solved in {solve_s:5.1f}s", t0)
    frontier = pd.DataFrame(rows)
    with np.errstate(divide="ignore", invalid="ignore"):
        frontier["avoided_per_buffer_min"] = frontier["avoided_test"] / frontier["buffer_test"]
    _write(frontier, "buffer_frontier.csv")

    keys = sorted(cells, key=cells.get)
    alloc = pd.DataFrame(
        {
            "station": [k[0] for k in keys],
            "dep_hour_local": [k[1] for k in keys],
            "turns_per_day": nbar,
            "buffer_min": allocations[cfg.lp_reference_budget],
            "propagated_delay_per_turn": score,
        }
    )
    alloc["buffer_min_per_day"] = alloc["turns_per_day"] * alloc["buffer_min"]
    alloc = alloc[alloc.buffer_min > 1e-6].sort_values("buffer_min_per_day", ascending=False)
    _write(alloc, "buffer_allocation.csv")

    _write(
        scenario_size_study(con, cfg, train_days, legs_te, base_te, tau, beta, t0),
        "buffer_scenario_sizes.csv",
    )

    ref = frontier[frontier.budget == cfg.lp_reference_budget].set_index("policy")
    summary["buffer"] = {
        "carrier": cfg.lp_carrier,
        "tau": tau,
        "beta": beta,
        "b_max": cfg.lp_b_max,
        "scenario_days": len(scen_days),
        "test_days": int(sc_te.n_days),
        "lp_rows": lp.shape[0],
        "lp_cols": lp.shape[1],
        "cells": n_cells,
        "turns_per_day_train": float(nbar.sum()),
        "baseline_delay_per_day_test": float(base_te.mean()),
        "reference_budget": cfg.lp_reference_budget,
        "reference": json.loads(
            ref.drop(columns=["budget"]).to_json(orient="index", double_precision=6)
        ),
    }


def run(con: duckdb.DuckDBPyConnection, cfg: Config | None = None) -> dict:
    cfg = cfg or Config()
    t0 = time.perf_counter()
    all_months = months(con)
    train, test = all_months[: -cfg.n_test_months], all_months[-cfg.n_test_months :]
    summary: dict = {"config": cfg.__dict__}
    run_descriptive(con, summary)
    _log("descriptive done", t0)
    run_propagation(con, cfg, train, test, summary)
    _log("propagation done", t0)
    run_contagion(con, cfg, all_months, summary)
    _log("contagion done", t0)
    run_buffer(con, cfg, train, test, summary, t0)
    _log("buffer LP done", t0)
    # Rounded so that a re-run on the same data rewrites byte-identical files.
    text = json.dumps(_round(summary), indent=2, default=str)
    (RESULTS / "summary.json").write_text(text + "\n")
    return summary
