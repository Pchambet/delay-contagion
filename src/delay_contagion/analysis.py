"""End-to-end analysis on the warehouse marts; writes the small result tables in `results/`.

Figures and the report are rendered from `results/` alone, so they can be rebuilt without
the ~350 MB of raw data.

Time split: the 12 months are cut chronologically into 9 training months and 3 held-out
months. Every fitted quantity (hinge parameters, LP buffers, heuristic rankings) sees only
training months; every reported performance number is computed on held-out months.
"""

from __future__ import annotations

import datetime as dt
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
    # SAA scenarios: every training day (None) or a random sample of that many days; the
    # scenario-size study shows what fewer days cost.
    lp_scenarios: int | None = None
    # HiGHS interior point (with crossover): each budget is a cold solve of a few minutes;
    # warm-started dual simplex slows down sharply as the budget grows on the full year.
    lp_solver: str = "ipm"
    # Scenario-size study; the full training set is always added as the last size.
    lp_scenario_sizes: list[int] = field(default_factory=lambda: [10, 30, 60, 90, 150, 210])
    lp_size_replicates: int = 3
    lp_b_max: float = 20.0  # max extra minutes on any one turn
    lp_round_step: float = 5.0  # timetables move in 5-minute steps
    lp_budgets: list[float] = field(
        default_factory=lambda: [0, 150, 300, 600, 900, 1200, 1800, 2400]
    )
    lp_reference_budget: float = 600
    # Evaluation-model sensitivity: re-score the same buffers under other (tau, beta).
    sens_tau_offsets: list[float] = field(default_factory=lambda: [-10, -5, 5, 10])
    sens_beta: float = 0.8
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


def _round(obj: object, digits: int = 6) -> object:
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


Con = duckdb.DuckDBPyConnection
Summary = dict[str, object]


def run_descriptive(con: Con, summary: Summary) -> None:
    summary["network"] = descriptive.network_summary(con)
    _write(descriptive.cause_shares_by_carrier(con), "cause_shares_by_carrier.csv")
    _write(descriptive.reactionary_by_hour(con), "reactionary_by_hour.csv")
    _write(descriptive.carrier_rotations(con), "carrier_rotations.csv")


MODELS = ("constant", "linear", "linear_slack", "hinge_tau0", "gbm", "hinge")


def run_propagation(
    con: Con, cfg: Config, train: list[str], test: list[str], summary: Summary, t0: float
) -> pd.DataFrame:
    """Fit and score the hinge; returns the per-carrier fits at full precision (the LP's
    input, so it never depends on the rounded publication table)."""
    carriers = propagation.fit_groups(con, "carrier", train, n_boot=cfg.n_boot, seed=cfg.seed)
    train_list = ", ".join(f"'{m}'" for m in train)
    hubs = [
        r[0]
        for r in con.execute(
            f"select station from marts.fct_turns where flight_month in ({train_list}) "
            "group by 1 order by count(*) desc limit ?",
            [cfg.n_hub_airports],
        ).fetchall()
    ]
    airports = propagation.fit_groups(con, "station", train, hubs, n_boot=cfg.n_boot,
                                      seed=cfg.seed)  # fmt: skip
    _log("hinge fits done", t0)
    turns_tr = propagation.load_turns(con, train)
    turns_te = propagation.load_turns(con, test)
    turns_te["p_gbm"] = propagation.gbm_predictions(turns_tr, turns_te, seed=cfg.seed)
    _log("gradient-boosted benchmark done", t0)
    scores_c = propagation.holdout_scores(carriers, "carrier", turns_tr, turns_te)
    scores_a = propagation.holdout_scores(airports, "station", turns_tr, turns_te)
    del turns_tr, turns_te
    fitted = carriers
    # Inner merge: a carrier with no held-out turns (Hawaiian, merged into Alaska) has no
    # score; it is listed in the summary instead of disappearing silently.
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
        "test_filter_excluded_share": propagation.filter_excluded_share(con, test),
        "mae": {m: float((carriers[f"mae_{m}"] * n).sum() / n.sum()) for m in MODELS},
        "rmse": {
            m: float(np.sqrt((carriers[f"rmse_{m}"] ** 2 * n).sum() / n.sum())) for m in MODELS
        },
        "carriers_hinge_beats_linear_mae": int(
            (carriers["mae_hinge"] < carriers["mae_linear"]).sum()
        ),
        "carriers_hinge_beats_linear_slack_mae": int(
            (carriers["mae_hinge"] < carriers["mae_linear_slack"]).sum()
        ),
        "carriers_hinge_beats_gbm_mae": int((carriers["mae_hinge"] < carriers["mae_gbm"]).sum()),
        "n_carriers": len(carriers),
        "carriers_without_test": sorted(set(fitted["group"]) - set(carriers["group"])),
        "hubs": hubs,
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
    return fitted


def run_contagion(con: Con, cfg: Config, all_months: list[str], summary: Summary) -> None:
    n = contagion.build_events(con, all_months)
    seed = cfg.seed
    curve = contagion.multiplier_curve(con, seed=seed)
    by_hour = contagion.multiplier_by(con, "dep_hour_local", min_events=500, seed=seed)
    by_carrier = contagion.multiplier_by(con, "carrier", seed=seed)
    by_airport = contagion.multiplier_by(con, "origin", min_events=cfg.min_events_airport,
                                         seed=seed)  # fmt: skip
    adjusted = contagion.carrier_adjusted_by(con, "origin", by_carrier, seed=seed)
    by_airport = by_airport.merge(
        adjusted[["origin", "carrier_mix", "excess_over_mix", "excess_over_mix_lo",
                  "excess_over_mix_hi"]], on="origin", how="left"
    )  # fmt: skip
    coords = con.execute(
        "select iata as origin, name, city, state, lat, lon from main.airports"
    ).df()
    by_airport = by_airport.merge(coords, on="origin", how="left")
    by_airport = by_airport.sort_values("excess_over_mix", ascending=False).reset_index(drop=True)
    _write(curve, "contagion_curve.csv")
    _write(by_hour, "contagion_by_hour.csv")
    _write(by_carrier, "contagion_by_carrier.csv")
    _write(by_airport, "superspreaders.csv")
    summary["contagion"] = {
        "matched_events": int(n),
        "overall": contagion.overall(con, seed=seed),
        "late_aircraft_coded": contagion.late_aircraft_coded(con),
        "n_airports_ranked": len(by_airport),
    }
    # Last: rebuilds the event table once per matching variant.
    _write(contagion.robustness(con, all_months, seed=seed), "contagion_robustness.csv")


def _days(con: Con, carrier: str, months_: list[str]) -> list[dt.date]:
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


def _cells(legs: pd.DataFrame) -> dict[tuple[str, int], int]:
    """Decision cells = (station, local hour) of every turn seen in the training scenarios."""
    turns = legs[legs.leg_seq > 1]
    keys = sorted(set(zip(turns["origin"], turns["dep_hour_local"], strict=True)))
    return {k: i for i, k in enumerate(keys)}


def _avoided(
    sc: buffer_lp.Scenarios, base: np.ndarray, b: np.ndarray, tau: float, beta: float
) -> np.ndarray:
    """Delay minutes avoided per day under buffers `b`, relative to the observed day."""
    return base - buffer_lp.daily_delay(sc, buffer_lp.simulate(sc, b, tau, beta))


def _ci(boot: np.ndarray, per_day: np.ndarray) -> tuple[float, float]:
    draws = boot @ per_day / boot.sum(1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def week_index(days: list[dt.date] | np.ndarray) -> np.ndarray:
    """7-day block of each day, counted from the first day."""
    d = pd.to_datetime(pd.Series(days))
    return ((d - d.min()).dt.days // 7).to_numpy()


def week_bootstrap(days: list[dt.date] | np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Day weights (n_boot, n_days) from resampling whole weeks.

    Days are not independent (weekly schedules, multi-day disruptions), so held-out CIs
    resample 7-day blocks of consecutive days.
    """
    week = week_index(days)
    n_weeks = int(week.max()) + 1
    draws = np.random.default_rng(seed).multinomial(
        n_weeks, np.full(n_weeks, 1 / n_weeks), size=n_boot
    )
    return draws[:, week]


def scenario_size_study(
    con: Con,
    cfg: Config,
    train_days: list[dt.date],
    legs_te: pd.DataFrame,
    base_te: np.ndarray,
    tau: float,
    beta: float,
    t0: float,
    full_solution: np.ndarray | None = None,
) -> pd.DataFrame:
    """Out-of-sample value of the LP as the scenario set grows, over independent draws.

    In-sample gains are optimistic (the LP fits the days it sees); the held-out gain is what
    a planner would actually get. Each replicate is a fresh random order of training days,
    with nested sets of the first S days; the full training set is solved once (or reused
    from the main run via `full_solution`, which must use the same cells).
    """
    rows = []
    sizes = [n for n in cfg.lp_scenario_sizes if n < len(train_days)]
    for rep in range(cfg.lp_size_replicates):
        order = np.random.default_rng(cfg.seed + 100 + rep).permutation(len(train_days))
        for n in sizes + ([len(train_days)] if rep == 0 else []):
            legs_n = buffer_lp.load_legs(con, cfg.lp_carrier, [train_days[i] for i in order[:n]])
            cells = _cells(legs_n)
            sc_n = buffer_lp.from_legs(legs_n, cells, tau, beta)
            sc_te = buffer_lp.from_legs(legs_te, cells, tau, beta)
            nbar = buffer_lp.cell_turns_per_day(sc_n, len(cells))
            if n == len(train_days) and full_solution is not None:
                b_lp = full_solution
            else:
                lp_n = buffer_lp.BufferLP(
                    sc_n, len(cells), tau, beta, cfg.lp_b_max, solver=cfg.lp_solver
                )
                b_lp, _ = lp_n.solve(cfg.lp_reference_budget)
            uni = buffer_lp.uniform_policy(nbar, cfg.lp_reference_budget, cfg.lp_b_max)
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


def _policies(
    lp: buffer_lp.BufferLP,
    budget: float,
    cfg: Config,
    greedy_score: np.ndarray,
    marginal_score: np.ndarray,
) -> dict[str, np.ndarray]:
    b_lp, _ = lp.solve(budget)
    nbar = lp.nbar
    return {
        "LP-optimised": b_lp,
        "LP, 5-min steps": buffer_lp.round_to_step(b_lp, cfg.lp_round_step),
        "Marginal greedy": buffer_lp.greedy_policy(marginal_score, nbar, budget, cfg.lp_b_max),
        "Greedy": buffer_lp.greedy_policy(greedy_score, nbar, budget, cfg.lp_b_max),
        "Uniform": buffer_lp.uniform_policy(nbar, budget, cfg.lp_b_max),
        "Uniform, 5-min steps": buffer_lp.uniform_step_policy(
            nbar, budget, cfg.lp_round_step, seed=cfg.seed
        ),
    }


def _schedule_metrics(
    sc: buffer_lp.Scenarios, b: np.ndarray, tau: float, beta: float
) -> dict[str, float]:
    """What a buffer does on the clock, not only against the padded timetable.

    `ontime15`: share of arrivals less than 15 min late against the padded schedule (the DOT
    on-time measure). `clock_late_per_day`: positive lateness against the *original*
    timetable, i.e. arrival delay plus the timetable shift, summed per day.
    """
    arr = buffer_lp.simulate(sc, b, tau, beta)
    clock = arr + buffer_lp.schedule_shift(sc, b)
    return {
        "ontime15": float((arr < 15).mean()),
        "ontime15_clock": float((clock < 15).mean()),
        "clock_late_per_day": float(buffer_lp.daily_delay(sc, clock).mean()),
    }


def run_buffer(
    con: Con,
    cfg: Config,
    train: list[str],
    test: list[str],
    hinge: pd.DataFrame,
    summary: Summary,
    t0: float,
) -> None:
    """`hinge`: per-carrier fits from `run_propagation` (columns group, tau, beta)."""
    params = hinge.set_index("group").loc[cfg.lp_carrier]
    tau, beta = float(params["tau"]), float(params["beta"])
    rng = np.random.default_rng(cfg.seed)
    train_days = _days(con, cfg.lp_carrier, train)
    test_days = _days(con, cfg.lp_carrier, test)
    if cfg.lp_scenarios is None:
        scen_days = train_days
    else:
        pick = rng.choice(len(train_days), size=cfg.lp_scenarios, replace=False)
        scen_days = sorted(train_days[i] for i in pick)
    legs_tr = buffer_lp.load_legs(con, cfg.lp_carrier, scen_days)
    legs_te = buffer_lp.load_legs(con, cfg.lp_carrier, test_days)
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
    lp = buffer_lp.BufferLP(sc_tr, n_cells, tau, beta, cfg.lp_b_max, solver=cfg.lp_solver)
    nbar = lp.nbar
    greedy_score = buffer_lp.propagated_delay_score(sc_tr, n_cells, tau, beta)
    marginal_score = buffer_lp.marginal_value_score(sc_tr, nbar, tau, beta, cfg.lp_b_max)
    _log("marginal-value scores done", t0)
    # Same day order as the scenario day index (sorted chain dates).
    te_dates = np.sort(legs_te["flight_date"].unique())
    boot = week_bootstrap(te_dates, 2000, cfg.seed + 1)

    rows, reference = [], {}
    for budget in cfg.lp_budgets:
        t = time.perf_counter()
        policies = _policies(lp, budget, cfg, greedy_score, marginal_score)
        solve_s = time.perf_counter() - t
        if budget == cfg.lp_reference_budget:
            reference = policies
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
        _log(f"budget {budget:>6.0f} min/day: policies built in {solve_s:5.1f}s", t0)
    frontier = pd.DataFrame(rows)
    with np.errstate(divide="ignore", invalid="ignore"):
        frontier["avoided_per_buffer_min"] = frontier["avoided_test"] / frontier["buffer_test"]
    _write(frontier, "buffer_frontier.csv")

    keys = sorted(cells, key=lambda k: cells[k])
    alloc = pd.DataFrame(
        {
            "station": [k[0] for k in keys],
            "dep_hour_local": [k[1] for k in keys],
            "turns_per_day": nbar,
            "buffer_min": reference["LP-optimised"],
            "propagated_delay_per_turn": greedy_score,
            "marginal_value_per_min": marginal_score,
        }
    )
    alloc["buffer_min_per_day"] = alloc["turns_per_day"] * alloc["buffer_min"]
    alloc = alloc[alloc.buffer_min > 1e-6].sort_values("buffer_min_per_day", ascending=False)
    _write(alloc, "buffer_allocation.csv")

    # Clock-time view and the DOT on-time share at the reference budget.
    schedule = pd.DataFrame(
        [{"policy": "No buffer", **_schedule_metrics(sc_te, zero, tau, beta)}]
        + [{"policy": p, **_schedule_metrics(sc_te, b, tau, beta)} for p, b in reference.items()]
    )
    schedule["clock_late_change_per_day"] = (
        schedule["clock_late_per_day"] - schedule.loc[0, "clock_late_per_day"]
    )
    _write(schedule, "buffer_schedule_metrics.csv")

    # The held-out replay uses the hinge the LP optimised against: re-score the same buffers
    # under other evaluation parameters, re-deriving primary delays so each replay stays exact.
    sens = []
    settings = [(tau + d, beta) for d in cfg.sens_tau_offsets] + [(tau, cfg.sens_beta)]
    for tau_e, beta_e in [(tau, beta), *settings]:
        sc_e = buffer_lp.from_legs(legs_te, cells, tau_e, beta_e)
        base_e = buffer_lp.daily_delay(sc_e, sc_e.obs_arr)
        for name, b in reference.items():
            avoided_e = _avoided(sc_e, base_e, b, tau_e, beta_e).mean()
            spent = buffer_lp.daily_buffer(sc_e, b).mean()
            sens.append({"tau_eval": tau_e, "beta_eval": beta_e, "policy": name,
                         "avoided_test": avoided_e, "avoided_per_buffer_min": avoided_e / spent})  # fmt: skip
    sens = pd.DataFrame(sens)
    uni = sens[sens.policy == "Uniform"].set_index(["tau_eval", "beta_eval"])
    sens["per_min_vs_uniform"] = [
        r.avoided_per_buffer_min / uni.loc[(r.tau_eval, r.beta_eval), "avoided_per_buffer_min"]
        for r in sens.itertuples()
    ]
    _write(sens, "buffer_sensitivity.csv")
    _log("schedule metrics and evaluation sensitivity done", t0)

    _write(
        scenario_size_study(
            con,
            cfg,
            train_days,
            legs_te,
            base_te,
            tau,
            beta,
            t0,
            full_solution=reference["LP-optimised"] if cfg.lp_scenarios is None else None,
        ),
        "buffer_scenario_sizes.csv",
    )

    ref = frontier[frontier.budget == cfg.lp_reference_budget].set_index("policy")
    uni_curve = frontier[frontier.policy == "Uniform"].sort_values("buffer_test")
    lp_spend = float(ref.loc["LP-optimised", "buffer_test"])
    summary["buffer"] = {
        "carrier": cfg.lp_carrier,
        "tau": tau,
        "beta": beta,
        "b_max": cfg.lp_b_max,
        "scenario_days": len(scen_days),
        "train_days_available": len(train_days),
        "test_days": int(sc_te.n_days),
        "test_weeks": int(week_index(te_dates).max()) + 1,
        "lp_rows": lp.shape[0],
        "lp_cols": lp.shape[1],
        "cells": n_cells,
        "turns_per_day_train": float(nbar.sum()),
        "turns_per_day_test": float((sc_te.cell >= 0).sum() / sc_te.n_days),
        "baseline_delay_per_day_test": float(base_te.mean()),
        "reference_budget": cfg.lp_reference_budget,
        # Uniform padding interpolated at the LP's held-out spend: equal-spend comparison.
        "uniform_at_lp_spend": float(
            np.interp(lp_spend, uni_curve["buffer_test"], uni_curve["avoided_test"])
        ),
        "reference": json.loads(
            ref.drop(columns=["budget"]).to_json(orient="index", double_precision=6)
        ),
    }


def run(con: duckdb.DuckDBPyConnection, cfg: Config | None = None) -> dict:
    cfg = cfg or Config()
    t0 = time.perf_counter()
    all_months = months(con)
    train, test = all_months[: -cfg.n_test_months], all_months[-cfg.n_test_months :]
    summary: Summary = {"config": cfg.__dict__}
    run_descriptive(con, summary)
    _log("descriptive done", t0)
    hinge = run_propagation(con, cfg, train, test, summary, t0)
    _log("propagation done", t0)
    run_contagion(con, cfg, all_months, summary)
    _log("contagion done", t0)
    run_buffer(con, cfg, train, test, hinge, summary, t0)
    _log("buffer LP done", t0)
    # Rounded so that a re-run on the same data rewrites byte-identical files.
    text = json.dumps(_round(summary), indent=2, default=str)
    (RESULTS / "summary.json").write_text(text + "\n")
    return summary
