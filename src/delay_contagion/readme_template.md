# delay-contagion

How a late flight infects the rest of the day through aircraft rotations, and where a limited budget of turnaround buffer absorbs the contagion best.

[![ci](https://github.com/Pchambet/delay-contagion/actions/workflows/ci.yml/badge.svg)](https://github.com/Pchambet/delay-contagion/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-0d9488)
![License MIT](https://img.shields.io/badge/license-MIT-64748b)
[![Report](https://img.shields.io/badge/report-live-d97706)](https://pchambet.github.io/delay-contagion/)

![Delay avoided against buffer spent on held-out days: LP-optimised vs marginal-value greedy vs greedy vs uniform padding](docs/figures/hero_frontier.png)

<!-- Generated from src/delay_contagion/readme_template.md by `make report`; edit the template. -->

## TL;DR

- **$reactionary_share of cause-coded US delay minutes are reactionary** (cause codes exist for arrivals 15+ min late): the aircraft arrived late from its previous leg. Late-aircraft delay per departing flight grows from $reactionary_6am min at 6:00 to $reactionary_peak by $reactionary_peak_hour ($flights flights, $window).
- **A turn absorbs delay up to a threshold, then passes it on about one-for-one.** $lp_carrier_name's effective minimum turn time is **τ = $lp_carrier_tau min**, with pass-through β = $lp_carrier_beta. On held-out months this two-parameter hinge reaches $mae_hinge min MAE against $mae_linear_slack for a linear model in inbound delay and slack; boosted trees do better ($mae_gbm), and the hinge is kept because it is interpretable and stays linear inside the decision model.
- **Each primary minute of departure delay adds $multiplier $multiplier_ci minutes of arrival delay later in the same aircraft's day**, from $events delayed departures matched to on-time ones of the same carrier, day, part of day and legs left ($mult_origin when the origin airport is matched too).
- **Placement beats volume.** Placed by a scenario LP, a buffer minute avoids **$lp_per_min min** of arrival delay on $test_days held-out days, **$lp_vs_uniform as much as uniform padding** ($uniform_per_min); padding the historically most-delayed turns does worse than uniform ($greedy_per_min).
- **Padding buys predictability, not speed.** "Delay" here is arrival delay against the padded schedule (the DOT on-time definition). The LP's $lp_avoided min/day are $lp_share_of_baseline of held-out delay minutes, no aircraft lands earlier on the clock, and on the share of arrivals under 15 minutes late fractional uniform padding scores slightly higher ($ontime_uniform vs $ontime_lp), largely a threshold artefact (its 5-minute version: $ontime_uniform_step).

## Why it matters

Most airline punctuality work forecasts the delay of one flight. Operations teams fight something else: one late aircraft drags its whole rotation, and schedule slack is the cheapest vaccine they control. Padding every turn is expensive (aircraft time is the scarcest resource an airline has); padding nothing makes the network fragile. The decision is *where* a few hundred minutes of slack buy the most punctuality, and that needs three things measured properly: how much delay is contagious, how a turn transmits it, and how a limited budget should be spread.

## Approach

```mermaid
flowchart LR
  A[BTS on-time files<br/>12 months, 7M flights] -->|DuckDB| B[Parquet cache]
  B --> C[dbt marts<br/>UTC legs, rotations, turns]
  C --> F[Hinge propagation model<br/>profiled tau, day bootstrap]
  C --> G[Matched contagion multiplier<br/>super-spreader airports]
  F --> H[SAA buffer LP<br/>HiGHS, $scenario_days scenario days]
  H --> I[Held-out evaluation<br/>vs heuristics, sensitivity]
```

The dbt layer (dbt-duckdb; 34 schema tests and 3 singular tests; `make dbt-fixture` builds it in CI on a committed fixture of $fixture_flights real flights around the March 2026 DST switch). The chart mirrors the dbt DAG; `make docs` generates the dbt docs site with the same lineage.

```mermaid
flowchart LR
  S[(flights parquet)] --> stg[stg_flights<br/>typed, local to UTC]
  seed[(seed: airports<br/>IANA zones)] --> stg
  stg --> seq[int_leg_sequence<br/>aircraft chains]
  seq --> rot[int_rotations<br/>turns]
  seq --> legs[fct_legs]
  rot --> turns[fct_turns]
  stg --> agg[agg_airport_hour<br/>agg_carrier_month<br/>agg_delay_causes]
```

1. **Data and clock.** The latest 12 published months of the BTS Reporting Carrier On-Time Performance files ($window) are discovered by probing the server, cached as Parquet and modelled in **dbt** on DuckDB. Times are local, so every leg is moved to UTC with its airport's IANA zone (first occurrence inside the repeated fall-back hour); the arrival date is resolved against the published block time, which handles red-eyes, the date line and both DST nights (hand-made test cases).
2. **Rotations.** Legs of the same tail are chained when the aircraft physically continues: same station, scheduled ground time between 0 and 5 h, non-negative actual ground time (otherwise the tail was swapped). This yields $turns turns.
3. **Propagation model.** `outbound = a + β · max(0, inbound − (slack − τ))`, fitted per carrier and per hub on the first 9 months, τ profiled on a 1-minute grid, CIs from a bootstrap over whole days, scored on the last 3 months against a constant, two linear models and boosted trees; at the $n_hubs busiest hubs (chosen on training months) τ spans $hub_tau_range min (`results/hinge_airports.csv`).
4. **Contagion multiplier.** For *clean starts* (aircraft ready, flight still late) the downstream arrival delay of the same aircraft is compared with on-time clean starts of the same carrier, day, part of day and number of legs left (4+ pooled), with stricter matching designs as robustness checks.
5. **Decision.** A sample-average-approximation LP chooses extra minutes per (station, departure hour) for $lp_carrier_name, subject to a daily budget, with the delay recursion along every aircraft chain as linear constraints ($lp_size). Scenarios are all $scenario_days training days; evaluation replays $test_days held-out days. The objective and the headline metric are arrival delay against the padded schedule.

## Results

**Reactionary delay builds through the day.** Late-arriving aircraft are a minor cause at dawn and match all primary causes combined by the evening.

![Reactionary vs primary delay minutes per flight by hour](docs/figures/reactionary_by_hour.png)

**A turn is a hinge, with a rounded knee.** Held-out data (dots) follow the training-month fit (lines): flat while the late inbound still leaves τ minutes on the ground, then a slope of about one. The knee is softer than the model's corner, and in summer Delta's plateau sits a few minutes above its fit. Slack only helps through the threshold: pinning τ at zero gives $mae_hinge_tau0 MAE, and adding slack linearly barely moves the linear model ($mae_linear to $mae_linear_slack). The boosted trees are better on $gbm_wins carriers with held-out data (Hawaiian, merged into Alaska, has none); the hinge keeps $hinge_share_of_gbm of their gain over the slack-aware linear model with two interpretable parameters, which is what lets the decision model stay linear.

![Hinge model: outbound delay vs ground time left](docs/figures/hinge_fit.png)

$carrier_table_md

τ intervals come from a 1-minute grid, so an interval like [57, 57] means "within a minute", not certainty.

**Contagion.** Excess downstream delay scales almost linearly with the primary delay (dashed: one-for-one), and delays that start in the morning travel furthest. By carrier the multiplier runs from $carrier_mult_low to $carrier_mult_high; $lp_carrier_name, whose aircraft fly $lp_carrier_legs_rank legs per aircraft-day ($lp_carrier_legs_per_day; median scheduled turn $lp_carrier_median_turn min, against $shortest_median_turn, the shortest), sits at $lp_carrier_multiplier (`results/carrier_rotations.csv`).

![Contagion dose-response and time-of-day multiplier](docs/figures/contagion.png)

The headline holds under stricter matching (`results/contagion_robustness.csv`). $coded_share_events of the "clean" starts still carry a late-aircraft cause code ($coded_share_min of their minutes), so "primary" is an approximation; dropping them changes little. Matching within the origin airport mainly removes the excess of small slips, which cluster at congested airports.

$robustness_table_md

**Super-spreaders.** Raw airport multipliers mostly track the carrier mix: $top_raw_spreader tops the raw ranking at $top_raw_multiplier, but its airlines alone predict $top_raw_mix, close to $lp_carrier_name's network multiplier ($lp_carrier_multiplier): the raw ranking is largely a carrier effect. Compared with each event's own carrier multiplier, the stations that add contagion of their own are congested hubs; the top three are $top_spreaders ($top_spreaders_range per primary minute above their carrier mix), and $top_spreaders_ci_note. Interactive map in the [report](https://pchambet.github.io/delay-contagion/).

![Super-spreader airports after carrier mix](docs/figures/superspreaders.png)

**Where buffer pays.** The budget is set at $budget min/day on training-day traffic; busier summer days fly more turns, so on the $test_days held-out days the LP schedules $lp_spend min/day and uniform padding $uniform_spend. Per buffer minute the comparison is at equal spend. 95% CIs from a bootstrap over whole weeks:

$policy_table_md

Delay avoided is arrival delay against the padded schedule, summed per day; clock lateness is lateness against the original timetable (baseline $clock_base min/day). A buffer on one turn moves every later departure of that aircraft's day back in the timetable, so the same minutes count as avoided on each later leg that would have been late: that is why a buffer minute can avoid more than one minute of delay, and why on the clock every policy adds lateness. The LP's edge over uniform padding is $lp_gain $lp_gain_ci min/day, paired over days. The most-delayed-turns rule loses to uniform ($greedy_gain $greedy_gain_ci), because those turns are late through their inbound, not through short slack. Fractional uniform padding is $uniform_seconds seconds per turn; the implementable version, 5 minutes on randomly chosen station-hours, gets $uniform_step_per_min per buffer minute. Fractional uniform padding's slight edge on the on-time share is mostly a threshold effect: a fraction of a minute on every turn nudges arrivals of exactly 15 minutes under the line, and the 5-minute version loses it ($ontime_uniform_step).

## Reproduce

```bash
make setup     # uv sync --locked (Python 3.12)
make data      # download + cache 12 months of BTS files
make run       # dbt build + models + LP + result tables + figures
make report    # site/index.html and this README, from results/
```

`make data` downloads about 350 MB of zip files (kept in `data/raw/`) and writes 80 MB of Parquet; the DuckDB warehouse needs about 4 GB. On a busy 10-core laptop `make run` took about 3 minutes for dbt and 26 for the analysis (the LP budget sweep over all training days and the scenario-size study dominate). DuckDB is capped at 3 GB and 3 threads, HiGHS and LightGBM at 3 threads. LightGBM is a runtime dependency only for the boosted-tree benchmark. `make test lint` (pytest, ruff, mypy) runs offline in under a minute; `make docs` builds the dbt docs on the fixture.

## Repository layout

```
dbt/                 dbt project: staging -> intermediate -> marts, schema + singular tests
  fixtures/          real-flight fixture used by CI
  seeds/airports.csv IATA -> IANA time zone + coordinates
src/delay_contagion/
  ingest.py          discover, download and cache BTS months
  propagation.py     hinge model, sufficient statistics, day bootstrap, baselines
  contagion.py       matched multiplier, robustness designs, carrier-adjusted ranking
  buffer_lp.py       delay recursion, SAA LP (HiGHS), heuristic policies
  analysis.py        chronological train/test pipeline -> results/
  figures.py, report.py, headlines.py, readme_template.md, report_template.html
results/             small result tables behind every number in this README
docs/figures/        static figures
site/index.html      report page (GitHub Pages)
tests/               UTC/DST, chaining, hinge recovery, LP toy optimum, contagion recovery
```

## Methodology notes and limitations

- **What is out of sample.** Hinge parameters, hub choice, LP buffers and heuristic rankings only see $train_window; every reported error or delay reduction is on $test_window. The held-out months are summer months with more traffic and delay than the average training day, which raises absolute minutes avoided for every policy alike.
- **The metric is schedule-relative, and the gain is small.** Delay avoided is arrival delay against the padded schedule. The LP's $lp_avoided min/day are $lp_share_of_baseline of the $baseline_delay held-out delay minutes per day. Against the original timetable every policy adds lateness ($clock_lp min/day for the LP, $clock_uniform for uniform), and the on-time share (A15) moves from $ontime_base to $ontime_lp (LP), $ontime_rounded (LP in 5-minute steps), $ontime_marginal (marginal greedy), $ontime_uniform (fractional uniform) and $ontime_uniform_step (uniform in 5-minute steps). The LP minimises delay minutes, so it targets long cascades rather than the 15-minute line. A budget-neutral variant that moves slack from loose turns to tight ones, keeping each timetable's length fixed (AhmadBeygi, Cohn and Lapp, 2010), would turn the reduction into clock time; it needs turn-level decisions and is not implemented here.
- **The decision evaluation is in-model.** Buffers are scored by replaying held-out days through the fitted recursion: with zero buffer it reproduces every observed delay exactly, but the LP optimised against the same hinge. Re-scoring the same buffers under other replay parameters moves the LP's absolute gain between $sens_lp_range min/day, while its per-minute edge over uniform padding stays in the range $sens_ratio_range. The ranking is robust; absolute minutes depend on the model.

$sensitivity_table_md

- **β and τ are not clean causal effects.** Several carriers and hubs have β above 1 with intervals that exclude 1, which a turn cannot do physically: a common shock (weather or ATC at the station) delays both the inbound and the outbound's own departure. The same confounding can shift τ. The LP reads β as the causal effect of slack, so its absolute gains inherit this bias; the sensitivity table above bounds how much it matters for the ranking.
- **Buffers are continuous.** The LP is a relaxation: its buffers are fractional minutes. Rounded to 5-minute schedule steps, they avoid $rounded_per_min min per buffer minute ($rounded_spend min/day spent). Crew legality, gate use and the commercial value of a departure time are outside the model.
- **Scenario count.** Re-solving the LP on random draws of $size_small to $size_full training days, its in-sample edge over uniform padding falls from $size_train_small to $size_train_large while the held-out edge rises from $size_test_small to $size_test_large ($size_test_90 at 90 days, $size_test_prev at $size_prev): beyond about 90 days the held-out edge levels off. Read the gap with care: in-sample and held-out days are in different delay regimes (summer is busier), so the two curves meeting is not a test of SAA adequacy. No optimality-gap estimate (independent replications in the style of Mak, Morton and Wood) was run.

![In-sample vs out-of-sample LP gain as the scenario set grows](docs/figures/scenario_sizes.png)

- **The contagion multiplier is observational.** Matching on carrier, day, part of day and legs left removes carrier-wide day-level conditions, not local weather or a ground stop at one airport, nor aircraft-level confounders; with the origin airport added to the match the multiplier is $mult_origin. The first bin of the dose-response (slips of 1 to 14 minutes, $small_bin_mult) is inflated by airport-level confounding: matched within the origin it drops from $small_slip_mult to $small_slip_origin. The headline uses 15+ minute delays. Legs left is counted on the realised chain, after cancellations and swaps that the delay itself may cause, so it is partly a post-treatment quantity; matching on scheduled legs would need the cancelled legs kept in the chain.
- **Uncertainty is conditional on the models.** Policy CIs resample whole weeks of held-out days ($test_weeks blocks); hinge and contagion CIs resample days.
- **Chains are conservative.** They break at cancellations, diversions, tail swaps and ground times over 5 hours, so delay that jumps aircraft (swaps, crew connections) is not counted; in that respect the multiplier understates network-wide contagion.
- **Cause codes are self-reported** by carriers and only exist for arrivals 15+ minutes late; the late-aircraft share ranges from $cause_share_range across carriers, so cross-carrier comparisons of cause shares need care.
- **Data window.** The fitting window drops turns with inbound delay outside −90 to 600 minutes or outbound outside −60 to 600 ($filter_excluded of held-out turns). $carriers_without_test reports under Alaska from January 2026 (no held-out turns; $n_carriers_test carriers are scored) and Spirit's reporting ends in May 2026 ($spirit_test_turns held-out turns).

## References

- US DOT Bureau of Transportation Statistics, *Reporting Carrier On-Time Performance (1987-present)*, TranStats. Public domain. https://www.transtats.bts.gov/
- Beatty, R., Hsu, R., Berry, L., Rome, J. (1999). Preliminary evaluation of flight delay propagation through an airline schedule. *Air Traffic Control Quarterly* 7(4).
- AhmadBeygi, S., Cohn, A., Guan, Y., Belobaba, P. (2008). Analysis of the potential for delay propagation in passenger airline networks. *Journal of Air Transport Management* 14(5).
- AhmadBeygi, S., Cohn, A., Lapp, M. (2010). Decreasing airline delay propagation by re-allocating scheduled slack. *IIE Transactions* 42(7).
- Lan, S., Clarke, J.-P., Barnhart, C. (2006). Planning for robust airline operations: optimizing aircraft routings and flight departure times to minimize passenger disruptions. *Transportation Science* 40(1).
- Fleurquin, P., Ramasco, J. J., Eguiluz, V. M. (2013). Systemic delay propagation in the US airport network. *Scientific Reports* 3, 1159.
- Shapiro, A., Dentcheva, D., Ruszczynski, A. (2014). *Lectures on Stochastic Programming*, 2nd ed. SIAM (sample average approximation).
- Mak, W.-K., Morton, D. P., Wood, R. K. (1999). Monte Carlo bounding techniques for determining solution quality in stochastic programs. *Operations Research Letters* 24(1-2).
- Huangfu, Q., Hall, J. A. J. (2018). Parallelizing the dual revised simplex method. *Mathematical Programming Computation* 10 (HiGHS).
- Ke, G. et al. (2017). LightGBM: a highly efficient gradient boosting decision tree. *NeurIPS* 30.
- `airportsdata` (airport IANA time zones), dbt-duckdb, DuckDB.

---

Built by [Pierre Chambet](https://github.com/Pchambet) — decision science for operations under uncertainty.
