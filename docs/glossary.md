# Glossary

Core domain terms of the pipeline (candidates → residuals → simulator), defined as the
code implements them. Process terms (epic, milestone, regression harness, …) live in
[CONTRIBUTING.md](../CONTRIBUTING.md). Changes to an existing definition need explicit
approval.

**As-of date** (`asof_date`) — The outer refit date at which a candidate's weights were
fitted. The weights stay frozen at this date for the candidate's lifetime, and the
simulator rebuilds the candidate's residuals with the residual model fitted at this
date, never at the simulation day. Outer refit dates come from a frequency walk or an
explicit date list. Distinct from *fit date*. Defined in: `src/simulator/types.py`
`CandidateRef`; `src/candidates/pair_candidate_panel_creator.py`
`resolve_asof_datetimes`.

**Candidate** — One pair scored at one as-of date with one hedge ratio method: frozen
weights plus mean-reversion diagnostics, identified by `candidate_id` (a 16-hex hash
of group, candidate type, hedge ratio method, as-of datetime and spread id). The
candidate panel holds every scored candidate including invalid ones (`is_valid`,
`why_invalid`); `select_candidates` filters it to the selected panel, which the
simulator turns into `CandidateRef`s. Defined in:
`src/candidates/pair_candidate_panel_creator.py` `_build_pair_candidate_rows_for_date`;
`src/candidates/candidate_panel.py` `make_candidate_id`; `src/simulator/types.py`
`CandidateRef`.

**Entry / exit** — An entry is an `OpenCandidateAction` the trader emits for an active,
signal-ready candidate with no open position on the same `(spread_id, residual_key)`:
long (direction +1) if z-score ≤ −`entry_z`, short (−1) if z-score ≥ `entry_z`. An
exit is a `CloseCandidateAction` with reason `exit_rule`, `time_stop_<n>d`
(`days_open` ≥ `max_holding_days`) or `z_cross` (long: z ≥ −`exit_z`; short:
z ≤ `exit_z`). Defined in: `src/simulator/traders/pair_spread_mean_reversion.py`
`_threshold_entries`, `_generate_closes`.

**Fit date** (`fit_date`) — The date a residual model was fitted, using only data up to
and including that date; the key of `residual_params.parquet`. The offline panel path
fits on every business day; the in-simulator candidate generator fits on outer refit
dates only. Defined in: `src/residuals/causal_residuals.py` `fit_causal_residual_model`,
`save_residual_params`.

**Group** (`group_id`) — A set of member tickers within which pairs are formed,
together with a proxy ETF, a benchmark and a risk-free series; residual models are
fitted per group. A ticker belongs to exactly one group. The market data layer calls
these sectors. Defined in: `src/data/returns.py` `GroupReturnBundle`;
`src/types/common.py` `GroupId`.

**Hedge ratio** — The weight β of a pair's right leg relative to its left leg, giving
weights `[1, −β]`. Chosen by `hedge_ratio_method`: `unit` (β = 1), `ols` (OLS without
intercept of left on right residual returns) or `pca` (second principal component of
the 2×2 covariance, scaled to left = 1, right negative). Estimated on the trailing
`hedge_ratio_lb` window of the pair's own non-NaN rows, and only if at least `min_obs`
rows remain. Defined in: `src/candidates/pair_candidate_panel_creator.py`
`_compute_pair_weights`; `src/residuals/spreads.py` `ols_beta_no_intercept`.

**Mean-reversion diagnostics** (`kappa`, `half_life`, `mr_score`) — From the regression
Δx_t = a + b·x_{t−1} on the demeaned spread level x: `kappa` = −b,
`half_life` = ln 2 / `kappa`, `mr_score` = `kappa` / std of the regression residuals.
Computed at panel build over `mr_diag_lb` with validity gates (`min_kappa`,
`max_half_life`, …), and in the simulator over `MRDiagnosticsConfig.lookback`. Defined
in: `src/candidates/pair_candidate_panel_creator.py` `_fast_pair_diagnostics`;
`src/simulator/candidate_signals.py` `_compute_mr_diagnostics`.

**Pair** — A spread with exactly two legs (`candidate_type == "pair"`, `n_legs == 2`).
Its spread id is `A|B` with tickers sorted alphabetically; the first ticker is the
left leg (weight 1). The only candidate type the pipeline produces. Defined in:
`src/candidates/pair_candidate_panel_creator.py` `PairSpreadConfig`;
`src/residuals/spreads.py` `generate_spread_ids`.

**Position** — An open holding in one candidate (`LiveCandidatePosition`), keyed by
`candidate_id`. It carries a direction (+1 long spread, −1 short spread), a
`pair_notional` (gross dollar notional set at entry by `SizingEngine`) and integer
units per ticker: direction × `pair_notional` × wᵢ / Σ|w| / priceᵢ, with the
candidate's weights frozen at entry. Defined in: `src/simulator/types.py`
`LiveCandidatePosition`; `src/simulator/position_translator.py` `translate_open`;
`src/simulator/sizing_engine.py` `SizingEngine`.

**Residual** (residual return) — A member ticker's return minus a weighted
least-squares fit on a constant, the benchmark return and the proxy's own residual
(the proxy regressed on constant + benchmark). Optionally on excess returns
(risk-free subtracted) and with leading residual principal components projected out.
Configured by `CausalResidualConfig` (modes `eq_rolling`, `eq_expanding`,
`decay_expanding`) and identified by its `residual_key`, e.g. `exp_hl504_mh1008_rf`.
Fitted causally at a fit date, then applied unchanged to any return slice. Defined
in: `src/residuals/causal_residuals.py` `fit_causal_residual_model`,
`apply_causal_residual_model`, `CausalResidualConfig.key`.

**Signal** — A candidate's per-day z-score together with `is_signal_ready`, which is
true only if every z-score lookback yields a finite mean and a positive std. The
simulator collects these per day into the signal frame (`candidate_states`) that the
trader reads; there is no separate signal object. Defined in:
`src/simulator/candidate_signals.py` `CandidateSignalGenerator.build_signal_frame`;
`src/simulator/types.py` `CandidateMarketSnapshot`.

**Spread** — A weighted sum of member tickers' residual returns, Σ wᵢ rᵢ (spread
return); its level is the cumulative sum of spread returns. Identified by `spread_id`,
the member tickers sorted and joined by `|`; the spread id names the members, not the
weights. Defined in: `src/residuals/spreads.py` `spread_id`;
`src/analytics/spread_primitives.py` `compute_spread_level`.

**Trade** — One position from entry to exit, identified by `trade_id` =
`<entry date>:<first 8 characters of candidate_id>` and recorded at exit as
`ClosedCandidateTrade` with gross and net realised PnL, transaction and borrow costs.
Defined in: `src/simulator/types.py` `ClosedCandidateTrade`;
`src/simulator/simulator.py` `_make_trade_id`.

**Z-score** — (level − mean) / std of the spread level, with mean and std rolling over
`lookback` days or EWM with halflife `lookback` (`ZScoreConfig.method`). With several
lookbacks the z-score is the weighted sum of the per-lookback z-scores (weights
normalised to 1), and `roll_mean` / `roll_std` report the first lookback. A
`timescale_label` (`<residual_key>__zlb<n>` or `__zhl<n>`) names one residual-key and
z-score-config combination. Defined in: `src/simulator/candidate_signals.py`
`_finalize_batch_states`, `_compute_z_score`; `src/simulator/config.py` `ZScoreConfig`.
