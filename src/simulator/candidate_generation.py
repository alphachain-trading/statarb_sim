"""
F2 C6a: the in-simulator candidate generator — outer-date-only, not wired in.

Reuses the exact same scoring functions the offline walk
(pair_candidate_panel_creator.create_pair_candidate_panel) already calls —
resolve_asof_datetimes, fit_causal_residual_model, _build_pair_candidate_rows_for_date
— so this is not a second implementation of panel scoring, only a thinner walk over
it: outer refit dates only, no daily residual-fit grid.

Outer-date-only is justified by two independent facts, per F2_spec.md:
- D5: fit_causal_residual_model is a pure function of (bundle, date, cfg) with no
  state carried across the walk (see F2_spec.md's D5 note for the path:line
  evidence) -- so a fit at an outer date d gives the identical result whether or
  not intermediate daily fits happened in between. Removing them changes nothing
  at the dates that still get fit.
- P3 (F2 pre-check): after C3, zero remaining consumers read fits at non-outer
  dates, and the PC-variance-ratio mechanism that was the only other daily-grid
  reader was already dead code, deleted at C1.

Not wired into Simulator.run() yet (C6b's job). This module exists to be tested
for exact equality against the offline path's own output on the harness
universes (test_candidate_generation_equivalence.py) before anything depends on
it.

`resolve_asof_datetimes` can return a date that is not actually in
`bundle.aligned_returns.index` (found while building this test —
`docs/refactor/found.md`, "make_ranking_dates returns resample bin labels, not
the actual last-data date"): a resample-and-take-last bin label can be a market
holiday that literally has no row. The offline walk never notices, because it
only builds candidate rows on dates that are *also* in its daily fit grid
(also from `make_ranking_dates`, and empty for a holiday) — this generator has
no daily grid to filter through, so it must filter phantom dates itself, or it
would produce a candidate keyed to a date with no real return.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.candidates.pair_candidate_panel_creator import (
    PairSpreadConfig,
    _build_pair_candidate_rows_for_date,
    resolve_asof_datetimes,
)
from src.data.returns import GroupReturnBundle
from src.residuals.causal_residuals import (
    CausalResidualConfig,
    FittedCausalResidualModel,
    fit_causal_residual_model,
)


def generate_candidates(
    *,
    bundle: GroupReturnBundle,
    residual_cfg: CausalResidualConfig,
    pair_cfg: PairSpreadConfig,
    hedge_ratio_lb: int,
    mr_diag_lb: int,
    frequency: str | None = None,
    dates: list[str | pd.Timestamp] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    max_steps: int | None = None,
    debug: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[pd.Timestamp, FittedCausalResidualModel]]:
    """
    Generate candidate rows, weight rows, and fitted residual params for one
    group, fitting only on outer refit dates.

    Returns (rows, weight_rows, fitted_params) — the same three products
    create_pair_candidate_panel's walk produces (panel rows, weights.parquet
    rows, residual_params), but keyed only by outer asof_date, not the daily
    grid.
    """
    asof_datetimes = resolve_asof_datetimes(
        bundle=bundle,
        residual_cfg=residual_cfg,
        frequency=frequency,
        dates=dates,
        start_date=start_date,
        end_date=end_date,
        max_steps=max_steps,
    )
    # See module docstring: a resample bin label need not be an actual date
    # in the data (e.g. a market holiday). Drop any that aren't -- the
    # offline walk's daily-grid gating already excludes these; this is the
    # equivalent filter for a walk with no daily grid to filter through.
    aligned_index = bundle.aligned_returns.index
    asof_datetimes = asof_datetimes[asof_datetimes.isin(aligned_index)]

    rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    fitted_params: dict[pd.Timestamp, FittedCausalResidualModel] = {}

    for dt in asof_datetimes:
        dt = pd.Timestamp(dt)
        model = fit_causal_residual_model(bundle=bundle, date=dt, cfg=residual_cfg)
        fitted_params[dt] = model

        date_rows, date_weight_rows = _build_pair_candidate_rows_for_date(
            bundle=bundle,
            asof_datetime=dt,
            residual_cfg=residual_cfg,
            pair_cfg=pair_cfg,
            hedge_ratio_lb=hedge_ratio_lb,
            mr_diag_lb=mr_diag_lb,
            debug=debug,
            fitted_model=model,
            progress=False,
        )
        rows.extend(date_rows)
        weight_rows.extend(date_weight_rows)

    return rows, weight_rows, fitted_params
