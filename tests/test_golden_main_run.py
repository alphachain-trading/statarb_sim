"""
Characterisation (golden-master) tests for the main pipeline on fixed synthetic data.

C1: ``run_from_config`` end to end (live candidate generation -> signals -> sizing ->
    execution -> performance) — trades and metrics.
C2: ``generate_candidate_panels_by_group`` — scored candidate panels, leg weights and
    fitted residual params.

These pin *current* behaviour; they do not assert that it is correct. A diff means
"something moved" — review it, then regenerate deliberately with ``UPDATE_GOLDEN=1``.
Data is fully synthetic (tests/_synthetic.py), so the tests are hermetic and do not
need the gitignored ``data/`` directory.
"""

import unittest
from typing import Any

import pandas as pd
import pytest

from src.data.universe_marketdata import UniverseMarketData
from src.simulator.config import SimulatorConfig
from src.simulator.simulator_factory import generate_candidate_panels_by_group, run_from_config
from tests._golden import assert_matches_golden
from tests._synthetic import build_sim_config, make_umd, patched_universe

GROUPS = ["alpha", "beta"]
TRADE_COLS = [
    "trade_id",
    "group_id",
    "spread_id",
    "entry_date",
    "exit_date",
    "days_open",
    "direction",
    "pair_notional",
    "entry_z_score",
    "exit_z_score",
    "realized_pnl_gross",
    "transaction_costs",
    "borrow_costs",
    "realized_pnl_net",
]


def _ts(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


@pytest.mark.slow
class TestGoldenMainRun(unittest.TestCase):
    umd: UniverseMarketData
    cfg: SimulatorConfig

    @classmethod
    def setUpClass(cls) -> None:
        cls.umd = make_umd()
        cls.cfg = build_sim_config(GROUPS)

    def test_c1_run_from_config_trades_and_metrics(self):
        with patched_universe(self.umd):
            result = run_from_config(self.cfg)
        trades = result.closed_trades_df().sort_values("trade_id")
        records = trades[TRADE_COLS].to_dict("records")
        for rec in records:
            rec["entry_date"] = _ts(rec["entry_date"])
            rec["exit_date"] = _ts(rec["exit_date"])
        metrics = dict(result.performance.metrics)  # type: ignore[union-attr]
        assert_matches_golden(
            "c1_main_run",
            {"n_trades": len(trades), "trades": records, "metrics": metrics},
        )

    def test_c2_candidate_panels(self):
        with patched_universe(self.umd):
            by_group = generate_candidate_panels_by_group(self.cfg)
        summary: dict[str, Any] = {}
        for (group_id, residual_key), v in sorted(by_group.items()):
            panel: pd.DataFrame = v["panel"]
            numeric = panel.select_dtypes("number")
            first_date = panel["asof_date"].min()
            first = panel[panel["asof_date"] == first_date].sort_values("spread_id")
            weights = v["weight_rows"]
            summary[f"{group_id}|{residual_key}"] = {
                "shape": list(panel.shape),
                "columns": list(panel.columns),
                "numeric_col_sums": {c: float(numeric[c].sum()) for c in numeric.columns},
                "valid_by_asof": {
                    _ts(d): int(n) for d, n in panel.groupby("asof_date")["is_valid"].sum().items()
                },
                "first_date_rows": [
                    {
                        "spread_id": r.spread_id,
                        "mr_score": r.mr_score,
                        "kappa": r.kappa,
                        "half_life": r.half_life,
                        "is_valid": bool(r.is_valid),
                    }
                    for r in first.itertuples(index=False)
                ],
                "n_weight_rows": len(weights),
                "weight_sum": float(sum(w["weight"] for w in weights)),
                "fitted_param_dates": [_ts(d) for d in sorted(v["fitted_params"])],
            }
        assert_matches_golden("c2_candidate_panels", summary)


if __name__ == "__main__":
    unittest.main()
