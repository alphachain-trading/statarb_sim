"""Unit tests for select_candidates: filters, reference-relative metrics, metadata."""

import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.candidates.candidate_selector import (
    CandidateSelectionConfig,
    SelectedCandidatePanelResult,
    _choose_reference_row,
    _extract_asof_date_series,
    load_selected_candidate_panel_result,
    save_selected_candidate_panel_result,
    select_candidates,
)

D1, D2 = pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-12")


def _panel(rows: list[dict[str, Any]]) -> pd.DataFrame:
    base = {
        "candidate_subtype": "knee",
        "candidate_type": "pair",
        "weight_model": "pca",
        "group_id": "G1",
        "is_valid": True,
        "success": True,
        "mr_score": 1.0,
        "half_life": 10.0,
        "asof_date": D1,
    }
    df = pd.DataFrame([{**base, **r} for r in rows])
    if "spread_id" not in df.columns:
        df["spread_id"] = [f"S{i}|T{i}" for i in range(len(df))]
    return df


def _cfg(**kw: Any) -> CandidateSelectionConfig:
    kw.setdefault("require_success", True)
    return CandidateSelectionConfig(**kw)


def _sel(panel: pd.DataFrame, **kw: Any) -> pd.DataFrame:
    return select_candidates(panel, _cfg(**kw)).panel


class TestBasicFilters(unittest.TestCase):
    def test_subtype_filter_and_empty_tuple_means_no_filter(self):
        p = _panel(
            [
                {"candidate_subtype": "knee"},
                {"candidate_subtype": "dense_knee"},
                {"candidate_subtype": "pca"},
            ]
        )
        self.assertEqual(
            list(_sel(p, allowed_candidate_subtypes=("knee",))["candidate_subtype"]), ["knee"]
        )
        self.assertEqual(len(_sel(p)), 2)  # default ("knee", "dense_knee")
        self.assertEqual(len(_sel(p, allowed_candidate_subtypes=())), 3)

    def test_is_valid_and_success_flags_with_nan_counting_as_false(self):
        p = _panel(
            [
                {"is_valid": True, "success": True},
                {"is_valid": False},
                {"is_valid": None},
                {"success": False},
                {"success": None},
            ]
        )
        p["is_valid"] = p["is_valid"].astype("object")
        p["success"] = p["success"].astype("object")
        self.assertEqual(list(_sel(p).index), [0])
        loose = _sel(p, require_is_valid=False, require_success=False)
        self.assertEqual(len(loose), 5)

    def test_missing_success_column_is_assumed_true(self):
        # Pins current behaviour (documented in the code): no `success` column -> all True.
        p = _panel([{}, {}]).drop(columns=["success"])
        self.assertEqual(len(_sel(p)), 2)

    def test_other_missing_required_columns_raise(self):
        p = _panel([{}]).drop(columns=["mr_score"])
        with self.assertRaises(ValueError):
            _sel(p)

    def test_excluded_tickers_drop_any_spread_containing_them(self):
        p = _panel(
            [
                {"spread_id": "A|B"},
                {"spread_id": "C|D"},
                {"spread_id": "B|E"},
                {"spread_id": "AB|X"},
            ]
        )
        got = _sel(p, excluded_tickers=["B"])
        self.assertEqual(list(got["spread_id"]), ["C|D", "AB|X"])  # exact ticker match only

    def test_adf_threshold(self):
        p = _panel([{}, {}, {}])
        p["adf_pvalue"] = [0.01, 0.05, np.nan]
        got = _sel(p, adf_pvalue_max=0.05)
        self.assertEqual(list(got.index), [0, 1])  # <= is inclusive, NaN is excluded

    def test_adf_threshold_on_all_nan_column_raises_instead_of_selecting_nothing(self):
        p = _panel([{}, {}])
        p["adf_pvalue"] = np.nan
        with self.assertRaises(ValueError):
            _sel(p, adf_pvalue_max=0.05)


class TestReferenceRelativeMetrics(unittest.TestCase):
    def test_mr_retention_relative_to_best_mr_score(self):
        # reference = highest mr_score (10) -> retention 1.0, 0.8, 0.4
        p = _panel([{"mr_score": 10.0}, {"mr_score": 8.0}, {"mr_score": 4.0}])
        got = _sel(p, mr_retention_min=0.5)
        self.assertEqual(list(got["mr_score"]), [10.0, 8.0])
        self.assertEqual(list(got["mr_retention_vs_ref"]), [1.0, 0.8])
        self.assertEqual(list(got["is_reference_candidate"]), [True, False])

    def test_dense_candidate_is_the_reference_even_if_not_best(self):
        p = _panel(
            [
                {"candidate_subtype": "knee", "mr_score": 10.0},
                {"candidate_subtype": "dense_knee", "mr_score": 5.0},
            ]
        )
        got = _sel(p, mr_retention_min=0.1)
        by_sub = got.set_index("candidate_subtype")
        self.assertEqual(by_sub.loc["knee", "mr_retention_vs_ref"], 2.0)  # 10 / 5
        self.assertEqual(by_sub.loc["dense_knee", "mr_retention_vs_ref"], 1.0)
        self.assertTrue(by_sub.loc["dense_knee", "is_reference_candidate"])

    def test_half_life_multiple_relative_to_reference(self):
        # reference = highest mr_score row (half_life 10); multiples 1, 1.5, 3
        p = _panel(
            [
                {"mr_score": 9.0, "half_life": 10.0},
                {"mr_score": 5.0, "half_life": 15.0},
                {"mr_score": 4.0, "half_life": 30.0},
            ]
        )
        got = _sel(p, half_life_mult_max=2.0)
        self.assertEqual(list(got["half_life_mult_vs_ref"]), [1.0, 1.5])

    def test_scopes_are_independent_per_date_and_group(self):
        p = _panel(
            [
                {"mr_score": 10.0, "asof_date": D1, "group_id": "G1"},
                {"mr_score": 5.0, "asof_date": D1, "group_id": "G1"},
                {"mr_score": 5.0, "asof_date": D2, "group_id": "G1"},  # alone on D2 -> ref
                {"mr_score": 2.0, "asof_date": D1, "group_id": "G2"},  # alone in G2 -> ref
            ]
        )
        got = _sel(p, mr_retention_min=0.9)
        self.assertEqual(sorted(got.index), [0, 2, 3])

    def test_zero_or_nan_reference_makes_relative_metric_unusable(self):
        p = _panel([{"mr_score": 0.0}, {"mr_score": 0.0}])
        self.assertEqual(len(_sel(p, mr_retention_min=0.1)), 0)  # NaN retention is excluded
        p2 = _panel([{"half_life": np.nan}])
        self.assertEqual(len(_sel(p2, half_life_mult_max=5.0)), 0)

    def test_reference_tie_break_prefers_shorter_half_life(self):
        g = _panel([{"mr_score": 3.0, "half_life": 20.0}, {"mr_score": 3.0, "half_life": 8.0}])
        self.assertEqual(_choose_reference_row(g)["half_life"], 8.0)

    def test_reference_falls_back_to_first_row_when_no_valid_mr(self):
        g = _panel([{"mr_score": np.nan}, {"mr_score": np.nan}])
        self.assertEqual(_choose_reference_row(g).name, 0)

    def test_reference_scope_cols_change_the_grouping(self):
        p = _panel(
            [
                {"mr_score": 10.0, "weight_model": "pca"},
                {"mr_score": 5.0, "weight_model": "ols"},
            ]
        )
        # default scope includes weight_model -> both are alone -> retention 1.0 each
        self.assertEqual(len(_sel(p, mr_retention_min=0.9)), 2)
        # without it they share a scope: ols retention = 0.5
        got = _sel(p, mr_retention_min=0.9, reference_scope_cols=("group_id",))
        self.assertEqual(len(got), 1)


class TestMetadataAndIO(unittest.TestCase):
    def test_counts(self):
        p = _panel(
            [
                {"asof_date": D1},
                {"asof_date": D1, "is_valid": False},
                {"asof_date": D2, "is_valid": False},
            ]
        )
        meta = select_candidates(p, _cfg(), source_stem="stem").metadata
        self.assertEqual((meta["n_input_rows"], meta["n_selected_rows"]), (3, 1))
        self.assertEqual((meta["n_input_dates"], meta["n_selected_dates"]), (2, 1))
        self.assertEqual(meta["source_stem"], "stem")
        self.assertEqual(meta["selection_cfg"]["require_success"], True)

    def test_empty_panel_returns_empty_result_with_zero_counts(self):
        res = select_candidates(pd.DataFrame(), _cfg())
        self.assertTrue(res.panel.empty)
        self.assertEqual((res.metadata["n_input_rows"], res.metadata["n_selected_rows"]), (0, 0))

    def test_input_panel_is_not_mutated(self):
        p = _panel([{"mr_score": 10.0}, {"mr_score": 4.0}])
        before = p.copy()
        select_candidates(p, _cfg(mr_retention_min=0.5))
        pd.testing.assert_frame_equal(p, before)

    def test_asof_date_can_come_from_a_multiindex_level(self):
        p = _panel([{}]).set_index(["asof_date", "spread_id"])
        out = _extract_asof_date_series(p)
        self.assertEqual(out.iloc[0], D1)
        with self.assertRaises(ValueError):
            _extract_asof_date_series(pd.DataFrame({"x": [1]}))

    def test_save_load_round_trip(self):
        res = SelectedCandidatePanelResult(
            panel=_panel([{}, {}]), metadata={"n": 2, "when": "today", "cfg": {"a": 1}}
        )
        with tempfile.TemporaryDirectory() as tmp:
            save_selected_candidate_panel_result(res, Path(tmp), "stem")
            back = load_selected_candidate_panel_result(Path(tmp), "stem")
        pd.testing.assert_frame_equal(back.panel, res.panel)
        self.assertEqual(back.metadata["n"], 2)
        self.assertEqual(back.metadata["artifact_type"], "selected_candidate_panel")


class TestConfigValidation(unittest.TestCase):
    def test_invalid_values(self):
        for kw in (
            {"adf_pvalue_max": 0.0},
            {"adf_pvalue_max": 1.5},
            {"mr_retention_min": 0.0},
            {"mr_retention_min": 2.0},
            {"half_life_mult_max": 0.0},
            {"reference_scope_cols": ("group_id", "group_id")},
            {"reference_scope_cols": ("nonsense",)},
        ):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                _cfg(**kw)

    def test_require_success_is_mandatory(self):
        with self.assertRaises(TypeError):
            CandidateSelectionConfig()  # type: ignore[call-arg]  # mandatory field omitted


if __name__ == "__main__":
    unittest.main()
