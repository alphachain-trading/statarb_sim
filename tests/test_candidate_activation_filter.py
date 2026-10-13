"""CandidateActivation (which candidates are tracked/active) and CandidateFilter."""

import unittest
from typing import Any, ClassVar

import pandas as pd

from src.candidates.candidate_selector import CandidateSelectionConfig
from src.simulator.candidate_activation import CandidateActivation
from src.simulator.candidate_filter import CandidateFilter
from src.simulator.config import ActivationConfig, ZScoreConfig
from tests._synthetic import make_ref


def _ref(cid: str, spread: str = "A|B", group: str = "G1", tsl: str = "", **kw: Any):
    return make_ref(cid, spread_id=spread, group_id=group, timescale_label=tsl, **kw)


def _pair() -> CandidateActivation:
    return CandidateActivation(
        ActivationConfig(one_active_per_group=False, switch_only_when_flat=False)
    )


def _portfolio(switch_only_when_flat: bool) -> CandidateActivation:
    cfg = ActivationConfig(one_active_per_group=True, switch_only_when_flat=switch_only_when_flat)
    return CandidateActivation(cfg)


def _ids(refs) -> list[str]:
    return [r.candidate_id for r in refs]


class TestPairActivation(unittest.TestCase):
    def test_new_arrival_becomes_active_and_replaces_stale_same_spread(self):
        act = _pair()
        act.process_new_arrivals([_ref("old")], is_flat_by_candidate_id={}, open_spread_ids=set())
        act.process_new_arrivals([_ref("new")], is_flat_by_candidate_id={}, open_spread_ids=set())
        self.assertEqual(_ids(act.get_active_candidates()), ["new"])

    def test_old_ref_with_open_position_is_kept_alongside_the_new_one(self):
        act = _pair()
        act.process_new_arrivals([_ref("old")], is_flat_by_candidate_id={}, open_spread_ids=set())
        # multi-timescale form: (spread_id, timescale_label)
        act.process_new_arrivals(
            [_ref("new")], is_flat_by_candidate_id={}, open_spread_ids={("A|B", "")}
        )
        self.assertEqual(_ids(act.get_active_candidates()), ["old", "new"])

    def test_legacy_string_set_of_open_spreads(self):
        act = _pair()
        act.process_new_arrivals([_ref("old")], is_flat_by_candidate_id={}, open_spread_ids=set())
        act.process_new_arrivals([_ref("new")], is_flat_by_candidate_id={}, open_spread_ids={"A|B"})
        self.assertEqual(_ids(act.get_active_candidates()), ["old", "new"])

    def test_other_spreads_are_untouched(self):
        act = _pair()
        act.process_new_arrivals([_ref("a", "A|B"), _ref("c", "C|D")], is_flat_by_candidate_id={})
        act.process_new_arrivals([_ref("a2", "A|B")], is_flat_by_candidate_id={})
        self.assertEqual(sorted(_ids(act.get_active_candidates())), ["a2", "c"])

    def test_open_position_on_another_timescale_does_not_protect_the_old_ref(self):
        act = _pair()
        act.process_new_arrivals([_ref("old", tsl="t1")], is_flat_by_candidate_id={})
        act.process_new_arrivals(
            [_ref("new", tsl="t1")], is_flat_by_candidate_id={}, open_spread_ids={("A|B", "t2")}
        )
        self.assertEqual(_ids(act.get_active_candidates()), ["new"])

    def test_groups_and_timescales_have_independent_slots(self):
        act = _pair()
        act.process_new_arrivals(
            [_ref("g1", group="G1"), _ref("g2", group="G2"), _ref("g1t", group="G1", tsl="t1")],
            is_flat_by_candidate_id={},
        )
        self.assertEqual(sorted(act.active_by_group), ["G1", "G1|t1", "G2"])
        self.assertEqual(sorted(_ids(act.get_active_candidates("G1"))), ["g1", "g1t"])
        self.assertEqual(_ids(act.get_active_candidates("G2")), ["g2"])
        self.assertEqual(act.get_active_candidates("nope"), [])


class TestPortfolioActivation(unittest.TestCase):
    def test_first_candidate_is_activated(self):
        act = _portfolio(True)
        act.process_new_arrivals([_ref("a")], is_flat_by_candidate_id={})
        self.assertEqual(act.get_active_candidate("G1").candidate_id, "a")  # type: ignore[union-attr]
        self.assertIsNone(act.get_active_candidate("G2"))

    def test_switch_only_when_flat_keeps_a_busy_incumbent(self):
        act = _portfolio(True)
        act.process_new_arrivals([_ref("a")], is_flat_by_candidate_id={})
        act.process_new_arrivals([_ref("b")], is_flat_by_candidate_id={"a": False})
        self.assertEqual(_ids(act.get_active_candidates()), ["a"])
        act.process_new_arrivals([_ref("b")], is_flat_by_candidate_id={"a": True})
        self.assertEqual(_ids(act.get_active_candidates()), ["b"])

    def test_unknown_incumbent_counts_as_flat(self):
        act = _portfolio(True)
        act.process_new_arrivals([_ref("a")], is_flat_by_candidate_id={})
        act.process_new_arrivals([_ref("b")], is_flat_by_candidate_id={})
        self.assertEqual(_ids(act.get_active_candidates()), ["b"])

    def test_without_flat_requirement_the_newest_always_wins(self):
        act = _portfolio(False)
        act.process_new_arrivals([_ref("a")], is_flat_by_candidate_id={})
        act.process_new_arrivals([_ref("b")], is_flat_by_candidate_id={"a": False})
        self.assertEqual(_ids(act.get_active_candidates()), ["b"])

    def test_same_candidate_again_is_a_no_op(self):
        act = _portfolio(False)
        act.process_new_arrivals([_ref("a")], is_flat_by_candidate_id={})
        act.process_new_arrivals([_ref("a")], is_flat_by_candidate_id={})
        self.assertEqual(_ids(act.get_active_candidates()), ["a"])

    def test_timescales_are_independent_slots(self):
        act = _portfolio(True)
        act.process_new_arrivals(
            [_ref("a", tsl="t1"), _ref("b", tsl="t2")], is_flat_by_candidate_id={}
        )
        self.assertEqual(sorted(_ids(act.get_active_candidates())), ["a", "b"])
        self.assertEqual(act.get_active_candidate("G1", "t2").candidate_id, "b")  # type: ignore[union-attr]


class TestActivationFrame(unittest.TestCase):
    def test_flags_and_sorting(self):
        act = _pair()
        act.process_new_arrivals(
            [_ref("act", "Z|Y"), _ref("act2", "A|B", "G0")], is_flat_by_candidate_id={}
        )
        frame = act.build_activation_frame([_ref("fresh", "M|N"), _ref("act", "Z|Y")])
        self.assertEqual(frame.index.name, "candidate_id")
        self.assertEqual(list(frame.index), ["act2", "fresh", "act"])  # sorted by group, spread

        def flags(cid: str) -> list[bool]:
            return [bool(frame.at[cid, "is_new_arrival"]), bool(frame.at[cid, "is_active"])]

        self.assertEqual(flags("act"), [True, True])
        self.assertEqual(flags("fresh"), [True, False])
        self.assertEqual(flags("act2"), [False, True])

    def test_empty_frame_keeps_its_columns(self):
        frame = _pair().build_activation_frame([])
        self.assertTrue(frame.empty)
        self.assertEqual(
            list(frame.columns), ["spread_id", "group_id", "is_new_arrival", "is_active"]
        )
        self.assertEqual(frame.index.name, "candidate_id")


def _selected(rows: list[dict[str, Any]]) -> pd.DataFrame:
    base = {"group_id": "G1", "residual_key": "", "subtype": "pca"}
    return pd.DataFrame([{**base, **r} for r in rows])


def _filter(zcs: dict[str, list[ZScoreConfig]] | None = None) -> CandidateFilter:
    return CandidateFilter(CandidateSelectionConfig(require_success=True), zcs or {})


def _zc(lookback: int, rkey: str) -> ZScoreConfig:
    return ZScoreConfig(lookback=lookback, ddof=1, method="ewm", residual_key=rkey)


D = pd.Timestamp("2024-01-05")


class TestBuildCandidateRefs(unittest.TestCase):
    WEIGHTS: ClassVar[dict[tuple[str, pd.Timestamp], dict[str, float]]] = {
        ("B|A", D): {"A": -0.5, "B": 0.5}
    }

    def test_members_and_weights_follow_spread_id_order(self):
        rows = _selected([{"candidate_id": "c1", "spread_id": "B|A", "asof_date": D}])
        (ref,) = _filter().build_candidate_refs(rows, self.WEIGHTS)
        self.assertEqual((ref.members, ref.weights), (("B", "A"), (0.5, -0.5)))
        self.assertEqual((ref.asof_date, ref.group_id, ref.timescale_label), (D, "G1", ""))

    def test_missing_weights_key_raises(self):
        rows = _selected([{"candidate_id": "c1", "spread_id": "X|Y", "asof_date": D}])
        with self.assertRaises(KeyError):
            _filter().build_candidate_refs(rows, self.WEIGHTS)

    def test_single_zscore_config_per_residual_key_sets_the_label(self):
        zc = _zc(21, "exp_hl63_mh126")
        rows = _selected(
            [
                {
                    "candidate_id": "c1",
                    "spread_id": "B|A",
                    "asof_date": D,
                    "residual_key": zc.residual_key,
                }
            ]
        )
        (ref,) = _filter({zc.residual_key: [zc]}).build_candidate_refs(rows, self.WEIGHTS)
        self.assertEqual(
            (ref.residual_key, ref.timescale_label), (zc.residual_key, "exp_hl63_mh126__zhl21")
        )

    def test_several_zscore_configs_fan_out_one_ref_each(self):
        rkey = "exp_hl63_mh126"
        zcs = [_zc(21, rkey), _zc(63, rkey)]
        rows = _selected(
            [{"candidate_id": "c1", "spread_id": "B|A", "asof_date": D, "residual_key": rkey}]
        )
        refs = _filter({rkey: zcs}).build_candidate_refs(rows, self.WEIGHTS)
        self.assertEqual([r.timescale_label for r in refs], [f"{rkey}__zhl21", f"{rkey}__zhl63"])
        self.assertEqual({r.candidate_id for r in refs}, {"c1"})  # identity stays clean

    def test_unknown_residual_key_gets_no_label(self):
        rows = _selected(
            [{"candidate_id": "c1", "spread_id": "B|A", "asof_date": D, "residual_key": "other"}]
        )
        (ref,) = _filter({"exp": [_zc(5, "exp")]}).build_candidate_refs(rows, self.WEIGHTS)
        self.assertEqual(ref.timescale_label, "")


class TestFilterRunAndSelection(unittest.TestCase):
    def test_get_selected_on_date_matches_exactly(self):
        panel = _selected(
            [
                {"candidate_id": "a", "spread_id": "A|B", "asof_date": D},
                {"candidate_id": "b", "spread_id": "C|D", "asof_date": D + pd.Timedelta(days=7)},
            ]
        )
        got = _filter().get_selected_on_date(panel, D)
        self.assertEqual(list(got["candidate_id"]), ["a"])
        self.assertEqual(len(_filter().get_selected_on_date(panel, D + pd.Timedelta(days=1))), 0)

    def test_run_normalises_and_sorts_the_selected_panel(self):
        panel = pd.DataFrame(
            {
                "candidate_id": ["b", "a", "c"],
                "spread_id": ["B|C", "A|B", "C|D"],
                "group_id": ["G2", "G1", "G1"],
                "asof_date": [D, D, D - pd.Timedelta(days=7)],
                "candidate_subtype": ["knee"] * 3,
                "candidate_type": ["pair"] * 3,
                "weight_model": ["pca"] * 3,
                "is_valid": [True] * 3,
                "success": [True] * 3,
                "mr_score": [1.0] * 3,
                "half_life": [5.0] * 3,
            }
        )
        out = _filter().run(panel)
        self.assertEqual(list(out["candidate_id"]), ["c", "a", "b"])  # asof, group, id
        self.assertEqual(list(out["subtype"]), ["knee"] * 3)

    def test_run_rejects_panels_missing_required_columns(self):
        panel = pd.DataFrame(
            {
                "candidate_id": ["a"],
                "spread_id": ["A|B"],
                "asof_date": [D],
                "candidate_subtype": ["knee"],
                "candidate_type": ["pair"],
                "is_valid": [True],
                "success": [True],
                "mr_score": [1.0],
                "half_life": [5.0],
            }
        )
        with self.assertRaises(ValueError):
            _filter().run(panel)  # no group_id


if __name__ == "__main__":
    unittest.main()
