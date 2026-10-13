"""
PanelBatchConfig window broadcasting and the zip(strict=True) guards.

Unequal lengths are not possible after construction: __post_init__ rejects lists whose
length differs from len(residual_configs) and broadcasts scalars. The only way to get
there is mutating the (non-frozen) config afterwards — which must fail loudly rather
than silently truncate (zip strict=True).
"""

import unittest
from typing import cast
from unittest import mock

from src.candidates.pair_candidate_panel_creator import PairSpreadConfig
from src.candidates.panel_batch import PanelBatchConfig, run_panel_batch
from src.residuals.causal_residuals import CausalResidualConfig, ResidualMode
from tests._synthetic import make_ref

_PAIR = PairSpreadConfig(
    hedge_ratio_methods=["pca"],
    min_obs=10,
    min_return_std=1e-8,
    min_level_std=1e-8,
    min_kappa=1e-6,
    max_half_life=126.0,
    tiny_weight_threshold=1e-6,
)


def _rc(lb: int) -> CausalResidualConfig:
    return CausalResidualConfig(mode=ResidualMode.EQ_ROLLING, subtract_risk_free=True, lb=lb)


def _cfg(n: int, hedge, diag) -> PanelBatchConfig:
    return PanelBatchConfig(
        residual_configs=[_rc(20 + i) for i in range(n)],
        hedge_ratio_lb=hedge,
        mr_diag_lb=diag,
        pair_cfg=_PAIR,
    )


class TestResolvedWindows(unittest.TestCase):
    def test_scalars_are_broadcast_to_every_config(self):
        self.assertEqual(_cfg(3, 252, 126).resolved_windows(), [(252, 126)] * 3)

    def test_single_config_scalars(self):
        self.assertEqual(_cfg(1, 252, 126).resolved_windows(), [(252, 126)])

    def test_lists_are_matched_by_position(self):
        self.assertEqual(_cfg(2, [100, 200], [10, 20]).resolved_windows(), [(100, 10), (200, 20)])

    def test_mixed_scalar_and_list(self):
        self.assertEqual(_cfg(2, [100, 200], 50).resolved_windows(), [(100, 50), (200, 50)])
        self.assertEqual(_cfg(2, 100, [5, 6]).resolved_windows(), [(100, 5), (100, 6)])


class TestConstructionRejectsMismatches(unittest.TestCase):
    def test_list_length_must_equal_number_of_configs(self):
        with self.assertRaises(ValueError):
            _cfg(3, [1, 2], 5)
        with self.assertRaises(ValueError):
            _cfg(2, 5, [1, 2, 3])

    def test_list_not_allowed_for_a_single_config(self):
        with self.assertRaises(ValueError):
            _cfg(1, [252], 126)

    def test_empty_residual_configs(self):
        with self.assertRaises(ValueError):
            PanelBatchConfig(residual_configs=[], hedge_ratio_lb=1, mr_diag_lb=1, pair_cfg=_PAIR)

    def test_residual_keys_as_strings_are_resolved_and_mixing_is_rejected(self):
        cfg = PanelBatchConfig(
            residual_configs=["rol_lb21", "rol_lb42"],
            hedge_ratio_lb=[10, 20],
            mr_diag_lb=5,
            pair_cfg=_PAIR,
        )
        self.assertTrue(all(isinstance(rc, CausalResidualConfig) for rc in cfg.residual_configs))
        with self.assertRaises(ValueError):
            PanelBatchConfig(
                residual_configs=["rol_lb21", _rc(42)],  # type: ignore[arg-type]  # mixing on purpose
                hedge_ratio_lb=1,
                mr_diag_lb=1,
                pair_cfg=_PAIR,
            )


class TestStrictZipAfterMutation(unittest.TestCase):
    def test_mutating_residual_configs_afterwards_fails_loudly_in_run_panel_batch(self):
        # resolved_windows() alone cannot see this (both lists still match each other);
        # the zip in run_panel_batch pairing configs with windows is the guard. With
        # strict=False it would silently run only the first two configs; with
        # strict=True it stops right after group discovery, before any data is loaded.
        cfg = _cfg(2, [100, 200], [10, 20])
        cast("list[CausalResidualConfig]", cfg.residual_configs).append(_rc(99))
        with (
            mock.patch(
                "src.candidates.panel_batch.discover_group_yamls", return_value=[("g", "x")]
            ),
            self.assertRaises(ValueError) as ctx,
        ):
            run_panel_batch(cfg)
        self.assertIn("zip()", str(ctx.exception))

    def test_mutating_a_window_list_afterwards_fails_loudly(self):
        cfg = _cfg(2, [100, 200], 50)
        cfg.hedge_ratio_lb = [100]
        with self.assertRaises(ValueError):
            cfg.resolved_windows()

    def test_zero_mismatch_does_not_raise(self):
        cfg = _cfg(2, [100, 200], [10, 20])
        self.assertEqual(len(cfg.resolved_windows()), len(cfg.residual_configs))


class TestCandidateRefLengthGuard(unittest.TestCase):
    """CandidateRef guarantees members/weights have equal length, which makes the
    zip(ref.members, ref.weights, strict=True) call sites safe."""

    def test_equal_lengths_are_accepted(self):
        ref = make_ref()
        self.assertEqual(len(ref.members), len(ref.weights))

    def test_unequal_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            make_ref(members=("A", "B", "C"), weights=(1.0, -1.0))
        with self.assertRaises(ValueError):
            make_ref(members=(), weights=())


if __name__ == "__main__":
    unittest.main()
