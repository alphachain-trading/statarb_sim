"""
Batch candidate-panel creation over a list of residual configs.

Usage (notebook):

    from src.candidates.panel_batch import PanelBatchConfig, run_panel_batch
    from src.candidates.pair_candidate_panel_creator import PairSpreadConfig
    from src.residuals.causal_residuals import CausalResidualConfig, ResidualMode, AbsOrMult

    cfg = PanelBatchConfig(
        residual_configs=[
            CausalResidualConfig(
                mode=ResidualMode.DECAY_EXPANDING,
                subtract_risk_free=True,
                hl=504,
                min_lb_type_dec_exp=AbsOrMult.MULTIPLIER,
                min_lb_dec_exp=2,
            ),
        ],
        hedge_ratio_lb=252,
        mr_diag_lb=252,
        selected_groups=["materials"],
        pair_cfg=PairSpreadConfig(
            hedge_ratio_methods=["pca"],
            min_obs=252,
            min_return_std=1e-8,
            min_level_std=1e-8,
            min_kappa=1e-6,
            max_half_life=126.0,
            tiny_weight_threshold=1e-6,
        ),
    )
    results = run_panel_batch(cfg)

The sweep IS the residual_configs list — build it however you like (a plain
list comprehension over CausalResidualConfig instances). hedge_ratio_lb /
mr_diag_lb may be a scalar (broadcast across every residual config) or a
list of matching length (one value per residual config), but never a list
when residual_configs has length 1.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from src.settings import CONFIG_UNIVERSE, DATA_UNIVERSES, CANDIDATE_PANELS_ROOT

from src.simulator.config import GroupDataSource, group_abbrev
from src.candidates.pair_candidate_panel_creator import (
    PairSpreadConfig,
    create_pair_candidate_panel,
    CandidatePanelResult,
)
from src.data.returns import build_group_return_bundle
from src.data.universe_loader import UniverseDataLoader
from src.data.universe_config import UniverseConfig
from src.residuals.causal_residuals import CausalResidualConfig


# ── group discovery ─────────────────────────────────────────────────────


def discover_group_yamls(
    universe_dir: str | Path,
    selected_groups: list[str] | None = None,
    excluded_groups: list[str] | None = None,
) -> list[tuple[str, Path]]:
    """
    Discover (group_id, yaml_path) pairs from a universe directory
    (config/universes/{universe_name}/), one yaml per group, filename stem
    is the group_id.

    Returns sorted list of (group_id, path) tuples.
    """
    if selected_groups is not None and excluded_groups is not None:
        raise ValueError("Cannot specify both selected_groups and excluded_groups.")

    universe_dir = Path(universe_dir)
    results = []

    for yaml_path in sorted(universe_dir.glob("*.yaml")):
        group_id = yaml_path.stem

        if selected_groups is not None and group_id not in selected_groups:
            continue
        if excluded_groups is not None and group_id in excluded_groups:
            continue

        results.append((group_id, yaml_path))

    return results


# ── config ──────────────────────────────────────────────────────────────


@dataclass(kw_only=True)
class PanelBatchConfig:
    """
    Central config for batch candidate panel creation.

    Holds a list of already-constructed, self-validating CausalResidualConfigs
    — the sweep IS this list, built by the caller (a plain list comprehension,
    no factory). PanelBatchConfig has no residual-shaped fields of its own and
    no mirrored mode/validation logic; that lives exactly once, in
    CausalResidualConfig.__post_init__.

    residual_configs may also be given as list[str] of residual_key strings
    (e.g. "exp_hl504_mh1008_rf") — each is resolved via
    CausalResidualConfig.from_key at construction time, fail-fast on a
    malformed key. Mixed lists (some str, some CausalResidualConfig) are
    rejected.

    hedge_ratio_lb / mr_diag_lb are the two independent candidate-scoring
    windows (hedge-ratio fit vs mean-reversion diagnostics), applied to every
    panel in residual_configs — each may be a scalar (broadcast) or a list of
    matching length (one value per residual config).
    """

    residual_configs: list[CausalResidualConfig] | list[str]
    hedge_ratio_lb: int | list[int]
    mr_diag_lb: int | list[int]

    # Pair spread config (shared across all groups/residual configs). No
    # default: PairSpreadConfig.min_obs is itself mandatory with no default
    # (Track B), and a dataclass default here would just be a second place
    # for the same result-affecting number to silently live. Callers must
    # state it explicitly — see D_config_hygiene.md's "no numeric defaults
    # in dataclasses" principle, which this field was the one exception to
    # until now.
    pair_cfg: PairSpreadConfig

    # Panel frequency
    frequency: str = "W-FRI"

    # Group selection
    selected_groups: list[str] | None = None
    excluded_groups: list[str] | None = None

    # Paths
    universe_name: str = "sp500_v1"  # config/universes/{universe_name}/{group_id}.yaml
    universe_dir: str | Path = ""  # "" → CONFIG_UNIVERSE / universe_name; full override otherwise
    data_path: str | Path = ""     # "" → DATA_UNIVERSES from settings

    # Data loading
    force_download: bool = False
    check_for_corruptions: bool = False
    start_after_nan: bool = True

    # Panel creation
    start_date: str | None = None
    end_date: str | None = None
    max_steps: int | None = None
    debug: bool = False

    # Persistence
    persist_result: bool = True
    persist_residual_params: bool = True
    persist_dir_template: str = "V2.pair"  # subdirectory under CANDIDATE_PANELS_ROOT

    # ── validation ────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        if not self.residual_configs:
            raise ValueError("residual_configs must be non-empty.")

        is_str = [isinstance(rc, str) for rc in self.residual_configs]
        if any(is_str) and not all(is_str):
            raise ValueError(
                "residual_configs must not mix str keys and CausalResidualConfig "
                "instances — pass either list[str] or list[CausalResidualConfig]."
            )
        if all(is_str):
            self.residual_configs = [
                CausalResidualConfig.from_key(rc) for rc in self.residual_configs
            ]

        n = len(self.residual_configs)
        for name in ("hedge_ratio_lb", "mr_diag_lb"):
            val = getattr(self, name)
            if isinstance(val, list):
                if n == 1:
                    raise ValueError(
                        f"{name} must be a scalar when residual_configs has length 1."
                    )
                if len(val) != n:
                    raise ValueError(
                        f"{name} list length {len(val)} != len(residual_configs) ({n})."
                    )

    # ── path helpers ─────────────────────────────────────────────────────

    def resolved_universe_dir(self) -> Path:
        return Path(self.universe_dir) if self.universe_dir else Path(CONFIG_UNIVERSE) / self.universe_name

    def resolved_data_path(self) -> Path:
        return Path(self.data_path) if self.data_path else Path(DATA_UNIVERSES)

    def persist_dir_for(self) -> str:
        """Persist directory for all panels (flat, residual key in stem)."""
        return self.persist_dir_template

    # ── per-point window resolution ──────────────────────────────────────

    def resolved_windows(self) -> list[tuple[int, int]]:
        """Broadcast hedge_ratio_lb / mr_diag_lb to one (hedge_lb, diag_lb) pair per residual config."""
        n = len(self.residual_configs)

        def _broadcast(val: int | list[int]) -> list[int]:
            return list(val) if isinstance(val, list) else [val] * n

        return list(zip(_broadcast(self.hedge_ratio_lb), _broadcast(self.mr_diag_lb)))


# ── batch runner ────────────────────────────────────────────────────────


def _make_panel_stem(
    group_id: str,
    residual_cfg: CausalResidualConfig,
    pair_cfg: PairSpreadConfig,
    frequency: str,
    batch_timestamp: str,
) -> str:
    """
    Build a compact, canonical file stem for panel artifacts.

    Format: {group_abbrev}_pairs_{methods}_{freq}_{window}_hl{hl}_{timestamp}
    Example: dsc_pairs_pca_W-FRI_exp_hl126_20260505_2128
    """
    abbrev = group_abbrev(group_id)
    methods = "+".join(pair_cfg.hedge_ratio_methods)
    window = "exp" if residual_cfg.window_mode == "expanding" else f"lb{residual_cfg.lookback}"
    hl = f"hl{residual_cfg.half_life}" if residual_cfg.half_life else "nohl"
    return f"{abbrev}_pairs_{methods}_{frequency}_{window}_{hl}_{batch_timestamp}"


def run_panel_batch(
    cfg: PanelBatchConfig,
) -> dict[tuple[str, str], CandidatePanelResult]:
    """
    Run batch panel creation across groups × cfg.residual_configs.

    Returns
    -------
    dict mapping (group_id, residual_key) → CandidatePanelResult
    """
    batch_timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")

    # Discover groups
    group_entries = discover_group_yamls(
        universe_dir=cfg.resolved_universe_dir(),
        selected_groups=cfg.selected_groups,
        excluded_groups=cfg.excluded_groups,
    )

    if not group_entries:
        raise FileNotFoundError(
            f"No group YAMLs found in {cfg.resolved_universe_dir()} "
            f"(selected={cfg.selected_groups}, excluded={cfg.excluded_groups})"
        )

    points = list(zip(cfg.residual_configs, cfg.resolved_windows()))

    group_labels = [group_abbrev(gid) for gid, _ in group_entries]
    print(f"Batch {batch_timestamp}")
    print(f"Groups: {len(group_entries)} [{', '.join(group_labels)}]")
    print(f"Residual configs: {[rc.key for rc in cfg.residual_configs]}")
    total_jobs = len(group_entries) * len(points)
    print(f"Total jobs: {total_jobs}")

    results: dict[tuple[str, str], CandidatePanelResult] = {}

    group_bar = tqdm(group_entries, desc="Groups", unit="group")

    for group_id, yaml_path in group_bar:
        abbrev = group_abbrev(group_id)
        group_bar.set_postfix_str(abbrev)

        # Load UMD once per group
        ucfg = UniverseConfig.from_yaml(yaml_path)
        loader = UniverseDataLoader(
            ucfg,
            data_path=cfg.resolved_data_path(),
            progress=False,
        )
        umd = loader.load(
            force_download=cfg.force_download,
            check_for_corruptions=cfg.check_for_corruptions,
            start_after_nan=cfg.start_after_nan,
        )

        bundle = build_group_return_bundle(
            umd=umd,
            group_id=group_id,
            field="Close",
            return_method="log",
            dropna="any",
        )

        # Inner loop: residual_configs
        point_iter = points
        if len(points) > 1:
            point_iter = tqdm(points, desc=f"  {abbrev} sweep", leave=False, unit="pt")

        for residual_cfg, (hedge_ratio_lb, mr_diag_lb) in point_iter:
            persist_dir = cfg.persist_dir_for()

            stem = _make_panel_stem(
                group_id=group_id,
                residual_cfg=residual_cfg,
                pair_cfg=cfg.pair_cfg,
                frequency=cfg.frequency,
                batch_timestamp=batch_timestamp,
            )

            result = create_pair_candidate_panel(
                bundle=bundle,
                residual_cfg=residual_cfg,
                pair_cfg=cfg.pair_cfg,
                frequency=cfg.frequency,
                hedge_ratio_lb=hedge_ratio_lb,
                mr_diag_lb=mr_diag_lb,
                start_date=cfg.start_date,
                end_date=cfg.end_date,
                max_steps=cfg.max_steps,
                debug=cfg.debug,
                progress=False,  # tqdm handles progress
                persist_result=cfg.persist_result,
                persist_residual_params=cfg.persist_residual_params,
                persist_result_dir=persist_dir,
                persist_result_file_stem=stem,
            )

            key = (group_id, residual_cfg.key)
            results[key] = result

            n_rows = len(result.panel) if result.panel is not None else 0
            n_valid = int(result.panel["is_valid"].sum()) if n_rows > 0 else 0
            tqdm.write(f"  {abbrev} / {residual_cfg.key}: {n_valid}/{n_rows} valid candidates")

    print(f"\nDone. {len(results)} panels created.")
    return results