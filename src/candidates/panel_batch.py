"""
Batch panel creation with multi-timescale support.

Usage (notebook):

    from src.candidates.panel_batch import PanelBatchConfig, TimescaleConfig, run_panel_batch

    cfg = PanelBatchConfig(
        timescales=[
            TimescaleConfig(residual_half_life=126),
            TimescaleConfig(residual_half_life=63),
        ],
    )
    results = run_panel_batch(cfg)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from src.settings import CONFIG_UNIVERSE, DATA_UNIVERSES, CANDIDATE_PANELS_ROOT

from src.simulator.config import TimescaleConfig, SectorDataSource, sector_abbrev
from src.candidates.pair_candidate_panel_creator import (
    PairSpreadConfig,
    create_pair_candidate_panel,
    CandidatePanelResult,
)
from src.data.returns import build_group_return_bundle
from src.data.universe_loader import UniverseDataLoader
from src.data.universe_config import UniverseConfig
from src.residuals.causal_residuals import CausalResidualConfig


# ── sector discovery ────────────────────────────────────────────────────


def discover_sector_yamls(
    universe_dir: str | Path,
    selected_sectors: list[str] | None = None,
    excluded_sectors: list[str] | None = None,
) -> list[tuple[str, Path]]:
    """
    Discover (group_id, yaml_path) pairs from universe config directory.

    Returns sorted list of (group_id, path) tuples.
    """
    if selected_sectors is not None and excluded_sectors is not None:
        raise ValueError("Cannot specify both selected_sectors and excluded_sectors.")

    universe_dir = Path(universe_dir)
    results = []

    for yaml_path in sorted(universe_dir.glob("universe.*_only.*.yaml")):
        name = yaml_path.name
        # Parse: universe.{group_id}_only.{version}.yaml
        prefix = "universe."
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix):]
        if "_only." not in rest:
            continue
        group_id = rest.split("_only.", 1)[0]
        if not group_id:
            continue

        if selected_sectors is not None and group_id not in selected_sectors:
            continue
        if excluded_sectors is not None and group_id in excluded_sectors:
            continue

        results.append((group_id, yaml_path))

    return results


# ── config ──────────────────────────────────────────────────────────────


@dataclass
class PanelBatchConfig:
    """
    Central config for batch candidate panel creation.

    Controls sector discovery, data loading, timescales, and persistence.
    Analogous to SimulatorConfig for the simulation pipeline.
    """
    # Timescales to sweep
    timescales: list[TimescaleConfig] = field(
        default_factory=lambda: [TimescaleConfig()],
    )

    # Pair spread config (shared across all sectors/timescales)
    pair_cfg: PairSpreadConfig = field(
        default_factory=lambda: PairSpreadConfig(
            hedge_ratio_methods=["pca"],
            skip_adf=False,
        ),
    )

    # Residual config overrides (applied per-timescale)
    residual_window_mode: str = "expanding"
    residual_min_history_multiplier: int = 2  # min_history = residual_hl * this
    remove_residual_pcs: int = 0
    subtract_risk_free: bool = False

    # Panel frequency
    frequency: str = "W-FRI"

    # Sector selection
    selected_sectors: list[str] | None = None
    excluded_sectors: list[str] | None = None

    # Paths
    universe_dir: str | Path = ""  # "" → CONFIG_UNIVERSE from settings
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

    def resolved_universe_dir(self) -> Path:
        return Path(self.universe_dir) if self.universe_dir else Path(CONFIG_UNIVERSE)

    def resolved_data_path(self) -> Path:
        return Path(self.data_path) if self.data_path else Path(DATA_UNIVERSES)

    def persist_dir_for(self) -> str:
        """Persist directory for all panels (flat, timescale in stem)."""
        return self.persist_dir_template

    def make_residual_cfg(self, ts: TimescaleConfig) -> CausalResidualConfig:
        """Build CausalResidualConfig for a specific timescale."""
        return ts.to_residual_cfg(
            window_mode=self.residual_window_mode,
            min_history=ts.residual_half_life * self.residual_min_history_multiplier,
            remove_residual_pcs=self.remove_residual_pcs,
            subtract_risk_free=self.subtract_risk_free,
        )


# ── batch runner ────────────────────────────────────────────────────────


def _make_panel_stem(
    group_id: str,
    ts: TimescaleConfig,
    residual_cfg: CausalResidualConfig,
    pair_cfg: PairSpreadConfig,
    frequency: str,
    batch_timestamp: str,
) -> str:
    """
    Build a compact, canonical file stem for panel artifacts.

    Format: {sector_abbrev}_pairs_{methods}_{freq}_{window}_hl{hl}_{timestamp}
    Example: dsc_pairs_pca_W-FRI_exp_hl126_20260505_2128
    """
    abbrev = sector_abbrev(group_id)
    methods = "+".join(pair_cfg.hedge_ratio_methods)
    window = "exp" if residual_cfg.window_mode == "expanding" else f"lb{residual_cfg.lookback}"
    hl = f"hl{residual_cfg.half_life}" if residual_cfg.half_life else "nohl"
    return f"{abbrev}_pairs_{methods}_{frequency}_{window}_{hl}_{batch_timestamp}"


def run_panel_batch(
    cfg: PanelBatchConfig,
) -> dict[tuple[str, str], CandidatePanelResult]:
    """
    Run batch panel creation across sectors × timescales.

    Returns
    -------
    dict mapping (group_id, timescale_label) → CandidatePanelResult
    """
    batch_timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")

    # Discover sectors
    sector_entries = discover_sector_yamls(
        universe_dir=cfg.resolved_universe_dir(),
        selected_sectors=cfg.selected_sectors,
        excluded_sectors=cfg.excluded_sectors,
    )

    if not sector_entries:
        raise FileNotFoundError(
            f"No sector YAMLs found in {cfg.resolved_universe_dir()} "
            f"(selected={cfg.selected_sectors}, excluded={cfg.excluded_sectors})"
        )

    sector_labels = [sector_abbrev(gid) for gid, _ in sector_entries]
    print(f"Batch {batch_timestamp}")
    print(f"Sectors: {len(sector_entries)} [{', '.join(sector_labels)}]")
    print(f"Timescales: {[ts.label for ts in cfg.timescales]}")
    total_jobs = len(sector_entries) * len(cfg.timescales)
    print(f"Total jobs: {total_jobs}")

    results: dict[tuple[str, str], CandidatePanelResult] = {}

    sector_bar = tqdm(sector_entries, desc="Sectors", unit="sector")

    for group_id, yaml_path in sector_bar:
        abbrev = sector_abbrev(group_id)
        sector_bar.set_postfix_str(abbrev)

        # Load UMD once per sector
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

        # Inner loop: timescales
        ts_iter = cfg.timescales
        if len(cfg.timescales) > 1:
            ts_iter = tqdm(cfg.timescales, desc=f"  {abbrev} timescales", leave=False, unit="ts")

        for ts in ts_iter:
            residual_cfg = cfg.make_residual_cfg(ts)
            persist_dir = cfg.persist_dir_for()

            stem = _make_panel_stem(
                group_id=group_id,
                ts=ts,
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

            key = (group_id, ts.label)
            results[key] = result

            n_rows = len(result.panel) if result.panel is not None else 0
            n_valid = int(result.panel["is_valid"].sum()) if n_rows > 0 else 0
            tqdm.write(f"  {abbrev} / {ts.label}: {n_valid}/{n_rows} valid candidates")

    print(f"\nDone. {len(results)} panels created.")
    return results