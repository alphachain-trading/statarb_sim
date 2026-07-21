"""
Batch candidate-panel creation over a mode-based residual configuration.

Usage (notebook):

    from src.candidates.panel_batch import PanelBatchConfig, run_panel_batch
    from src.simulator.config import ResidualMode, AbsOrMult

    cfg = PanelBatchConfig(
        residual_mode=ResidualMode.DECAY_EXPANDING,
        residual_hl=504,
        hedge_ratio_lb=252,
        mr_diag_lb=252,
        residual_min_lb_type_dec_exp=AbsOrMult.MULTIPLIER,
        residual_min_lb_dec_exp=2,
        subtract_risk_free=True,
        selected_sectors=["materials"],
    )
    results = run_panel_batch(cfg)

residual_hl / residual_lb may be lists to sweep the residual timescale; when
swept, hedge_ratio_lb / mr_diag_lb may be per-point lists of matching length
(or scalars broadcast across the sweep).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from tqdm.auto import tqdm

from src.settings import CONFIG_UNIVERSE, DATA_UNIVERSES, CANDIDATE_PANELS_ROOT

from src.simulator.config import (
    ResidualMode,
    AbsOrMult,
    SectorDataSource,
    sector_abbrev,
)
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


@dataclass(frozen=True)
class ResidualSpec:
    """One resolved sweep point: the residual config plus its scoring windows."""
    residual_cfg: CausalResidualConfig
    hedge_ratio_lb: int
    mr_diag_lb: int
    sweep_value: int | None  # residual_hl / residual_lb for this point; None for eq_expanding


@dataclass(kw_only=True)
class PanelBatchConfig:
    """
    Central config for batch candidate panel creation.

    The residual model is selected via ``residual_mode`` (see ResidualMode).
    The residual timescale (``residual_hl`` for decay_expanding, ``residual_lb``
    for eq_rolling) may be a list to sweep multiple panels in one batch;
    ``hedge_ratio_lb`` / ``mr_diag_lb`` are the two independent candidate-scoring
    windows (hedge-ratio fit vs mean-reversion diagnostics).
    """

    # ── residual shape (spec fields) ─────────────────────────────────────
    residual_mode: ResidualMode
    residual_hl: int | list[int] | None = None
    residual_lb: int | list[int] | None = None
    hedge_ratio_lb: int | list[int] = field(default=None)  # required (validated)
    mr_diag_lb: int | list[int] = field(default=None)       # required (validated)
    residual_min_lb_eq_exp: int | None = None                # required for eq_expanding
    residual_min_lb_type_dec_exp: AbsOrMult | None = None    # required for decay_expanding
    residual_min_lb_dec_exp: int | None = None               # required for decay_expanding
    residual_window_mode: Literal["rolling", "expanding"] = field(init=False)

    # ── residual model extras ────────────────────────────────────────────
    remove_residual_pcs: int = 0
    subtract_risk_free: bool = False

    # Pair spread config (shared across all sectors/sweep points)
    pair_cfg: PairSpreadConfig = field(
        default_factory=lambda: PairSpreadConfig(
            hedge_ratio_methods=["pca"],
            skip_adf=False,
        ),
    )

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

    # ── derivation + validation ──────────────────────────────────────────

    def __post_init__(self) -> None:
        # residual_window_mode derives from residual_mode.
        self.residual_window_mode = (
            "rolling" if self.residual_mode is ResidualMode.EQ_ROLLING else "expanding"
        )

        if self.hedge_ratio_lb is None:
            raise ValueError("hedge_ratio_lb is required.")
        if self.mr_diag_lb is None:
            raise ValueError("mr_diag_lb is required.")

        # 1. Exactly one of residual_hl / residual_lb set, matching the mode.
        mode = self.residual_mode
        if mode is ResidualMode.EQ_ROLLING:
            if self.residual_lb is None:
                raise ValueError("EQ_ROLLING requires residual_lb.")
            if self.residual_hl is not None:
                raise ValueError("EQ_ROLLING requires residual_hl to be None.")
        elif mode is ResidualMode.DECAY_EXPANDING:
            if self.residual_hl is None:
                raise ValueError("DECAY_EXPANDING requires residual_hl.")
            if self.residual_lb is not None:
                raise ValueError("DECAY_EXPANDING requires residual_lb to be None.")
        elif mode is ResidualMode.EQ_EXPANDING:
            if self.residual_hl is not None or self.residual_lb is not None:
                raise ValueError(
                    "EQ_EXPANDING requires both residual_hl and residual_lb to be None."
                )
        else:  # pragma: no cover - exhaustive
            raise ValueError(f"Unknown residual_mode: {mode!r}")

        # 3. Mode-specific min-lb fields: required for their mode, forbidden elsewhere.
        if mode is ResidualMode.EQ_EXPANDING:
            if self.residual_min_lb_eq_exp is None:
                raise ValueError("EQ_EXPANDING requires residual_min_lb_eq_exp.")
        elif self.residual_min_lb_eq_exp is not None:
            raise ValueError(
                "residual_min_lb_eq_exp is only valid for EQ_EXPANDING."
            )

        if mode is ResidualMode.DECAY_EXPANDING:
            if self.residual_min_lb_type_dec_exp is None or self.residual_min_lb_dec_exp is None:
                raise ValueError(
                    "DECAY_EXPANDING requires residual_min_lb_type_dec_exp and "
                    "residual_min_lb_dec_exp."
                )
        elif (
            self.residual_min_lb_type_dec_exp is not None
            or self.residual_min_lb_dec_exp is not None
        ):
            raise ValueError(
                "residual_min_lb_type_dec_exp / residual_min_lb_dec_exp are only "
                "valid for DECAY_EXPANDING."
            )

        # 2. hedge_ratio_lb / mr_diag_lb list-length must match the swept axis.
        swept = self._swept_value()
        if isinstance(swept, list):
            n = len(swept)
            for name in ("hedge_ratio_lb", "mr_diag_lb"):
                val = getattr(self, name)
                if isinstance(val, list) and len(val) != n:
                    raise ValueError(
                        f"{name} list length {len(val)} != swept residual length {n}."
                    )
        else:
            for name in ("hedge_ratio_lb", "mr_diag_lb"):
                if isinstance(getattr(self, name), list):
                    raise ValueError(
                        f"{name} must be a scalar when the residual timescale is not swept."
                    )

    def _swept_value(self) -> int | list[int] | None:
        """The residual timescale for this mode (residual_hl or residual_lb)."""
        if self.residual_mode is ResidualMode.EQ_ROLLING:
            return self.residual_lb
        if self.residual_mode is ResidualMode.DECAY_EXPANDING:
            return self.residual_hl
        return None  # eq_expanding has no swept timescale

    # ── path helpers ─────────────────────────────────────────────────────

    def resolved_universe_dir(self) -> Path:
        return Path(self.universe_dir) if self.universe_dir else Path(CONFIG_UNIVERSE)

    def resolved_data_path(self) -> Path:
        return Path(self.data_path) if self.data_path else Path(DATA_UNIVERSES)

    def persist_dir_for(self) -> str:
        """Persist directory for all panels (flat, sweep point in stem)."""
        return self.persist_dir_template

    # ── residual-config resolution ───────────────────────────────────────

    def _min_history(self, sweep_value: int | None) -> int:
        """Resolve min_history for one sweep point per the residual mode."""
        mode = self.residual_mode
        if mode is ResidualMode.EQ_ROLLING:
            return int(sweep_value)  # min_history == lookback
        if mode is ResidualMode.DECAY_EXPANDING:
            if self.residual_min_lb_type_dec_exp is AbsOrMult.MULTIPLIER:
                return int(sweep_value) * int(self.residual_min_lb_dec_exp)
            return int(self.residual_min_lb_dec_exp)
        # eq_expanding
        return int(self.residual_min_lb_eq_exp)

    def resolved_specs(self) -> list[ResidualSpec]:
        """Expand the (possibly swept) config into one ResidualSpec per panel."""
        swept = self._swept_value()
        if isinstance(swept, list):
            n = len(swept)
            sweep_values: list[int | None] = list(swept)
        elif swept is None:
            n = 1
            sweep_values = [None]
        else:
            n = 1
            sweep_values = [swept]

        def _pick(val: int | list[int], i: int) -> int:
            return int(val[i]) if isinstance(val, list) else int(val)

        specs: list[ResidualSpec] = []
        for i in range(n):
            sv = sweep_values[i]
            min_history = self._min_history(sv)
            if self.residual_mode is ResidualMode.EQ_ROLLING:
                residual_cfg = CausalResidualConfig(
                    window_mode="rolling",
                    lookback=int(sv),
                    half_life=None,
                    min_history=min_history,
                    remove_residual_pcs=self.remove_residual_pcs,
                    subtract_risk_free=self.subtract_risk_free,
                )
            elif self.residual_mode is ResidualMode.DECAY_EXPANDING:
                residual_cfg = CausalResidualConfig(
                    window_mode="expanding",
                    half_life=int(sv),
                    min_history=min_history,
                    remove_residual_pcs=self.remove_residual_pcs,
                    subtract_risk_free=self.subtract_risk_free,
                )
            else:  # eq_expanding
                residual_cfg = CausalResidualConfig(
                    window_mode="expanding",
                    half_life=None,
                    min_history=min_history,
                    remove_residual_pcs=self.remove_residual_pcs,
                    subtract_risk_free=self.subtract_risk_free,
                )
            specs.append(ResidualSpec(
                residual_cfg=residual_cfg,
                hedge_ratio_lb=_pick(self.hedge_ratio_lb, i),
                mr_diag_lb=_pick(self.mr_diag_lb, i),
                sweep_value=sv,
            ))
        return specs


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
) -> dict[tuple[str, str, int | None], CandidatePanelResult]:
    """
    Run batch panel creation across sectors × residual sweep points.

    Returns
    -------
    dict mapping (group_id, residual_mode, sweep_value) → CandidatePanelResult
    where sweep_value is the residual_hl / residual_lb for that point (None for
    eq_expanding).
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

    specs = cfg.resolved_specs()
    mode = cfg.residual_mode.value

    sector_labels = [sector_abbrev(gid) for gid, _ in sector_entries]
    print(f"Batch {batch_timestamp}")
    print(f"Sectors: {len(sector_entries)} [{', '.join(sector_labels)}]")
    print(f"Residual mode: {mode}")
    print(f"Sweep points: {[s.sweep_value for s in specs]} ({[s.residual_cfg.key for s in specs]})")
    total_jobs = len(sector_entries) * len(specs)
    print(f"Total jobs: {total_jobs}")

    results: dict[tuple[str, str, int | None], CandidatePanelResult] = {}

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

        # Inner loop: residual sweep points
        spec_iter = specs
        if len(specs) > 1:
            spec_iter = tqdm(specs, desc=f"  {abbrev} sweep", leave=False, unit="pt")

        for spec in spec_iter:
            residual_cfg = spec.residual_cfg
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
                hedge_ratio_lb=spec.hedge_ratio_lb,
                mr_diag_lb=spec.mr_diag_lb,
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

            label = f"{mode}/{spec.sweep_value}" if spec.sweep_value is not None else mode
            key = (group_id, mode, spec.sweep_value)
            results[key] = result

            n_rows = len(result.panel) if result.panel is not None else 0
            n_valid = int(result.panel["is_valid"].sum()) if n_rows > 0 else 0
            tqdm.write(f"  {abbrev} / {label}: {n_valid}/{n_rows} valid candidates")

    print(f"\nDone. {len(results)} panels created.")
    return results