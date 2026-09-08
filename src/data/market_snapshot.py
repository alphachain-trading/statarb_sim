"""
Market data snapshot layer (Track C).

A snapshot freezes a download session: one `dl_ts` applied to every group in
it, immutable once written. This is the boundary against "input revision" —
corporate actions retroactively re-adjusting historical closes so the same
tickers/range yield different values on a later download. Causality (row t
never reads data after t) does not protect against this; only binding a run
to one frozen, hash-verified block of data does.

`universe_version` is a NAME, not a versioned directory. Deliberately not a
copy of config/universe/: copying would give the group yamls a second place
to drift. Instead, the manifest records `universe_version` plus, per group,
the resolved yaml content verbatim — that is the versioned record, colocated
with the data it produced.

Deviation from the brief's literal signature: `ensure_market_snapshot` takes
an explicit `groups` list. The brief's `mint_universe_version` mechanism
(directory copy) was the only place a universe_version could resolve to a
group set; without it, nothing else supplies that mapping. `groups` fills
the gap deliberately rather than reintroducing a directory copy.

Layout — dl_ts outermost, one snapshot holds many groups:

    {snapshot_root}/{universe_version}_{dl_ts}/
        manifest.json
        {universe_name}/            # e.g. materials_only_v1, one per group
            prices_daily.parquet
            ticker_info.parquet
            group_info.parquet
            membership.parquet

Immutability: build under a temp dir beside the snapshot root, atomic
`os.rename` into position, then chmod the whole tree read-only. The write
path refuses to target an existing snapshot directory.

The snapshot build must never take `UniverseDataLoader.load`'s cache-hit
resync branch (universe_loader.py, `load()`, the `to_remove`/`to_add`
in-place rebuild) — that branch mutates a cache in place with no record of
the event, exactly the failure mode a snapshot exists to prevent. Structural
guarantee: each group downloads into a freshly created temp directory, so
`_cache_exists()` is always False and `load()` always takes the fresh-
download branch. `force_download=True` is passed as well, so the branch
taken does not even depend on that being true. `_download_group` also
asserts the cache does not exist first and raises loudly if it ever would —
belt and suspenders around a case that should be unreachable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.data.universe_config import UniverseConfig
from src.data.universe_loader import UniverseDataLoader
from src.data.universe_marketdata import UniverseMarketData
from src.settings import CONFIG_UNIVERSE, DATA_MARKET_SNAPSHOTS

UNIVERSE_VERSION_PATTERN = r"[A-Za-z0-9][A-Za-z0-9_]*"
# Microsecond resolution, not just seconds: two mint calls in the same
# process (e.g. a sweep script minting several universe_versions back to
# back) can otherwise land in the same second and collide on snapshot_id.
DL_TS_PATTERN = r"\d{8}T\d{12}"
SNAPSHOT_ID_RE = re.compile(rf"^{UNIVERSE_VERSION_PATTERN}_{DL_TS_PATTERN}$")
_UNIVERSE_VERSION_ONLY_RE = re.compile(rf"^{UNIVERSE_VERSION_PATTERN}$")


class SnapshotExistsError(FileExistsError):
    """Raised when a write would land on an existing snapshot directory."""


class SnapshotHashMismatchError(RuntimeError):
    """Raised when a loaded snapshot's on-disk content no longer matches its manifest."""


class SnapshotResyncGuardError(RuntimeError):
    """
    Raised if a snapshot build ever finds a pre-existing cache where it
    expects a fresh temp directory. Should be unreachable; see module
    docstring. Guards against silently taking UniverseDataLoader.load's
    in-place resync branch, which is incompatible with snapshot immutability.
    """


def _dl_ts_now() -> str:
    return pd.Timestamp.now().strftime("%Y%m%dT%H%M%S%f")


def _group_yaml_path(universe_dir: Path, group_id: str) -> Path:
    return universe_dir / f"universe.{group_id}_only.v1.yaml"


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_ticker_set(tickers: list[str]) -> str:
    payload = "\n".join(sorted(set(tickers))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_prices_df(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic re-normalization of a prices frame read back off disk."""
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Snapshot prices must have MultiIndex columns.")

    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)
    df.index = pd.DatetimeIndex(idx).normalize()
    df.index.name = "Date"

    df = df.sort_index().sort_index(axis=1)
    df.columns = pd.MultiIndex.from_tuples(
        [(str(a), str(b)) for a, b in df.columns], names=["ticker", "field"]
    )
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df


def _make_tree_read_only(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_dir():
            path.chmod(0o555)
        else:
            path.chmod(0o444)
    root.chmod(0o555)


def _finalize_snapshot(tmp_dir: Path, target_dir: Path) -> None:
    if target_dir.exists():
        raise SnapshotExistsError(
            f"refusing to write snapshot: target directory already exists: {target_dir}"
        )
    os.rename(tmp_dir, target_dir)
    _make_tree_read_only(target_dir)


def _download_group(group_id: str, yaml_path: Path, tmp_dir: Path, progress: bool) -> tuple[UniverseConfig, UniverseMarketData]:
    config = UniverseConfig.from_yaml(yaml_path)
    loader = UniverseDataLoader(config, data_path=tmp_dir, progress=progress)

    if loader._cache_exists():
        raise SnapshotResyncGuardError(
            f"snapshot download for group '{group_id}' found an existing cache at "
            f"{loader.cache_dir} inside what should be a fresh temp directory. "
            "Refusing to proceed: this would route through UniverseDataLoader.load's "
            "in-place cache-hit resync, which mutates data underneath a snapshot and "
            "destroys the immutability guarantee. This should be unreachable."
        )

    umd = loader.load(force_download=True, check_for_corruptions=True, start_after_nan=False)
    return config, umd


def _build_group_manifest_entry(config: UniverseConfig, umd: UniverseMarketData, group_dir: Path) -> dict:
    # The ticker set is the group membership roster (what membership.parquet
    # holds), not umd.tickers() — the latter also includes proxy/benchmark/
    # risk-free tickers, which are not sector membership and would make this
    # hash mismatch on every load_market_snapshot call for no real reason.
    tickers = sorted(umd.membership["ticker"].unique())
    d = config.loader_defaults
    return {
        "universe_name": config.universe_name,
        "resolved_config": config.raw,
        "content_hash": _hash_file(group_dir / "prices_daily.parquet"),
        "ticker_set_hash": _hash_ticker_set(tickers),
        "tickers": tickers,
        "requested_range": {"start": d.get("start_date"), "end": d.get("end_date")},
        "realized_range": {
            "start": str(umd.prices.index.min().date()),
            "end": str(umd.prices.index.max().date()),
        },
    }


def _mint_snapshot(
    universe_version: str,
    groups: list[str],
    universe_dir: Path,
    snapshot_root: Path,
    progress: bool,
) -> str:
    if not _UNIVERSE_VERSION_ONLY_RE.match(universe_version):
        raise ValueError(
            f"universe_version must match {_UNIVERSE_VERSION_ONLY_RE.pattern!r}, got "
            f"{universe_version!r}"
        )

    dl_ts = _dl_ts_now()
    snapshot_id = f"{universe_version}_{dl_ts}"

    snapshot_root.mkdir(parents=True, exist_ok=True)
    target_dir = snapshot_root / snapshot_id
    if target_dir.exists():
        raise SnapshotExistsError(
            f"refusing to write snapshot: target directory already exists: {target_dir}"
        )

    tmp_dir = Path(tempfile.mkdtemp(dir=snapshot_root, prefix=f".tmp-{snapshot_id}-"))
    try:
        manifest_groups: dict[str, dict] = {}
        group_ids = sorted(set(groups))
        iterator = tqdm(group_ids, desc=f"snapshot {snapshot_id}", unit="group") if progress else group_ids
        for group_id in iterator:
            yaml_path = _group_yaml_path(universe_dir, group_id)
            if not yaml_path.exists():
                raise FileNotFoundError(
                    f"missing universe config for group '{group_id}': {yaml_path}"
                )
            config, umd = _download_group(group_id, yaml_path, tmp_dir, progress)
            group_dir = tmp_dir / config.universe_name
            manifest_groups[group_id] = _build_group_manifest_entry(config, umd, group_dir)

        manifest = {
            "snapshot_id": snapshot_id,
            "universe_version": universe_version,
            "dl_ts": dl_ts,
            "groups": manifest_groups,
        }
        (tmp_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        _finalize_snapshot(tmp_dir, target_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    return snapshot_id


def _read_manifest(snapshot_dir: Path) -> dict:
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no manifest.json in snapshot directory: {snapshot_dir}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _list_snapshots(universe_version: str, snapshot_root: Path) -> list[str]:
    if not snapshot_root.exists():
        return []
    pattern = re.compile(rf"^{re.escape(universe_version)}_({DL_TS_PATTERN})$")
    found: list[tuple[str, str]] = []
    for child in snapshot_root.iterdir():
        if not child.is_dir():
            continue
        m = pattern.match(child.name)
        if m:
            found.append((m.group(1), child.name))
    found.sort(reverse=True)  # dl_ts is sortable; newest first
    return [snapshot_id for _dl_ts, snapshot_id in found]


def _merge_group_market_data(per_group: list[UniverseMarketData]) -> UniverseMarketData:
    """
    Concatenate single-group UniverseMarketData into one. No cross-group
    ticker-overlap policy here — a snapshot may deliberately hold overlapping
    groups; whether that is permitted for a given run is a simulator-layer
    decision (see src.simulator.simulator_factory._merge_umds), not a
    snapshot-loading one.
    """
    merged_prices = per_group[0].prices.copy()
    for u in per_group[1:]:
        new_cols = [c for c in u.prices.columns if c not in merged_prices.columns]
        if new_cols:
            merged_prices = merged_prices.join(u.prices[new_cols], how="outer")

    merged_ticker_info = pd.concat([u.ticker_info for u in per_group], axis=0)
    merged_ticker_info = merged_ticker_info[~merged_ticker_info.index.duplicated(keep="first")]

    merged_group_info = pd.concat([u.group_info for u in per_group], axis=0)
    merged_group_info = merged_group_info[~merged_group_info.index.duplicated(keep="first")]

    merged_membership = pd.concat([u.membership for u in per_group], axis=0, ignore_index=True)
    merged_membership = merged_membership.drop_duplicates()

    return UniverseMarketData(
        prices=merged_prices.sort_index(axis=1),
        ticker_info=merged_ticker_info.sort_index(),
        group_info=merged_group_info.sort_index(),
        membership=merged_membership.sort_values(["group_id", "ticker"]).reset_index(drop=True),
    )


def ensure_market_snapshot(
    universe_version: str,
    groups: list[str],
    *,
    universe_dir: str | Path | None = None,
    snapshot_root: str | Path | None = None,
    force: bool = False,
    progress: bool = True,
) -> list[str]:
    """
    Ensure a snapshot exists for `universe_version` covering exactly `groups`.

    Returns every existing snapshot id for `universe_version`, chronologically
    sorted newest first — always a list, never a single "latest" pick, so
    there is no branch at the call site.

    Reuse: if `force` is False and an existing snapshot's group set exactly
    equals `groups`, nothing is downloaded. Otherwise a new snapshot is
    minted with exactly the requested group set (never appended to an
    existing snapshot directory — that would mix adjustment vintages within
    one manifest and destroy the hash's meaning). This mint-not-append rule
    is what "adding a group raises" means in practice: there is no append
    operation for a caller to reach.
    """
    if not groups:
        raise ValueError("ensure_market_snapshot requires a non-empty groups list.")

    universe_dir = Path(universe_dir) if universe_dir else CONFIG_UNIVERSE
    snapshot_root = Path(snapshot_root) if snapshot_root else DATA_MARKET_SNAPSHOTS
    requested = set(groups)

    existing = _list_snapshots(universe_version, snapshot_root)

    if not force:
        for snapshot_id in existing:
            manifest = _read_manifest(snapshot_root / snapshot_id)
            if set(manifest["groups"].keys()) == requested:
                return existing

    _mint_snapshot(universe_version, sorted(requested), universe_dir, snapshot_root, progress)
    return _list_snapshots(universe_version, snapshot_root)


def load_market_snapshot(
    snapshot_id: str,
    *,
    snapshot_root: str | Path | None = None,
) -> UniverseMarketData:
    """
    Load a snapshot by its full id. Recomputes both manifest hashes per group
    and raises on any mismatch — never warns; a warning scrolls past in a
    sweep. `snapshot_id` must be a full id: never resolves a prefix.
    """
    if not SNAPSHOT_ID_RE.match(snapshot_id):
        raise ValueError(
            f"'{snapshot_id}' is not a full snapshot id (expected "
            f"'{{universe_version}}_{{dl_ts}}', e.g. "
            "'sp500_v3_20260825T143012481923'); partial/prefix ids are never resolved."
        )

    snapshot_root = Path(snapshot_root) if snapshot_root else DATA_MARKET_SNAPSHOTS
    snapshot_dir = snapshot_root / snapshot_id
    if not snapshot_dir.exists():
        raise FileNotFoundError(f"no snapshot at {snapshot_dir}")

    manifest = _read_manifest(snapshot_dir)

    per_group: dict[str, UniverseMarketData] = {}
    for group_id, entry in manifest["groups"].items():
        group_dir = snapshot_dir / entry["universe_name"]
        prices_path = group_dir / "prices_daily.parquet"

        actual_content_hash = _hash_file(prices_path)
        if actual_content_hash != entry["content_hash"]:
            raise SnapshotHashMismatchError(
                f"content hash mismatch for group '{group_id}' in snapshot "
                f"'{snapshot_id}': expected {entry['content_hash']}, got "
                f"{actual_content_hash}. {prices_path} was modified after the "
                "snapshot was written."
            )

        membership = pd.read_parquet(group_dir / "membership.parquet")
        actual_ticker_set_hash = _hash_ticker_set(sorted(membership["ticker"].unique()))
        if actual_ticker_set_hash != entry["ticker_set_hash"]:
            raise SnapshotHashMismatchError(
                f"ticker-set hash mismatch for group '{group_id}' in snapshot "
                f"'{snapshot_id}': expected {entry['ticker_set_hash']}, got "
                f"{actual_ticker_set_hash}. membership.parquet was modified after "
                "the snapshot was written."
            )

        prices = _normalize_prices_df(pd.read_parquet(prices_path))
        ticker_info = pd.read_parquet(group_dir / "ticker_info.parquet").sort_index()
        group_info = pd.read_parquet(group_dir / "group_info.parquet").sort_index()
        membership = membership.sort_values(["group_id", "ticker"]).reset_index(drop=True)

        per_group[group_id] = UniverseMarketData(
            prices=prices,
            ticker_info=ticker_info,
            group_info=group_info,
            membership=membership,
        )

    values = list(per_group.values())
    if len(values) == 1:
        return values[0]
    return _merge_group_market_data(values)
