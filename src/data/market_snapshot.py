"""
Market data snapshot layer (Track C).

A snapshot freezes a download session: one `dl_ts` applied to every group in
it, immutable once written. This is the boundary against "input revision" —
corporate actions retroactively re-adjusting historical closes so the same
tickers/range yield different values on a later download. Causality (row t
never reads data after t) does not protect against this; only binding a run
to one frozen, hash-verified block of data does.

THE UNIVERSE MODEL

A universe is a DIRECTORY of group yamls, one per group:

    config/universes/{universe_name}/{group_id}.yaml

The directory name IS the universe name. A universe name identifies exactly
one set of groups and tickers — any change to either (a group added or
removed, a ticker list edited) requires a new universe version (e.g.
sp500_v8 -> sp500_v9), never an edit in place. `ensure_market_snapshot`
therefore takes only `universe_version`; the group set is not a parameter,
it is read off the directory. Group *selection* for a run (a subset of a
universe's groups) is a run concern (DataConfig.selected_groups/
excluded_groups), not a snapshot concern — a snapshot always covers every
group in its universe directory.

This corrects the module's first version, which took an explicit `groups`
list and treated `universe_version` as a bare name with mint_universe_version
copying a directory to version it. That directory-copy design is dropped:
the universe directory under config/universes/ *is* the versioned record,
so there is nothing left for the snapshot layer to copy. See
docs/refactor/C_market_snapshot.md for the full model.

A snapshot freezes one download of one universe. Two snapshots of the same
universe may differ in realized end date and in price values (corporate
actions re-adjust history) — that is what dl_ts and the content hash are
for. What two snapshots of the same universe_version may NOT differ in is
group/ticker composition: `_mint_snapshot` enforces this by diffing the
current directory's yaml content against the verbatim yamls stored in the
most recent snapshot of the same universe_version, and raises if they
differ (see UniverseChangedError). With no prior snapshot there is nothing
to compare against — the first mint is definitional.

Layout — dl_ts outermost, one snapshot holds every group in its universe:

    {snapshot_root}/{universe_version}_{dl_ts}/
        manifest.json
        {universe_name}/            # e.g. materials_only_v1, one per group
            prices_daily.parquet
            ticker_info.parquet
            group_info.parquet
            membership.parquet

(The per-group subdirectory name is UniverseConfig.universe_name — the
group yaml's own `meta.universe_name` field, unrelated to and predating the
config/universes/{universe_name}/ directory-level name above. Both are
called "universe_name" for historical reasons; see found.md.)

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

Incomplete downloads: a ticker listed in a group yaml (member, proxy_etf,
benchmark, or risk_free) that yfinance returns no data for aborts the mint.
Every group is attempted before raising, so the single error lists every
missing ticker across every group, grouped by group_id, rather than
surfacing one at a time across repeated retries.
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


class UniverseChangedError(RuntimeError):
    """
    Raised at mint time when a universe directory's group yamls no longer
    match the most recent snapshot minted for the same universe_version.
    A universe_version identifies exactly one group/ticker set; edit the
    yaml(s) under a new universe_version instead of in place.
    """


class IncompleteUniverseDownloadError(RuntimeError):
    """
    Raised when one or more tickers listed in a universe's group yamls
    returned no data. Carries every missing ticker across every group, not
    just the first — see module docstring.
    """


def _dl_ts_now() -> str:
    return pd.Timestamp.now().strftime("%Y%m%dT%H%M%S%f")


def _validate_universe_version(universe_version: str) -> None:
    if not _UNIVERSE_VERSION_ONLY_RE.match(universe_version):
        raise ValueError(
            f"universe_version must match {_UNIVERSE_VERSION_ONLY_RE.pattern!r}, got "
            f"{universe_version!r}"
        )


def _discover_group_yamls(universe_path: Path) -> list[tuple[str, Path]]:
    """(group_id, yaml_path) pairs for every group in a universe directory."""
    return [(p.stem, p) for p in sorted(universe_path.glob("*.yaml"))]


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


def _check_universe_unchanged(group_entries: list[tuple[str, Path]], prior_manifest: dict) -> None:
    """
    Raise if the current universe directory's yaml content differs from the
    yamls stored (verbatim, per group) in `prior_manifest` — added group,
    removed group, or any content change to a shared group's yaml.
    """
    prior_groups: dict = prior_manifest["groups"]
    current_ids = {group_id for group_id, _ in group_entries}
    prior_ids = set(prior_groups.keys())

    added = sorted(current_ids - prior_ids)
    removed = sorted(prior_ids - current_ids)
    changed = []
    for group_id, yaml_path in group_entries:
        if group_id not in prior_groups:
            continue
        current_raw = UniverseConfig.from_yaml(yaml_path).raw
        if current_raw != prior_groups[group_id]["resolved_config"]:
            changed.append(group_id)

    if not (added or removed or changed):
        return

    parts = []
    if added:
        parts.append(f"group(s) added: {added}")
    if removed:
        parts.append(f"group(s) removed: {removed}")
    if changed:
        parts.append(f"group(s) with edited yaml content: {sorted(changed)}")

    raise UniverseChangedError(
        f"universe '{prior_manifest['universe_version']}' has changed since its most recent "
        f"snapshot ({prior_manifest['snapshot_id']}): {'; '.join(parts)}. A universe_version "
        "identifies exactly one group/ticker set. Mint a new universe_version directory "
        "(e.g. copy config/universes/ to a new version name with the desired edits) instead "
        "of editing this one in place."
    )


def _download_group(
    group_id: str, yaml_path: Path, tmp_dir: Path, progress: bool
) -> tuple[UniverseConfig, UniverseMarketData | None, list[str]]:
    """
    Returns (config, umd, missing_tickers). umd is None only when every
    requested ticker failed to download (UniverseDataLoader.load's "All
    ticker downloads failed" case) — missing_tickers is then every ticker
    requested for this group. Never raises on a ticker-availability gap;
    the caller aggregates across groups first. Still raises hard (and
    always) on SnapshotResyncGuardError — that is a structural bug, not a
    data-availability gap, and must never be folded into the aggregated
    missing-ticker report.
    """
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

    requested = set(config.all_symbols())
    try:
        umd = loader.load(force_download=True, check_for_corruptions=True, start_after_nan=False)
    except RuntimeError:
        # Every ticker in this group failed (UniverseDataLoader._download_prices
        # raises "All ticker downloads failed." in this case). Continue to the
        # next group rather than aborting — the caller wants every group's
        # gaps in one report, not this one first.
        return config, None, sorted(requested)

    missing = sorted(requested - set(umd.tickers()))
    return config, umd, missing


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


def _format_missing_tickers_error(universe_version: str, missing_by_group: dict[str, list[str]]) -> str:
    total = sum(len(v) for v in missing_by_group.values())
    lines = [
        f"mint aborted for universe '{universe_version}': {total} ticker(s) missing data "
        f"across {len(missing_by_group)} group(s):",
    ]
    for group_id in sorted(missing_by_group):
        lines.append(f"  {group_id}: {', '.join(missing_by_group[group_id])}")
    lines.append(
        "This is a data-provider survivorship gap (yfinance has no data for these "
        "tickers over the requested range), not a transient failure — retrying will not "
        "help. Create a new universe version with these tickers removed from the "
        "affected group yaml(s) and mint under that new universe_version."
    )
    return "\n".join(lines)


def _mint_snapshot(
    universe_version: str,
    universe_dir: Path,
    snapshot_root: Path,
    progress: bool,
) -> str:
    _validate_universe_version(universe_version)

    universe_path = universe_dir / universe_version
    if not universe_path.is_dir():
        raise FileNotFoundError(f"no universe directory: {universe_path}")

    group_entries = _discover_group_yamls(universe_path)
    if not group_entries:
        raise ValueError(f"universe directory has no group yaml files: {universe_path}")

    existing = _list_snapshots(universe_version, snapshot_root)
    if existing:
        prior_manifest = _read_manifest(snapshot_root / existing[0])
        _check_universe_unchanged(group_entries, prior_manifest)

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
        missing_by_group: dict[str, list[str]] = {}

        iterator = (
            tqdm(group_entries, desc=f"snapshot {snapshot_id}", unit="group")
            if progress else group_entries
        )
        for group_id, yaml_path in iterator:
            config, umd, missing = _download_group(group_id, yaml_path, tmp_dir, progress)
            if missing:
                missing_by_group[group_id] = missing
                continue
            group_dir = tmp_dir / config.universe_name
            manifest_groups[group_id] = _build_group_manifest_entry(config, umd, group_dir)

        if missing_by_group:
            raise IncompleteUniverseDownloadError(
                _format_missing_tickers_error(universe_version, missing_by_group)
            )

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
    """
    Every snapshot id for `universe_version`, chronologically sorted newest
    first. No group-set filtering: a universe_version identifies exactly one
    group set by construction (enforced at mint time by
    _check_universe_unchanged), so every snapshot returned here already
    shares the same group set.
    """
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
    *,
    universe_dir: str | Path | None = None,
    snapshot_root: str | Path | None = None,
    force: bool = False,
    progress: bool = True,
) -> list[str]:
    """
    Ensure a snapshot exists for `universe_version` — every group in
    config/universes/{universe_version}/, by construction.

    Returns every existing snapshot id for `universe_version`, chronologically
    sorted newest first — always a list, never a single "latest" pick, so
    there is no branch at the call site.

    Reuse: if `force` is False and at least one snapshot already exists for
    `universe_version`, nothing is downloaded. Otherwise a new snapshot is
    minted (never appended to an existing snapshot directory — that would
    mix adjustment vintages within one manifest and destroy the hash's
    meaning). Minting always enforces universe immutability first: the
    current directory's yaml content must match the most recent existing
    snapshot's, or the mint raises (UniverseChangedError) rather than
    silently drifting what universe_version means.
    """
    universe_dir = Path(universe_dir) if universe_dir else CONFIG_UNIVERSE
    snapshot_root = Path(snapshot_root) if snapshot_root else DATA_MARKET_SNAPSHOTS

    existing = _list_snapshots(universe_version, snapshot_root)
    if not force and existing:
        return existing

    _mint_snapshot(universe_version, universe_dir, snapshot_root, progress)
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
