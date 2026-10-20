"""Guards for tests that read real market data out of a local market snapshot."""

from __future__ import annotations

import unittest

from src.settings import DATA_MARKET_SNAPSHOTS


def require_market_snapshot(snapshot_id: str) -> None:
    """
    Skip the calling test unless the pinned snapshot `snapshot_id` exists.

    Tests never mint over the network, and binding never mints either (#12):
    without this guard, bind_market_snapshot would raise on a missing snapshot.
    """
    if not (DATA_MARKET_SNAPSHOTS / snapshot_id).is_dir():
        raise unittest.SkipTest(
            f"no market snapshot '{snapshot_id}' under {DATA_MARKET_SNAPSHOTS}; "
            "tests never mint over the network. Restore it from backup (#54) "
            "to run this test."
        )
