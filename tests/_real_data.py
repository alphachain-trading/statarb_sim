"""Guards for tests that read real market data out of a local market snapshot."""

from __future__ import annotations

import unittest

from src.data.market_snapshot import list_market_snapshots
from src.settings import DATA_MARKET_SNAPSHOTS


def require_market_snapshot(universe_version: str) -> None:
    """
    Skip the calling test unless a snapshot of `universe_version` exists.

    Tests never mint over the network: without this guard,
    bind_market_snapshot would mint one on first use.
    """
    if not list_market_snapshots(universe_version):
        raise unittest.SkipTest(
            f"no market snapshot of '{universe_version}' under {DATA_MARKET_SNAPSHOTS}; "
            "tests never mint over the network. Mint one with "
            f"ensure_market_snapshot('{universe_version}') to run this test."
        )
