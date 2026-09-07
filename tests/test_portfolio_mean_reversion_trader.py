"""
Track D: PortfolioMeanReversionTrader.default_abs_target_group_exposure lost
its class default (was 1.0, sitting beside PortfolioMeanReversionConfig as a
second, ungated numeric default with no way to override it at the one
construction site, simulator_factory.py). Dormant on B's baseline (which
never constructs a PortfolioMeanReversionConfig/-Trader), so there is no
B_baseline.txt signal for this one — this test is the verification instead.
"""
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.simulator.config import PortfolioMeanReversionConfig
from src.simulator.traders.portfolio_mean_reversion import PortfolioMeanReversionTrader


class TestDefaultAbsTargetGroupExposureIsMandatory(unittest.TestCase):
    def test_omitting_it_raises(self):
        with self.assertRaises(TypeError):
            PortfolioMeanReversionTrader(
                config=PortfolioMeanReversionConfig(entry_z=2.0, exit_z=0.0),
            )

    def test_stating_it_succeeds(self):
        trader = PortfolioMeanReversionTrader(
            config=PortfolioMeanReversionConfig(entry_z=2.0, exit_z=0.0),
            default_abs_target_group_exposure=1.0,
        )
        self.assertEqual(trader.default_abs_target_group_exposure, 1.0)


if __name__ == "__main__":
    unittest.main()
