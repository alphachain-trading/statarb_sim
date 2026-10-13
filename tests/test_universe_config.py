"""UniverseConfig: structural validation, symbol accessors, YAML loading."""

import tempfile
import unittest
from pathlib import Path
from typing import Any

import yaml

from src.data.universe_config import UniverseConfig
from src.settings import CONFIG_UNIVERSE


def _raw() -> dict[str, Any]:
    return {
        "meta": {"universe_name": "mini"},
        "hierarchy": {"root": "root"},
        "tickers": {
            "AAA": {"kind": "equity"},
            "BBB": {"kind": "equity"},
            "XLE": {"kind": "etf"},
            "SPY": {"kind": "etf"},
            "IRX": {"kind": "rate"},
        },
        "groups": {
            "root": {"type": "root", "children": ["energy"]},
            "energy": {
                "type": "sector",
                "parent": "root",
                "members": ["AAA", "BBB"],
                "proxy_etf": "XLE",
                "benchmark": "SPY",
                "risk_free": "IRX",
            },
        },
    }


def _cfg(**mutate: Any) -> UniverseConfig:
    raw = _raw()
    for path, value in mutate.items():
        section, key = path.split("__")
        raw[section][key] = value
    return UniverseConfig(raw=raw)


class TestValidation(unittest.TestCase):
    def test_valid_config_passes(self):
        _cfg().validate()

    def test_unknown_root_group(self):
        raw = _raw()
        raw["hierarchy"]["root"] = "nowhere"
        with self.assertRaises(ValueError):
            UniverseConfig(raw=raw).validate()

    def test_unknown_parent(self):
        raw = _raw()
        raw["groups"]["energy"]["parent"] = "ghost"
        with self.assertRaises(ValueError):
            UniverseConfig(raw=raw).validate()

    def test_member_must_be_a_known_ticker(self):
        raw = _raw()
        raw["groups"]["energy"]["members"].append("ZZZ")
        with self.assertRaises(ValueError) as ctx:
            UniverseConfig(raw=raw).validate()
        self.assertIn("ZZZ", str(ctx.exception))

    def test_proxy_benchmark_and_risk_free_must_be_known_tickers(self):
        for key in ("proxy_etf", "benchmark", "risk_free"):
            with self.subTest(key=key):
                raw = _raw()
                raw["groups"]["energy"][key] = "ZZZ"
                with self.assertRaises(ValueError):
                    UniverseConfig(raw=raw).validate()

    def test_ticker_in_two_groups_is_rejected(self):
        raw = _raw()
        raw["groups"]["root"]["members"] = ["AAA"]
        with self.assertRaises(ValueError) as ctx:
            UniverseConfig(raw=raw).validate()
        self.assertIn("multiple groups", str(ctx.exception))


class TestAccessors(unittest.TestCase):
    def setUp(self):
        self.cfg = _cfg()

    def test_names_and_members(self):
        self.assertEqual(self.cfg.universe_name, "mini")
        self.assertEqual(self.cfg.root_group, "root")
        self.assertEqual(self.cfg.group_members("energy"), ["AAA", "BBB"])
        self.assertEqual(self.cfg.group_members("root"), [])
        self.assertEqual(
            (
                self.cfg.group_proxy("energy"),
                self.cfg.group_benchmark("energy"),
                self.cfg.group_risk_free("energy"),
            ),
            ("XLE", "SPY", "IRX"),
        )
        self.assertIsNone(self.cfg.group_proxy("root"))

    def test_symbol_lists(self):
        self.assertEqual(self.cfg.all_symbols(), ["AAA", "BBB", "IRX", "SPY", "XLE"])
        self.assertEqual(self.cfg.equity_symbols(), ["AAA", "BBB"])

    def test_loader_defaults_default_to_empty(self):
        self.assertEqual(self.cfg.loader_defaults, {})
        raw = _raw()
        raw["loader_defaults"] = {"interval": "1d"}
        self.assertEqual(UniverseConfig(raw=raw).loader_defaults, {"interval": "1d"})

    def test_structure_rendering(self):
        text = UniverseConfig.universe_structure(self.cfg)
        self.assertEqual(
            text.splitlines(),
            [
                "mini",
                "`-- root [root]",
                "    `-- energy [sector]  proxy=XLE  bench=SPY",
                "        |-- AAA",
                "        `-- BBB",
            ],
        )
        self.assertIn("universe_name=mini", self.cfg.to_string())


class TestFromYaml(unittest.TestCase):
    def test_accepts_str_and_path_and_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "good.yaml"
            good.write_text(yaml.safe_dump(_raw()))
            self.assertEqual(UniverseConfig.from_yaml(str(good)).universe_name, "mini")
            self.assertEqual(UniverseConfig.from_yaml(good).universe_name, "mini")
            bad_raw = _raw()
            bad_raw["groups"]["energy"]["members"].append("ZZZ")
            bad = Path(tmp) / "bad.yaml"
            bad.write_text(yaml.safe_dump(bad_raw))
            with self.assertRaises(ValueError):
                UniverseConfig.from_yaml(bad)

    def test_every_committed_universe_file_is_valid(self):
        files = sorted(Path(CONFIG_UNIVERSE).rglob("*.yaml"))
        self.assertTrue(files, "no committed universe configs found")
        for f in files:
            with self.subTest(file=f.name):
                UniverseConfig.from_yaml(f)


if __name__ == "__main__":
    unittest.main()
