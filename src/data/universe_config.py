from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.types.common import GroupId


# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class UniverseConfig:
    raw: dict[str, Any]
    def to_string(self) -> str:
        out = ''
        out += 'UniverseConfig.universe_structure(cfg)\n'+UniverseConfig.universe_structure(self)+'\n'
        out += f'universe_name={self.universe_name}'+'\n'
        out += f'tickers={self.tickers}'+'\n'
        out += f'groups={self.groups}'+'\n'
        out += f'root_group={self.root_group}'+'\n'
        out += f'all_symbols()={self.all_symbols()}'+'\n'
        out += f'equity_symbols()={self.equity_symbols()}'+'\n'
        for g in self.groups:
            out += f'group_members({g})={self.group_members(g)}'+'\n'
            out += f'group_proxy({g})={self.group_proxy(g)}'+'\n'
            out += f'group_benchmark({g})={self.group_benchmark(g)}'+'\n'
        return out
        

    @classmethod
    def from_yaml(cls, path: str | Path) -> "UniverseConfig":
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        cfg = cls(raw=raw)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        groups = self.raw["groups"]
        tickers = self.raw["tickers"]

        if self.root_group not in groups:
            raise ValueError(f"Root group '{self.root_group}' not found in groups.")
        for id_, group in groups.items():
            parent = group.get("parent")
            if parent is not None and parent not in groups:
                raise ValueError(f"Group '{id_}' has unknown parent '{parent}'.")

            for symbol in group.get("members", []):
                if symbol not in tickers:
                    raise ValueError(f"Group '{id_}' references unknown ticker '{symbol}'.")

            for key in ("proxy_etf", "benchmark", "risk_free"):
                sym = group.get(key)
                if sym is not None and sym not in tickers:
                    raise ValueError(f"Group '{id_}' references unknown {key} '{sym}'.")

        seen: dict[str, str] = {}
        for id_, group in groups.items():
            for symbol in group.get("members", []):
                if symbol in seen:
                    raise ValueError(
                        f"Ticker '{symbol}' appears in multiple groups: "
                        f"'{seen[symbol]}' and '{id_}'."
                    )
                seen[symbol] = id_

    @property
    def universe_name(self) -> str:
        return self.raw["meta"]["universe_name"]

    @property
    def root_group(self) -> str:
        return self.raw["hierarchy"]["root"]

    @property
    def loader_defaults(self) -> dict[str, Any]:
        return self.raw.get("loader_defaults", {})

    @property
    def groups(self) -> dict[str, dict[str, Any]]:
        return self.raw["groups"]

    @property
    def tickers(self) -> dict[str, dict[str, Any]]:
        return self.raw["tickers"]

    def all_symbols(self) -> list[str]:
        out = set(self.tickers.keys())
        for g in self.groups.values():
            out.update(g.get("members", []))
            if g.get("proxy_etf"):
                out.add(g["proxy_etf"])
            if g.get("benchmark"):
                out.add(g["benchmark"])
            if g.get("risk_free"):
                out.add(g["risk_free"])
        return sorted(out)

    def equity_symbols(self) -> list[str]:
        return sorted(
            s for s, meta in self.tickers.items()
            if meta.get("kind") == "equity"
        )

    def group_members(self, id_: GroupId) -> list[str]:
        return list(self.groups[id_].get("members", []))

    def group_proxy(self, id_: GroupId) -> str | None:
        return self.groups[id_].get("proxy_etf")

    def group_benchmark(self, id_: GroupId) -> str | None:
        return self.groups[id_].get("benchmark")

    def group_risk_free(self, id_: GroupId) -> str | None:
        return self.groups[id_].get("risk_free")

    @staticmethod
    def universe_structure(cfg: UniverseConfig) -> str:
        """
        Render the universe hierarchy as simple ASCII art.

        Example output:

        epat_sector_hierarchy_v1
        `-- us_equities [root]
            |-- financials [sector]  proxy=XLF  bench=SPY
            |   |-- JPM
            |   |-- BAC
            |   `-- ...
            `-- utilities [sector]  proxy=XLU  bench=SPY
                |-- NEE
                `-- SO
        """
        groups = cfg.groups
        root = cfg.root_group
        title = cfg.universe_name

        def group_children(group_id: str) -> list[str]:
            return list(groups[group_id].get("children", []))

        def render_group(group_id: str, prefix: str = "", is_last: bool = True) -> list[str]:
            g = groups[group_id]
            connector = "`-- " if is_last else "|-- "
            line = f"{prefix}{connector}{group_id} [{g.get('type', '?')}]"

            proxy = g.get("proxy_etf")
            bench = g.get("benchmark")
            extras = []
            if proxy:
                extras.append(f"proxy={proxy}")
            if bench:
                extras.append(f"bench={bench}")
            if extras:
                line += "  " + "  ".join(extras)

            lines = [line]

            child_prefix = prefix + ("    " if is_last else "|   ")

            children = group_children(group_id)
            members = list(g.get("members", []))

            n = len(children) + len(members)
            idx = 0

            for child in children:
                idx += 1
                lines.extend(render_group(child, child_prefix, is_last=(idx == n)))

            for member in members:
                idx += 1
                member_connector = "`-- " if idx == n else "|-- "
                lines.append(f"{child_prefix}{member_connector}{member}")

            return lines

        lines = [title]
        lines.extend(render_group(root, prefix="", is_last=True))
        return "\n".join(lines)