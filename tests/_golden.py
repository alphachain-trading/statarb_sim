"""
Golden-master helper: compare a JSON-serialisable structure with a committed file.

Floats compare with a relative tolerance (BLAS / platform differences), everything
else exactly. Regenerate deliberately with ``UPDATE_GOLDEN=1 pytest <test>`` and
review the diff — goldens are never rewritten implicitly.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

GOLDEN_DIR = Path(__file__).parent / "golden"
REL_TOL = 1e-7
ABS_TOL = 1e-9


def _diff(expected: Any, actual: Any, path: str, out: list[str]) -> None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected:
                out.append(f"{path}.{key}: unexpected key")
            elif key not in actual:
                out.append(f"{path}.{key}: missing key")
            else:
                _diff(expected[key], actual[key], f"{path}.{key}", out)
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            out.append(f"{path}: length {len(actual)} != golden {len(expected)}")
            return
        for i, (e, a) in enumerate(zip(expected, actual, strict=True)):
            _diff(e, a, f"{path}[{i}]", out)
    elif isinstance(expected, float) or isinstance(actual, float):
        both_nan = (
            isinstance(expected, float)
            and isinstance(actual, float)
            and math.isnan(expected)
            and math.isnan(actual)
        )
        if both_nan:
            return
        if not (
            isinstance(expected, (int, float))
            and isinstance(actual, (int, float))
            and math.isclose(expected, actual, rel_tol=REL_TOL, abs_tol=ABS_TOL)
        ):
            out.append(f"{path}: {actual!r} != golden {expected!r}")
    elif expected != actual:
        out.append(f"{path}: {actual!r} != golden {expected!r}")


def assert_matches_golden(name: str, actual: Any) -> None:
    """Fail with a readable diff if ``actual`` differs from tests/golden/<name>.json."""
    path = GOLDEN_DIR / f"{name}.json"
    normalised = json.loads(json.dumps(actual, default=str, allow_nan=True))
    if os.environ.get("UPDATE_GOLDEN") == "1":
        GOLDEN_DIR.mkdir(exist_ok=True)
        path.write_text(json.dumps(normalised, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        return
    if not path.exists():
        raise AssertionError(f"golden file {path} missing; create it with UPDATE_GOLDEN=1")
    expected = json.loads(path.read_text(encoding="utf-8"))
    problems: list[str] = []
    _diff(expected, normalised, name, problems)
    if problems:
        shown = "\n".join(problems[:20])
        more = f"\n... and {len(problems) - 20} more" if len(problems) > 20 else ""
        raise AssertionError(f"golden mismatch for {name}:\n{shown}{more}")
