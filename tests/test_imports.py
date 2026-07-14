"""Guard against source files that exist on disk but never reach a commit.

The original `.gitignore` carried unanchored `data/` and `artifacts/` patterns,
so git silently ignored the `src/data/` package at every tree level. Every
entry path died at import with `ModuleNotFoundError: No module named
'src.data'`. Importing each module under `src/` turns that class of defect into
a test failure instead of a broken clone.
"""

import importlib
import pkgutil
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"

# Resolve `src` against this repo rather than whatever an ambient PYTHONPATH
# happens to point at, so the test cannot end up grading a sibling checkout.
sys.path.insert(0, str(REPO_ROOT))


def iter_module_names():
    """Yield the dotted name of every module and subpackage under src/."""
    for module_info in pkgutil.walk_packages([str(SRC_ROOT)], prefix="src."):
        yield module_info.name


class TestImports(unittest.TestCase):
    def test_src_is_importable(self):
        names = sorted(iter_module_names())
        self.assertTrue(names, f"no modules discovered under {SRC_ROOT}")

        failures = []
        for name in names:
            try:
                importlib.import_module(name)
            except Exception as exc:
                failures.append(f"{name}: {type(exc).__name__}: {exc}")

        self.assertEqual(
            [],
            failures,
            "modules under src/ failed to import:\n" + "\n".join(failures),
        )


if __name__ == "__main__":
    unittest.main()
