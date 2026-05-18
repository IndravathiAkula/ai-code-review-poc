"""Wrapper script — generate the trend dashboard from a directory of
artifact JSONs.

Mirrors the ``scripts/review_local.py`` pattern: the reviewer package
lives at ``src/reviewer`` and isn't installed system-wide, so this
shim makes ``src/`` importable before delegating to
``reviewer.dashboard.main``.

Usage:
    python scripts/dashboard.py --inputs ./artifacts --out dashboard.html
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reviewer.dashboard import main


if __name__ == "__main__":
    raise SystemExit(main())
