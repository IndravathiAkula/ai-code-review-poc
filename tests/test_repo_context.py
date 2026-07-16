"""Tests for reviewer.repo_context — git grep-driven context retrieval."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reviewer import repo_context
from reviewer.repo_context import (
    build_related_code_block,
    extract_new_symbols,
    gather_context,
)


PY_DIFF = """diff --git a/svc/processor.py b/svc/processor.py
--- /dev/null
+++ b/svc/processor.py
@@ -0,0 +1,4 @@
+def process_order(order_id: str) -> dict:
+    return {"id": order_id}
+
+class OrderProcessor:
+    pass
"""


TS_DIFF = """diff --git a/web/api.ts b/web/api.ts
--- /dev/null
+++ b/web/api.ts
@@ -0,0 +1,3 @@
+export function fetchUserOrders(userId: string) {
+    return fetch(`/api/users/${userId}/orders`);
+}
"""


# ---- extract_new_symbols -------------------------------------------------

def test_extract_new_symbols_python_functions_and_classes():
    symbols = extract_new_symbols(PY_DIFF, "svc/processor.py")
    assert "process_order" in symbols
    assert "OrderProcessor" in symbols


def test_extract_new_symbols_typescript_export_function():
    symbols = extract_new_symbols(TS_DIFF, "web/api.ts")
    assert "fetchUserOrders" in symbols


def test_extract_new_symbols_drops_short_and_stopwords():
    diff = """diff --git a/a.py b/a.py
--- /dev/null
+++ b/a.py
@@ -0,0 +1,4 @@
+def a():
+    pass
+def get():
+    pass
"""
    # ``a`` too short; ``get`` is a stopword
    assert extract_new_symbols(diff, "a.py") == set()


def test_extract_new_symbols_drops_underscore_prefixed():
    """Private helpers aren't worth grepping for — usually only one caller."""
    diff = """diff --git a/a.py b/a.py
--- /dev/null
+++ b/a.py
@@ -0,0 +1,2 @@
+def _internal_helper():
+    pass
"""
    assert extract_new_symbols(diff, "a.py") == set()


def test_extract_new_symbols_unknown_language_returns_empty():
    diff = """diff --git a/a.unknown b/a.unknown
--- /dev/null
+++ b/a.unknown
@@ -0,0 +1,1 @@
+def process_order():
"""
    # Unknown language -> no patterns -> empty
    assert extract_new_symbols(diff, "a.unknown") == set()


# ---- build_related_code_block -------------------------------------------

def _init_git_repo(root: Path) -> None:
    """Initialise a throwaway git repo in ``root`` so git grep works."""
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)


def test_build_related_code_block_finds_caller(tmp_path):
    if not _git_available():
        pytest.skip("git not available")
    _init_git_repo(tmp_path)
    # Create the file under review AND a caller elsewhere in the repo.
    (tmp_path / "svc").mkdir()
    (tmp_path / "svc" / "processor.py").write_text(
        "def process_order(order_id):\n    return {}\n"
    )
    (tmp_path / "handlers.py").write_text(
        "from svc.processor import process_order\n\n"
        "def handle(req):\n    return process_order(req.id)\n"
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True,
    )

    block = build_related_code_block(PY_DIFF, "svc/processor.py", tmp_path)
    assert "Callers of `process_order`" in block
    # The snippet should show the caller file
    assert "handlers.py" in block


def test_build_related_code_block_returns_empty_when_no_callers(tmp_path):
    if not _git_available():
        pytest.skip("git not available")
    _init_git_repo(tmp_path)
    (tmp_path / "svc").mkdir()
    (tmp_path / "svc" / "processor.py").write_text(
        "def process_order(order_id):\n    return {}\n"
    )
    # No caller elsewhere and no sibling file
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True,
    )
    block = build_related_code_block(PY_DIFF, "svc/processor.py", tmp_path)
    # No callers, no siblings -> empty block
    assert block == ""


def test_build_related_code_block_includes_sibling_for_style(tmp_path):
    if not _git_available():
        pytest.skip("git not available")
    _init_git_repo(tmp_path)
    (tmp_path / "svc").mkdir()
    (tmp_path / "svc" / "processor.py").write_text(
        "def process_order(order_id):\n    return {}\n"
    )
    # A sibling to demonstrate style conventions
    (tmp_path / "svc" / "notifier.py").write_text(
        "import logging\nlogger = logging.getLogger(__name__)\n\n"
        "def notify(user):\n    logger.info('notify %s', user)\n"
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True,
    )
    block = build_related_code_block(PY_DIFF, "svc/processor.py", tmp_path)
    assert "Sibling file" in block
    assert "notifier.py" in block


def test_build_related_code_block_respects_max_chars(tmp_path):
    if not _git_available():
        pytest.skip("git not available")
    _init_git_repo(tmp_path)
    (tmp_path / "svc").mkdir()
    (tmp_path / "svc" / "processor.py").write_text(
        "def process_order(): pass\n"
    )
    (tmp_path / "sibling.py").write_text("x = 1\n" * 5000)  # huge
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True,
    )
    block = build_related_code_block(
        PY_DIFF, "svc/processor.py", tmp_path, max_chars=200,
    )
    assert len(block) <= 200 + len("\n... (context truncated)")


# ---- gather_context ------------------------------------------------------

def test_gather_context_swallows_errors():
    """Nonexistent root shouldn't raise — it just returns ''."""
    assert gather_context(PY_DIFF, "svc/x.py", "/does/not/exist") == ""


def test_gather_context_empty_diff():
    assert gather_context("", "x.py", "/tmp") == ""


# ---- helpers -------------------------------------------------------------

def _git_available() -> bool:
    from shutil import which
    return which("git") is not None
