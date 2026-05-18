"""Tests for the MCP server tools.

The MCP server boots a JSON-RPC loop, but the registered tools are just
ordinary functions — we exercise them directly without spinning up a
client/server pair."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from reviewer.mcp_server import (
    _format_findings, _get_diff, mcp, review_diff, review_working_tree,
)


# ---------- registration ----------

def test_both_tools_registered_with_fastmcp():
    names = {t.name for t in mcp._tool_manager.list_tools()}
    assert "review_diff" in names
    assert "review_working_tree" in names


# ---------- _format_findings ----------

def test_format_findings_empty_returns_clean_message():
    out = _format_findings([])
    assert "No findings" in out


def test_format_findings_singular_finding_uses_singular_label():
    out = _format_findings([{
        "path": "a.py", "line": 1, "severity": "low",
        "category": "maintainability", "title": "x", "confidence": 0.7,
        "explanation": "y", "suggested_fix": None,
    }])
    assert "1 finding)" in out
    assert "1 findings)" not in out


def test_format_findings_uses_uppercase_severity_label():
    out = _format_findings([{
        "path": "a.py", "line": 1, "severity": "critical",
        "category": "security", "title": "x", "confidence": 0.9,
        "explanation": "y", "suggested_fix": None,
    }])
    assert "[CRITICAL/security]" in out


def test_format_findings_includes_suggested_fix_block():
    out = _format_findings([{
        "path": "a.py", "line": 1, "severity": "high",
        "category": "security", "title": "SQLi", "confidence": 0.9,
        "explanation": "string concat", "suggested_fix": "use params",
    }])
    assert "**Suggested fix:**" in out
    assert "use params" in out


def test_format_findings_omits_fix_block_when_none():
    out = _format_findings([{
        "path": "a.py", "line": 1, "severity": "low",
        "category": "maintainability", "title": "x", "confidence": 0.7,
        "explanation": "y", "suggested_fix": None,
    }])
    assert "**Suggested fix:**" not in out


def test_format_findings_unknown_severity_falls_back_to_low_label():
    out = _format_findings([{
        "path": "a.py", "line": 1, "severity": "???",
        "category": "maintainability", "title": "x", "confidence": 0.7,
        "explanation": "y", "suggested_fix": None,
    }])
    # Falls back to LOW (our default in _SEVERITY_LABEL.get)
    assert "[LOW/" in out


# ---------- _get_diff (subprocess shim) ----------

def test_get_diff_raises_on_missing_path(tmp_path):
    nope = tmp_path / "does-not-exist"
    with pytest.raises(RuntimeError, match="does not exist"):
        _get_diff(str(nope), staged=False)


def test_get_diff_runs_git_diff_cached_when_staged(tmp_path):
    """Verify the --cached flag is passed through. We don't need a real
    git repo — just check argv."""
    captured = {}

    class _FakeResult:
        returncode = 0
        stdout = "fake diff"
        stderr = ""

    def _fake_run(args, cwd, capture_output, text, check):
        captured["args"] = list(args)
        return _FakeResult()

    with patch("reviewer.mcp_server.subprocess.run", _fake_run):
        out = _get_diff(str(tmp_path), staged=True)
    assert out == "fake diff"
    assert captured["args"] == ["git", "diff", "--cached"]


def test_get_diff_raises_when_git_returncode_nonzero(tmp_path):
    class _FakeResult:
        returncode = 128
        stdout = ""
        stderr = "fatal: not a git repository"

    def _fake_run(*a, **kw):
        return _FakeResult()

    with patch("reviewer.mcp_server.subprocess.run", _fake_run):
        with pytest.raises(RuntimeError, match="not a git repository"):
            _get_diff(str(tmp_path), staged=False)


# ---------- review_diff tool ----------

SAMPLE_DIFF = """diff --git a/x.py b/x.py
index 1..2 100644
--- a/x.py
+++ b/x.py
@@ -0,0 +1,3 @@
+import os
+TOKEN = "abc"
+x = 1
"""


def test_review_diff_returns_markdown_with_findings(monkeypatch):
    """End-to-end of the review_diff tool with review_patch mocked."""
    expected = [{
        "path": "x.py", "line": 2, "severity": "critical",
        "category": "security", "title": "Hardcoded secret",
        "explanation": "TOKEN is hardcoded.", "suggested_fix": None,
        "confidence": 0.95,
    }]
    monkeypatch.setattr(
        "reviewer.mcp_server.review_patch",
        lambda diff_text, **kw: expected,
    )
    monkeypatch.setattr(
        "reviewer.mcp_server.build_provider",
        lambda name: type("P", (), {"name": name, "complete": None})(),
    )

    out = review_diff(SAMPLE_DIFF)
    assert "## AI Review (1 finding)" in out
    assert "[CRITICAL/security]" in out
    assert "Hardcoded secret" in out


def test_review_diff_empty_input_short_circuits_without_calling_provider(monkeypatch):
    """An empty diff should be cheap — no provider build, no model call."""
    calls = []
    monkeypatch.setattr(
        "reviewer.mcp_server.build_provider",
        lambda name: calls.append(name) or None,
    )
    out = review_diff("")
    assert out == "No changes to review."
    assert calls == []


def test_review_diff_passes_provider_and_thresholds(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "reviewer.mcp_server.build_provider",
        lambda name: captured.setdefault("provider", name) or type(
            "P", (), {"name": name})(),
    )
    monkeypatch.setattr(
        "reviewer.mcp_server.review_patch",
        lambda diff_text, **kw: captured.update(kw) or [],
    )
    review_diff(
        SAMPLE_DIFF, model="claude-sonnet-4-6", provider="anthropic",
        min_confidence=0.85, min_severity="high",
    )
    assert captured["provider"] == "anthropic"
    assert captured["model"] == "claude-sonnet-4-6"
    assert captured["min_confidence"] == 0.85
    assert captured["min_severity"] == "high"


# ---------- review_working_tree tool ----------

def test_review_working_tree_returns_friendly_message_on_clean_tree(monkeypatch):
    monkeypatch.setattr("reviewer.mcp_server._get_diff",
                        lambda repo, staged: "")
    out = review_working_tree(repo_path=".")
    assert "No working-tree changes" in out


def test_review_working_tree_says_staged_when_staged_clean(monkeypatch):
    monkeypatch.setattr("reviewer.mcp_server._get_diff",
                        lambda repo, staged: "")
    out = review_working_tree(staged=True)
    assert "No staged changes" in out


def test_review_working_tree_returns_error_on_git_failure(monkeypatch):
    def _boom(repo, staged):
        raise RuntimeError("fatal: not a git repository")
    monkeypatch.setattr("reviewer.mcp_server._get_diff", _boom)
    out = review_working_tree(repo_path="/not/a/repo")
    assert out.startswith("**Error:**")
    assert "not a git repository" in out


def test_review_working_tree_reviews_diff_when_changes_present(monkeypatch):
    monkeypatch.setattr("reviewer.mcp_server._get_diff",
                        lambda repo, staged: SAMPLE_DIFF)
    monkeypatch.setattr(
        "reviewer.mcp_server.build_provider",
        lambda name: type("P", (), {"name": name})(),
    )
    monkeypatch.setattr(
        "reviewer.mcp_server.review_patch",
        lambda diff_text, **kw: [{
            "path": "x.py", "line": 2, "severity": "high",
            "category": "security", "title": "secret",
            "explanation": "...", "suggested_fix": None, "confidence": 0.9,
        }],
    )
    out = review_working_tree(repo_path=".")
    assert "## AI Review (1 finding)" in out
    assert "[HIGH/security]" in out
