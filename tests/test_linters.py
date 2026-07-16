"""Unit tests for reviewer.linters.

We stub ``subprocess.run`` and ``shutil.which`` so tests don't need real
ruff / eslint / mypy / tsc installations.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reviewer import linters
from reviewer.linters import (
    _filter_to_added,
    _parse_mypy_line,
    _parse_tsc_line,
    added_lines_by_path,
    run_all,
    run_bandit,
    run_eslint,
    run_mypy,
    run_ruff,
    run_semgrep,
    run_tsc,
)


SIMPLE_DIFF = """diff --git a/a.py b/a.py
index e69de29..4a1b2c3 100644
--- a/a.py
+++ b/a.py
@@ -0,0 +1,3 @@
+import os
+import sys
+print("hi")
"""


TS_DIFF = """diff --git a/app.ts b/app.ts
index e69de29..4a1b2c3 100644
--- a/app.ts
+++ b/app.ts
@@ -0,0 +1,2 @@
+const x: any = 1;
+console.log(x);
"""


# ---- added_lines_by_path -------------------------------------------------

def test_added_lines_by_path_returns_only_added_lines():
    added = added_lines_by_path(SIMPLE_DIFF)
    assert added == {"a.py": {1, 2, 3}}


def test_added_lines_by_path_ignores_context_lines():
    diff = """diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1,3 +1,4 @@
 line1
-line2
+line2_new
+added
 line3
"""
    added = added_lines_by_path(diff)
    # Only the two "+" lines, not context lines.
    assert added == {"a.py": {2, 3}}


# ---- _filter_to_added ----------------------------------------------------

def test_filter_to_added_keeps_only_matching_line_numbers():
    findings = [
        {"path": "a.py", "line": 1},
        {"path": "a.py", "line": 99},   # not in the added set
        {"path": "b.py", "line": 5},    # unknown path
    ]
    added = {"a.py": {1, 2, 3}}
    out = _filter_to_added(findings, added)
    assert out == [{"path": "a.py", "line": 1}]


def test_filter_to_added_ignores_non_integer_lines():
    findings = [{"path": "a.py", "line": "not-a-number"}]
    added = {"a.py": {1}}
    assert _filter_to_added(findings, added) == []


# ---- run_ruff ------------------------------------------------------------

def _fake_run(monkeypatch, *, rc: int, stdout: str = "", stderr: str = ""):
    """Install a fake ``subprocess.run`` for the duration of the test."""
    class R:
        returncode = rc
        def __init__(self):
            self.stdout = stdout
            self.stderr = stderr
    def fake(*args, **kwargs):
        return R()
    monkeypatch.setattr(subprocess, "run", fake)


def test_run_ruff_returns_empty_when_binary_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(linters.shutil, "which", lambda _: None)
    assert run_ruff(["a.py"], {"a.py": {1}}, tmp_path) == []


def test_run_ruff_parses_json_findings(tmp_path, monkeypatch):
    monkeypatch.setattr(linters.shutil, "which", lambda _: "/usr/bin/ruff")
    payload = json.dumps([
        {
            "code": "F401",
            "message": "`os` imported but unused",
            "filename": str(tmp_path / "a.py"),
            "location": {"row": 1, "column": 1},
            "fix": {"message": "Remove unused import"},
        },
        {
            "code": "E501",
            "message": "Line too long",
            "filename": str(tmp_path / "a.py"),
            "location": {"row": 99, "column": 1},  # not in added set -> dropped
        },
    ])
    _fake_run(monkeypatch, rc=1, stdout=payload)
    findings = run_ruff(["a.py"], {"a.py": {1, 2, 3}}, tmp_path)
    assert len(findings) == 1
    f = findings[0]
    assert f["path"] == "a.py"
    assert f["line"] == 1
    assert f["source"] == "lint"
    assert f["tool"] == "ruff"
    assert f["category"] == "maintainability"
    assert "F401" in f["title"]
    assert f["suggested_fix"] == "Remove unused import"


def test_run_ruff_tool_error_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(linters.shutil, "which", lambda _: "/usr/bin/ruff")
    _fake_run(monkeypatch, rc=2, stderr="ruff: bad config")
    assert run_ruff(["a.py"], {"a.py": {1}}, tmp_path) == []


def test_run_ruff_no_files_short_circuits(tmp_path, monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(linters.shutil, "which",
                        lambda _: (called.__setitem__("n", called["n"] + 1) or "ruff"))
    assert run_ruff([], {}, tmp_path) == []
    assert called["n"] == 0  # short-circuit before which()


# ---- run_bandit ----------------------------------------------------------

def test_run_bandit_missing_binary_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(linters.shutil, "which", lambda _: None)
    assert run_bandit(["a.py"], {"a.py": {1}}, tmp_path) == []


def test_run_bandit_high_confidence_maps_to_security(tmp_path, monkeypatch):
    monkeypatch.setattr(linters.shutil, "which", lambda _: "/usr/bin/bandit")
    payload = json.dumps({
        "results": [{
            "filename": str(tmp_path / "a.py"),
            "line_number": 1,
            "issue_severity": "HIGH",
            "issue_confidence": "HIGH",
            "issue_text": "Use of assert detected",
            "test_id": "B101",
            "issue_cwe": {"id": 703},
        }]
    })
    _fake_run(monkeypatch, rc=1, stdout=payload)
    findings = run_bandit(["a.py"], {"a.py": {1}}, tmp_path)
    assert len(findings) == 1
    f = findings[0]
    assert f["category"] == "security"    # HIGH confidence -> real vuln
    assert f["severity"] == "high"
    assert f["source"] == "lint"
    assert f["tool"] == "bandit"
    assert "B101" in f["title"]
    assert "CWE-703" in f["title"]


def test_run_bandit_medium_confidence_maps_to_hotspot(tmp_path, monkeypatch):
    monkeypatch.setattr(linters.shutil, "which", lambda _: "/usr/bin/bandit")
    payload = json.dumps({
        "results": [{
            "filename": str(tmp_path / "a.py"),
            "line_number": 2,
            "issue_severity": "MEDIUM",
            "issue_confidence": "MEDIUM",
            "issue_text": "subprocess with shell=True",
            "test_id": "B602",
        }]
    })
    _fake_run(monkeypatch, rc=1, stdout=payload)
    findings = run_bandit(["a.py"], {"a.py": {1, 2}}, tmp_path)
    assert findings[0]["category"] == "security_hotspot"
    assert findings[0]["severity"] == "medium"


def test_run_bandit_filters_to_added_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(linters.shutil, "which", lambda _: "/usr/bin/bandit")
    payload = json.dumps({
        "results": [
            {"filename": str(tmp_path / "a.py"), "line_number": 5,
             "issue_severity": "HIGH", "issue_confidence": "HIGH",
             "issue_text": "x", "test_id": "B1"},
            {"filename": str(tmp_path / "a.py"), "line_number": 99,
             "issue_severity": "HIGH", "issue_confidence": "HIGH",
             "issue_text": "y", "test_id": "B2"},
        ]
    })
    _fake_run(monkeypatch, rc=1, stdout=payload)
    findings = run_bandit(["a.py"], {"a.py": {5}}, tmp_path)
    assert len(findings) == 1
    assert findings[0]["line"] == 5


# ---- run_semgrep ---------------------------------------------------------

def test_run_semgrep_missing_binary_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(linters.shutil, "which", lambda _: None)
    assert run_semgrep(["a.py"], {"a.py": {1}}, tmp_path) == []


def test_run_semgrep_maps_error_to_high_security(tmp_path, monkeypatch):
    monkeypatch.setattr(linters.shutil, "which", lambda _: "/usr/bin/semgrep")
    payload = json.dumps({
        "results": [{
            "check_id": "python.lang.security.audit.dangerous-subprocess",
            "path": str(tmp_path / "a.py"),
            "start": {"line": 1, "col": 5},
            "extra": {
                "severity": "ERROR",
                "message": "Detected subprocess with shell=True",
                "metadata": {"category": "security", "cwe": ["CWE-78"]},
            },
        }]
    })
    _fake_run(monkeypatch, rc=1, stdout=payload)
    findings = run_semgrep(["a.py"], {"a.py": {1}}, tmp_path)
    assert len(findings) == 1
    f = findings[0]
    assert f["category"] == "security"
    assert f["severity"] == "high"
    assert f["tool"] == "semgrep"
    assert "CWE-78" in f["title"]


def test_run_semgrep_warning_security_becomes_hotspot(tmp_path, monkeypatch):
    """WARNING-severity security rules are hotspots, not vulnerabilities."""
    monkeypatch.setattr(linters.shutil, "which", lambda _: "/usr/bin/semgrep")
    payload = json.dumps({
        "results": [{
            "check_id": "js.audit.something",
            "path": str(tmp_path / "a.py"),
            "start": {"line": 1},
            "extra": {
                "severity": "WARNING",
                "message": "audit-worthy pattern",
                "metadata": {"category": "security"},
            },
        }]
    })
    _fake_run(monkeypatch, rc=1, stdout=payload)
    findings = run_semgrep(["a.py"], {"a.py": {1}}, tmp_path)
    assert findings[0]["category"] == "security_hotspot"


def test_run_semgrep_filters_to_added_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(linters.shutil, "which", lambda _: "/usr/bin/semgrep")
    payload = json.dumps({
        "results": [
            {"check_id": "r1", "path": str(tmp_path / "a.py"),
             "start": {"line": 1}, "extra": {"severity": "ERROR", "message": "x"}},
            {"check_id": "r2", "path": str(tmp_path / "a.py"),
             "start": {"line": 99}, "extra": {"severity": "ERROR", "message": "y"}},
        ]
    })
    _fake_run(monkeypatch, rc=1, stdout=payload)
    findings = run_semgrep(["a.py"], {"a.py": {1}}, tmp_path)
    assert len(findings) == 1
    assert findings[0]["line"] == 1


# ---- run_eslint ----------------------------------------------------------

def test_run_eslint_requires_config(tmp_path, monkeypatch):
    monkeypatch.setattr(linters, "_npx_bin", lambda: "/usr/bin/npx")
    # no .eslintrc*/eslint.config.* file -> no run
    assert run_eslint(["a.ts"], {"a.ts": {1}}, tmp_path) == []


def test_run_eslint_parses_findings(tmp_path, monkeypatch):
    (tmp_path / ".eslintrc.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(linters, "_npx_bin", lambda: "/usr/bin/npx")
    payload = json.dumps([
        {
            "filePath": str(tmp_path / "app.ts"),
            "messages": [
                {"ruleId": "no-explicit-any", "severity": 2,
                 "message": "Unexpected any", "line": 1},
                {"ruleId": "no-console", "severity": 1,
                 "message": "Unexpected console", "line": 99},  # dropped
            ],
        }
    ])
    _fake_run(monkeypatch, rc=1, stdout=payload)
    findings = run_eslint(["app.ts"], {"app.ts": {1, 2}}, tmp_path)
    assert len(findings) == 1
    f = findings[0]
    assert f["line"] == 1
    assert f["severity"] == "medium"   # severity==2 -> medium
    assert f["source"] == "lint"
    assert f["tool"] == "eslint"
    assert "no-explicit-any" in f["title"]


# ---- mypy / tsc parsers --------------------------------------------------

def test_parse_mypy_line_error_with_column():
    f = _parse_mypy_line(
        "src/foo.py:42:5: error: Incompatible types  [assignment]",
        Path("/tmp"),
    )
    assert f is not None
    assert f["line"] == 42
    assert f["severity"] == "medium"
    assert f["source"] == "type"
    assert "Incompatible types" in f["explanation"]


def test_parse_mypy_line_warning_no_column():
    f = _parse_mypy_line(
        "src/foo.py:7: warning: Something odd", Path("/tmp"),
    )
    assert f is not None
    assert f["line"] == 7
    assert f["severity"] == "low"


def test_parse_mypy_line_ignores_non_matches():
    assert _parse_mypy_line("random garbage", Path("/tmp")) is None
    # notes are informational, not findings
    assert _parse_mypy_line("a.py:1: note: hint", Path("/tmp")) is None


def test_parse_tsc_line_error():
    f = _parse_tsc_line(
        "src/app.ts(12,5): error TS2322: Type 'string' is not assignable to type 'number'.",
        Path("/tmp"),
    )
    assert f is not None
    assert f["line"] == 12
    assert f["severity"] == "medium"
    assert f["tool"] == "tsc"


def test_parse_tsc_line_ignores_non_matches():
    assert _parse_tsc_line("no match here", Path("/tmp")) is None


# ---- run_all -------------------------------------------------------------

def test_run_all_returns_empty_when_no_added_lines(tmp_path):
    # Empty diff -> nothing to check
    assert run_all("", tmp_path) == []


def test_run_all_dispatches_by_extension(tmp_path, monkeypatch):
    called: list[str] = []

    def stub(name):
        def fn(files, added, root):
            called.append(name)
            return []
        return fn
    monkeypatch.setattr(linters, "run_ruff", stub("ruff"))
    monkeypatch.setattr(linters, "run_bandit", stub("bandit"))
    monkeypatch.setattr(linters, "run_eslint", stub("eslint"))
    monkeypatch.setattr(linters, "run_semgrep", stub("semgrep"))
    monkeypatch.setattr(linters, "run_mypy", stub("mypy"))
    monkeypatch.setattr(linters, "run_tsc", stub("tsc"))

    run_all(SIMPLE_DIFF, tmp_path, enable_lint=True, enable_type_check=False)
    assert "ruff" in called
    assert "bandit" in called
    assert "mypy" not in called
    assert "semgrep" not in called   # opt-in

    called.clear()
    run_all(TS_DIFF, tmp_path,
            enable_lint=True, enable_type_check=True, enable_semgrep=True)
    assert "eslint" in called and "tsc" in called
    assert "semgrep" in called


def test_run_all_missing_root_returns_empty():
    assert run_all(SIMPLE_DIFF, "/does/not/exist/anywhere") == []
