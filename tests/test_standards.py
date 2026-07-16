"""Tests for reviewer.standards (structured YAML rules DSL)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reviewer.standards import apply_rules, load_rules


PY_DIFF = """diff --git a/svc/app.py b/svc/app.py
--- a/svc/app.py
+++ b/svc/app.py
@@ -0,0 +1,4 @@
+import os
+print("hello")
+x = eval("1+1")
+def f(): pass
"""

TS_DIFF = """diff --git a/web/App.tsx b/web/App.tsx
--- a/web/App.tsx
+++ b/web/App.tsx
@@ -0,0 +1,3 @@
+const x: any = 1;
+console.log(x);
+// nothing suspicious
"""


# ---- load_rules ----------------------------------------------------------

def test_load_rules_returns_empty_for_none_or_empty():
    assert load_rules(None) == []
    assert load_rules([]) == []


def test_load_rules_compiles_valid_rule():
    rules = load_rules([{
        "id": "no-print",
        "pattern": r"^\s*print\(",
        "severity": "medium",
        "category": "maintainability",
    }])
    assert len(rules) == 1
    assert rules[0]["id"] == "no-print"
    assert rules[0]["severity"] == "medium"


def test_load_rules_drops_rule_missing_pattern(capsys):
    rules = load_rules([{"id": "bad", "severity": "low"}])
    assert rules == []
    assert "missing" in capsys.readouterr().err.lower()


def test_load_rules_drops_rule_with_bad_regex(capsys):
    rules = load_rules([{"id": "bad", "pattern": "["}])
    assert rules == []
    assert "invalid regex" in capsys.readouterr().err.lower()


def test_load_rules_normalizes_string_language(capsys):
    rules = load_rules([{
        "id": "r",
        "pattern": "x",
        "languages": "python",  # string instead of list
        "paths": "src/**/*.py",
    }])
    assert rules[0]["languages"] == ["python"]
    assert rules[0]["paths"] == ["src/**/*.py"]


def test_load_rules_defaults_unknown_severity_to_low(capsys):
    rules = load_rules([{
        "id": "r", "pattern": "x", "severity": "bananas",
    }])
    assert rules[0]["severity"] == "low"
    assert "unknown severity" in capsys.readouterr().err.lower()


# ---- apply_rules ---------------------------------------------------------

def test_apply_rules_matches_added_lines():
    rules = load_rules([{
        "id": "no-print",
        "pattern": r"^\s*print\(",
        "languages": ["python"],
        "severity": "low",
        "category": "maintainability",
        "message": "Use logging instead of print()",
    }])
    findings = apply_rules(PY_DIFF, rules)
    assert len(findings) == 1
    f = findings[0]
    assert f["path"] == "svc/app.py"
    assert f["line"] == 2   # the `+print(...)` line
    assert f["source"] == "standards"
    assert f["confidence"] == 1.0
    assert "no-print" in f["title"]


def test_apply_rules_multiple_rules_same_file():
    rules = load_rules([
        {"id": "no-print", "pattern": r"^\s*print\("},
        {"id": "no-eval",  "pattern": r"\beval\("},
    ])
    findings = apply_rules(PY_DIFF, rules)
    assert len(findings) == 2
    ids = sorted(f["title"].split(":")[0] for f in findings)
    assert ids == ["no-eval", "no-print"]


def test_apply_rules_language_filter():
    """A python-only rule must not fire on a TypeScript file."""
    rules = load_rules([{
        "id": "no-print",
        "pattern": r"^\s*print\(",
        "languages": ["python"],
    }])
    assert apply_rules(TS_DIFF, rules) == []


def test_apply_rules_path_filter():
    """A path-scoped rule must not fire outside its glob."""
    rules = load_rules([{
        "id": "web-any",
        "pattern": r"\bany\b",
        "paths": ["web/**/*.tsx"],
    }])
    assert len(apply_rules(TS_DIFF, rules)) == 1
    # Same pattern but a path glob that excludes App.tsx:
    rules2 = load_rules([{
        "id": "srv-any",
        "pattern": r"\bany\b",
        "paths": ["server/**/*.ts"],
    }])
    assert apply_rules(TS_DIFF, rules2) == []


def test_apply_rules_ignores_context_lines():
    diff = """diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1,2 +1,3 @@
 print("existing")
+print("new")
 unchanged
"""
    rules = load_rules([{"id": "no-print", "pattern": r"^\s*print\("}])
    findings = apply_rules(diff, rules)
    # Only the `+print("new")` line, not the pre-existing one.
    assert len(findings) == 1
    assert findings[0]["line"] == 2


def test_apply_rules_no_rules_returns_empty():
    assert apply_rules(PY_DIFF, []) == []
