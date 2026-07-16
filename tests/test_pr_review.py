"""Tests for reviewer.pr_review — PR-level architectural review."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reviewer.pr_review import (
    build_summary,
    is_config_path,
    is_doc_path,
    is_test_path,
    missing_tests_finding,
    review_pr_level,
)
from reviewer.providers import ChatResponse, Usage


PROD_DIFF = """diff --git a/svc/handlers.py b/svc/handlers.py
--- a/svc/handlers.py
+++ b/svc/handlers.py
@@ -0,0 +1,10 @@
+def process_order(order_id: str) -> dict:
+    order = fetch_order(order_id)
+    if order.status == "pending":
+        charge_card(order.card_token, order.total)
+        order.status = "paid"
+        save(order)
+        send_confirmation(order.email)
+        return {"status": "ok"}
+    return {"status": "already-processed"}
+
"""


DIFF_WITH_TESTS = PROD_DIFF + """diff --git a/tests/test_handlers.py b/tests/test_handlers.py
--- a/tests/test_handlers.py
+++ b/tests/test_handlers.py
@@ -0,0 +1,3 @@
+def test_process_order():
+    result = process_order("42")
+    assert result["status"] == "ok"
"""


DOCS_ONLY_DIFF = """diff --git a/README.md b/README.md
--- a/README.md
+++ b/README.md
@@ -0,0 +1,2 @@
+# New docs
+content
"""


TINY_TYPO_DIFF = """diff --git a/svc/util.py b/svc/util.py
--- a/svc/util.py
+++ b/svc/util.py
@@ -0,0 +1,1 @@
+# fix typo
"""


# ---- path classifiers ----------------------------------------------------

def test_is_test_path_recognizes_common_layouts():
    assert is_test_path("tests/test_foo.py")
    assert is_test_path("src/__tests__/App.test.tsx")
    assert is_test_path("web/foo.spec.ts")
    assert is_test_path("test_utils.py")


def test_is_test_path_rejects_prod_code():
    assert not is_test_path("src/app.py")
    assert not is_test_path("lib/handler.ts")


def test_is_doc_path():
    assert is_doc_path("README.md")
    assert is_doc_path("docs/guide.rst")
    assert is_doc_path("CHANGELOG")
    assert not is_doc_path("src/main.py")


def test_is_config_path():
    assert is_config_path("package.json")
    assert is_config_path("pyproject.toml")
    assert is_config_path("requirements.txt")
    assert not is_config_path("src/app.py")


# ---- build_summary -------------------------------------------------------

def test_build_summary_counts_added_and_removed():
    s = build_summary(PROD_DIFF)
    assert s.files == ["svc/handlers.py"]
    assert s.total_added == 10
    assert s.total_removed == 0
    assert s.file_stats[0]["path"] == "svc/handlers.py"
    assert s.file_stats[0]["added"] == 10


def test_build_summary_truncates_excerpt():
    long = PROD_DIFF + "x" * 20_000
    s = build_summary(long, max_diff_chars=200)
    assert len(s.excerpt) <= 200 + len("\n... (truncated)")
    assert "truncated" in s.excerpt


def test_build_summary_orders_files_by_change_size():
    small = "diff --git a/small.py b/small.py\n--- a/small.py\n+++ b/small.py\n@@ -0,0 +1,1 @@\n+x = 1\n"
    diff = small + PROD_DIFF
    s = build_summary(diff)
    # Big file (10 adds) should come before the 1-line file.
    assert s.file_stats[0]["path"] == "svc/handlers.py"
    assert s.file_stats[1]["path"] == "small.py"


# ---- missing_tests_finding ----------------------------------------------

def test_missing_tests_fires_when_code_changed_no_tests():
    s = build_summary(PROD_DIFF)
    f = missing_tests_finding(s)
    assert f is not None
    assert f["source"] == "pr_review"
    assert f["category"] == "maintainability"
    assert "svc/handlers.py" in f["explanation"]


def test_missing_tests_silent_when_tests_included():
    s = build_summary(DIFF_WITH_TESTS)
    assert missing_tests_finding(s) is None


def test_missing_tests_silent_on_docs_only_pr():
    s = build_summary(DOCS_ONLY_DIFF)
    assert missing_tests_finding(s) is None


def test_missing_tests_silent_on_tiny_change():
    """A 1-line typo fix shouldn't demand a test."""
    s = build_summary(TINY_TYPO_DIFF)
    assert missing_tests_finding(s) is None


# ---- review_pr_level -----------------------------------------------------

class FakeProvider:
    name = "fake"

    def __init__(self, content: str, usage: Usage | None = None):
        self._content = content
        self._usage = usage or Usage(prompt_tokens=100, completion_tokens=20)

    def complete(self, **kwargs) -> ChatResponse:
        return ChatResponse(content=self._content, usage=self._usage)


def test_review_pr_level_returns_missing_tests_plus_llm(monkeypatch):
    llm_payload = json.dumps({
        "findings": [{
            "path": "svc/handlers.py",
            "severity": "high",
            "category": "correctness",
            "title": "No error handling around charge_card",
            "explanation": "If charge_card raises, order gets stuck in pending.",
            "confidence": 0.85,
        }]
    })
    provider = FakeProvider(llm_payload)
    findings = review_pr_level(
        PROD_DIFF, provider=provider, model="test-model",
        pr_title="Add process_order",
    )
    # Missing-tests heuristic + LLM finding
    assert len(findings) == 2
    titles = [f["title"] for f in findings]
    assert any("Missing tests" in t for t in titles)
    assert any("charge_card" in t for t in titles)
    # LLM finding should be tagged pr_review and line=0
    llm_f = next(f for f in findings if "charge_card" in f["title"])
    assert llm_f["source"] == "pr_review"
    assert llm_f["line"] == 0


def test_review_pr_level_disables_missing_tests():
    provider = FakeProvider('{"findings": []}')
    findings = review_pr_level(
        PROD_DIFF, provider=provider, model="test-model",
        enable_missing_tests=False,
    )
    assert findings == []


def test_review_pr_level_handles_llm_failure_gracefully(monkeypatch):
    class BrokenProvider:
        name = "broken"
        def complete(self, **kwargs):
            raise RuntimeError("network died")
    findings = review_pr_level(
        PROD_DIFF, provider=BrokenProvider(), model="m",
    )
    # LLM crashed but the deterministic missing-tests check still fires.
    assert len(findings) == 1
    assert "Missing tests" in findings[0]["title"]


def test_review_pr_level_logs_usage(monkeypatch):
    provider = FakeProvider(
        '{"findings": []}',
        usage=Usage(prompt_tokens=500, completion_tokens=50),
    )
    usage_log: list[dict] = []
    review_pr_level(
        DIFF_WITH_TESTS,  # no missing-tests -> only LLM finding
        provider=provider, model="test-model",
        usage_log=usage_log,
    )
    assert len(usage_log) == 1
    assert usage_log[0]["path"] == "(pr-level)"
    assert usage_log[0]["prompt_tokens"] == 500
