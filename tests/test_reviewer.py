import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest
from azure.core.exceptions import HttpResponseError, ServiceRequestError

from reviewer import review_patch
from reviewer import reviewer as reviewer_mod


SAMPLE_DIFF = """diff --git a/app/auth.py b/app/auth.py
index 1111111..2222222 100644
--- a/app/auth.py
+++ b/app/auth.py
@@ -0,0 +1,5 @@
+import sqlite3
+DB_PATH = "users.db"
+ADMIN_TOKEN = "sk-live-9f8a7b6c5d4e3f2a1b0c"
+def authenticate(u, p):
+    return True
"""


LOCK_DIFF = """diff --git a/yarn.lock b/yarn.lock
index 1111111..2222222 100644
--- a/yarn.lock
+++ b/yarn.lock
@@ -0,0 +1,2 @@
+foo
+bar
"""


SENSITIVE_DIFF = """diff --git a/.env b/.env
index 1111111..2222222 100644
--- a/.env
+++ b/.env
@@ -0,0 +1,2 @@
+DB_PASSWORD=hunter2
+API_KEY=sk-live-deadbeef
"""


SENSITIVE_DIR_DIFF = """diff --git a/secrets/api.json b/secrets/api.json
index 1111111..2222222 100644
--- a/secrets/api.json
+++ b/secrets/api.json
@@ -0,0 +1,2 @@
+{
+  "key": "abc"
+}
"""


class FakeUsage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content, usage=None):
        self.choices = [FakeChoice(content)]
        self.usage = usage


class FakeClient:
    """Records every .complete() call. Optionally raises on the first call
    (used to exercise the response_format-fallback path in review_patch)."""

    def __init__(self, content, usage=None, raise_on_first=None):
        self._content = content
        self._usage = usage
        self._raise_on_first = raise_on_first
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        if self._raise_on_first and len(self.calls) == 1:
            raise self._raise_on_first("response_format unsupported")
        return FakeResponse(self._content, self._usage)


def _payload(findings):
    return json.dumps({"findings": findings})


def test_returns_valid_finding():
    client = FakeClient(_payload([{
        "line": 3, "severity": "critical", "category": "security",
        "title": "Hardcoded secret", "explanation": "ADMIN_TOKEN is hardcoded",
        "suggested_fix": "load from env", "confidence": 0.9,
    }]))
    out = review_patch(SAMPLE_DIFF, model="x", client=client)
    assert len(out) == 1
    assert out[0]["path"] == "app/auth.py"
    assert out[0]["line"] == 3
    assert out[0]["severity"] == "critical"


def test_drops_finding_outside_diff():
    client = FakeClient(_payload([{
        "line": 999, "severity": "high", "category": "security",
        "title": "Imaginary", "explanation": "...", "confidence": 0.9,
    }]))
    out = review_patch(SAMPLE_DIFF, model="x", client=client)
    assert out == []


def test_filters_below_confidence_threshold():
    client = FakeClient(_payload([{
        "line": 3, "severity": "high", "category": "security",
        "title": "Low conf", "explanation": "...", "confidence": 0.4,
    }]))
    out = review_patch(SAMPLE_DIFF, model="x", client=client, min_confidence=0.6)
    assert out == []


def test_dedupes_same_path_line_title():
    client = FakeClient(_payload([
        {"line": 3, "severity": "high", "category": "security",
         "title": "Dup", "explanation": "first", "confidence": 0.9},
        {"line": 3, "severity": "high", "category": "security",
         "title": "dup", "explanation": "different text", "confidence": 0.9},
    ]))
    out = review_patch(SAMPLE_DIFF, model="x", client=client)
    assert len(out) == 1


def test_skips_lock_files_without_calling_model():
    client = FakeClient(_payload([]))
    out = review_patch(LOCK_DIFF, model="x", client=client)
    assert out == []
    assert client.calls == []


def test_skips_sensitive_dotenv_without_calling_model():
    client = FakeClient(_payload([]))
    out = review_patch(SENSITIVE_DIFF, model="x", client=client)
    assert out == []
    assert client.calls == []


def test_skips_sensitive_secrets_directory_without_calling_model():
    client = FakeClient(_payload([]))
    out = review_patch(SENSITIVE_DIR_DIFF, model="x", client=client)
    assert out == []
    assert client.calls == []


def test_user_block_pattern_via_env_skips_path(monkeypatch):
    monkeypatch.setenv("REVIEWER_BLOCK_PATTERNS", "*custom_block*")
    custom_diff = """diff --git a/src/custom_block_me.py b/src/custom_block_me.py
index 1111111..2222222 100644
--- a/src/custom_block_me.py
+++ b/src/custom_block_me.py
@@ -0,0 +1,1 @@
+x = 1
"""
    client = FakeClient(_payload([]))
    review_patch(custom_diff, model="x", client=client)
    assert client.calls == []


def test_records_usage_log_with_cost():
    client = FakeClient(_payload([]), usage=FakeUsage(100, 50))
    log = []
    review_patch(SAMPLE_DIFF, model="openai/gpt-4o-mini",
                 client=client, usage_log=log)
    assert len(log) == 1
    entry = log[0]
    assert entry["prompt_tokens"] == 100
    assert entry["completion_tokens"] == 50
    assert entry["total_tokens"] == 150
    assert entry["cost_usd"] is not None
    assert entry["cost_usd"] > 0


def test_records_usage_log_with_latency():
    client = FakeClient(_payload([]), usage=FakeUsage(10, 5))
    log = []
    review_patch(SAMPLE_DIFF, model="openai/gpt-4o-mini",
                 client=client, usage_log=log)
    assert "latency_seconds" in log[0]
    assert log[0]["latency_seconds"] >= 0


def test_records_usage_log_with_unknown_model():
    client = FakeClient(_payload([]), usage=FakeUsage(10, 5))
    log = []
    review_patch(SAMPLE_DIFF, model="unknown/model-xyz",
                 client=client, usage_log=log)
    assert log[0]["cost_usd"] is None


def test_retries_without_response_format_on_typeerror():
    client = FakeClient(_payload([]), raise_on_first=TypeError)
    out = review_patch(SAMPLE_DIFF, model="x", client=client)
    assert out == []
    assert len(client.calls) == 2
    assert "response_format" in client.calls[0]
    assert "response_format" not in client.calls[1]


def test_retries_without_response_format_on_valueerror():
    client = FakeClient(_payload([]), raise_on_first=ValueError)
    out = review_patch(SAMPLE_DIFF, model="x", client=client)
    assert out == []
    assert len(client.calls) == 2


TWO_HUNK_DIFF = """diff --git a/x.py b/x.py
index 1111111..2222222 100644
--- a/x.py
+++ b/x.py
@@ -1,3 +1,3 @@
 a
-b
+B
 c
@@ -10,3 +10,3 @@
 x
-y
+Y
 z
"""


TWO_FILE_DIFF = """diff --git a/a.py b/a.py
index 1111111..2222222 100644
--- a/a.py
+++ b/a.py
@@ -0,0 +1,2 @@
+import os
+TOKEN = "abc"
diff --git a/b.py b/b.py
index 3333333..4444444 100644
--- a/b.py
+++ b/b.py
@@ -0,0 +1,2 @@
+import sys
+SECRET = "xyz"
"""


def test_small_file_uses_single_chunk():
    client = FakeClient(_payload([]))
    review_patch(SAMPLE_DIFF, model="x", client=client)
    assert len(client.calls) == 1


def test_large_file_splits_per_hunk():
    client = FakeClient(_payload([]))
    review_patch(TWO_HUNK_DIFF, model="x", client=client, max_diff_chars=20)
    assert len(client.calls) == 2


def test_large_file_kept_as_one_chunk_when_under_threshold():
    client = FakeClient(_payload([]))
    review_patch(TWO_HUNK_DIFF, model="x", client=client, max_diff_chars=10000)
    assert len(client.calls) == 1


class _CountingClient(FakeClient):
    """Tracks max simultaneous in-flight calls — used to verify parallelism."""

    def __init__(self, content, hold_seconds=0.05):
        super().__init__(content)
        self._hold = hold_seconds
        self._lock = __import__("threading").Lock()
        self.in_flight = 0
        self.peak_in_flight = 0

    def complete(self, **kwargs):
        import time
        with self._lock:
            self.calls.append(kwargs)
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            time.sleep(self._hold)
            return FakeResponse(self._content, self._usage)
        finally:
            with self._lock:
                self.in_flight -= 1


def test_multiple_files_run_in_parallel():
    client = _CountingClient(_payload([]))
    review_patch(TWO_FILE_DIFF, model="x", client=client, concurrency=2)
    assert len(client.calls) == 2
    assert client.peak_in_flight == 2


def test_concurrency_one_runs_serially():
    client = _CountingClient(_payload([]))
    review_patch(TWO_FILE_DIFF, model="x", client=client, concurrency=1)
    assert len(client.calls) == 2
    assert client.peak_in_flight == 1


def test_usage_log_records_one_entry_per_chunk():
    client = FakeClient(_payload([]), usage=FakeUsage(10, 5))
    log = []
    review_patch(TWO_FILE_DIFF, model="openai/gpt-4o-mini",
                 client=client, usage_log=log, concurrency=2)
    assert len(log) == 2
    assert {e["path"] for e in log} == {"a.py", "b.py"}


def test_findings_collected_from_all_files_under_concurrency():
    client = FakeClient(_payload([{
        "line": 2, "severity": "high", "category": "security",
        "title": "secret", "explanation": "...", "confidence": 0.9,
    }]))
    out = review_patch(TWO_FILE_DIFF, model="x", client=client, concurrency=2)
    assert {f["path"] for f in out} == {"a.py", "b.py"}
    assert len(out) == 2


def _http_error(status_code, retry_after=None):
    err = HttpResponseError(message=f"status {status_code}")
    err.status_code = status_code
    if retry_after is not None:
        class _Resp:
            headers = {"Retry-After": str(retry_after)}
        err.response = _Resp()
    return err


class ScriptedClient:
    """Raises a scripted sequence of exceptions, then returns success.

    Each entry in ``script`` is either an exception instance to raise or
    ``None`` to mean 'return the success response'.
    """

    def __init__(self, script, content, usage=None):
        self.script = list(script)
        self._content = content
        self._usage = usage
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        if not self.script:
            return FakeResponse(self._content, self._usage)
        action = self.script.pop(0)
        if action is None:
            return FakeResponse(self._content, self._usage)
        raise action


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch):
    """Replace time.sleep so retry tests don't actually wait."""
    monkeypatch.setattr(reviewer_mod, "_sleep", lambda _s: None)


def test_retries_on_429_then_succeeds():
    client = ScriptedClient(
        [_http_error(429), None],
        _payload([{"line": 3, "severity": "high", "category": "security",
                   "title": "x", "explanation": "...", "confidence": 0.9}]),
    )
    out = review_patch(SAMPLE_DIFF, model="x", client=client,
                       max_retries=3, retry_base_seconds=0)
    assert len(out) == 1
    assert len(client.calls) == 2


def test_retries_on_503_then_succeeds():
    client = ScriptedClient(
        [_http_error(503), _http_error(503), None],
        _payload([]),
    )
    review_patch(SAMPLE_DIFF, model="x", client=client,
                 max_retries=3, retry_base_seconds=0)
    assert len(client.calls) == 3


def test_retries_on_service_request_error():
    client = ScriptedClient(
        [ServiceRequestError(message="connection reset"), None],
        _payload([]),
    )
    review_patch(SAMPLE_DIFF, model="x", client=client,
                 max_retries=3, retry_base_seconds=0)
    assert len(client.calls) == 2


def test_persistent_429_skips_chunk_returns_empty():
    client = ScriptedClient(
        [_http_error(429), _http_error(429), _http_error(429), _http_error(429)],
        _payload([]),
    )
    out = review_patch(SAMPLE_DIFF, model="x", client=client,
                       max_retries=2, retry_base_seconds=0)
    assert out == []
    # 1 initial + 2 retries = 3 calls; the 4th-status entry is never used.
    assert len(client.calls) == 3


def test_non_retryable_4xx_skips_chunk_without_retry():
    client = ScriptedClient(
        [_http_error(401)],
        _payload([]),
    )
    out = review_patch(SAMPLE_DIFF, model="x", client=client,
                       max_retries=3, retry_base_seconds=0)
    assert out == []
    assert len(client.calls) == 1


def test_retry_after_header_is_honored(monkeypatch):
    captured = []
    monkeypatch.setattr(reviewer_mod, "_sleep", lambda s: captured.append(s))
    client = ScriptedClient(
        [_http_error(429, retry_after=7), None],
        _payload([]),
    )
    review_patch(SAMPLE_DIFF, model="x", client=client,
                 max_retries=2, retry_base_seconds=99.0)
    assert captured == [7.0]  # Retry-After wins over exponential backoff


def test_one_chunk_failing_does_not_kill_other_chunks():
    """File 'a.py' rate-limits forever; 'b.py' succeeds. We should still
    get b.py's findings."""
    finding = {"line": 2, "severity": "high", "category": "security",
               "title": "secret", "explanation": "...", "confidence": 0.9}

    class _RouteClient:
        """First call (whichever it is) sees a.py fail; b.py always wins."""
        def __init__(self):
            self.calls = []

        def complete(self, **kwargs):
            self.calls.append(kwargs)
            user_msg = kwargs["messages"][1].content
            if "File: a.py" in user_msg:
                raise _http_error(429)
            return FakeResponse(json.dumps({"findings": [finding]}))

    client = _RouteClient()
    out = review_patch(TWO_FILE_DIFF, model="x", client=client,
                       max_retries=1, retry_base_seconds=0, concurrency=1)
    assert {f["path"] for f in out} == {"b.py"}


def test_response_format_fallback_still_works_with_retries():
    """The TypeError fallback should not consume the retry budget."""
    client = ScriptedClient(
        [TypeError("response_format unsupported"), _http_error(429), None],
        _payload([]),
    )
    review_patch(SAMPLE_DIFF, model="x", client=client,
                 max_retries=1, retry_base_seconds=0)
    # 1: with response_format -> TypeError -> drop param
    # 2: without response_format -> 429 -> sleep, retry attempt #1
    # 3: without response_format -> success
    assert len(client.calls) == 3
    # First call had response_format; subsequent ones did not.
    assert "response_format" in client.calls[0]
    assert "response_format" not in client.calls[1]
    assert "response_format" not in client.calls[2]


def test_min_severity_filter_drops_low_severity_findings():
    client = FakeClient(_payload([
        {"line": 1, "severity": "low", "category": "maintainability",
         "title": "nit", "explanation": "...", "confidence": 0.9},
        {"line": 2, "severity": "high", "category": "security",
         "title": "rce", "explanation": "...", "confidence": 0.9},
    ]))
    out = review_patch(SAMPLE_DIFF, model="x", client=client,
                       min_severity="high")
    assert [f["severity"] for f in out] == ["high"]


PY_AND_TS_DIFF = """diff --git a/a.py b/a.py
index 1111111..2222222 100644
--- a/a.py
+++ b/a.py
@@ -0,0 +1,1 @@
+x = 1
diff --git a/b.ts b/b.ts
index 3333333..4444444 100644
--- a/b.ts
+++ b/b.ts
@@ -0,0 +1,1 @@
+const y = 2;
"""


def test_models_by_language_routes_per_file():
    """Python file goes to one model, TypeScript file to another."""
    client = FakeClient(_payload([]))
    review_patch(
        PY_AND_TS_DIFF, model="default-model", client=client,
        models_by_language={"python": "py-model", "typescript": "ts-model"},
        concurrency=1,
    )
    models_seen = [c["model"] for c in client.calls]
    assert sorted(models_seen) == ["py-model", "ts-model"]


def test_models_by_language_falls_back_to_default():
    """A language not in the override map uses the run-wide default."""
    client = FakeClient(_payload([]))
    review_patch(
        PY_AND_TS_DIFF, model="default-model", client=client,
        models_by_language={"python": "py-model"},  # ts unset
        concurrency=1,
    )
    models_seen = sorted(c["model"] for c in client.calls)
    assert models_seen == ["default-model", "py-model"]


def test_min_severity_default_keeps_everything():
    client = FakeClient(_payload([
        {"line": 1, "severity": "low", "category": "maintainability",
         "title": "nit", "explanation": "...", "confidence": 0.9},
    ]))
    out = review_patch(SAMPLE_DIFF, model="x", client=client)
    assert len(out) == 1


THREE_FILE_DIFF = """diff --git a/a.py b/a.py
index 1111111..2222222 100644
--- a/a.py
+++ b/a.py
@@ -0,0 +1,1 @@
+x = 1
diff --git a/b.py b/b.py
index 3333333..4444444 100644
--- a/b.py
+++ b/b.py
@@ -0,0 +1,1 @@
+y = 2
diff --git a/c.py b/c.py
index 5555555..6666666 100644
--- a/c.py
+++ b/c.py
@@ -0,0 +1,1 @@
+z = 3
"""


def test_max_files_per_pr_truncates_task_list_pre_flight():
    """Excess files should be skipped before any model call —
    expected behavior for a 200-file PR hitting a max_files_per_pr=20 cap."""
    client = FakeClient(_payload([]))
    review_patch(THREE_FILE_DIFF, model="x", client=client,
                 max_files_per_pr=1, concurrency=1)
    assert len(client.calls) == 1


def test_max_files_per_pr_zero_means_unlimited():
    client = FakeClient(_payload([]))
    review_patch(THREE_FILE_DIFF, model="x", client=client,
                 max_files_per_pr=0, concurrency=1)
    assert len(client.calls) == 3


def test_max_tokens_per_pr_stops_subsequent_chunks_after_cap_hit():
    """First chunk reports 1000 tokens, blowing past the 500-token cap.
    Remaining chunks must skip without spending."""
    client = FakeClient(_payload([]), usage=FakeUsage(800, 200))
    review_patch(THREE_FILE_DIFF, model="openai/gpt-4o-mini",
                 client=client, max_tokens_per_pr=500, concurrency=1)
    # Exactly one model call — the rest were skipped due to cap.
    assert len(client.calls) == 1


def test_max_tokens_per_pr_zero_means_unlimited():
    client = FakeClient(_payload([]), usage=FakeUsage(800, 200))
    review_patch(THREE_FILE_DIFF, model="openai/gpt-4o-mini",
                 client=client, max_tokens_per_pr=0, concurrency=1)
    assert len(client.calls) == 3


def test_max_tokens_per_pr_allows_chunks_under_cap():
    """If running total stays under the cap, all chunks proceed."""
    client = FakeClient(_payload([]), usage=FakeUsage(50, 25))
    review_patch(THREE_FILE_DIFF, model="openai/gpt-4o-mini",
                 client=client, max_tokens_per_pr=10_000, concurrency=1)
    assert len(client.calls) == 3


def test_max_tokens_cap_with_concurrency_caps_at_least_some_chunks():
    """Under concurrency, a few chunks may have already started when the
    cap trips — but we should still skip at least some. Verifies the
    event-flag plumbing works through the threadpool."""
    client = FakeClient(_payload([]), usage=FakeUsage(800, 200))
    # cap=100 → very tight, every chunk pushes past it.
    review_patch(THREE_FILE_DIFF, model="openai/gpt-4o-mini",
                 client=client, max_tokens_per_pr=100, concurrency=2)
    # At least one of the three chunks should have skipped.
    assert len(client.calls) < 3


def test_findings_sorted_by_severity():
    client = FakeClient(_payload([
        {"line": 1, "severity": "low", "category": "maintainability",
         "title": "nit", "explanation": "...", "confidence": 0.9},
        {"line": 2, "severity": "critical", "category": "security",
         "title": "rce", "explanation": "...", "confidence": 0.9},
        {"line": 3, "severity": "medium", "category": "correctness",
         "title": "edge", "explanation": "...", "confidence": 0.9},
    ]))
    out = review_patch(SAMPLE_DIFF, model="x", client=client)
    assert [f["severity"] for f in out] == ["critical", "medium", "low"]
