import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from unidiff import PatchSet

from reviewer.utils import (
    language_for, should_skip, valid_new_lines,
    dedupe, filter_by_confidence, filter_by_severity,
    sort_findings, format_comment, AI_COMMENT_TAG,
    is_sensitive_path, find_skip_label,
)


SAMPLE_DIFF = """diff --git a/a.py b/a.py
index e69de29..4a1b2c3 100644
--- a/a.py
+++ b/a.py
@@ -0,0 +1,3 @@
+def f():
+    return 1 / 0
+
"""


def test_language_for():
    assert language_for("a.py") == "python"
    assert language_for("a.tsx") == "typescript"
    assert language_for("a.unknown") == "text"


def test_should_skip():
    assert should_skip("yarn.lock")
    assert should_skip("bundle.min.js")
    assert not should_skip("src/a.py")


def test_valid_new_lines():
    patch = PatchSet(SAMPLE_DIFF)
    lines = valid_new_lines(patch[0])
    assert lines == {1, 2, 3}


def test_dedupe():
    a = {"path": "x.py", "line": 1, "title": "Bug"}
    b = {"path": "x.py", "line": 1, "title": "bug"}
    c = {"path": "x.py", "line": 2, "title": "Bug"}
    assert len(dedupe([a, b, c])) == 2


def test_filter_by_confidence():
    items = [{"confidence": 0.5}, {"confidence": 0.7}, {"confidence": 0.9}]
    assert len(filter_by_confidence(items, 0.6)) == 2


def test_filter_by_severity_keeps_at_or_above_threshold():
    items = [
        {"severity": "low"}, {"severity": "medium"},
        {"severity": "high"}, {"severity": "critical"},
    ]
    assert [f["severity"] for f in filter_by_severity(items, "medium")] == [
        "medium", "high", "critical"]


def test_filter_by_severity_low_keeps_everything():
    items = [{"severity": s} for s in ("low", "medium", "high", "critical")]
    assert len(filter_by_severity(items, "low")) == 4


def test_filter_by_severity_unknown_severity_is_kept():
    """An unknown severity string shouldn't silently drop a finding."""
    items = [{"severity": "??"}, {"severity": "high"}]
    assert len(filter_by_severity(items, "high")) == 2


def test_filter_by_severity_is_case_insensitive():
    items = [{"severity": "HIGH"}, {"severity": "Medium"}]
    out = filter_by_severity(items, "high")
    assert len(out) == 1
    assert out[0]["severity"] == "HIGH"


def test_sort_findings_orders_by_severity_then_confidence():
    items = [
        {"severity": "low", "confidence": 0.9},
        {"severity": "critical", "confidence": 0.7},
        {"severity": "high", "confidence": 0.95},
    ]
    out = sort_findings(items)
    assert out[0]["severity"] == "critical"
    assert out[1]["severity"] == "high"
    assert out[2]["severity"] == "low"


def test_is_sensitive_path_blocks_dotenv_files():
    assert is_sensitive_path(".env")
    assert is_sensitive_path("config/.env")
    assert is_sensitive_path(".env.production")
    assert is_sensitive_path(".env.local")


def test_is_sensitive_path_allows_dotenv_example():
    """`.env.example` is meant to be checked in — don't block it."""
    assert not is_sensitive_path(".env.example")
    assert not is_sensitive_path("docs/.env.example")


def test_is_sensitive_path_blocks_secrets_directory():
    assert is_sensitive_path("secrets/api.json")
    assert is_sensitive_path("app/secrets/keys.txt")
    assert is_sensitive_path("a/b/credentials/token")
    assert is_sensitive_path(".ssh/config")
    assert is_sensitive_path(".aws/credentials")


def test_is_sensitive_path_blocks_keys_and_certs():
    assert is_sensitive_path("certs/server.pem")
    assert is_sensitive_path("private.key")
    assert is_sensitive_path("client.p12")
    assert is_sensitive_path("auth.pfx")


def test_is_sensitive_path_blocks_ssh_keys():
    assert is_sensitive_path(".ssh/id_rsa")
    assert is_sensitive_path("id_rsa.pub")
    assert is_sensitive_path("id_ed25519")
    assert is_sensitive_path("deploy_rsa")


def test_is_sensitive_path_blocks_credentials_json():
    assert is_sensitive_path("credentials.json")
    assert is_sensitive_path("service-account.json")
    assert is_sensitive_path("service-account-prod.json")


def test_is_sensitive_path_allows_normal_source():
    assert not is_sensitive_path("src/main.py")
    assert not is_sensitive_path("README.md")
    assert not is_sensitive_path("tests/test_foo.py")
    assert not is_sensitive_path("config/app.json")  # not "credentials"


def test_is_sensitive_path_handles_windows_separators():
    assert is_sensitive_path("app\\secrets\\api.json")
    assert is_sensitive_path("config\\.env")


def test_is_sensitive_path_extra_patterns_match_full_path():
    assert is_sensitive_path("infra/prod/db.sql",
                             extra_patterns=("*infra/prod/*",))
    assert not is_sensitive_path("infra/dev/db.sql",
                                 extra_patterns=("*infra/prod/*",))


def test_find_skip_label_returns_first_match():
    matched = find_skip_label(
        ["bug", "skip-ai-review", "needs-design"],
        ["skip-ai-review", "wip"],
    )
    assert matched == "skip-ai-review"


def test_find_skip_label_returns_none_when_no_overlap():
    assert find_skip_label(["bug", "wip"], ["skip-ai-review"]) is None


def test_find_skip_label_empty_pr_labels():
    assert find_skip_label([], ["skip-ai-review"]) is None


def test_find_skip_label_empty_skip_list():
    assert find_skip_label(["skip-ai-review", "bug"], []) is None


def test_find_skip_label_preserves_pr_label_order():
    """When multiple labels match, the one that appears first in pr_labels
    wins — keeps the log message predictable."""
    matched = find_skip_label(
        ["second-skip", "first-skip", "bug"],
        ["first-skip", "second-skip"],
    )
    assert matched == "second-skip"


def test_format_comment_has_tag_and_fix():
    f = {
        "severity": "high", "category": "security", "title": "SQLi",
        "explanation": "string concat", "confidence": 0.9,
        "suggested_fix": "use params",
    }
    body = format_comment(f)
    assert AI_COMMENT_TAG in body
    assert "Suggested fix" in body
    assert "use params" in body
