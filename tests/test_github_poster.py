"""Tests for the pure dedup logic in github_poster.

Network-touching paths (PyGithub, requests) are not exercised here; only
``partition_findings`` is in scope, since that's the part that decides
what to post, keep, and delete on a re-run.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reviewer.github_poster import partition_findings
from reviewer.utils import format_comment


def _f(path, line, title, **extra):
    return {
        "path": path, "line": line, "title": title,
        "severity": extra.get("severity", "high"),
        "category": extra.get("category", "security"),
        "explanation": extra.get("explanation", "test"),
        "confidence": extra.get("confidence", 0.9),
        "suggested_fix": extra.get("suggested_fix"),
    }


class FakeComment:
    """Stands in for a PyGithub PullRequestComment."""
    def __init__(self, path, line, body):
        self.path = path
        self.line = line
        self.body = body


def _existing_for(finding):
    """Build a FakeComment whose body matches what we'd post for ``finding``."""
    return FakeComment(finding["path"], finding["line"], format_comment(finding))


def test_partition_no_existing_means_all_fresh():
    findings = [_f("a.py", 1, "x"), _f("b.py", 2, "y")]
    fresh, stale, kept = partition_findings(findings, [])
    assert fresh == findings
    assert stale == []
    assert kept == 0


def test_partition_unchanged_finding_is_kept_and_not_re_posted():
    f = _f("a.py", 5, "x")
    existing = [_existing_for(f)]
    fresh, stale, kept = partition_findings([f], existing)
    assert fresh == []
    assert stale == []
    assert kept == 1


def test_partition_stale_existing_is_deleted():
    """A finding that no longer exists -> its old comment is stale."""
    old = _f("a.py", 5, "old finding")
    existing = [_existing_for(old)]
    fresh, stale, kept = partition_findings([], existing)
    assert fresh == []
    assert stale == existing
    assert kept == 0


def test_partition_changed_body_treated_as_stale_and_fresh():
    """Same path/line but body changed (e.g. confidence drift) -> both
    delete the old comment and post the new one."""
    f1 = _f("a.py", 5, "title", confidence=0.8)
    f2 = _f("a.py", 5, "title", confidence=0.95)  # different body
    existing = [_existing_for(f1)]
    fresh, stale, kept = partition_findings([f2], existing)
    assert fresh == [f2]
    assert stale == existing
    assert kept == 0


def test_partition_moved_finding_treated_as_stale_and_fresh():
    """Body identical but line moved -> repost at new line."""
    f_old = _f("a.py", 5, "title")
    f_new = _f("a.py", 6, "title")  # same content, different line
    existing = [_existing_for(f_old)]
    fresh, stale, kept = partition_findings([f_new], existing)
    assert fresh == [f_new]
    assert stale == existing


def test_partition_ignores_non_ai_comments():
    """Human comments must never be touched."""
    f = _f("a.py", 1, "x")
    human = FakeComment("a.py", 1, "Looks good to me")
    fresh, stale, kept = partition_findings([f], [human])
    assert fresh == [f]
    assert stale == []
    assert kept == 0


def test_partition_outdated_comment_with_no_line_is_stale():
    """A PyGithub comment whose .line is None (outdated) can't match a
    new finding -- always treat as stale so we re-post at the new line."""
    f = _f("a.py", 5, "x")
    outdated = FakeComment("a.py", None, format_comment(f))
    fresh, stale, kept = partition_findings([f], [outdated])
    assert fresh == [f]
    assert stale == [outdated]


def test_partition_mixed():
    keep_f = _f("a.py", 1, "kept")
    new_f = _f("b.py", 2, "new")
    stale_f = _f("c.py", 3, "going away")
    existing = [_existing_for(keep_f), _existing_for(stale_f)]
    fresh, stale, kept = partition_findings([keep_f, new_f], existing)
    assert fresh == [new_f]
    assert len(stale) == 1
    assert stale[0].path == "c.py"
    assert kept == 1


def test_collect_large_pr_diff_local_git(monkeypatch):
    from reviewer.github_poster import _collect_large_pr_diff
    import subprocess

    class DummySha:
        sha = "abc1234"
        ref = "main"

    class DummyPR:
        number = 1381
        base = DummySha()
        head = DummySha()

    def fake_run(cmd, **kwargs):
        class DummyResult:
            returncode = 0
            stdout = "diff --git a/foo.py b/foo.py\n+print('hello')\n"
        return DummyResult()

    monkeypatch.setattr(subprocess, "run", fake_run)
    diff = _collect_large_pr_diff(DummyPR())
    assert "diff --git a/foo.py b/foo.py" in diff
    assert "+print('hello')" in diff


def test_collect_large_pr_diff_get_files_fallback(monkeypatch):
    from reviewer.github_poster import _collect_large_pr_diff
    import subprocess

    class DummyFile:
        def __init__(self, filename, patch):
            self.filename = filename
            self.patch = patch

    class DummyPR:
        number = 1381
        base = None
        head = None

        def get_files(self):
            return [
                DummyFile("service.py", "@@ -1 +1 @@\n-old\n+new"),
                DummyFile("binary.png", None),  # no patch, skipped
            ]

    # Force git run to fail or not return output
    def fake_run(cmd, **kwargs):
        class DummyResult:
            returncode = 1
            stdout = ""
        return DummyResult()

    monkeypatch.setattr(subprocess, "run", fake_run)
    diff = _collect_large_pr_diff(DummyPR())
    assert "diff --git a/service.py b/service.py" in diff
    assert "+new" in diff
    assert "binary.png" not in diff


def test_fetch_pr_diff_handles_406(monkeypatch):
    from reviewer.github_poster import fetch_pr_diff
    import requests

    class DummyResponse:
        status_code = 406
        text = ""

        def raise_for_status(self):
            raise requests.exceptions.HTTPError("406 Client Error: Not Acceptable")

    class DummyPR:
        number = 1381
        url = "https://api.github.com/repos/test/repo/pulls/1381"
        title = "Scrum 2418 insurance dashboard"
        body = "PR description"

    class DummyRepo:
        def get_pull(self, num):
            return DummyPR()

    class DummyGithub:
        def __init__(self, token):
            pass

        def get_repo(self, name):
            return DummyRepo()

    monkeypatch.setattr("reviewer.github_poster.Github", DummyGithub)
    monkeypatch.setattr(requests, "get", lambda url, **kwargs: DummyResponse())
    monkeypatch.setattr(
        "reviewer.github_poster._collect_large_pr_diff",
        lambda pr: "diff --git a/main.py b/main.py\n+fixed\n"
    )

    diff, title, body = fetch_pr_diff("fake_token", "test/repo", 1381)
    assert "diff --git a/main.py b/main.py" in diff
    assert title == "Scrum 2418 insurance dashboard"
    assert body == "PR description"

