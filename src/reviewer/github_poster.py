from __future__ import annotations
from typing import Any, Iterable

from github import Github
from github.PullRequest import PullRequest

from .reporting import SUMMARY_TAG
from .utils import AI_COMMENT_TAG, format_comment


def _latest_commit_sha(pr: PullRequest) -> str:
    commits = list(pr.get_commits())
    if not commits:
        raise RuntimeError(f"PR #{pr.number} has no commits.")
    return commits[-1].sha


def _finding_key(f: dict, body: str) -> tuple[str, int, str]:
    return (f["path"], int(f["line"]), body)


def _comment_key(c: Any) -> tuple[str, int, str] | None:
    """Return the (path, line, body) key for an existing inline comment, or
    ``None`` if the comment is no longer anchored to a line in the diff."""
    line = getattr(c, "line", None)
    path = getattr(c, "path", None)
    body = getattr(c, "body", None)
    if line is None or path is None or body is None:
        return None
    try:
        return (path, int(line), body)
    except (TypeError, ValueError):
        return None


def partition_findings(
    findings: Iterable[dict],
    existing_comments: Iterable[Any],
) -> tuple[list[dict], list[Any], int]:
    """Split findings into (fresh_to_post, stale_to_delete, kept_count).

    A finding is "kept" — neither posted nor deleted — when there is already
    an ``[AI-REVIEW]`` comment on the same path + line whose body matches
    exactly. This makes re-runs idempotent: pushing a new commit that
    doesn't change a finding leaves its comment untouched instead of
    deleting + re-posting it.

    Pure function over hashable keys so it can be unit-tested without
    PyGithub or the network.
    """
    new_pairs: list[tuple[tuple[str, int, str], dict]] = []
    for f in findings:
        body = format_comment(f)
        new_pairs.append((_finding_key(f, body), f))

    existing_pairs: list[tuple[tuple[str, int, str], Any]] = []
    for c in existing_comments:
        body = getattr(c, "body", "") or ""
        if not body.lstrip().startswith(AI_COMMENT_TAG):
            continue
        key = _comment_key(c)
        if key is None:
            # Outdated comment (no current line) — always treat as stale so
            # we delete it and post the finding fresh at the new line.
            existing_pairs.append((None, c))
            continue
        existing_pairs.append((key, c))

    new_keys = {k for k, _ in new_pairs}
    matched_keys: set[tuple[str, int, str]] = set()
    stale: list[Any] = []
    for key, comment in existing_pairs:
        if key is not None and key in new_keys:
            matched_keys.add(key)
        else:
            stale.append(comment)

    fresh = [f for k, f in new_pairs if k not in matched_keys]
    return fresh, stale, len(matched_keys)


def cleanup_stale_ai_comments(pr: PullRequest) -> int:
    """Delete every ``[AI-REVIEW]`` inline comment regardless of content.

    Kept for backwards-compatible callers; the default ``post_findings``
    flow now uses ``partition_findings`` for surgical cleanup.
    """
    removed = 0
    for c in pr.get_review_comments():
        if c.body and c.body.lstrip().startswith(AI_COMMENT_TAG):
            try:
                c.delete()
                removed += 1
            except Exception:
                pass
    return removed


def post_findings(
    token: str,
    repo_full_name: str,
    pr_number: int,
    findings: Iterable[dict],
    *,
    cleanup: bool = True,
    dry_run: bool = False,
) -> dict:
    """Post findings as inline review comments on the PR.

    Behavior:
    - Existing ``[AI-REVIEW]`` comments whose (path, line, body) matches a
      new finding are kept untouched (idempotent re-runs).
    - Comments that no longer match any finding are deleted (when
      ``cleanup=True``).
    - Findings with no matching existing comment are posted.

    Returns ``{"posted", "skipped", "removed_stale", "kept"}``.
    """
    gh = Github(token)
    repo = gh.get_repo(repo_full_name)
    pr = repo.get_pull(pr_number)
    findings = list(findings)

    if dry_run:
        return _dry_run_report(findings)

    existing = list(pr.get_review_comments()) if cleanup else []
    fresh, stale, kept = partition_findings(findings, existing)

    removed = 0
    if cleanup:
        for c in stale:
            try:
                c.delete()
                removed += 1
            except Exception as e:
                print(f"[warn] failed to delete stale comment: {e}")

    sha = _latest_commit_sha(pr)
    commit = repo.get_commit(sha)

    posted, skipped = 0, 0
    for f in fresh:
        path = f.get("path")
        line = f.get("line")
        if not path or not line:
            skipped += 1
            continue
        try:
            pr.create_review_comment(
                body=format_comment(f),
                commit=commit,
                path=path,
                line=line,
                side="RIGHT",
            )
            posted += 1
        except Exception as e:
            skipped += 1
            print(f"[warn] failed to post on {path}:{line} — {e}")

    return {"posted": posted, "skipped": skipped,
            "removed_stale": removed, "kept": kept}


def _dry_run_report(findings: list[dict]) -> dict:
    for f in findings:
        path = f.get("path")
        line = f.get("line")
        body = format_comment(f)
        print(f"[dry-run] would comment on {path}:{line}\n{body}\n")
    return {"posted": len(findings), "skipped": 0,
            "removed_stale": 0, "kept": 0}


def upsert_summary_comment(
    token: str,
    repo_full_name: str,
    pr_number: int,
    body: str,
) -> str:
    """Update the existing AI summary comment in place (matched by tag), or
    create one if none exists. Returns 'updated' | 'created'."""
    gh = Github(token)
    repo = gh.get_repo(repo_full_name)
    pr = repo.get_pull(pr_number)
    for c in pr.get_issue_comments():
        if c.body and c.body.lstrip().startswith(SUMMARY_TAG):
            c.edit(body)
            return "updated"
    pr.create_issue_comment(body)
    return "created"


def fetch_pr_diff(token: str, repo_full_name: str, pr_number: int) -> tuple[str, str, str]:
    """Return (diff_text, title, body) for a PR."""
    import requests
    gh = Github(token)
    repo = gh.get_repo(repo_full_name)
    pr = repo.get_pull(pr_number)
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3.diff",
    }
    r = requests.get(pr.url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.text, pr.title or "", pr.body or ""
