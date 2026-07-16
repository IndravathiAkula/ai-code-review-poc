"""PR-level architectural review — the pass a senior engineer does after
reading every file individually.

The per-file reviewer in :mod:`reviewer` runs one call per chunk and
sees ~one file at a time. That's efficient but structurally blind: it
can't spot a missing test, a breaking API change that only becomes
obvious across three files, a new dependency added without
justification, or a PR that grew scope beyond its title.

This module does one additional LLM call on a compact PR-level summary
(file list, added-line counts, a curated slice of the diff) and asks
for exactly the checks a per-file pass misses.

Findings emitted here have ``line: 0`` and ``source: "pr_review"`` —
they can't be posted inline, so :mod:`reporting` renders them in the
PR-level summary comment instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from unidiff import PatchSet

from .prompts import PR_LEVEL_SYSTEM, PR_LEVEL_USER_TEMPLATE
from .providers import ChatResponse, Provider
from .reviewer import _call_with_retry, _parse_json
from .utils import is_config_path, is_doc_path, is_test_path


DEFAULT_MAX_DIFF_CHARS_PR = 12000  # leaves room for the system prompt


@dataclass
class PRLevelSummary:
    """Compact digest of the PR that fits in one prompt."""
    files: list[str]
    total_added: int
    total_removed: int
    file_stats: list[dict]  # {path, added, removed}
    excerpt: str            # bounded slice of the raw diff

    def to_dict(self) -> dict:
        return {
            "files": self.files,
            "total_added": self.total_added,
            "total_removed": self.total_removed,
            "file_stats": self.file_stats,
            "excerpt": self.excerpt,
        }


def build_summary(
    diff_text: str, max_diff_chars: int = DEFAULT_MAX_DIFF_CHARS_PR,
) -> PRLevelSummary:
    """Digest ``diff_text`` into a per-file stats block plus an excerpt.

    The excerpt is the raw diff truncated to ``max_diff_chars`` — the
    LLM sees at least the file structure and some real code even on a
    huge PR. Files are ordered by change size (biggest first) so the
    most impactful chunks land in the excerpt window.
    """
    patch = PatchSet(diff_text)
    stats: list[dict] = []
    total_added = total_removed = 0
    for pf in patch:
        if pf.is_binary_file:
            continue
        added = sum(1 for h in pf for line in h if line.is_added)
        removed = sum(1 for h in pf for line in h if line.is_removed)
        stats.append({
            "path": pf.path,
            "added": added,
            "removed": removed,
        })
        total_added += added
        total_removed += removed
    stats.sort(key=lambda s: -(s["added"] + s["removed"]))
    excerpt = diff_text
    if len(excerpt) > max_diff_chars:
        excerpt = excerpt[:max_diff_chars] + "\n... (truncated)"
    return PRLevelSummary(
        files=[s["path"] for s in stats],
        total_added=total_added,
        total_removed=total_removed,
        file_stats=stats,
        excerpt=excerpt,
    )


def missing_tests_finding(summary: PRLevelSummary) -> dict | None:
    """Deterministic heuristic: prod code touched, no tests touched.

    Returns a PR-level finding when it's plausible the PR should have
    included tests but didn't. Skips docs-only, config-only, and
    test-only PRs so we don't nag on non-code changes.
    """
    if not summary.files:
        return None
    code_files = [
        p for p in summary.files
        if not is_test_path(p) and not is_doc_path(p) and not is_config_path(p)
    ]
    test_files = [p for p in summary.files if is_test_path(p)]
    if not code_files or test_files:
        return None
    # Also skip when the only "code" change is trivially small (< 5
    # added lines total) — a one-line typo fix shouldn't demand a
    # test.
    code_added = sum(
        s["added"] for s in summary.file_stats
        if s["path"] in code_files
    )
    if code_added < 5:
        return None
    return {
        "path": "(pull request)",
        "line": 0,
        "severity": "medium",
        "category": "maintainability",
        "title": f"Missing tests: {len(code_files)} code file(s) changed with "
                 f"no test updates",
        "explanation": (
            f"This PR modifies production code ({code_added} added line(s) "
            f"across {len(code_files)} file(s)) but does not add or update "
            f"any tests. A senior reviewer would ask whether the new "
            f"behaviour is exercised by CI. Files without matching test "
            f"changes: {', '.join(code_files[:5])}"
            f"{'…' if len(code_files) > 5 else ''}."
        ),
        "suggested_fix": None,
        "confidence": 0.9,
        "source": "pr_review",
        "tool": "missing-tests-heuristic",
    }


def review_pr_level(
    diff_text: str,
    *,
    provider: Provider,
    model: str,
    pr_title: str = "",
    pr_description: str = "",
    repo: str = "unknown/unknown",
    max_diff_chars: int = DEFAULT_MAX_DIFF_CHARS_PR,
    max_retries: int = 3,
    retry_base_seconds: float = 1.0,
    usage_log: list[dict] | None = None,
    enable_missing_tests: bool = True,
) -> list[dict]:
    """Ask the LLM for cross-file, PR-shaped concerns. Returns findings.

    One extra chunk-sized call. Cost is bounded by ``max_diff_chars``.
    Deterministic PR-level checks (missing tests) run first and don't
    consume tokens; they compose with the LLM output.
    """
    summary = build_summary(diff_text, max_diff_chars=max_diff_chars)
    findings: list[dict] = []
    if enable_missing_tests:
        mt = missing_tests_finding(summary)
        if mt is not None:
            findings.append(mt)

    if not summary.files:
        return findings

    stats_lines = "\n".join(
        f"- {s['path']}: +{s['added']} -{s['removed']}"
        for s in summary.file_stats
    )
    user = PR_LEVEL_USER_TEMPLATE.format(
        repo=repo,
        title=pr_title or "(no title)",
        description=pr_description or "(no description)",
        file_count=len(summary.files),
        total_added=summary.total_added,
        total_removed=summary.total_removed,
        file_stats=stats_lines,
        excerpt=summary.excerpt,
    )
    call_kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": PR_LEVEL_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    try:
        resp: ChatResponse = _call_with_retry(
            provider.complete,
            max_retries=max_retries,
            base_seconds=retry_base_seconds,
            **call_kwargs,
        )
    except Exception as exc:
        # PR-level review is best-effort; don't fail the whole run if
        # it errors — the per-file findings still ship.
        import sys
        print(f"[warn] PR-level review failed: {exc}", file=sys.stderr)
        return findings

    if usage_log is not None and resp.usage is not None:
        from .pricing import cost_usd
        u = resp.usage
        usage_log.append({
            "provider": provider.name,
            "model": model,
            "path": "(pr-level)",
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "cache_read_tokens": u.cache_read_tokens,
            "cache_creation_tokens": u.cache_creation_tokens,
            "total_tokens": (u.prompt_tokens + u.completion_tokens
                             + u.cache_read_tokens + u.cache_creation_tokens),
            "cost_usd": cost_usd(
                model, u.prompt_tokens, u.completion_tokens,
                provider=provider.name,
                cache_read_tokens=u.cache_read_tokens,
                cache_creation_tokens=u.cache_creation_tokens,
            ),
            "latency_seconds": 0.0,
        })

    data = _parse_json(resp.content)
    for item in data.get("findings", []) or []:
        # PR-level findings have no line anchor. Force line=0 so the
        # renderer knows to put them in the summary instead of an
        # inline comment.
        item["path"] = str(item.get("path") or "(pull request)")
        item["line"] = 0
        item["source"] = "pr_review"
        item.setdefault("tool", "pr-level-llm")
        findings.append(item)
    return findings
