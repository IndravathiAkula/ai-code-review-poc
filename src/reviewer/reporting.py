"""Per-run observability: aggregate ``usage_log`` entries into a summary.

The output of ``summarize_run`` feeds three sinks in CI:
- A JSON artifact written to disk (full raw detail for offline analysis).
- A markdown blob appended to ``$GITHUB_STEP_SUMMARY`` (visible in the
  Actions run page).
- An idempotent PR-level comment (so reviewers see token/cost cost without
  digging into Actions).

Pure-Python with no PyGithub or HTTP dependencies — the I/O happens in
``review_pr.py`` and ``github_poster.py``.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


SUMMARY_TAG = "[AI-REVIEW-SUMMARY]"


@dataclass
class RunSummary:
    model: str
    repo: str
    pr_number: int | None
    posted: int
    kept: int
    skipped: int
    removed_stale: int
    severity_counts: dict[str, int]
    total_calls: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float | None
    wall_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def _aggregate_usage(usage_log: Iterable[dict]) -> tuple[int, int, int, int, float | None]:
    calls = 0
    pt = ct = tt = 0
    cost = 0.0
    cost_known = False
    for u in usage_log:
        calls += 1
        pt += int(u.get("prompt_tokens", 0) or 0)
        ct += int(u.get("completion_tokens", 0) or 0)
        tt += int(u.get("total_tokens", 0) or 0)
        c = u.get("cost_usd")
        if c is not None:
            cost += float(c)
            cost_known = True
    return calls, pt, ct, tt, (cost if cost_known else None)


def summarize_run(
    *,
    model: str,
    repo: str,
    pr_number: int | None,
    findings: list[dict],
    post_report: dict,
    usage_log: list[dict],
    wall_seconds: float,
) -> RunSummary:
    """Roll up findings + post-report + usage_log into a single summary."""
    severity_counts: Counter = Counter(
        (f.get("severity", "low") or "low").lower() for f in findings)
    calls, pt, ct, tt, cost = _aggregate_usage(usage_log)
    return RunSummary(
        model=model,
        repo=repo,
        pr_number=pr_number,
        posted=int(post_report.get("posted", 0)),
        kept=int(post_report.get("kept", 0)),
        skipped=int(post_report.get("skipped", 0)),
        removed_stale=int(post_report.get("removed_stale", 0)),
        severity_counts=dict(severity_counts),
        total_calls=calls,
        prompt_tokens=pt,
        completion_tokens=ct,
        total_tokens=tt,
        cost_usd=cost,
        wall_seconds=wall_seconds,
    )


def _fmt_cost(cost: float | None) -> str:
    if cost is None:
        return "—"
    if cost < 0.0001:
        return f"${cost:.6f}"
    return f"${cost:.4f}"


def _fmt_severity_breakdown(counts: dict[str, int]) -> str:
    """Stable ordering: critical → high → medium → low → other."""
    order = ["critical", "high", "medium", "low"]
    parts = []
    for sev in order:
        n = counts.get(sev, 0)
        if n:
            parts.append(f"{n} {sev}")
    extras = sorted(k for k in counts if k not in order)
    for sev in extras:
        n = counts.get(sev, 0)
        if n:
            parts.append(f"{n} {sev}")
    return ", ".join(parts) if parts else "none"


def render_markdown(summary: RunSummary) -> str:
    """Render a RunSummary as a markdown blob suitable for both the GitHub
    step summary and a PR-level issue comment."""
    pr_label = f"PR #{summary.pr_number}" if summary.pr_number else "diff"
    lines = [
        f"{SUMMARY_TAG}",
        "",
        "## AI Review Summary",
        "",
        f"`{summary.model}` reviewed {pr_label} in {summary.wall_seconds:.2f}s.",
        "",
        "| Metric | Count |",
        "|---|---|",
        f"| Findings posted | {summary.posted} |",
        f"| Findings kept (unchanged) | {summary.kept} |",
        f"| Stale comments removed | {summary.removed_stale} |",
        f"| Skipped (errors) | {summary.skipped} |",
        "",
        f"**Severity breakdown:** {_fmt_severity_breakdown(summary.severity_counts)}  ",
        f"**Tokens:** {summary.total_tokens:,} "
        f"({summary.prompt_tokens:,} prompt + {summary.completion_tokens:,} completion) "
        f"across {summary.total_calls} call(s)  ",
        f"**Estimated cost:** {_fmt_cost(summary.cost_usd)}",
    ]
    return "\n".join(lines)


def write_step_summary(markdown: str, *, env: dict | None = None) -> bool:
    """Append ``markdown`` to ``$GITHUB_STEP_SUMMARY`` if set in ``env``.
    Returns True on success, False if the env var isn't set (running locally)."""
    import os
    env = env if env is not None else os.environ
    path = env.get("GITHUB_STEP_SUMMARY")
    if not path:
        return False
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(markdown)
        f.write("\n")
    return True


def write_artifact(
    path: str | Path,
    summary: RunSummary,
    usage_log: list[dict],
) -> Path:
    """Write the full run report (summary + per-chunk usage_log) as JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary.to_dict(), "usage_log": usage_log}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
