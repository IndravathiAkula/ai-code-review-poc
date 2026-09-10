"""MCP server — expose the reviewer as a tool for Claude Code / Cursor /
Continue / any Model Context Protocol client.

Lets a developer run an AI review on their local working-tree diff
*before* pushing, without leaving the IDE. Same review pipeline that
runs in CI (severity floor, hallucination guard, sensitive-path
blocklist, provider routing, idempotent-by-design output) — just
without the PR-posting layer at the end.

Two tools are registered:

- ``review_diff(diff_text, ...)``: review an arbitrary unified diff
  string. The model decides which language each file is, validates
  every reported line against the diff, applies the confidence and
  severity floors, and returns markdown.

- ``review_working_tree(repo_path=".", staged=False, ...)``: convenience
  wrapper that runs ``git diff`` (or ``git diff --cached`` if
  ``staged=True``) under ``repo_path`` and reviews the result. Returns
  a friendly "no changes to review" when the tree is clean.

Run with:  ``python -m reviewer.mcp_server``
"""
from __future__ import annotations

import subprocess
from pathlib import Path

try:
    from mcp.server.fastmcp import FastMCP
except Exception:
    from mcp.server.mcpserver import MCPServer as FastMCP

from . import review_patch
from .providers import build_provider


mcp = FastMCP("AI Code Reviewer")


# ---- formatters ----

_SEVERITY_LABEL = {
    "critical": "CRITICAL",
    "high": "HIGH",
    "medium": "MEDIUM",
    "low": "LOW",
}


def _format_findings(findings: list[dict]) -> str:
    """Render review findings as IDE-friendly markdown.

    Sorted by severity (critical first) before this function is called
    by ``review_patch``; we just lay them out for human consumption.
    """
    if not findings:
        return "**No findings.** The diff looks clean to the model."

    lines = [f"## AI Review ({len(findings)} finding{'s' if len(findings) != 1 else ''})", ""]
    for i, f in enumerate(findings, 1):
        sev = _SEVERITY_LABEL.get(str(f.get("severity", "low")).lower(), "LOW")
        cat = f.get("category", "maintainability")
        path = f.get("path", "?")
        line = f.get("line", "?")
        title = (f.get("title") or "").strip() or "(untitled)"
        conf = float(f.get("confidence", 0.0))
        explanation = (f.get("explanation") or "").strip()

        lines.append(f"### {i}. [{sev}/{cat}] `{path}:{line}` — {title}")
        lines.append(f"_confidence {conf:.2f}_")
        lines.append("")
        if explanation:
            lines.append(explanation)
            lines.append("")
        fix = f.get("suggested_fix")
        if fix:
            lines.append("**Suggested fix:**")
            lines.append("```")
            lines.append(str(fix).strip())
            lines.append("```")
            lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines).rstrip()


def _get_diff(repo_path: str, staged: bool) -> str:
    """Run ``git diff`` (or ``--cached``) in ``repo_path`` and return stdout.

    Raises ``RuntimeError`` if git fails (not a repo, missing path, etc.).
    """
    args = ["git", "diff"]
    if staged:
        args.append("--cached")
    p = Path(repo_path).expanduser().resolve()
    if not p.exists():
        raise RuntimeError(f"path does not exist: {p}")
    result = subprocess.run(
        args, cwd=str(p), capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git diff failed in {p}: {result.stderr.strip() or 'unknown error'}")
    return result.stdout


def _run_review(
    diff_text: str,
    model: str,
    provider_name: str,
    min_confidence: float,
    min_severity: str,
) -> str:
    """Shared review-and-format pipeline used by both MCP tools."""
    if not diff_text.strip():
        return "No changes to review."
    provider = build_provider(provider_name)
    findings = review_patch(
        diff_text,
        model=model,
        provider=provider,
        min_confidence=min_confidence,
        min_severity=min_severity,
        repo="mcp/local",
        pr_title="Local diff review",
    )
    return _format_findings(findings)


# ---- MCP tools ----

@mcp.tool()
def review_diff(
    diff_text: str,
    model: str = "openai/gpt-4o-mini",
    provider: str = "github-models",
    min_confidence: float = 0.6,
    min_severity: str = "low",
) -> str:
    """Review a unified diff string and return findings as markdown.

    Use this when you have a diff already in hand (paste-buffer, the
    output of ``git diff``, a saved ``.diff`` file). For the common
    case of reviewing your current working-tree changes, prefer
    ``review_working_tree`` instead.

    Args:
        diff_text: Unified diff produced by ``git diff`` or equivalent.
        model: Model identifier accepted by the chosen provider
            (e.g. ``openai/gpt-4o-mini`` for github-models,
            ``llama-3.3-70b-versatile`` for groq,
            ``claude-sonnet-4-6`` for anthropic).
        provider: One of ``github-models`` (default), ``openai``,
            ``anthropic``, ``groq``, ``openrouter``, ``nvidia``,
            ``together``, ``anyscale``, ``cerebras``, ``ollama``.
            Each requires its API key env var to be set.
        min_confidence: Drop findings below this confidence (0.0-1.0).
        min_severity: Drop findings below this severity level
            (``low`` | ``medium`` | ``high`` | ``critical``).
    """
    return _run_review(diff_text, model, provider, min_confidence, min_severity)


@mcp.tool()
def review_working_tree(
    repo_path: str = ".",
    staged: bool = False,
    model: str = "openai/gpt-4o-mini",
    provider: str = "github-models",
    min_confidence: float = 0.6,
    min_severity: str = "low",
) -> str:
    """Review your current working-tree (or staged) changes.

    Runs ``git diff`` in ``repo_path`` and feeds the result through the
    same review pipeline as CI. Great for catching obvious issues
    before opening a PR.

    Args:
        repo_path: Directory containing the git repo. Defaults to the
            current working directory.
        staged: If True, review only staged changes (``git diff --cached``)
            instead of all working-tree changes.
        model: Model identifier (see ``review_diff`` for examples).
        provider: Provider name (see ``review_diff`` for the full list).
        min_confidence: Confidence floor (0.0-1.0).
        min_severity: Severity floor (low|medium|high|critical).
    """
    try:
        diff_text = _get_diff(repo_path, staged)
    except RuntimeError as exc:
        return f"**Error:** {exc}"
    if not diff_text.strip():
        which = "staged" if staged else "working-tree"
        return f"No {which} changes to review."
    return _run_review(diff_text, model, provider, min_confidence, min_severity)


def main() -> None:
    """Entrypoint for ``python -m reviewer.mcp_server``."""
    mcp.run()


if __name__ == "__main__":
    main()
