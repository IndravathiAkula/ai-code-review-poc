"""Repo-level configuration loaded from ``.ai-review.yml`` at repo root.

Precedence (highest first):

1. Explicit kwargs passed by the script (e.g. CLI flags).
2. ``.ai-review.yml`` at the repo root — the place repo owners customize.
3. Environment variables set by the workflow.
4. Built-in defaults.

Order is intentional: the workflow yaml ships with sensible defaults
exposed as env vars; ``.ai-review.yml`` is how a repo overrides those
without forking the workflow; CLI flags are last-mile overrides.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import yaml


CONFIG_FILENAME = ".ai-review.yml"


@dataclass
class ReviewConfig:
    """Single source of truth for tunable review settings.

    Mirrors the ``review_patch`` kwargs and the post-step toggles so that
    ``review_pr.py`` can fan one ``ReviewConfig`` out across the whole
    pipeline.
    """
    provider: str = "github-models"
    model: str = "openai/gpt-4o-mini"
    min_confidence: float = 0.6
    min_severity: str = "low"
    concurrency: int = 4
    max_diff_chars: int = 8000
    max_retries: int = 3
    retry_base_seconds: float = 1.0
    block_patterns: tuple[str, ...] = ()
    post_summary: bool = True
    models_by_language: dict[str, str] = field(default_factory=dict)
    # Per-language extra rules appended to the system prompt. Maps
    # language name (matching ``utils.language_for``: ``python``,
    # ``typescript``, etc.) to a multi-line rules block. Useful when
    # mainstream prompts don't cover a stack-specific bug class
    # (React useEffect deps, Python mutable default args, etc.).
    prompt_extras_by_language: dict[str, str] = field(default_factory=dict)
    # Labels that cause the reviewer to exit immediately with no model
    # calls. Useful for huge refactors, generated-code PRs, dep bumps,
    # etc. Default includes the common ``skip-ai-review`` convention.
    skip_labels: tuple[str, ...] = ("skip-ai-review",)
    # Cost caps — 0 means unlimited. ``max_files_per_pr`` is enforced
    # pre-flight (excess chunks are skipped before any model call);
    # ``max_tokens_per_pr`` is enforced at runtime against the running
    # token total reported by completed chunks.
    max_files_per_pr: int = 0
    max_tokens_per_pr: int = 0
    # Opt-in: also surface lint-class maintainability findings (unused
    # vars, redeclaration in same scope, deep nesting, redundant logic).
    # Off by default because deterministic linters (ruff, eslint, etc.)
    # already cover this cheaply; enable it for repos that don't run a
    # linter in PR CI.
    include_maintainability_findings: bool = False
    # Deterministic tooling that runs alongside the LLM. Lint is on by
    # default (ruff for .py, eslint via ``npx --no-install`` for JS/TS —
    # both no-op when the tool or config isn't present). Type checking
    # is opt-in because mypy/tsc walk broader graphs and can be slow.
    enable_linters: bool = True
    enable_type_check: bool = False
    # Structured coding-standards rules. Each entry is a mapping with at
    # least ``pattern`` (regex); see ``reviewer.standards`` for the full
    # schema. Matches on ADDED diff lines produce PR comments with no
    # model call.
    standards_rules: list = field(default_factory=list)
    # PR-level architectural review — one extra LLM call that looks at
    # the whole diff and flags cross-file / structural concerns a
    # per-file pass can't see (missing tests, breaking changes,
    # architectural smells, scope creep). On by default because the
    # cost is one extra chunk-sized call and the signal is high.
    enable_pr_level_review: bool = True
    # Deterministic "missing tests" heuristic that fires when a PR
    # touches production code but not tests. Independent of the LLM
    # PR-level pass so it works even when the model call is disabled.
    enable_missing_tests_check: bool = True
    # Repository context: retrieve callers of newly-defined symbols +
    # one sibling file for style reference and inject into the per-file
    # user prompt. Requires ``git`` on PATH (guaranteed in Actions).
    enable_repo_context: bool = True
    max_context_files: int = 3
    max_context_chars: int = 3000
    # Route test-file diffs to a specialized "review the tests, not the
    # code under test" system prompt (tautological asserts, missing edge
    # cases, over-mocking, etc.). Off routes tests through the normal
    # reviewer prompt.
    enable_test_review: bool = True
    # Semgrep: multi-language SAST + best-practice rules. Opt-in because
    # it's a bigger dep (~50MB) and needs network for --config auto.
    enable_semgrep: bool = False
    # New-code coverage gate: parse a coverage report and flag when the
    # PR's added lines fall below the threshold. Off by default — needs
    # CI to write the report file first.
    enable_coverage_gate: bool = False
    coverage_report_path: str = "coverage.xml"
    new_code_coverage_threshold: float = 0.8


_KNOWN_KEYS: set[str] = {f.name for f in fields(ReviewConfig)}


# Each known config field can also be set via an environment variable.
_ENV_OVERRIDES: dict[str, str] = {
    "provider": "REVIEWER_PROVIDER",
    "model": "MODEL",
    "min_confidence": "MIN_CONFIDENCE",
    "min_severity": "MIN_SEVERITY",
    "concurrency": "REVIEWER_CONCURRENCY",
    "max_diff_chars": "REVIEWER_MAX_DIFF_CHARS",
    "max_retries": "REVIEWER_MAX_RETRIES",
    "retry_base_seconds": "REVIEWER_RETRY_BASE_SECONDS",
    "block_patterns": "REVIEWER_BLOCK_PATTERNS",
    "post_summary": "REVIEWER_POST_SUMMARY",
    "max_files_per_pr": "REVIEWER_MAX_FILES_PER_PR",
    "max_tokens_per_pr": "REVIEWER_MAX_TOKENS_PER_PR",
    "skip_labels": "REVIEWER_SKIP_LABELS",
    "include_maintainability_findings": "REVIEWER_INCLUDE_MAINTAINABILITY",
    "enable_linters": "REVIEWER_ENABLE_LINTERS",
    "enable_type_check": "REVIEWER_ENABLE_TYPE_CHECK",
    "enable_pr_level_review": "REVIEWER_ENABLE_PR_LEVEL_REVIEW",
    "enable_missing_tests_check": "REVIEWER_ENABLE_MISSING_TESTS",
    "enable_repo_context": "REVIEWER_ENABLE_REPO_CONTEXT",
    "max_context_files": "REVIEWER_MAX_CONTEXT_FILES",
    "max_context_chars": "REVIEWER_MAX_CONTEXT_CHARS",
    "enable_test_review": "REVIEWER_ENABLE_TEST_REVIEW",
    "enable_semgrep": "REVIEWER_ENABLE_SEMGREP",
    "enable_coverage_gate": "REVIEWER_ENABLE_COVERAGE_GATE",
    "coverage_report_path": "REVIEWER_COVERAGE_REPORT_PATH",
    "new_code_coverage_threshold": "REVIEWER_NEW_CODE_COVERAGE_THRESHOLD",
}


def _coerce(name: str, value: Any) -> Any:
    """Best-effort type coercion. Strings from env need int/float/bool
    conversion; YAML usually arrives already typed."""
    if name in {"min_confidence", "retry_base_seconds",
                "new_code_coverage_threshold"}:
        return float(value)
    if name in {"concurrency", "max_diff_chars", "max_retries",
                "max_files_per_pr", "max_tokens_per_pr",
                "max_context_files", "max_context_chars"}:
        return int(value)
    if name in {"post_summary", "include_maintainability_findings",
                "enable_linters", "enable_type_check",
                "enable_pr_level_review", "enable_missing_tests_check",
                "enable_repo_context", "enable_test_review",
                "enable_semgrep", "enable_coverage_gate"}:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if name == "standards_rules":
        if value is None:
            return []
        if not isinstance(value, list):
            raise TypeError(
                f"standards_rules must be a list, got {type(value).__name__}")
        return list(value)
    if name in {"block_patterns", "skip_labels"}:
        if isinstance(value, str):
            return tuple(p.strip() for p in value.split(",") if p.strip())
        if value is None:
            return ()
        return tuple(value)
    if name in {"models_by_language", "prompt_extras_by_language"}:
        return dict(value or {})
    return value  # str fields pass through


def load_config(path: str | Path | None = None) -> dict:
    """Read ``.ai-review.yml`` from ``path`` (or repo root if ``None``).

    Returns an empty dict when the file doesn't exist. Top-level types
    that aren't mappings, and unknown keys, get a warning printed to
    stderr but don't fail the run.
    """
    if path is None:
        path = Path.cwd() / CONFIG_FILENAME
    p = Path(path)
    if not p.exists():
        return {}
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        print(
            f"[warn] {p}: top-level YAML must be a mapping, got "
            f"{type(raw).__name__}; ignoring",
            file=sys.stderr,
        )
        return {}
    unknown = set(raw) - _KNOWN_KEYS
    if unknown:
        print(
            f"[warn] {p}: unknown config keys ignored: {sorted(unknown)}",
            file=sys.stderr,
        )
    return {k: v for k, v in raw.items() if k in _KNOWN_KEYS}


def effective_config(
    *,
    config_file: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> ReviewConfig:
    """Layer overrides on top of defaults: env first, config-file on top.

    Type errors on individual fields are logged and the offending field
    falls back to whichever lower-priority value already won.
    """
    cfg = ReviewConfig()
    config_file = config_file or {}
    env = env if env is not None else os.environ

    # Layer 1: env vars over defaults.
    for key, env_name in _ENV_OVERRIDES.items():
        raw = env.get(env_name)
        if raw is None or raw == "":
            continue
        try:
            setattr(cfg, key, _coerce(key, raw))
        except (TypeError, ValueError) as e:
            print(
                f"[warn] env {env_name}: invalid value {raw!r} ({e}); "
                f"keeping current value",
                file=sys.stderr,
            )

    # Layer 2: config-file values over env (repo owner has the final say).
    for key in _KNOWN_KEYS:
        if key in config_file:
            try:
                setattr(cfg, key, _coerce(key, config_file[key]))
            except (TypeError, ValueError) as e:
                print(
                    f"[warn] config.{key}: invalid value "
                    f"{config_file[key]!r} ({e}); keeping current value",
                    file=sys.stderr,
                )

    return cfg
