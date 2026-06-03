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
}


def _coerce(name: str, value: Any) -> Any:
    """Best-effort type coercion. Strings from env need int/float/bool
    conversion; YAML usually arrives already typed."""
    if name in {"min_confidence", "retry_base_seconds"}:
        return float(value)
    if name in {"concurrency", "max_diff_chars", "max_retries",
                "max_files_per_pr", "max_tokens_per_pr"}:
        return int(value)
    if name in {"post_summary", "include_maintainability_findings"}:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
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
