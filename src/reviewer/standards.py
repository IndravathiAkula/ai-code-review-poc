"""Coding-standards validation via structured YAML rules.

Rules live in ``.ai-review.yml`` under ``standards_rules``. Each rule
scans **added** diff lines with a regex and produces a finding on every
match. Unlike ``prompt_extras_by_language`` (which feeds free text to
the LLM), standards rules are deterministic and cost nothing at runtime.

Rule schema (all fields except ``pattern`` are optional):

.. code-block:: yaml

    standards_rules:
      - id: no-print-in-src
        pattern: '^\\s*print\\('
        languages: [python]           # utils.language_for output
        paths:    ["src/**/*.py"]     # fnmatch globs vs. the file path
        severity: low                 # critical | high | medium | low
        category: maintainability
        message:  Use logging instead of print()
        explanation: |
          print() bypasses the configured log handlers, breaks structured
          logging pipelines, and can't be silenced in production.

The comment posted to the PR uses ``message`` as the short title and
``explanation`` as the body. Both are copy-pasted verbatim; no LLM call.
"""
from __future__ import annotations

import re
import sys
from typing import Any

from unidiff import PatchSet

from .utils import language_for


def _glob_to_regex(pattern: str) -> re.Pattern:
    """Compile a shell-style glob to a regex.

    ``**`` matches zero or more path segments (``a/**/b`` matches ``a/b``,
    ``a/x/b``, ``a/x/y/b``). ``*`` matches within a single segment.
    Regular ``fnmatch`` doesn't do globstar — users writing
    ``web/**/*.tsx`` in ``.ai-review.yml`` expect bash semantics, so we
    implement it here.
    """
    tokens = re.split(r"(\*\*/|/\*\*|\*\*|\*|\?)", pattern)
    parts: list[str] = []
    for tok in tokens:
        if tok == "**/":
            parts.append("(?:.*/)?")   # zero or more directory segments
        elif tok == "/**":
            parts.append("(?:/.*)?")
        elif tok == "**":
            parts.append(".*")
        elif tok == "*":
            parts.append("[^/]*")
        elif tok == "?":
            parts.append("[^/]")
        elif tok == "":
            continue
        else:
            parts.append(re.escape(tok))
    return re.compile("^" + "".join(parts) + "$")


_ALLOWED_SEVERITIES = {"critical", "high", "medium", "low"}
_ALLOWED_CATEGORIES = {
    "correctness", "security", "security_hotspot",
    "performance", "maintainability",
}


def _warn(rid: str, msg: str) -> None:
    print(f"[warn] standards_rules[{rid}]: {msg}", file=sys.stderr)


def _validate_rule(raw: Any, idx: int) -> dict | None:
    """Return a compiled rule dict, or ``None`` on malformed input.

    Bad rules warn to stderr and are dropped — one broken rule shouldn't
    take out the whole set. Every warning names the rule id so users can
    find it in the CI log.
    """
    if not isinstance(raw, dict):
        print(
            f"[warn] standards_rules[{idx}]: expected mapping, got "
            f"{type(raw).__name__}; skipping", file=sys.stderr,
        )
        return None
    rid = str(raw.get("id") or f"rule-{idx}")
    pattern = raw.get("pattern")
    if not pattern or not isinstance(pattern, str):
        _warn(rid, "missing or non-string 'pattern'; skipping")
        return None
    try:
        compiled = re.compile(pattern)
    except re.error as e:
        _warn(rid, f"invalid regex ({e}); skipping")
        return None
    severity = str(raw.get("severity", "low")).lower()
    if severity not in _ALLOWED_SEVERITIES:
        _warn(rid, f"unknown severity {severity!r}; defaulting to 'low'")
        severity = "low"
    category = str(raw.get("category", "maintainability")).lower()
    if category not in _ALLOWED_CATEGORIES:
        _warn(rid, f"unknown category {category!r}; defaulting to 'maintainability'")
        category = "maintainability"
    languages = raw.get("languages") or []
    if isinstance(languages, str):
        languages = [languages]
    paths = raw.get("paths") or []
    if isinstance(paths, str):
        paths = [paths]
    return {
        "id": rid,
        "regex": compiled,
        "languages": [str(x).lower() for x in languages],
        "paths": [str(p) for p in paths],
        "path_regexes": [_glob_to_regex(str(p)) for p in paths],
        "severity": severity,
        "category": category,
        "message": str(raw.get("message") or rid),
        "explanation": str(raw.get("explanation") or ""),
    }


def load_rules(raw_rules: list | None) -> list[dict]:
    """Validate + compile a raw list of rule dicts from YAML."""
    if not raw_rules:
        return []
    compiled: list[dict] = []
    for i, raw in enumerate(raw_rules):
        rule = _validate_rule(raw, i)
        if rule is not None:
            compiled.append(rule)
    return compiled


def _rule_applies(rule: dict, path: str, lang: str) -> bool:
    if rule["languages"] and lang not in rule["languages"]:
        return False
    if rule["path_regexes"]:
        norm = path.replace("\\", "/")
        if not any(rx.match(norm) for rx in rule["path_regexes"]):
            return False
    return True


def apply_rules(diff_text: str, rules: list[dict]) -> list[dict]:
    """Scan added lines in ``diff_text`` against ``rules``. Returns findings.

    A finding is emitted per (rule, matched-line) — a rule that matches
    twice in the same diff yields two findings so both get comments.
    """
    if not rules:
        return []
    findings: list[dict] = []
    for pf in PatchSet(diff_text):
        if pf.is_binary_file:
            continue
        path = pf.path
        lang = language_for(path)
        active = [r for r in rules if _rule_applies(r, path, lang)]
        if not active:
            continue
        for hunk in pf:
            for line in hunk:
                if not line.is_added or line.target_line_no is None:
                    continue
                content = line.value.rstrip("\n").rstrip("\r")
                for rule in active:
                    if rule["regex"].search(content):
                        findings.append({
                            "path": path,
                            "line": line.target_line_no,
                            "severity": rule["severity"],
                            "category": rule["category"],
                            "title": f"{rule['id']}: {rule['message'][:100]}",
                            "explanation": rule["explanation"] or rule["message"],
                            "suggested_fix": None,
                            "confidence": 1.0,
                            "source": "standards",
                            "tool": f"standards/{rule['id']}",
                        })
    return findings
