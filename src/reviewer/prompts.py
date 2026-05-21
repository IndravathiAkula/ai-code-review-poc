from __future__ import annotations


def build_system_prompt(
    lang: str,
    extras_by_language: dict[str, str] | None = None,
) -> str:
    """Return the system prompt with optional per-language rule extras
    appended.

    ``extras_by_language`` maps a normalized language name (matching
    ``utils.language_for`` output — ``python``, ``typescript``, etc.) to a
    block of additional rules. When a match is found, those rules are
    appended to the base ``SYSTEM`` prompt under a header so the model
    knows they're language-specific.

    Returns ``SYSTEM`` unchanged when ``extras_by_language`` is empty or
    has no entry for ``lang``.
    """
    if not extras_by_language:
        return SYSTEM
    extras = extras_by_language.get(lang)
    if not extras or not extras.strip():
        return SYSTEM
    return f"{SYSTEM}\n\nAdditional rules for {lang} code:\n{extras.rstrip()}"


SYSTEM = """You are a staff software engineer performing code review.
Review ONLY the added/modified lines in the provided unified diff.
Prioritize findings in this order: correctness > security > performance > maintainability.
Ignore pure style nits unless they cause a bug.
Return STRICT JSON matching the schema below. No prose outside the JSON.
If nothing is wrong, return {"findings": []}."""

USER_TEMPLATE = """Repository: {repo}
File: {path}
Language: {lang}
PR title: {title}
PR description:
{description}

Unified diff (hunks for this file only):
```diff
{diff}
```

Schema:
{{
  "findings": [
    {{
      "line": <int — line number in the NEW file, must appear as an added or context line in the diff>,
      "severity": "critical|high|medium|low",
      "category": "correctness|security|performance|maintainability",
      "title": "<short, <80 chars>",
      "explanation": "<2-4 sentences — cite the exact risk and why>",
      "suggested_fix": "<code block or null>",
      "confidence": <float 0.0-1.0>
    }}
  ]
}}

Rules:
- Only include findings with confidence >= 0.6.
- Do not invent APIs or functions. If unsure, lower your confidence.
- Do not flag style/formatting unless it introduces a bug.
- Prefer one high-quality finding over three weak ones.
- Code that has been deliberately commented out (whole JSX/HTML blocks,
  whole functions, debug prints) is not a correctness issue and must not
  be flagged unless the surrounding context shows it was unintentional
  (e.g. half-finished comment, partial syntax that breaks compilation).
- Configuration changes that reference external resources you cannot
  verify from the diff alone (URLs, repo slugs, image tags, package
  versions, environment variable names) must have confidence capped at
  0.5. The model cannot tell whether the new value is correct.
- Hardcoded credentials, API keys, tokens, secrets, passwords, or
  private keys appearing in source code are ALWAYS critical security
  findings — emit them with severity "critical", category "security",
  and confidence >= 0.9. This rule overrides the "one high-quality
  finding" preference: secrets must always be reported, even alongside
  other findings."""
