"""Scoring strategies for the eval harness.

Two scorers, same return shape:

- ``score_case_keyword`` (default, free): a model finding "hits" an expected
  issue if it lands on the same file, within ``line_range``, and the
  title+explanation contains one of the labelled ``keywords``.

- ``score_case_judge`` (opt-in, costs tokens): same file + line-range
  pre-filter, but each candidate is scored by an LLM judge that decides
  whether the finding semantically describes the expected issue. Catches
  cases where the model is correct but doesn't use the labelled keywords.

Both return:
    {
        "expected": <int>,
        "hits": <int>,
        "findings_emitted": <int>,
        "extras": <int>,
        "by_category_hits": {cat: int},
        "by_category_expected": {cat: int},
    }
"""
from __future__ import annotations

import json
from typing import Any

from azure.ai.inference.models import SystemMessage, UserMessage


JUDGE_SYSTEM = (
    "You grade code-review findings against ground-truth bug labels. "
    "A candidate finding MATCHES an expected issue when it semantically "
    "describes the same root cause, even if the wording differs. "
    "Style, severity, or category mismatches alone are not disqualifying. "
    'Return strict JSON only: {"match": true|false}. No prose.'
)

JUDGE_USER_TEMPLATE = """Expected issue:
- File: {path}
- Lines: {lo}-{hi}
- Category: {category}
- Severity: {severity}
- Description hints (any one is sufficient — semantic match counts): {keywords}

Candidate finding from a code-review model:
- Title: {title}
- Explanation: {explanation}
- Reported line: {line}

Does the candidate correctly identify the expected issue? Return strict JSON only."""


def _haystack(f: dict) -> str:
    return f"{f.get('title', '')} {f.get('explanation', '')}".lower()


def _empty_result(findings_emitted: int, expected: int) -> dict:
    return {
        "expected": expected,
        "hits": 0,
        "findings_emitted": findings_emitted,
        "extras": findings_emitted,
        "by_category_hits": {},
        "by_category_expected": {},
    }


def _candidate_indices_in_range(remaining: list[dict], exp: dict) -> list[int]:
    """Indices into ``remaining`` of findings that share the file and fall
    inside the expected line range. Locked-in pre-filter for both scorers."""
    out = []
    lo, hi = exp["line_range"]
    path = exp["path"]
    for i, f in enumerate(remaining):
        if f.get("path") != path:
            continue
        try:
            line = int(f.get("line", -1))
        except (TypeError, ValueError):
            continue
        if lo <= line <= hi:
            out.append(i)
    return out


def score_case_keyword(findings: list[dict], labels: dict) -> dict:
    expected = labels.get("expected", [])
    remaining = list(findings)
    hits = 0
    by_cat_hits: dict[str, int] = {}
    by_cat_expected: dict[str, int] = {}

    for exp in expected:
        cat = exp.get("category", "uncategorized")
        by_cat_expected[cat] = by_cat_expected.get(cat, 0) + 1
        kws = [k.lower() for k in exp.get("keywords", [])]
        match_idx = None
        for i in _candidate_indices_in_range(remaining, exp):
            if any(k in _haystack(remaining[i]) for k in kws):
                match_idx = i
                break
        if match_idx is not None:
            hits += 1
            by_cat_hits[cat] = by_cat_hits.get(cat, 0) + 1
            remaining.pop(match_idx)

    return {
        "expected": len(expected),
        "hits": hits,
        "findings_emitted": len(findings),
        "extras": len(remaining),
        "by_category_hits": by_cat_hits,
        "by_category_expected": by_cat_expected,
    }


def _parse_judge_json(content: str) -> dict:
    content = (content or "").strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(content[start:end + 1])
            except json.JSONDecodeError:
                return {}
        return {}


def _judge_match(client, judge_model: str, exp: dict, cand: dict) -> bool:
    user = JUDGE_USER_TEMPLATE.format(
        path=exp.get("path", ""),
        lo=exp["line_range"][0],
        hi=exp["line_range"][1],
        category=exp.get("category", ""),
        severity=exp.get("severity", ""),
        keywords=", ".join(exp.get("keywords", [])) or "(none)",
        title=cand.get("title", ""),
        explanation=cand.get("explanation", ""),
        line=cand.get("line", ""),
    )
    try:
        resp = client.complete(
            model=judge_model,
            messages=[SystemMessage(JUDGE_SYSTEM), UserMessage(user)],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
    except (TypeError, ValueError):
        resp = client.complete(
            model=judge_model,
            messages=[SystemMessage(JUDGE_SYSTEM), UserMessage(user)],
            temperature=0.0,
        )
    data = _parse_judge_json(resp.choices[0].message.content or "")
    return bool(data.get("match", False))


def score_case_judge(
    findings: list[dict],
    labels: dict,
    *,
    judge_model: str,
    client,
) -> dict:
    expected = labels.get("expected", [])
    remaining = list(findings)
    hits = 0
    by_cat_hits: dict[str, int] = {}
    by_cat_expected: dict[str, int] = {}

    for exp in expected:
        cat = exp.get("category", "uncategorized")
        by_cat_expected[cat] = by_cat_expected.get(cat, 0) + 1
        match_idx = None
        for i in _candidate_indices_in_range(remaining, exp):
            if _judge_match(client, judge_model, exp, remaining[i]):
                match_idx = i
                break
        if match_idx is not None:
            hits += 1
            by_cat_hits[cat] = by_cat_hits.get(cat, 0) + 1
            remaining.pop(match_idx)

    return {
        "expected": len(expected),
        "hits": hits,
        "findings_emitted": len(findings),
        "extras": len(remaining),
        "by_category_hits": by_cat_hits,
        "by_category_expected": by_cat_expected,
    }
