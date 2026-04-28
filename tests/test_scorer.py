import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from scorer import score_case_keyword, score_case_judge


# ---------- keyword scorer ----------

LABELS = {
    "expected": [
        {"path": "a.py", "line_range": [1, 5],
         "category": "security", "severity": "critical",
         "keywords": ["sql injection", "sqli", "parameterized"]},
        {"path": "a.py", "line_range": [10, 12],
         "category": "correctness", "severity": "medium",
         "keywords": ["zero", "zerodivisionerror"]},
    ]
}


def test_keyword_scorer_full_match():
    findings = [
        {"path": "a.py", "line": 3, "title": "SQL injection",
         "explanation": "string concat sql"},
        {"path": "a.py", "line": 11, "title": "div",
         "explanation": "ZeroDivisionError when b is zero"},
    ]
    r = score_case_keyword(findings, LABELS)
    assert r["expected"] == 2
    assert r["hits"] == 2
    assert r["extras"] == 0
    assert r["by_category_hits"] == {"security": 1, "correctness": 1}
    assert r["by_category_expected"] == {"security": 1, "correctness": 1}


def test_keyword_scorer_misses_when_keyword_absent():
    findings = [{
        "path": "a.py", "line": 3,
        "title": "Use prepared statements",
        "explanation": "user input flows into query as raw string",
    }]
    r = score_case_keyword(findings, LABELS)
    assert r["hits"] == 0
    assert r["extras"] == 1
    assert r["by_category_hits"] == {}


def test_keyword_scorer_rejects_finding_outside_line_range():
    findings = [{
        "path": "a.py", "line": 99,
        "title": "SQL injection here",
        "explanation": "...",
    }]
    r = score_case_keyword(findings, LABELS)
    assert r["hits"] == 0


def test_keyword_scorer_rejects_finding_on_wrong_path():
    findings = [{
        "path": "b.py", "line": 3,
        "title": "SQL injection",
        "explanation": "sqli",
    }]
    r = score_case_keyword(findings, LABELS)
    assert r["hits"] == 0


def test_keyword_scorer_does_not_double_count_one_finding_for_two_expected():
    """If a single finding could match two expected entries, it counts once."""
    labels = {"expected": [
        {"path": "a.py", "line_range": [1, 10], "category": "security",
         "keywords": ["sqli"]},
        {"path": "a.py", "line_range": [1, 10], "category": "security",
         "keywords": ["sqli"]},
    ]}
    findings = [
        {"path": "a.py", "line": 3, "title": "sqli", "explanation": "..."},
    ]
    r = score_case_keyword(findings, labels)
    assert r["hits"] == 1
    assert r["expected"] == 2


# ---------- judge scorer ----------

class _JudgeMessage:
    def __init__(self, content):
        self.content = content


class _JudgeChoice:
    def __init__(self, content):
        self.message = _JudgeMessage(content)


class _JudgeResponse:
    def __init__(self, content):
        self.choices = [_JudgeChoice(content)]
        self.usage = None


class JudgeClient:
    """Returns a scripted match/no-match per call. Used to test that the
    judge scorer wires findings to the model and respects its verdict."""

    def __init__(self, verdicts):
        self._verdicts = list(verdicts)
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        verdict = self._verdicts.pop(0) if self._verdicts else False
        return _JudgeResponse(json.dumps({"match": verdict}))


def test_judge_scorer_match_when_judge_says_true():
    findings = [{
        "path": "a.py", "line": 3,
        "title": "Prepared statements", "explanation": "user input concat",
    }]
    client = JudgeClient(verdicts=[True])
    r = score_case_judge(findings, LABELS, judge_model="judge-x", client=client)
    assert r["hits"] == 1
    assert r["by_category_hits"] == {"security": 1}
    assert len(client.calls) == 1


def test_judge_scorer_no_match_when_judge_says_false():
    findings = [{
        "path": "a.py", "line": 3,
        "title": "irrelevant", "explanation": "...",
    }]
    client = JudgeClient(verdicts=[False])
    r = score_case_judge(findings, LABELS, judge_model="judge-x", client=client)
    assert r["hits"] == 0
    assert r["extras"] == 1


def test_judge_scorer_skips_findings_outside_line_range_without_calling_judge():
    """Pre-filter avoids spending judge tokens on out-of-range findings."""
    findings = [{
        "path": "a.py", "line": 99,
        "title": "Looks bad", "explanation": "...",
    }]
    client = JudgeClient(verdicts=[True])  # would say True if asked
    r = score_case_judge(findings, LABELS, judge_model="judge-x", client=client)
    assert r["hits"] == 0
    assert client.calls == []  # judge never invoked


def test_judge_scorer_stops_at_first_match_for_each_expected():
    """Once an expected is matched, the judge isn't asked again for it."""
    findings = [
        {"path": "a.py", "line": 2, "title": "first", "explanation": "..."},
        {"path": "a.py", "line": 3, "title": "second", "explanation": "..."},
    ]
    # Verdicts: first call (sqli expected vs first finding) → True.
    # Then divide-by-zero expected has no in-range candidates (lines 10-12).
    client = JudgeClient(verdicts=[True])
    r = score_case_judge(findings, LABELS, judge_model="judge-x", client=client)
    assert r["hits"] == 1
    assert len(client.calls) == 1


def test_judge_scorer_includes_keywords_in_prompt():
    findings = [{
        "path": "a.py", "line": 11,
        "title": "div", "explanation": "...",
    }]
    client = JudgeClient(verdicts=[True])
    score_case_judge(findings, LABELS, judge_model="judge-x", client=client)
    user_msg = client.calls[0]["messages"][1].content
    assert "zerodivisionerror" in user_msg.lower()
    assert "a.py" in user_msg
