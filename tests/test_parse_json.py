import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reviewer.reviewer import _parse_json


def test_clean_json():
    assert _parse_json('{"findings": []}') == {"findings": []}


def test_strips_plain_code_fence():
    assert _parse_json('```\n{"findings": []}\n```') == {"findings": []}


def test_strips_json_tagged_fence():
    assert _parse_json('```json\n{"findings": [{"line": 1}]}\n```') == {
        "findings": [{"line": 1}]
    }


def test_recovers_from_leading_prose():
    raw = 'Sure! Here is your response:\n{"findings": [{"line": 7}]}'
    assert _parse_json(raw) == {"findings": [{"line": 7}]}


def test_garbage_returns_empty_dict():
    assert _parse_json("not json at all") == {}


def test_braces_with_invalid_inner_returns_empty_dict():
    assert _parse_json("prose { not valid } more prose") == {}


def test_empty_string_returns_empty_dict():
    assert _parse_json("") == {}
