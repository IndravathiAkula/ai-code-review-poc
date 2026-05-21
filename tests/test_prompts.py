"""Tests for the system-prompt builder + per-language rule extras."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reviewer.prompts import SYSTEM, build_system_prompt


def test_build_system_prompt_returns_base_when_no_extras():
    assert build_system_prompt("python", None) == SYSTEM
    assert build_system_prompt("python", {}) == SYSTEM


def test_build_system_prompt_returns_base_when_language_not_in_map():
    extras = {"python": "- Mutable default args are bugs."}
    assert build_system_prompt("typescript", extras) == SYSTEM


def test_build_system_prompt_appends_matching_language_block():
    extras = {"typescript": "- useEffect missing deps is a bug.\n- any in public API is a finding."}
    out = build_system_prompt("typescript", extras)
    assert out.startswith(SYSTEM)
    assert "Additional rules for typescript code:" in out
    assert "useEffect missing deps is a bug." in out
    assert "any in public API is a finding." in out


def test_build_system_prompt_ignores_empty_extras_block():
    """Empty or whitespace-only string for a language should not append a
    header — keeps the rendered prompt clean when YAML accidentally has
    blank values."""
    assert build_system_prompt("python", {"python": ""}) == SYSTEM
    assert build_system_prompt("python", {"python": "   \n\n  "}) == SYSTEM


def test_build_system_prompt_trims_trailing_whitespace():
    out = build_system_prompt(
        "go", {"go": "- panic in init is a finding.\n\n\n"})
    # Should end on the rule line, not the trailing blank lines.
    assert out.endswith("panic in init is a finding.")


def test_build_system_prompt_preserves_base_intact():
    """Appending extras should never modify the original SYSTEM constant."""
    before = SYSTEM
    build_system_prompt("python", {"python": "- foo"})
    assert SYSTEM == before
