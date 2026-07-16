"""Tests for reviewer.coverage — cobertura + lcov + new-code gate."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reviewer.coverage import (
    CoverageReport,
    coverage_gate_finding,
    load_coverage,
    new_code_coverage,
    parse_cobertura,
    parse_lcov,
)


COBERTURA_XML = """<?xml version="1.0" ?>
<coverage>
  <packages>
    <package>
      <classes>
        <class filename="svc/a.py">
          <lines>
            <line number="1" hits="3"/>
            <line number="2" hits="0"/>
            <line number="3" hits="1"/>
          </lines>
        </class>
        <class filename="svc/b.py">
          <lines>
            <line number="10" hits="0"/>
            <line number="11" hits="5"/>
          </lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>
"""


LCOV_TEXT = """SF:web/api.ts
DA:1,3
DA:2,0
DA:3,1
end_of_record
SF:web/util.ts
DA:10,0
DA:11,5
end_of_record
"""


DIFF_PY = """diff --git a/svc/a.py b/svc/a.py
--- /dev/null
+++ b/svc/a.py
@@ -0,0 +1,3 @@
+def x():
+    y = 1
+    return y
"""


DIFF_TS = """diff --git a/web/api.ts b/web/api.ts
--- /dev/null
+++ b/web/api.ts
@@ -0,0 +1,3 @@
+function fetchUser() {
+    return 1;
+}
"""


# ---- parse_cobertura -----------------------------------------------------

def test_parse_cobertura_splits_covered_and_uncovered():
    r = parse_cobertura(COBERTURA_XML)
    assert r.covered == {"svc/a.py": {1, 3}, "svc/b.py": {11}}
    assert r.uncovered == {"svc/a.py": {2}, "svc/b.py": {10}}


def test_parse_cobertura_is_line_covered_helper():
    r = parse_cobertura(COBERTURA_XML)
    assert r.is_line_covered("svc/a.py", 1) is True
    assert r.is_line_covered("svc/a.py", 2) is False
    assert r.is_line_covered("svc/a.py", 999) is None  # not instrumented


def test_parse_cobertura_normalizes_windows_slashes():
    xml = COBERTURA_XML.replace("svc/a.py", "svc\\a.py")
    r = parse_cobertura(xml)
    assert "svc/a.py" in r.covered


# ---- parse_lcov ----------------------------------------------------------

def test_parse_lcov_splits_covered_and_uncovered():
    r = parse_lcov(LCOV_TEXT)
    assert r.covered == {"web/api.ts": {1, 3}, "web/util.ts": {11}}
    assert r.uncovered == {"web/api.ts": {2}, "web/util.ts": {10}}


def test_parse_lcov_ignores_unknown_directives():
    text = "SF:a.js\nBRDA:1,0,0,1\nDA:1,1\nend_of_record\n"
    r = parse_lcov(text)
    assert r.covered == {"a.js": {1}}


# ---- load_coverage -------------------------------------------------------

def test_load_coverage_reads_cobertura(tmp_path):
    p = tmp_path / "coverage.xml"
    p.write_text(COBERTURA_XML, encoding="utf-8")
    r = load_coverage(p)
    assert r is not None
    assert "svc/a.py" in r.covered


def test_load_coverage_reads_lcov(tmp_path):
    p = tmp_path / "lcov.info"
    p.write_text(LCOV_TEXT, encoding="utf-8")
    r = load_coverage(p)
    assert r is not None
    assert "web/api.ts" in r.covered


def test_load_coverage_returns_none_on_missing_file(tmp_path):
    assert load_coverage(tmp_path / "nope.xml") is None


def test_load_coverage_returns_none_on_bad_xml(tmp_path, capsys):
    p = tmp_path / "coverage.xml"
    p.write_text("<not really xml>", encoding="utf-8")
    assert load_coverage(p) is None
    assert "could not parse" in capsys.readouterr().err.lower()


# ---- new_code_coverage --------------------------------------------------

def test_new_code_coverage_measures_added_lines_only():
    report = parse_cobertura(COBERTURA_XML)
    ncc = new_code_coverage(DIFF_PY, report)
    # Diff added lines 1,2,3 in svc/a.py — 2 covered, 1 uncovered.
    assert ncc.total_instrumented == 3
    assert ncc.covered == 2
    assert ncc.uncovered_lines == {"svc/a.py": [2]}
    assert 0.66 < ncc.ratio < 0.67


def test_new_code_coverage_zero_when_no_matching_file():
    """If the coverage report doesn't cover the changed file, we return 0/0."""
    report = parse_cobertura(COBERTURA_XML)
    ncc = new_code_coverage(DIFF_TS, report)  # TS file, cobertura is python-only
    assert ncc.total_instrumented == 0
    assert ncc.ratio == 1.0  # by convention, nothing to cover -> pass


def test_new_code_coverage_fuzzy_path_matches_absolute_report():
    """CI coverage reports often have absolute paths; the diff has relative."""
    xml = COBERTURA_XML.replace(
        "svc/a.py", "/home/runner/work/repo/svc/a.py",
    )
    report = parse_cobertura(xml)
    ncc = new_code_coverage(DIFF_PY, report)
    assert ncc.total_instrumented == 3
    assert ncc.covered == 2


def test_new_code_coverage_ignores_docs_and_config():
    """A docs-only PR shouldn't fail the coverage gate."""
    diff = """diff --git a/README.md b/README.md
--- /dev/null
+++ b/README.md
@@ -0,0 +1,2 @@
+# Hello
+text
"""
    report = parse_cobertura(COBERTURA_XML)
    ncc = new_code_coverage(diff, report)
    assert ncc.total_instrumented == 0


# ---- coverage_gate_finding ---------------------------------------------

def test_coverage_gate_fires_below_threshold():
    report = parse_cobertura(COBERTURA_XML)
    ncc = new_code_coverage(DIFF_PY, report)  # 2/3 = 67%
    finding = coverage_gate_finding(ncc, threshold=0.80)
    assert finding is not None
    assert finding["source"] == "pr_review"
    assert finding["tool"] == "coverage-gate"
    assert "67%" in finding["title"]
    assert "svc/a.py" in finding["explanation"]


def test_coverage_gate_silent_above_threshold():
    report = parse_cobertura(COBERTURA_XML)
    ncc = new_code_coverage(DIFF_PY, report)  # 67%
    assert coverage_gate_finding(ncc, threshold=0.5) is None


def test_coverage_gate_silent_when_nothing_instrumented():
    """No matching lines -> nothing to gate."""
    ncc = new_code_coverage(DIFF_TS, parse_cobertura(COBERTURA_XML))
    assert coverage_gate_finding(ncc, threshold=0.99) is None


def test_coverage_gate_severity_scales_with_gap():
    ncc_large_gap = new_code_coverage(DIFF_PY, parse_cobertura(COBERTURA_XML))
    # 67% coverage, threshold 90% -> gap 23% -> high severity
    finding = coverage_gate_finding(ncc_large_gap, threshold=0.90)
    assert finding["severity"] == "high"

    # 67% coverage, threshold 72% -> gap 5% -> low severity
    finding_small = coverage_gate_finding(ncc_large_gap, threshold=0.72)
    assert finding_small["severity"] == "low"
