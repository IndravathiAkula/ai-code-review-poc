"""Coverage-diff quality gate — Sonar's "new code coverage" behaviour.

Read a coverage report from disk, overlay it onto the lines the PR
actually added, compute the % of new lines that are exercised, and
emit a PR-level finding when it falls below a threshold.

Supported formats:
- **Cobertura XML** (``coverage.xml`` — Python ``coverage``,
  JavaScript ``nyc``, Java Jacoco via cover2cover, etc.)
- **LCOV** (``lcov.info`` — JavaScript Jest, TypeScript ts-jest, etc.)

Both formats give per-line "hits" counts. A line with hits > 0 is
covered; hits == 0 (or absent) means uncovered.

Failure mode: if the file doesn't exist / can't be parsed, we return
``None`` and log — the reviewer still ships without the gate rather
than blocking CI on a missing artifact.
"""
from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from unidiff import PatchSet


@dataclass
class CoverageReport:
    """Which lines each file has recorded hits for.

    ``covered`` and ``uncovered`` are complementary sets: covered has
    hits>0, uncovered was in the coverage report with hits==0. Lines
    not in either set weren't instrumented (comments, blank lines,
    class-level statements the collector skipped) and don't count for
    or against the ratio.
    """
    covered: dict[str, set[int]]
    uncovered: dict[str, set[int]]

    def is_line_covered(self, path: str, line: int) -> bool | None:
        norm = path.replace("\\", "/")
        if norm in self.covered and line in self.covered[norm]:
            return True
        if norm in self.uncovered and line in self.uncovered[norm]:
            return False
        return None  # not instrumented — outside the coverage calculation

    def known_paths(self) -> set[str]:
        return set(self.covered) | set(self.uncovered)


def parse_cobertura(xml_text: str) -> CoverageReport:
    """Parse a Cobertura ``coverage.xml`` payload.

    Cobertura shape (simplified):

        <coverage>
          <packages>
            <package>
              <classes>
                <class filename="src/a.py">
                  <lines>
                    <line number="1" hits="3"/>
                    <line number="2" hits="0"/>
                  </lines>
                </class>
              </classes>
            </package>
          </packages>
        </coverage>
    """
    covered: dict[str, set[int]] = {}
    uncovered: dict[str, set[int]] = {}
    root = ET.fromstring(xml_text)
    for cls in root.iter("class"):
        filename = cls.get("filename", "").replace("\\", "/")
        if not filename:
            continue
        for line in cls.iter("line"):
            try:
                ln = int(line.get("number", 0))
                hits = int(line.get("hits", 0))
            except (TypeError, ValueError):
                continue
            if not ln:
                continue
            bucket = covered if hits > 0 else uncovered
            bucket.setdefault(filename, set()).add(ln)
    return CoverageReport(covered=covered, uncovered=uncovered)


_LCOV_SF = re.compile(r"^SF:(.+)$")
_LCOV_DA = re.compile(r"^DA:(\d+),(\d+)")


def parse_lcov(text: str) -> CoverageReport:
    """Parse ``lcov.info``.

    LCOV format is one record per file, records separated by ``end_of_record``:

        SF:src/a.js
        DA:1,3
        DA:2,0
        end_of_record

    Only ``SF`` (source file) and ``DA`` (hits per line) are needed for
    a new-code coverage gate.
    """
    covered: dict[str, set[int]] = {}
    uncovered: dict[str, set[int]] = {}
    current_file: str | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line == "end_of_record":
            current_file = None
            continue
        m_sf = _LCOV_SF.match(line)
        if m_sf:
            current_file = m_sf.group(1).replace("\\", "/")
            continue
        if current_file is None:
            continue
        m_da = _LCOV_DA.match(line)
        if not m_da:
            continue
        try:
            ln = int(m_da.group(1))
            hits = int(m_da.group(2))
        except ValueError:
            continue
        bucket = covered if hits > 0 else uncovered
        bucket.setdefault(current_file, set()).add(ln)
    return CoverageReport(covered=covered, uncovered=uncovered)


def load_coverage(path: str | Path) -> CoverageReport | None:
    """Read + auto-detect a coverage file. ``None`` on any failure."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"[warn] coverage: could not read {p}: {e}", file=sys.stderr)
        return None
    stripped = text.lstrip()
    try:
        if stripped.startswith("<"):
            return parse_cobertura(text)
        return parse_lcov(text)
    except (ET.ParseError, ValueError) as e:
        print(f"[warn] coverage: could not parse {p}: {e}", file=sys.stderr)
        return None


def _added_code_lines_by_path(diff_text: str) -> dict[str, set[int]]:
    """Return added-line sets, but only for non-doc non-config paths.

    Coverage on a README doesn't make sense; measuring new-code coverage
    across ``package-lock.json`` diffs would explode. Use the same path
    classifiers as the missing-tests heuristic to focus on real code.
    """
    from .utils import is_config_path, is_doc_path
    out: dict[str, set[int]] = {}
    for pf in PatchSet(diff_text):
        if pf.is_binary_file:
            continue
        path = pf.path
        if is_doc_path(path) or is_config_path(path):
            continue
        lines: set[int] = set()
        for hunk in pf:
            for line in hunk:
                if line.is_added and line.target_line_no is not None:
                    lines.add(line.target_line_no)
        if lines:
            out[path] = lines
    return out


@dataclass
class NewCodeCoverage:
    """Result of overlaying a coverage report on the PR's added lines."""
    total_instrumented: int   # added lines that were instrumented
    covered: int              # ... and had hits > 0
    uncovered_lines: dict[str, list[int]]  # path -> sorted lines with hits=0

    @property
    def ratio(self) -> float:
        if self.total_instrumented == 0:
            return 1.0  # nothing to cover -> not a failure
        return self.covered / self.total_instrumented


def new_code_coverage(
    diff_text: str, report: CoverageReport,
) -> NewCodeCoverage:
    """Compute % of the PR's added code lines that have coverage hits."""
    added = _added_code_lines_by_path(diff_text)
    total = 0
    covered_count = 0
    uncovered_lines: dict[str, list[int]] = {}
    for path, lines in added.items():
        # Try exact path match first; then try matching by basename or
        # by suffix, since coverage tools sometimes emit paths relative
        # to a different root than the diff (e.g. ``src/a.py`` vs.
        # ``/home/runner/work/repo/src/a.py``).
        report_lines_covered = report.covered.get(path)
        report_lines_uncovered = report.uncovered.get(path)
        if report_lines_covered is None and report_lines_uncovered is None:
            match = _fuzzy_match_path(path, report.known_paths())
            if match is not None:
                report_lines_covered = report.covered.get(match)
                report_lines_uncovered = report.uncovered.get(match)
        if report_lines_covered is None and report_lines_uncovered is None:
            continue
        for ln in sorted(lines):
            in_c = report_lines_covered is not None and ln in report_lines_covered
            in_u = report_lines_uncovered is not None and ln in report_lines_uncovered
            if not (in_c or in_u):
                continue
            total += 1
            if in_c:
                covered_count += 1
            else:
                uncovered_lines.setdefault(path, []).append(ln)
    return NewCodeCoverage(
        total_instrumented=total,
        covered=covered_count,
        uncovered_lines=uncovered_lines,
    )


def _fuzzy_match_path(diff_path: str, report_paths: set[str]) -> str | None:
    """Find the coverage entry that best matches a diff path.

    Coverage tools emit paths inconsistently — absolute in CI, relative
    to a subdir in local, sometimes with a ``./`` prefix. Try suffix match
    (report path ends with the diff path) which handles the common cases.
    """
    diff_norm = diff_path.replace("\\", "/")
    # Prefer the longest suffix match to avoid false positives on short paths.
    best: str | None = None
    for rp in report_paths:
        rp_norm = rp.replace("\\", "/")
        if rp_norm.endswith("/" + diff_norm) or rp_norm.endswith(diff_norm):
            if best is None or len(rp_norm) > len(best):
                best = rp
    return best


def coverage_gate_finding(
    ncc: NewCodeCoverage, threshold: float,
) -> dict | None:
    """Emit a PR-level finding when new-code coverage is below threshold.

    Returns ``None`` when the gate passes or when nothing new was
    instrumented. Severity scales with how far the ratio missed the bar.
    """
    if ncc.total_instrumented == 0:
        return None
    if ncc.ratio >= threshold:
        return None
    # Severity: >20 points below threshold = high; within 10 = medium; else low.
    gap = threshold - ncc.ratio
    if gap >= 0.20:
        severity = "high"
    elif gap >= 0.10:
        severity = "medium"
    else:
        severity = "low"
    top_files = sorted(
        ncc.uncovered_lines.items(),
        key=lambda kv: -len(kv[1]),
    )[:5]
    file_bullets = "\n".join(
        f"  - {p}: {len(lines)} uncovered line(s) ({', '.join(str(x) for x in lines[:8])}"
        f"{'…' if len(lines) > 8 else ''})"
        for p, lines in top_files
    )
    return {
        "path": "(pull request)",
        "line": 0,
        "severity": severity,
        "category": "maintainability",
        "title": (
            f"New-code coverage {ncc.ratio:.0%} below threshold "
            f"{threshold:.0%} ({ncc.covered}/{ncc.total_instrumented} lines)"
        ),
        "explanation": (
            f"This PR added {ncc.total_instrumented} instrumented line(s), "
            f"of which {ncc.covered} are exercised by tests "
            f"({ncc.ratio:.1%}). That's below the configured threshold of "
            f"{threshold:.0%}. Files with the most uncovered new code:\n"
            f"{file_bullets}"
        ),
        "suggested_fix": None,
        "confidence": 1.0,
        "source": "pr_review",
        "tool": "coverage-gate",
    }
