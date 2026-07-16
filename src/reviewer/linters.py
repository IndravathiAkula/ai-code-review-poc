"""Deterministic linters and type-checkers that run alongside the LLM.

Each tool is invoked as a subprocess against the files touched by the
PR, findings are reshaped into the same dict schema the LLM emits, and
filtered to lines that were actually **added** in the diff (never
context lines — we don't comment on unchanged code).

Missing tools skip silently: repos without eslint/mypy/tsc keep working
with just ruff (and vice versa). ruff is a hard dep in requirements.txt
so Python lint is always available in CI.

Finding schema addition: every finding here carries ``source`` (one of
``"lint"`` / ``"type"``) and ``tool`` (``ruff``/``eslint``/``mypy``/
``tsc``) so ``format_comment`` can tag the PR comment accordingly.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from unidiff import PatchSet


_SUBPROCESS_TIMEOUT = 90  # seconds — kill any linter that stalls


def added_lines_by_path(diff_text: str) -> dict[str, set[int]]:
    """Return ``{path: {line, ...}}`` of NEW-file lines *added* by the diff.

    Only ``+`` lines — context lines are excluded so lint findings on
    unchanged code don't get posted as new PR comments.
    """
    out: dict[str, set[int]] = {}
    for pf in PatchSet(diff_text):
        if pf.is_binary_file:
            continue
        lines: set[int] = set()
        for hunk in pf:
            for line in hunk:
                if line.is_added and line.target_line_no is not None:
                    lines.add(line.target_line_no)
        if lines:
            out[pf.path] = lines
    return out


def _files_by_ext(paths: Iterable[str], exts: tuple[str, ...]) -> list[str]:
    return [p for p in paths if p.lower().endswith(exts)]


def _has_any(root: Path, candidates: tuple[str, ...]) -> bool:
    return any((root / c).exists() for c in candidates)


def _run(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    """Run ``cmd`` with a 90s timeout. ``(rc, stdout, stderr)``.

    ``rc == 127`` means the binary wasn't found; ``124`` means timed out.
    """
    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True,
            timeout=_SUBPROCESS_TIMEOUT, check=False,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except FileNotFoundError:
        return 127, "", f"{cmd[0]}: not found"
    except subprocess.TimeoutExpired:
        return 124, "", f"{cmd[0]}: timed out after {_SUBPROCESS_TIMEOUT}s"


def _normalize_path(p: str, root: Path) -> str:
    """Make ``p`` relative to ``root`` with forward slashes.

    Falls back to the raw string (slashes normalized) when the path
    can't be resolved — happens on absolute paths pointing outside the
    workspace, which we surface as-is rather than dropping the finding.
    """
    if not p:
        return p
    try:
        rel = Path(p).resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        return p.replace("\\", "/")
    return str(rel).replace("\\", "/")


def _filter_to_added(
    findings: list[dict], added: dict[str, set[int]],
) -> list[dict]:
    """Drop any finding not anchored to a line the PR actually added."""
    out: list[dict] = []
    for f in findings:
        path = f.get("path")
        try:
            line = int(f.get("line", 0))
        except (TypeError, ValueError):
            continue
        if path in added and line in added[path]:
            out.append(f)
    return out


# ---- Python: ruff (lint) --------------------------------------------------

def run_ruff(
    py_files: list[str], added: dict[str, set[int]], root: Path,
) -> list[dict]:
    if not py_files or not shutil.which("ruff"):
        return []
    rc, out, err = _run(
        ["ruff", "check", "--output-format=json", "--no-fix", *py_files],
        cwd=root,
    )
    # 0 = clean, 1 = findings found — both are success. Anything else is a
    # tool error (bad config, unparseable file, install mismatch).
    if rc not in (0, 1):
        print(f"[warn] ruff: {err.strip() or f'exited {rc}'}", file=sys.stderr)
        return []
    if not out.strip():
        return []
    try:
        raw = json.loads(out)
    except json.JSONDecodeError as e:
        print(f"[warn] ruff: could not parse json ({e})", file=sys.stderr)
        return []
    findings: list[dict] = []
    for item in raw:
        loc = item.get("location") or {}
        try:
            line = int(loc.get("row", 0) or 0)
        except (TypeError, ValueError):
            continue
        if not line:
            continue
        code = item.get("code") or "ruff"
        message = (item.get("message") or "").strip()
        fix = item.get("fix") or {}
        findings.append({
            "path": _normalize_path(item.get("filename", ""), root),
            "line": line,
            "severity": "low",
            "category": "maintainability",
            "title": f"{code}: {message[:120]}",
            "explanation": message or code,
            "suggested_fix": fix.get("message"),
            "confidence": 0.99,
            "source": "lint",
            "tool": "ruff",
        })
    return _filter_to_added(findings, added)


# ---- Python: bandit (security SAST) --------------------------------------

# Bandit severity → our severity. Bandit uses HIGH/MEDIUM/LOW plus a
# confidence axis; we treat HIGH-severity-HIGH-confidence as ``high``,
# demote everything else so hotspots don't drown out real bugs.
_BANDIT_SEV_MAP = {"HIGH": "high", "MEDIUM": "medium", "LOW": "low"}


def run_bandit(
    py_files: list[str], added: dict[str, set[int]], root: Path,
) -> list[dict]:
    """Run bandit and map results to our finding schema.

    Bandit emits security-relevant findings (subprocess with shell=True,
    yaml.load on untrusted input, weak crypto, etc.) — these map to
    ``security`` or ``security_hotspot`` depending on confidence:
    HIGH-confidence → ``security``; MEDIUM/LOW → ``security_hotspot``,
    which is the "review this, it might be fine" bucket the SonarQube
    taxonomy uses.
    """
    if not py_files or not shutil.which("bandit"):
        return []
    rc, out, err = _run(
        ["bandit", "-f", "json", "-q", *py_files],
        cwd=root,
    )
    # bandit rc: 0 = clean, 1 = findings, other = tool error.
    if rc not in (0, 1):
        print(f"[warn] bandit: {err.strip() or f'exited {rc}'}", file=sys.stderr)
        return []
    if not out.strip():
        return []
    try:
        data = json.loads(out)
    except json.JSONDecodeError as e:
        print(f"[warn] bandit: could not parse json ({e})", file=sys.stderr)
        return []
    findings: list[dict] = []
    for item in data.get("results", []) or []:
        try:
            line = int(item.get("line_number", 0) or 0)
        except (TypeError, ValueError):
            continue
        if not line:
            continue
        raw_sev = str(item.get("issue_severity", "MEDIUM")).upper()
        raw_conf = str(item.get("issue_confidence", "MEDIUM")).upper()
        severity = _BANDIT_SEV_MAP.get(raw_sev, "medium")
        # Confidence axis decides real-vuln vs. hotspot: HIGH-conf =
        # bandit is sure this pattern is dangerous → security. Anything
        # else = hotspot for a human to eyeball.
        category = "security" if raw_conf == "HIGH" else "security_hotspot"
        test_id = item.get("test_id") or "bandit"
        message = (item.get("issue_text") or "").strip()
        cwe = (item.get("issue_cwe") or {}).get("id")
        cwe_suffix = f" (CWE-{cwe})" if cwe else ""
        findings.append({
            "path": _normalize_path(item.get("filename", ""), root),
            "line": line,
            "severity": severity,
            "category": category,
            "title": f"{test_id}: {message[:120]}{cwe_suffix}",
            "explanation": message or test_id,
            "suggested_fix": None,
            # Bandit's own confidence is a coarse label — surface it as a
            # numeric so downstream filtering behaves. HIGH→0.9, MED→0.7,
            # LOW→0.5. Findings below the reviewer's min_confidence get
            # dropped by the shared filter.
            "confidence": {"HIGH": 0.9, "MEDIUM": 0.7, "LOW": 0.5}.get(
                raw_conf, 0.7),
            "source": "lint",
            "tool": "bandit",
        })
    return _filter_to_added(findings, added)


# ---- Multi-language: semgrep (SAST + best practices) ---------------------

# Semgrep severity → our severity. Semgrep emits ERROR / WARNING / INFO.
_SEMGREP_SEV_MAP = {
    "ERROR": "high",
    "WARNING": "medium",
    "INFO": "low",
}

# Semgrep rule metadata often carries a ``category`` like ``security`` /
# ``best-practice`` / ``correctness``. Map to our taxonomy.
_SEMGREP_CATEGORY_MAP = {
    "security": "security",
    "correctness": "correctness",
    "performance": "performance",
    "best-practice": "maintainability",
    "maintainability": "maintainability",
}


def run_semgrep(
    files: list[str], added: dict[str, set[int]], root: Path,
) -> list[dict]:
    """Run semgrep with its auto-config and reshape findings.

    Semgrep is the industry-standard multi-language SAST engine — it
    ships thousands of rules for security + correctness + best-practices
    across ~30 languages. We invoke it via ``semgrep scan --json`` and
    map results into our finding schema. Skipped gracefully when
    semgrep isn't installed.

    Confidence axis: semgrep rules don't emit a confidence — we use the
    rule's severity to derive one (ERROR=0.9, WARNING=0.8, INFO=0.6).
    """
    if not files or not shutil.which("semgrep"):
        return []
    # ``--config auto`` uses the semgrep registry (needs network). Users
    # who set up their own ruleset get honoured via ``.semgrep.yml`` /
    # ``.semgrepignore`` in the repo root — semgrep picks those up
    # automatically when no --config is passed. Try --config auto first,
    # and if it errors (no network), fall back to the bare invocation
    # which uses local config if present.
    for cmd_variant in (
        ["semgrep", "scan", "--json", "--quiet", "--config", "auto", *files],
        ["semgrep", "scan", "--json", "--quiet", *files],
    ):
        rc, out, err = _run(cmd_variant, cwd=root)
        # 0 = clean, 1 = findings found, 2 = tool error.
        if rc in (0, 1) and out.strip():
            break
        # try next variant on error/no-output
    else:
        return []
    if rc not in (0, 1):
        print(f"[warn] semgrep: {err.strip() or f'exited {rc}'}", file=sys.stderr)
        return []
    try:
        data = json.loads(out)
    except json.JSONDecodeError as e:
        print(f"[warn] semgrep: could not parse json ({e})", file=sys.stderr)
        return []
    findings: list[dict] = []
    for item in data.get("results", []) or []:
        try:
            line = int((item.get("start") or {}).get("line", 0) or 0)
        except (TypeError, ValueError):
            continue
        if not line:
            continue
        check_id = item.get("check_id") or "semgrep"
        extra = item.get("extra") or {}
        raw_sev = str(extra.get("severity", "WARNING")).upper()
        severity = _SEMGREP_SEV_MAP.get(raw_sev, "medium")
        confidence = {"ERROR": 0.9, "WARNING": 0.8, "INFO": 0.6}.get(raw_sev, 0.8)
        message = (extra.get("message") or "").strip()
        metadata = extra.get("metadata") or {}
        raw_cat = str(metadata.get("category", "")).lower()
        category = _SEMGREP_CATEGORY_MAP.get(raw_cat, "maintainability")
        # Security rules that emit as WARNING/INFO are hotspots, not real
        # vulns — same distinction bandit uses.
        if category == "security" and raw_sev != "ERROR":
            category = "security_hotspot"
        # Trim the check_id — semgrep IDs are long
        # (``python.lang.security.audit.dangerous-subprocess-use-audit``);
        # keep the last two segments for readability.
        short_id = ".".join(check_id.split(".")[-2:]) if "." in check_id else check_id
        # CWE / OWASP surfaced in the title when present.
        cwe = metadata.get("cwe")
        cwe_suffix = ""
        if isinstance(cwe, list) and cwe:
            cwe_suffix = f" ({cwe[0]})"
        elif isinstance(cwe, str) and cwe:
            cwe_suffix = f" ({cwe})"
        findings.append({
            "path": _normalize_path(item.get("path", ""), root),
            "line": line,
            "severity": severity,
            "category": category,
            "title": f"{short_id}: {message[:120]}{cwe_suffix}",
            "explanation": message or short_id,
            "suggested_fix": (extra.get("fix") or None),
            "confidence": confidence,
            "source": "lint",
            "tool": "semgrep",
        })
    return _filter_to_added(findings, added)


# ---- Python: mypy (types) -------------------------------------------------

def run_mypy(
    py_files: list[str], added: dict[str, set[int]], root: Path,
) -> list[dict]:
    if not py_files or not shutil.which("mypy"):
        return []
    # Only run when the repo actually configures mypy — otherwise the
    # default settings emit noise that has nothing to do with the PR.
    if not _has_any(root, (
        "mypy.ini", ".mypy.ini", "pyproject.toml", "setup.cfg",
    )):
        return []
    rc, out, err = _run(
        ["mypy", "--show-column-numbers", "--no-error-summary",
         "--hide-error-context", "--no-color-output",
         "--no-pretty", *py_files],
        cwd=root,
    )
    if rc not in (0, 1):
        print(f"[warn] mypy: {err.strip() or f'exited {rc}'}", file=sys.stderr)
        return []
    findings: list[dict] = []
    for line in out.splitlines():
        parsed = _parse_mypy_line(line, root)
        if parsed:
            findings.append(parsed)
    return _filter_to_added(findings, added)


_MYPY_LINE_RE = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+)(?::\d+)?: "
    r"(?P<sev>error|warning): (?P<msg>.+?)$"
)


def _parse_mypy_line(line: str, root: Path) -> dict | None:
    """Parse ``path:line[:col]: severity: message [errcode]``."""
    m = _MYPY_LINE_RE.match(line)
    if not m:
        return None
    sev = m.group("sev")
    msg = m.group("msg").strip()
    return {
        "path": _normalize_path(m.group("path"), root),
        "line": int(m.group("line")),
        "severity": "medium" if sev == "error" else "low",
        "category": "correctness",
        "title": f"mypy: {msg[:120]}",
        "explanation": msg,
        "suggested_fix": None,
        "confidence": 0.9,
        "source": "type",
        "tool": "mypy",
    }


# ---- JS/TS: eslint (lint) -------------------------------------------------

def _npx_bin() -> str | None:
    """Locate ``npx`` on PATH (``npx.cmd`` on Windows)."""
    return shutil.which("npx") or shutil.which("npx.cmd")


_ESLINT_CONFIGS = (
    ".eslintrc", ".eslintrc.js", ".eslintrc.cjs", ".eslintrc.json",
    ".eslintrc.yaml", ".eslintrc.yml",
    "eslint.config.js", "eslint.config.mjs", "eslint.config.cjs",
    "eslint.config.ts",
)


def run_eslint(
    js_files: list[str], added: dict[str, set[int]], root: Path,
) -> list[dict]:
    if not js_files:
        return []
    if not _has_any(root, _ESLINT_CONFIGS):
        return []
    npx = _npx_bin()
    if not npx:
        return []
    rc, out, err = _run(
        [npx, "--no-install", "eslint", "-f", "json", *js_files],
        cwd=root,
    )
    # eslint: 0 = clean, 1 = lint problems, 2 = fatal (config error, etc.)
    if rc == 2:
        print(f"[warn] eslint: {err.strip() or 'exited 2'}", file=sys.stderr)
        return []
    if rc not in (0, 1) or not out.strip():
        return []
    try:
        raw = json.loads(out)
    except json.JSONDecodeError:
        return []
    findings: list[dict] = []
    for file_entry in raw:
        path = _normalize_path(file_entry.get("filePath", ""), root)
        for msg in file_entry.get("messages", []) or []:
            try:
                line = int(msg.get("line", 0) or 0)
            except (TypeError, ValueError):
                continue
            if not line:
                continue
            severity_num = int(msg.get("severity", 1) or 1)
            severity = "medium" if severity_num == 2 else "low"
            rule = msg.get("ruleId") or "eslint"
            text = (msg.get("message") or "").strip()
            fix = msg.get("fix") or {}
            findings.append({
                "path": path,
                "line": line,
                "severity": severity,
                "category": "maintainability",
                "title": f"{rule}: {text[:120]}",
                "explanation": text or rule,
                "suggested_fix": fix.get("text"),
                "confidence": 0.95,
                "source": "lint",
                "tool": "eslint",
            })
    return _filter_to_added(findings, added)


# ---- TypeScript: tsc (types) ---------------------------------------------

def run_tsc(
    ts_files: list[str], added: dict[str, set[int]], root: Path,
) -> list[dict]:
    if not ts_files:
        return []
    if not _has_any(root, ("tsconfig.json",)):
        return []
    npx = _npx_bin()
    if not npx:
        return []
    # tsc with tsconfig ignores explicit file args, so we type-check the
    # whole project and rely on _filter_to_added to trim to the PR.
    rc, out, err = _run(
        [npx, "--no-install", "tsc", "--noEmit", "--pretty", "false"],
        cwd=root,
    )
    if rc not in (0, 1, 2):
        # tsc exits nonzero on type errors; but rc=127 = npx couldn't find
        # tsc, and larger negatives = crash. Bail quietly.
        return []
    text = (out + "\n" + err)
    findings: list[dict] = []
    for line in text.splitlines():
        parsed = _parse_tsc_line(line, root)
        if parsed:
            findings.append(parsed)
    return _filter_to_added(findings, added)


_TSC_LINE_RE = re.compile(
    r"^(?P<path>.+?)\((?P<line>\d+),\d+\): "
    r"(?P<sev>error|warning) TS\d+: (?P<msg>.+?)$"
)


def _parse_tsc_line(line: str, root: Path) -> dict | None:
    """Parse ``path(line,col): error TSXXXX: message``."""
    m = _TSC_LINE_RE.match(line)
    if not m:
        return None
    sev = m.group("sev")
    msg = m.group("msg").strip()
    return {
        "path": _normalize_path(m.group("path"), root),
        "line": int(m.group("line")),
        "severity": "medium" if sev == "error" else "low",
        "category": "correctness",
        "title": f"tsc: {msg[:120]}",
        "explanation": msg,
        "suggested_fix": None,
        "confidence": 0.9,
        "source": "type",
        "tool": "tsc",
    }


# ---- Top-level entry -----------------------------------------------------

def run_all(
    diff_text: str,
    root: str | Path,
    *,
    enable_lint: bool = True,
    enable_type_check: bool = False,
    enable_semgrep: bool = False,
) -> list[dict]:
    """Run every applicable linter / type-checker on the PR diff.

    ``enable_type_check`` is opt-in because mypy/tsc are noticeably
    slower (mypy re-parses stubs, tsc walks the whole project) and can
    produce a lot of pre-existing noise that isn't PR-related. Lint
    findings are cheap and default-on.
    """
    root_path = Path(root)
    if not root_path.exists():
        return []
    added = added_lines_by_path(diff_text)
    if not added:
        return []
    all_paths = list(added.keys())
    py = _files_by_ext(all_paths, (".py",))
    js = _files_by_ext(all_paths, (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"))
    ts = _files_by_ext(all_paths, (".ts", ".tsx"))
    findings: list[dict] = []
    if enable_lint:
        findings += run_ruff(py, added, root_path)
        findings += run_bandit(py, added, root_path)
        findings += run_eslint(js, added, root_path)
    if enable_semgrep:
        # Semgrep works across ~30 languages; pass every changed file
        # and let semgrep decide what applies.
        findings += run_semgrep(all_paths, added, root_path)
    if enable_type_check:
        findings += run_mypy(py, added, root_path)
        findings += run_tsc(ts, added, root_path)
    return findings
