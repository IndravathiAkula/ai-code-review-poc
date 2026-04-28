from __future__ import annotations
import fnmatch
import hashlib
from typing import Iterable, Sequence


LANG_BY_EXT = {
    "py": "python", "js": "javascript", "jsx": "javascript",
    "ts": "typescript", "tsx": "typescript", "go": "go",
    "rs": "rust", "java": "java", "kt": "kotlin",
    "rb": "ruby", "php": "php", "cs": "csharp",
    "c": "c", "h": "c", "cc": "cpp", "cpp": "cpp", "hpp": "cpp",
    "sh": "bash", "yml": "yaml", "yaml": "yaml",
    "sql": "sql", "tf": "terraform",
}

SKIP_SUFFIXES = (
    ".lock", ".min.js", ".min.css", ".map",
    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".ico",
    ".svg", ".woff", ".woff2", ".ttf",
)


def language_for(path: str) -> str:
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return LANG_BY_EXT.get(ext, "text")


def should_skip(path: str) -> bool:
    return path.endswith(SKIP_SUFFIXES)


# ---- sensitive-path blocklist ----

SENSITIVE_DIR_NAMES = frozenset({
    "secrets", "credentials", ".ssh", ".aws", ".gcp", ".azure",
})

SENSITIVE_FILE_PATTERNS: tuple[str, ...] = (
    ".env",
    ".env.local", ".env.production", ".env.staging",
    ".env.dev", ".env.development", ".env.prod",
    "*.pem", "*.key", "*.p12", "*.pfx",
    "id_rsa", "id_rsa.pub", "id_dsa", "id_ed25519", "id_ecdsa",
    "*_rsa", "*_dsa", "*_ed25519", "*_ecdsa",
    "credentials.json", "service-account*.json",
    "*.priv", "*.privatekey",
)


def is_sensitive_path(
    path: str,
    *,
    extra_patterns: Sequence[str] = (),
) -> bool:
    """True if ``path`` looks like a secrets-bearing file we should never
    send to the model. Defaults block well-known secret stores (``.env``,
    private keys, ``secrets/`` directories); ``extra_patterns`` adds
    user-supplied globs matched against the full path."""
    norm = path.replace("\\", "/")
    parts = [p for p in norm.split("/") if p]
    name = parts[-1] if parts else ""
    for seg in parts[:-1]:
        if seg in SENSITIVE_DIR_NAMES:
            return True
    for pat in SENSITIVE_FILE_PATTERNS:
        if fnmatch.fnmatch(name, pat):
            return True
    for pat in extra_patterns:
        if fnmatch.fnmatch(norm, pat):
            return True
    return False


def valid_new_lines(patched_file) -> set[int]:
    """Return the set of NEW-file line numbers touched by this diff
    (added + context). Used to reject hallucinated line numbers."""
    lines: set[int] = set()
    for hunk in patched_file:
        for line in hunk:
            if line.target_line_no is not None and (line.is_added or line.is_context):
                lines.add(line.target_line_no)
    return lines


def dedupe(findings: Iterable[dict]) -> list[dict]:
    """Drop duplicates keyed by (path, line, title)."""
    seen: set[str] = set()
    out: list[dict] = []
    for f in findings:
        key = f"{f.get('path','')}|{f.get('line',0)}|{f.get('title','').strip().lower()}"
        h = hashlib.sha1(key.encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out.append(f)
    return out


def filter_by_confidence(findings: Iterable[dict], threshold: float) -> list[dict]:
    return [f for f in findings if float(f.get("confidence", 0.0)) >= threshold]


SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def filter_by_severity(findings: Iterable[dict], min_severity: str) -> list[dict]:
    """Drop findings below ``min_severity``. Unknown severities are kept."""
    threshold = SEVERITY_RANK.get(min_severity.lower(), 0)
    out = []
    for f in findings:
        sev = f.get("severity", "low").lower()
        if SEVERITY_RANK.get(sev, threshold) >= threshold:
            out.append(f)
    return out


def sort_findings(findings: list[dict]) -> list[dict]:
    return sorted(
        findings,
        key=lambda f: (
            SEVERITY_ORDER.get(f.get("severity", "low"), 9),
            -float(f.get("confidence", 0.0)),
            f.get("path", ""),
            int(f.get("line", 0)),
        ),
    )


AI_COMMENT_TAG = "[AI-REVIEW]"


def format_comment(f: dict) -> str:
    sev = f.get("severity", "low").upper()
    cat = f.get("category", "maintainability")
    title = f.get("title", "").strip()
    conf = float(f.get("confidence", 0.0))
    body = [f"{AI_COMMENT_TAG} **{sev} / {cat}** — {title} _(confidence {conf:.2f})_",
            "",
            f.get("explanation", "").strip()]
    fix = f.get("suggested_fix")
    if fix:
        body.append("")
        body.append("**Suggested fix:**")
        body.append("```")
        body.append(fix.strip())
        body.append("```")
    return "\n".join(body)
