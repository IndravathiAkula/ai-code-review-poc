"""Give the LLM the same "I've read the codebase" context a senior has.

The per-file reviewer sees each chunk in isolation. A human reviewer
knows what else exists in the repo: callers of the changed function,
sibling files' conventions, the module the file imports. We fake
"has read the codebase" by grep-retrieving a small, targeted bundle
of related code and pasting it under a *Related code* section in the
user prompt.

Design choices:
- ``git grep`` for retrieval — always fast, always available (this is
  a PR pipeline, so we're in a git checkout), no external index needed.
- Bounded budget per chunk (``max_files`` files, ``max_chars`` total)
  so a giant PR doesn't blow the prompt budget or the token bill.
- Symbol extraction is regex-based per language — cheap, wrong sometimes,
  but wrong-in-a-safe-way (we might miss a caller; we won't hallucinate one).
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from unidiff import PatchSet

from .utils import language_for


# Cap the total context we inject per chunk. The model already sees the
# diff (up to ~8000 chars) plus the SYSTEM prompt (~5KB); another ~3KB
# of context is a meaningful boost without doubling prompt cost.
DEFAULT_MAX_CONTEXT_FILES = 3
DEFAULT_MAX_CONTEXT_CHARS = 3000
DEFAULT_CALLER_SNIPPET_LINES = 12
DEFAULT_SIBLING_HEAD_LINES = 40


# Per-language patterns for newly-defined symbols (function/class/const).
# We match against ADDED lines from the diff and use the names to grep
# for callers elsewhere in the repo. False positives on the regex are
# cheap — they just cause an extra empty grep call.
_SYMBOL_PATTERNS: dict[str, tuple[re.Pattern, ...]] = {
    "python": (
        re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
        re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\s*[:\(]"),
        re.compile(r"^\s*async\s+def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
    ),
    "typescript": (
        re.compile(
            r"^\s*(?:export\s+)?(?:async\s+)?function\s+"
            r"([A-Za-z_$][A-Za-z0-9_$]*)\s*[\(<]"
        ),
        re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_$][A-Za-z0-9_$]*)"),
        re.compile(
            r"^\s*(?:export\s+)?(?:const|let|var)\s+"
            r"([A-Za-z_$][A-Za-z0-9_$]*)\s*="
        ),
    ),
    "javascript": (
        re.compile(
            r"^\s*(?:export\s+)?(?:async\s+)?function\s+"
            r"([A-Za-z_$][A-Za-z0-9_$]*)\s*\("
        ),
        re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_$][A-Za-z0-9_$]*)"),
    ),
    "go": (
        re.compile(r"^\s*func\s+(?:\([^)]+\)\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\("),
        re.compile(r"^\s*type\s+([A-Za-z_][A-Za-z0-9_]*)\s+(?:struct|interface)"),
    ),
    "java": (
        re.compile(
            r"^\s*(?:public|private|protected|static|final|\s)*\s+"
            r"[\w<>\[\],\s]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("
        ),
        re.compile(r"^\s*(?:public|private|protected)?\s*class\s+"
                   r"([A-Za-z_][A-Za-z0-9_]*)"),
    ),
}


# Common identifiers we should never grep for — too generic, would match
# thousands of unrelated files.
_STOPWORD_SYMBOLS = frozenset({
    "get", "set", "run", "do", "make", "new", "init", "main", "test",
    "handler", "call", "process", "handle", "start", "stop", "load",
    "save", "read", "write", "check", "validate", "parse", "build",
    "create", "update", "delete", "find", "list", "count", "sum",
    "map", "filter", "reduce", "each", "next", "prev", "log", "err",
    "self", "cls", "value", "data", "result", "item", "index",
    # Common one-letter globals that would grep the entire repo:
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
})


def extract_new_symbols(diff_text: str, path: str) -> set[str]:
    """Return the identifiers *newly defined* in ``path``'s added diff lines.

    Used to find callers elsewhere in the repo. Language-specific — falls
    back to an empty set for languages we don't have patterns for; the
    reviewer still runs, just without context for that file.
    """
    lang = language_for(path)
    patterns = _SYMBOL_PATTERNS.get(lang)
    if not patterns:
        return set()
    symbols: set[str] = set()
    for pf in PatchSet(diff_text):
        if pf.path != path or pf.is_binary_file:
            continue
        for hunk in pf:
            for line in hunk:
                if not line.is_added:
                    continue
                content = line.value.rstrip("\n").rstrip("\r")
                for pat in patterns:
                    m = pat.match(content)
                    if m:
                        name = m.group(1)
                        if (name and len(name) > 2
                                and name.lower() not in _STOPWORD_SYMBOLS
                                and not name.startswith("_")):
                            symbols.add(name)
    return symbols


def _run_git_grep(
    root: Path, symbol: str, exclude_path: str, max_hits: int = 5,
) -> list[tuple[str, int, str]]:
    """Return up to ``max_hits`` (path, line_no, matching_line) tuples.

    Skips the file the symbol was defined in (its own definition line
    isn't a caller). Uses ``git grep -n`` — fast, respects .gitignore,
    and available everywhere the reviewer runs.
    """
    try:
        proc = subprocess.run(
            ["git", "grep", "-n", "--fixed-strings",
             "--word-regexp", "--", symbol],
            cwd=str(root),
            capture_output=True, text=True, timeout=15, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if proc.returncode not in (0, 1):
        # 0 = matches, 1 = no matches, other = git error.
        return []
    hits: list[tuple[str, int, str]] = []
    exclude = exclude_path.replace("\\", "/")
    for raw in proc.stdout.splitlines():
        try:
            path, ln, content = raw.split(":", 2)
        except ValueError:
            continue
        if path.replace("\\", "/") == exclude:
            continue
        try:
            ln_i = int(ln)
        except ValueError:
            continue
        hits.append((path, ln_i, content))
        if len(hits) >= max_hits:
            break
    return hits


def _read_snippet(root: Path, path: str, center_line: int, radius: int) -> str:
    """Read ``radius`` lines above and below ``center_line``.

    Best-effort — files that don't exist (renamed, moved, submodule)
    just return empty and the caller drops them from the context bundle.
    """
    p = root / path
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError):
        return ""
    lines = text.splitlines()
    if not lines:
        return ""
    start = max(0, center_line - radius - 1)
    end = min(len(lines), center_line + radius)
    slice_ = lines[start:end]
    numbered = [f"{start + i + 1}: {ln}" for i, ln in enumerate(slice_)]
    return "\n".join(numbered)


def _read_head(root: Path, path: str, lines: int) -> str:
    p = root / path
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError):
        return ""
    head = text.splitlines()[:lines]
    return "\n".join(head)


def _sibling_files(root: Path, path: str) -> list[str]:
    """Files in the same directory as ``path`` — used for style/conventions."""
    parent = (root / path).parent
    same_ext = Path(path).suffix
    if not same_ext or not parent.exists():
        return []
    siblings: list[str] = []
    for p in parent.iterdir():
        if not p.is_file() or p.suffix != same_ext:
            continue
        rel = p.relative_to(root).as_posix()
        if rel == path.replace("\\", "/"):
            continue
        siblings.append(rel)
    return sorted(siblings)


def build_related_code_block(
    diff_text: str,
    path: str,
    root: Path,
    *,
    max_files: int = DEFAULT_MAX_CONTEXT_FILES,
    max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    caller_snippet_lines: int = DEFAULT_CALLER_SNIPPET_LINES,
    sibling_head_lines: int = DEFAULT_SIBLING_HEAD_LINES,
) -> str:
    """Return a bounded block of related code to inject into the user prompt.

    Structure:
        Callers of <symbol> (path:line):
            <numbered snippet>

        Sibling file <path> (first N lines for style reference):
            <numbered head>

    Returns an empty string when the repo has no related code, keeping
    the prompt tidy on truly isolated changes.
    """
    if not root.exists():
        return ""
    sections: list[str] = []
    total = 0
    files_used = 0

    # ---- Callers of newly-defined symbols ----
    symbols = extract_new_symbols(diff_text, path)
    for symbol in sorted(symbols):
        if files_used >= max_files or total >= max_chars:
            break
        hits = _run_git_grep(root, symbol, exclude_path=path, max_hits=3)
        if not hits:
            continue
        hit_path, hit_line, _ = hits[0]  # closest single caller only
        snippet = _read_snippet(root, hit_path, hit_line, caller_snippet_lines)
        if not snippet:
            continue
        header = f"Callers of `{symbol}` — {hit_path}:{hit_line}:"
        block = f"{header}\n{snippet}"
        if total + len(block) > max_chars:
            break
        sections.append(block)
        total += len(block) + 2
        files_used += 1

    # ---- One sibling file for style reference ----
    if files_used < max_files and total < max_chars:
        siblings = _sibling_files(root, path)
        if siblings:
            sib = siblings[0]
            head = _read_head(root, sib, sibling_head_lines)
            if head:
                block = (
                    f"Sibling file `{sib}` (first {sibling_head_lines} lines "
                    f"— for style/conventions reference only):\n{head}"
                )
                if total + len(block) <= max_chars:
                    sections.append(block)

    if not sections:
        return ""
    joined = "\n\n".join(sections)
    if len(joined) > max_chars:
        joined = joined[:max_chars] + "\n... (context truncated)"
    return joined


def gather_context(
    diff_text: str,
    path: str,
    root: str | Path,
    *,
    max_files: int = DEFAULT_MAX_CONTEXT_FILES,
    max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> str:
    """Public entry point. Returns ``""`` on any failure — never raises.

    The reviewer's per-chunk loop calls this once per file. Failures are
    treated as "no context available" so a repo where git isn't set up,
    or a Windows CI runner without ``git grep``, still gets normal reviews.
    """
    root_path = Path(root)
    if not root_path.exists():
        return ""
    try:
        return build_related_code_block(
            diff_text, path, root_path,
            max_files=max_files, max_chars=max_chars,
        )
    except Exception as exc:
        print(f"[warn] repo_context: {exc}", file=sys.stderr)
        return ""
