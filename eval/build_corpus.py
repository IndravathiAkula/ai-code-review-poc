"""Regenerate every .diff and .labels.json under eval/corpus/.

Each case is a dict with `before`/`after` content; the diff is produced with
difflib so hunk headers and line counts are always correct. This is the
single source of truth for the eval corpus — edit cases here, rerun:

    python eval/build_corpus.py

Line numbers in `expected.line_range` refer to the NEW file (post-change)
side of the diff, which is what `reviewer.valid_new_lines` validates against.
"""
from __future__ import annotations
import difflib
import json
from pathlib import Path
from textwrap import dedent

CORPUS = Path(__file__).resolve().parent / "corpus"


def _diff(before: str, after: str, path: str) -> str:
    b = before.splitlines(keepends=True)
    a = after.splitlines(keepends=True)
    # Large context so whole-file changes still produce a single hunk.
    return "".join(difflib.unified_diff(
        b, a,
        fromfile=f"a/{path}", tofile=f"b/{path}",
        n=max(len(b), len(a), 10),
    ))


def _git_header(path: str) -> str:
    return f"diff --git a/{path} b/{path}\nindex 1111111..2222222 100644\n"


def build_case(case: dict) -> tuple[str, dict]:
    path = case["path"]
    before = case.get("before", "")
    after = case["after"]
    diff = _git_header(path) + _diff(before, after, path)
    labels = {
        "description": case["description"],
        "expected": case.get("expected", []),
    }
    return diff, labels


CASES: list[dict] = [
    # ── Bug cases ────────────────────────────────────────────────────────
    {
        "id": "sample_01",
        "path": "app/auth.py",
        "before": "",
        "after": dedent('''\
            import sqlite3

            DB_PATH = "users.db"
            ADMIN_TOKEN = "sk-live-9f8a7b6c5d4e3f2a1b0c9d8e7f6a5b4c"


            def authenticate(username: str, password: str) -> bool:
                conn = sqlite3.connect(DB_PATH)
                cur = conn.cursor()
                query = "SELECT id FROM users WHERE username='" + username + "' AND password='" + password + "'"
                cur.execute(query)
                row = cur.fetchone()
                return row is not None


            def divide(a, b):
                return a / b
            '''),
        "description": "Hardcoded secret + SQL injection + unchecked division by zero.",
        "expected": [
            {"path": "app/auth.py", "line_range": [4, 4], "category": "security", "severity": "critical",
             "keywords": ["secret", "token", "hardcoded", "credential"]},
            {"path": "app/auth.py", "line_range": [10, 10], "category": "security", "severity": "critical",
             "keywords": ["sql injection", "sqli", "parameterized", "string concatenation"]},
            {"path": "app/auth.py", "line_range": [17, 17], "category": "correctness", "severity": "medium",
             "keywords": ["zero", "zerodivisionerror", "divide"]},
        ],
    },
    {
        "id": "sample_02",
        "path": "api/orders.py",
        "before": dedent('''\
            from .models import Order, Customer


            def total_revenue(customers):
                return sum(o.amount for c in customers for o in c.orders)


            def export_csv(path, customers):
                with open(path, "w") as f:
                    for c in customers:
                        f.write(c.name + "\\n")
            '''),
        "after": dedent('''\
            from .models import Order, Customer


            def total_revenue(customers):
                total = 0
                for c in customers:
                    orders = Order.query.filter_by(customer_id=c.id).all()
                    for o in orders:
                        total += o.amount
                return total


            def find_order(orders, target_id):
                for i in range(len(orders)):
                    if orders[i].id == target_id:
                        return orders[i]
                return None


            def export_csv(path, customers):
                with open(path, "w") as f:
                    for c in customers:
                        f.write(c.name + "\\n")
            '''),
        "description": "Classic N+1 query in total_revenue; find_order uses index-based iteration where next() or a dict lookup would be clearer.",
        "expected": [
            {"path": "api/orders.py", "line_range": [5, 10], "category": "performance", "severity": "high",
             "keywords": ["n+1", "query", "loop", "eager", "join"]},
            {"path": "api/orders.py", "line_range": [13, 17], "category": "maintainability", "severity": "low",
             "keywords": ["enumerate", "pythonic", "next", "comprehension"]},
        ],
    },
    {
        "id": "sample_03",
        "path": "app/net.py",
        "before": dedent('''\
            import subprocess


            def ping_host(host: str) -> bytes:
                return subprocess.check_output(["ping", "-c", "1", host])


            def resolve(host: str) -> str:
                return host
            '''),
        "after": dedent('''\
            import subprocess
            import shlex


            def ping_host(host: str) -> bytes:
                return subprocess.check_output(f"ping -c 1 {host}", shell=True)


            def run_diagnostic(cmd: str) -> str:
                return subprocess.getoutput(cmd)


            def resolve(host: str) -> str:
                return host
            '''),
        "description": "shell=True with untrusted input enables command injection; run_diagnostic is even worse.",
        "expected": [
            {"path": "app/net.py", "line_range": [6, 6], "category": "security", "severity": "critical",
             "keywords": ["command injection", "shell=true", "shell injection", "subprocess", "untrusted"]},
            {"path": "app/net.py", "line_range": [9, 10], "category": "security", "severity": "critical",
             "keywords": ["command injection", "getoutput", "shell", "untrusted"]},
        ],
    },
    {
        "id": "sample_04",
        "path": "app/lockfile.py",
        "before": dedent('''\
            import os


            def acquire_once(path: str, content: str) -> bool:
                """Return True if we created the lock file, False otherwise."""
                if os.path.exists(path):
                    return False
                with open(path, "w") as f:
                    f.write(content)
                return True
            '''),
        "after": dedent('''\
            import os
            import time


            def acquire_once(path: str, content: str) -> bool:
                """Return True if we created the lock file, False otherwise."""
                if os.path.exists(path):
                    return False
                # give other callers a chance to notice — keeps tests stable
                time.sleep(0.01)
                with open(path, "w") as f:
                    f.write(content)
                return True


            LAST_ACQUIRED_AT: float = 0.0
            '''),
        "description": "TOCTOU race: exists() + open() is not atomic. The added sleep widens the window. Should use O_EXCL / open(path, 'x').",
        "expected": [
            {"path": "app/lockfile.py", "line_range": [7, 13], "category": "correctness", "severity": "high",
             "keywords": ["toctou", "race", "atomic", "o_excl", "check-then-use"]},
        ],
    },
    {
        "id": "sample_05",
        "path": "web/user.ts",
        "before": dedent('''\
            interface User {
              id: string;
              profile: { name: string };
            }

            export function displayName(user?: User): string {
              return user?.profile.name.toUpperCase() ?? "";
            }
            '''),
        "after": dedent('''\
            interface User {
              id: string;
              profile?: { name?: string };
            }

            export function displayName(user?: User): string {
              return user.profile.name.toUpperCase();
            }

            export function greet(user?: User): string {
              const name = user!.profile!.name!;
              return `Hello, ${name}`;
            }

            export const DEFAULT_USER = null as unknown as User;
            '''),
        "description": "displayName lost its optional-chaining guards after profile became optional. greet silences the compiler with non-null assertions without fixing the hazard.",
        "expected": [
            {"path": "web/user.ts", "line_range": [6, 8], "category": "correctness", "severity": "high",
             "keywords": ["undefined", "null", "optional", "cannot read", "nullable"]},
            {"path": "web/user.ts", "line_range": [10, 13], "category": "correctness", "severity": "high",
             "keywords": ["non-null assertion", "!.", "undefined", "null", "runtime"]},
        ],
    },
    {
        "id": "sample_06",
        "path": "app/files.py",
        "before": dedent('''\
            import os
            from flask import Flask, send_file, request

            app = Flask(__name__)
            BASE = "/srv/files"
            '''),
        "after": dedent('''\
            import os
            from flask import Flask, send_file, request

            app = Flask(__name__)
            BASE = "/srv/files"


            @app.get("/download")
            def download():
                name = request.args.get("name", "")
                full = os.path.join(BASE, name)
                return send_file(full)
            '''),
        "description": "Path traversal — os.path.join does not prevent '../' escapes; an absolute name replaces BASE entirely.",
        "expected": [
            {"path": "app/files.py", "line_range": [8, 12], "category": "security", "severity": "critical",
             "keywords": ["path traversal", "directory traversal", "../", "realpath", "abspath", "send_from_directory", "lfi"]},
        ],
    },
    {
        "id": "sample_07",
        "path": "app/config.py",
        "before": dedent('''\
            import yaml


            def load_config(text: str) -> dict:
                return yaml.safe_load(text)
            '''),
        "after": dedent('''\
            import yaml
            import pickle


            def load_config(text: str) -> dict:
                return yaml.load(text)


            def load_cached(blob: bytes) -> dict:
                # deserialize a cache entry produced by an earlier run
                return pickle.loads(blob)
            '''),
        "description": "yaml.load without a safe Loader allows arbitrary object construction (RCE). pickle.loads on attacker-reachable bytes is also RCE.",
        "expected": [
            {"path": "app/config.py", "line_range": [6, 6], "category": "security", "severity": "critical",
             "keywords": ["yaml.load", "safe_load", "deserialization", "rce", "loader"]},
            {"path": "app/config.py", "line_range": [10, 11], "category": "security", "severity": "critical",
             "keywords": ["pickle", "deserialization", "rce", "untrusted", "arbitrary code"]},
        ],
    },
    {
        "id": "sample_08",
        "path": "api/documents.py",
        "before": dedent('''\
            from flask import Blueprint, jsonify, abort
            from flask_login import login_required, current_user

            from .models import Document

            bp = Blueprint("docs", __name__)


            @bp.get("/api/documents")
            @login_required
            def list_documents():
                docs = Document.query.filter_by(owner_id=current_user.id).all()
                return jsonify([d.to_dict() for d in docs])
            '''),
        "after": dedent('''\
            from flask import Blueprint, jsonify, abort
            from flask_login import login_required

            from .models import Document

            bp = Blueprint("docs", __name__)


            @bp.get("/api/documents/<int:doc_id>")
            @login_required
            def get_document(doc_id: int):
                doc = Document.query.get(doc_id)
                if not doc:
                    abort(404)
                return jsonify(doc.to_dict())


            @bp.get("/api/documents")
            @login_required
            def list_documents():
                docs = Document.query.filter_by(owner_id=current_user.id).all()
                return jsonify([d.to_dict() for d in docs])
            '''),
        "description": "IDOR — any logged-in user can fetch any document by id. Missing ownership check (doc.owner_id == current_user.id).",
        "expected": [
            {"path": "api/documents.py", "line_range": [9, 15], "category": "security", "severity": "critical",
             "keywords": ["idor", "authorization", "ownership", "current_user", "access control", "broken access"]},
        ],
    },
    {
        "id": "sample_09",
        "path": "app/pagination.py",
        "before": dedent('''\
            def page(items: list, page_num: int, page_size: int) -> list:
                start = (page_num - 1) * page_size
                end = start + page_size
                return items[start:end]


            def chunk(items: list, size: int) -> list:
                return [items[i:i + size] for i in range(0, len(items), size)]
            '''),
        "after": dedent('''\
            def page(items: list, page_num: int, page_size: int) -> list:
                # 1-indexed pages, inclusive end
                start = (page_num - 1) * page_size
                end = start + page_size + 1
                return items[start:end]


            def last_n(items: list, n: int) -> list:
                return items[len(items) - n - 1:]


            def chunk(items: list, size: int) -> list:
                return [items[i:i + size] for i in range(0, len(items), size)]
            '''),
        "description": "page() returns page_size+1 items (off-by-one from the `+ 1`). last_n() starts one index too early (should be len(items) - n).",
        "expected": [
            {"path": "app/pagination.py", "line_range": [2, 5], "category": "correctness", "severity": "high",
             "keywords": ["off-by-one", "end", "+ 1", "page_size", "slice"]},
            {"path": "app/pagination.py", "line_range": [8, 9], "category": "correctness", "severity": "high",
             "keywords": ["off-by-one", "last_n", "len(items) - n", "index", "slice"]},
        ],
    },
    {
        "id": "sample_10",
        "path": "app/retry.py",
        "before": dedent('''\
            import time


            def call_with_retry(fn, attempts: int = 3):
                last_exc = None
                for i in range(attempts):
                    try:
                        return fn()
                    except Exception as e:
                        last_exc = e
                        time.sleep(2 ** i)
                raise last_exc
            '''),
        "after": dedent('''\
            def call_with_retry(fn, attempts: int = 3):
                for _ in range(attempts):
                    try:
                        return fn()
                    except:
                        pass
                return None


            def fetch_or_default(fetch, default):
                try:
                    return fetch()
                except Exception:
                    return default
            '''),
        "description": "Retry rewrite silently swallows every exception (bare except), loses backoff, returns None on exhaustion. fetch_or_default also swallows without logging.",
        "expected": [
            {"path": "app/retry.py", "line_range": [1, 7], "category": "correctness", "severity": "high",
             "keywords": ["bare except", "swallow", "silent", "backoff", "none", "retry"]},
            {"path": "app/retry.py", "line_range": [10, 14], "category": "maintainability", "severity": "medium",
             "keywords": ["swallow", "log", "silent", "mask", "exception"]},
        ],
    },
    {
        "id": "sample_11",
        "path": "api/proxy.py",
        "before": dedent('''\
            import requests
            from flask import Blueprint, request, Response

            bp = Blueprint("proxy", __name__)
            '''),
        "after": dedent('''\
            import requests
            from flask import Blueprint, request, Response

            bp = Blueprint("proxy", __name__)


            @bp.get("/api/fetch")
            def fetch():
                url = request.args.get("url", "")
                r = requests.get(url, timeout=5)
                return Response(r.content, content_type=r.headers.get("Content-Type", "text/plain"))


            ALLOWED_HOSTS = ["api.example.com"]  # unused for now
            '''),
        "description": "SSRF — user-supplied URL is fetched with no allowlist. Attacker can reach cloud metadata (169.254.169.254), internal services, file:// etc.",
        "expected": [
            {"path": "api/proxy.py", "line_range": [7, 11], "category": "security", "severity": "critical",
             "keywords": ["ssrf", "server-side request forgery", "allowlist", "metadata", "internal", "url", "scheme"]},
        ],
    },
    {
        "id": "sample_12",
        "path": "web/render.py",
        "before": dedent('''\
            from jinja2 import Template
            from markupsafe import Markup


            def render_message(msg: str) -> str:
                template = Template("<div class=\\"msg\\">{{ content }}</div>")
                return template.render(content=msg)
            '''),
        "after": dedent('''\
            from jinja2 import Template
            from markupsafe import Markup


            def render_message(msg: str) -> str:
                template = Template("<div class=\\"msg\\">{{ content|safe }}</div>")
                return template.render(content=msg)


            def render_link(url: str, label: str) -> str:
                # users asked for raw HTML labels
                return Markup(f'<a href="{url}">{label}</a>')
            '''),
        "description": "XSS — |safe disables autoescape on user content. render_link builds HTML via f-string and wraps in Markup, bypassing escape for both url (javascript: scheme) and label.",
        "expected": [
            {"path": "web/render.py", "line_range": [5, 6], "category": "security", "severity": "high",
             "keywords": ["xss", "safe", "autoescape", "escape", "untrusted", "jinja"]},
            {"path": "web/render.py", "line_range": [10, 12], "category": "security", "severity": "high",
             "keywords": ["xss", "markup", "escape", "javascript:", "href", "injection"]},
        ],
    },
    # ── Clean cases (false-positive detection) ──────────────────────────
    {
        "id": "clean_01",
        "path": "app/greet.py",
        "before": dedent('''\
            def say_hi(person_name: str) -> str:
                return f"Hello, {person_name}!"


            def say_bye(person_name: str) -> str:
                return f"Bye, {person_name}!"


            def greet_all(people: list[str]) -> list[str]:
                return [say_hi(p) for p in people]
            '''),
        "after": dedent('''\
            def say_hi(name: str) -> str:
                return f"Hello, {name}!"


            def say_bye(name: str) -> str:
                return f"Bye, {name}!"


            def greet_all(names: list[str]) -> list[str]:
                return [say_hi(n) for n in names]
            '''),
        "description": "Pure parameter rename (person_name -> name). No behavior change, no hazards.",
        "expected": [],
    },
    {
        "id": "clean_02",
        "path": "tests/test_pagination.py",
        "before": "",
        "after": dedent('''\
            import pytest

            from app.pagination import page


            @pytest.mark.parametrize("page_num,page_size,expected", [
                (1, 2, [1, 2]),
                (2, 2, [3, 4]),
                (3, 2, [5]),
            ])
            def test_page_returns_slice(page_num, page_size, expected):
                items = [1, 2, 3, 4, 5]
                assert page(items, page_num, page_size) == expected


            def test_page_beyond_end_returns_empty():
                items = [1, 2, 3]
                assert page(items, 5, 2) == []
            '''),
        "description": "New parametrized pytest for an existing helper. Tests are clean and well-formed.",
        "expected": [],
    },
    {
        "id": "clean_03",
        "path": "app/math_utils.py",
        "before": dedent('''\
            def clamp(x, lo, hi):
                if x < lo:
                    return lo
                if x > hi:
                    return hi
                return x


            def mean(xs):
                return sum(xs) / len(xs)
            '''),
        "after": dedent('''\
            def clamp(x: float, lo: float, hi: float) -> float:
                """Return x bounded to the inclusive range [lo, hi]."""
                if x < lo:
                    return lo
                if x > hi:
                    return hi
                return x


            def mean(xs: list[float]) -> float:
                """Arithmetic mean. Raises ZeroDivisionError on empty input — callers
                are expected to guard, same as statistics.mean."""
                return sum(xs) / len(xs)
            '''),
        "description": "Adds type annotations and docstrings. Behavior (including documented empty-input behavior of mean) is unchanged.",
        "expected": [],
    },
    {
        "id": "clean_04",
        "path": "requirements.txt",
        "before": dedent('''\
            requests==2.31.0
            PyYAML==6.0.1
            flask==3.0.2
            SQLAlchemy==2.0.25
            pytest==8.1.1
            rich==13.7.0
            '''),
        "after": dedent('''\
            requests==2.32.3
            PyYAML==6.0.2
            flask==3.0.3
            SQLAlchemy==2.0.36
            pytest==8.3.3
            rich==13.9.4
            '''),
        "description": "Routine minor/patch version bumps in requirements.txt. No code change.",
        "expected": [],
    },
]


def main() -> int:
    CORPUS.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        diff, labels = build_case(case)
        (CORPUS / f"{case['id']}.diff").write_text(diff, encoding="utf-8")
        (CORPUS / f"{case['id']}.labels.json").write_text(
            json.dumps(labels, indent=2), encoding="utf-8")
        print(f"wrote {case['id']}")
    print(f"\n{len(CASES)} cases written to {CORPUS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
