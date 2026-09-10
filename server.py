"""ACP local review bridge — expose the reviewer as a tiny HTTP service.

The ACP Node backend (Gen-pilot) calls this instead of re-implementing the
review engine, so this repo stays the single source of truth for review
intelligence (prompts, concurrency, retries, test review, repo context,
per-language extras, confidence/severity floors).

Run (from the PoC root, with the project venv active):
    python server.py                 # binds 127.0.0.1:8762
    python server.py --port 9000     # custom port

Endpoints:
    GET  /health  -> {"ok": true}
    POST /review  -> review a unified diff, returns structured findings

POST /review body (all keys optional unless noted):
{
  "diff": "<unified diff>",              # required
  "provider": "openrouter",             # github-models|openai|anthropic|groq|
                                         # openrouter|nvidia|together|anyscale|
                                         # cerebras|ollama  (default: openrouter)
  "model": "z-ai/glm-5.2:free",         # single model (or use "models")
  "models": ["z-ai/glm-5.2:free",       # ordered fallback chain — first model
             "liquid/lfm-2.5-2.6b:free"],# that answers wins
  "api_key": "...",                     # per-request key; falls back to the
                                         # provider's env var / .env
  "min_confidence": 0.5,
  "min_severity": "low",                # low|medium|high|critical
  "include_maintainability_findings": true,
  "max_retries": 2,
  "repo_root": "D:/path/to/repo",       # enables repo-context features
  "repo": "owner/name"
}

Response 200:
{
  "findings": [ {path, line, severity, category, title, explanation,
                 suggested_fix, confidence}, ... ],
  "provider": "openrouter",
  "model": "<model that produced the findings>",
  "attempts": [ {"model", "ok", "error"}, ... ]
}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from reviewer import review_patch  # noqa: E402
from reviewer.providers import build_provider  # noqa: E402

DEFAULT_PORT = 8762
DEFAULT_PROVIDER = "openrouter"


class _ReviewError(Exception):
    """Client-facing error: message is safe to return in the response body."""


def _build_provider(name: str, api_key: str | None):
    """Build a provider, preferring the per-request key over env/.env.

    Mirrors build_provider()'s env lookups but layers the request key on
    top of os.environ so the server also works standalone with its own
    .env (e.g. OPENROUTER_API_KEY in D:/ai-code-review-poc/.env).
    """
    env = dict(os.environ)
    if api_key:
        env["OPENROUTER_API_KEY"] = api_key
        env["OPENAI_API_KEY"] = api_key
        env["GROQ_API_KEY"] = api_key
        env["MODEL_API_TOKEN"] = api_key
        env["GITHUB_TOKEN"] = api_key
        env["ANTHROPIC_API_KEY"] = api_key
        env["NVIDIA_API_KEY"] = api_key
        env["TOGETHER_API_KEY"] = api_key
        env["ANYSCALE_API_KEY"] = api_key
        env["CEREBRAS_API_KEY"] = api_key
    try:
        return build_provider(name or None, env=env)
    except (RuntimeError, ValueError) as exc:
        raise _ReviewError(str(exc)) from exc


def _review(body: dict) -> dict:
    diff = body.get("diff")
    if not isinstance(diff, str) or not diff.strip():
        raise _ReviewError("missing or empty 'diff'")

    provider_name = (body.get("provider") or DEFAULT_PROVIDER).strip().lower()
    # Accept either a single model or an ordered fallback list.
    models = body.get("models") or ([body.get("model")] if body.get("model") else [])
    if not models:
        raise _ReviewError("missing 'model' / 'models' (e.g. a ranked OpenRouter model id)")
    if isinstance(models, str):
        models = [models]

    provider = _build_provider(provider_name, body.get("api_key"))

    # Accepted-finding history: the reviewer previously ignored these findings,
    # so teach the model about them. The instruction differs per review mode:
    #  - "exclude" (Review Ignoring Previously Ignored): never report them again.
    #  - "include" (Review Everything): re-evaluate, but only re-report when the
    #    issue has materially worsened.
    ignored = body.get("ignored_context") or {}
    ignored_findings = ignored.get("findings") or []
    extras_by_language = None
    if ignored_findings:
        instruction = ignored.get("instruction") or (
            "These findings were already reviewed and intentionally accepted. "
            "Do NOT report them again.")
        lines = [
            "FINDINGS PREVIOUSLY REVIEWED AND ACCEPTED BY THE HUMAN REVIEWER:",
        ]
        for i in ignored_findings[:50]:
            entry = f"- {i.get('path', '?')}:{i.get('line', '?')} — {i.get('title', '')}"
            if i.get("reason"):
                entry += f" (accepted because: {i['reason']})"
            lines.append(entry)
        lines.append(instruction)
        extras_block = "\n".join(lines)
        from reviewer.utils import LANG_BY_EXT
        extras_by_language = {lang: extras_block for lang in set(LANG_BY_EXT.values())}

    started = time.time()
    print(f"[review] START: diff={len(diff)} chars, provider={provider_name}, "
          f"models={list(models)}, max_retries={body.get('max_retries', 2)}", file=sys.stderr, flush=True)
    attempts: list[dict] = []
    final_findings: list[dict] = []
    model_used: str | None = None
    for m in models:
        model_started = time.time()
        usage_log: list[dict] = []
        findings = []
        error = None
        try:
            findings = review_patch(
                diff_text=diff,
                model=m,
                provider=provider,
                repo=body.get("repo") or "local/workspace",
                min_confidence=float(body.get("min_confidence", 0.5)),
                min_severity=body.get("min_severity") or "low",
                include_maintainability_findings=bool(
                    body.get("include_maintainability_findings", True)),
                max_diff_chars=int(body.get("max_diff_chars", 16000)),
                max_retries=int(body.get("max_retries", 2)),
                usage_log=usage_log,
                repo_root=body.get("repo_root") or None,
                prompt_extras_by_language=extras_by_language,
            )
        except Exception as exc:  # noqa: BLE001 — provider crashed on this model
            error = f"{type(exc).__name__}: {exc}"
        # "ok" means the model call actually went through (usage_log gets one
        # entry per successful call) — NOT "it found something". A clean diff
        # is a successful call with zero findings and must not be reported as
        # a failed attempt.
        attempts.append({
            "model": m,
            "ok": error is None and bool(usage_log),
            "error": error,
        })
        print(f"[review]   model {m}: ok={error is None and bool(usage_log)}, "
              f"findings={len(findings) if error is None else 'n/a'}, "
              f"took {time.time() - model_started:.1f}s"
              f"{f', error={error[:200]}' if error else ''}", file=sys.stderr, flush=True)
        # Only accept as "no issues" when the model call actually succeeded.
        if error is None and usage_log:
            final_findings = [f.to_dict() if hasattr(f, "to_dict") else f for f in findings]
            model_used = m
            break
        # Keep the last successfully-reached model's result as a tiebreak (even empty).
        if error is None:
            final_findings = [f.to_dict() if hasattr(f, "to_dict") else f for f in findings]
            model_used = m

    attempt_summary = "; ".join(
        (a["model"] + ("=ok" if a["ok"] else "=FAIL(" + (a["error"] or "no usage")[:80] + ")"))
        for a in attempts
    )
    print(f"[review] DONE in {time.time() - started:.1f}s: model_used={model_used or 'NONE (all attempts failed)'}, "
          f"findings={len(final_findings)}, attempts=[{attempt_summary}]",
          file=sys.stderr, flush=True)
    return {
        "findings": final_findings,
        "provider": provider_name,
        "model": model_used,
        "attempts": attempts,
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send(200, {"ok": True, "engine": "ai-code-review-poc"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/review":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise _ReviewError("request body must be a JSON object")
            self._send(200, _review(body))
        except _ReviewError as exc:
            self._send(400, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 — surface LLM/provider failures
            print(f"[review] failed: {exc}", file=sys.stderr)
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, fmt: str, *args) -> None:  # keep stdout quiet-ish
        if "/health" not in (args[0] if args else ""):
            sys.stderr.write("[http] " + fmt % args + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("ACP_POC_PORT", DEFAULT_PORT)))
    args = ap.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f"[ACP review bridge] {args.host}:{args.port} — "
          f"engine: ai-code-review-poc, default provider: {DEFAULT_PROVIDER}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
