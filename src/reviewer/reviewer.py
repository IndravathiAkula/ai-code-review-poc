from __future__ import annotations
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Any, Iterable

from unidiff import PatchSet
from azure.core.exceptions import HttpResponseError, ServiceRequestError

from .pricing import cost_usd
from .prompts import SYSTEM, USER_TEMPLATE
from .providers import (
    ChatResponse, Provider, ProviderPermanentError, ProviderTransientError,
    Usage, build_provider,
)
from .utils import (
    language_for, should_skip, valid_new_lines,
    dedupe, filter_by_confidence, filter_by_severity, sort_findings,
    is_sensitive_path,
)


DEFAULT_CONCURRENCY = 4
DEFAULT_MAX_DIFF_CHARS = 8000
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_SECONDS = 1.0
DEFAULT_MAX_FILES_PER_PR = 0   # 0 = no cap
DEFAULT_MAX_TOKENS_PER_PR = 0  # 0 = no cap
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

# Indirection so tests can monkeypatch sleep without real-time delays.
_sleep = time.sleep


@dataclass
class Finding:
    path: str
    line: int
    severity: str
    category: str
    title: str
    explanation: str
    suggested_fix: str | None
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_json(content: str) -> dict:
    content = content.strip()
    # Some models wrap JSON in ``` fences despite instructions.
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(content[start:end + 1])
            except json.JSONDecodeError:
                return {}
        return {}


def _retry_after_seconds(exc) -> float | None:
    """Honor the Retry-After header when present (in seconds).

    Works on both legacy ``HttpResponseError`` and ``ProviderTransientError``.
    """
    if isinstance(exc, ProviderTransientError):
        return exc.retry_after_seconds
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response else None
    if not headers:
        return None
    raw = headers.get("Retry-After") if hasattr(headers, "get") else None
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, ProviderTransientError):
        return True
    if isinstance(exc, ProviderPermanentError):
        return False
    if isinstance(exc, ServiceRequestError):
        return True  # legacy: connection drop / DNS / TLS
    if isinstance(exc, HttpResponseError):
        return getattr(exc, "status_code", None) in RETRYABLE_STATUS
    return False


_RETRYABLE_EXC_TYPES = (
    ProviderTransientError, ProviderPermanentError,
    HttpResponseError, ServiceRequestError,
)


def _call_with_retry(
    callee,
    *,
    max_retries: int,
    base_seconds: float,
    **kwargs,
):
    """Call ``callee(**kwargs)`` with two layers of resilience:

    - One-shot fallback: if the SDK/model rejects ``response_format`` with
      ``TypeError``/``ValueError``, drop that kwarg and retry once. The
      prompt still asks for strict JSON; ``_parse_json`` handles fences.
    - Exponential backoff on transient HTTP/network errors (429, 5xx,
      connection drops). Honors ``Retry-After`` when present.

    ``callee`` is typically ``provider.complete`` or a legacy
    ``client.complete`` — both raise the same error families thanks to
    the provider abstraction (or the legacy adapter).
    """
    attempts = 0
    while True:
        try:
            return callee(**kwargs)
        except (TypeError, ValueError):
            if "response_format" in kwargs:
                kwargs.pop("response_format")
                continue
            raise
        except _RETRYABLE_EXC_TYPES as exc:
            if attempts >= max_retries or not _is_retryable(exc):
                raise
            delay = _retry_after_seconds(exc)
            if delay is None:
                delay = base_seconds * (2 ** attempts)
            _sleep(delay)
            attempts += 1


def _file_chunks(patched_file, max_chars: int) -> Iterable[tuple[str, str]]:
    """Yield (label, hunks_text) pairs for a single file.

    Small files go through as one chunk so the model sees the full diff
    context. Files whose combined hunks exceed ``max_chars`` are split per
    hunk to stay under the model's prompt budget.
    """
    full = "\n".join(str(h) for h in patched_file)
    if len(full) <= max_chars or len(patched_file) <= 1:
        yield "all", full
        return
    for i, hunk in enumerate(patched_file):
        yield f"hunk-{i}", str(hunk)


class _CostBudget:
    """Shared running token total + trip event for cost-cap enforcement.

    A single instance is passed to every ``_review_chunk`` worker. After
    each successful chunk, ``add()`` increments the total under the lock
    and trips ``exceeded`` (and emits a one-shot warning) when the cap
    is crossed. Workers consult ``exceeded`` before issuing their model
    call so we don't pay for chunks that arrive after the cap is hit.
    """

    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.total = 0
        self.exceeded = threading.Event()
        self._lock = threading.Lock()
        self._announced = False

    def add(self, tokens: int, *, path: str) -> None:
        if self.max_tokens <= 0:
            return
        with self._lock:
            self.total += tokens
            if self.total >= self.max_tokens and not self._announced:
                self._announced = True
                self.exceeded.set()
                print(
                    f"[warn] cost cap: token budget exceeded after {path} "
                    f"({self.total}/{self.max_tokens}); remaining chunks "
                    f"will be skipped",
                    file=sys.stderr,
                )


class _LegacyClientAdapter:
    """Adapter that lets the new Provider interface wrap a legacy
    ``.complete(model, messages, temperature, response_format)`` client.

    Test fakes still pass Azure SDK message objects to their mock client
    so existing tests keep working — we translate ``role/content`` dicts
    into ``SystemMessage`` / ``UserMessage`` before forwarding. The
    exceptions the legacy client raises (``HttpResponseError`` /
    ``ServiceRequestError`` / ``TypeError``) propagate as-is;
    ``_call_with_retry`` recognises both legacy and Provider types.
    """

    name = "legacy"

    def __init__(self, client):
        self._client = client

    def complete(
        self,
        *,
        model: str,
        messages: list[dict],
        temperature: float,
        response_format: dict | None = None,
    ) -> ChatResponse:
        from azure.ai.inference.models import SystemMessage, UserMessage
        sdk_msgs = []
        for m in messages:
            role = m["role"]
            content = m.get("content", "")
            if role == "system":
                sdk_msgs.append(SystemMessage(content))
            elif role == "user":
                sdk_msgs.append(UserMessage(content))
            else:
                raise ValueError(f"unsupported role {role!r}")
        kwargs: dict = {"model": model, "messages": sdk_msgs,
                        "temperature": temperature}
        if response_format is not None:
            kwargs["response_format"] = response_format
        resp = self._client.complete(**kwargs)
        content = resp.choices[0].message.content or ""
        u = getattr(resp, "usage", None)
        usage = None
        if u is not None:
            usage = Usage(
                prompt_tokens=int(getattr(u, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(u, "completion_tokens", 0) or 0),
            )
        return ChatResponse(content=content, usage=usage)


def _review_chunk(
    *,
    provider: Provider,
    model: str,
    repo: str,
    pr_title: str,
    pr_description: str,
    path: str,
    lang: str,
    hunks_text: str,
    new_lines: set[int],
    usage_log: list[dict] | None,
    log_lock: threading.Lock,
    max_retries: int,
    retry_base_seconds: float,
    task_model: str | None = None,
    budget: "_CostBudget | None" = None,
) -> list[dict]:
    # Cost cap: if a prior chunk pushed the running total past the cap,
    # skip this chunk entirely. The first chunk that trips it logs the
    # warning; subsequent skips stay quiet to avoid log spam.
    if budget is not None and budget.exceeded.is_set():
        return []
    # Per-task override (e.g. models_by_language) wins over the run-wide model.
    model = task_model or model
    user = USER_TEMPLATE.format(
        repo=repo,
        path=path,
        lang=lang,
        title=pr_title or "(no title)",
        description=pr_description or "(no description)",
        diff=hunks_text,
    )

    t0 = time.perf_counter()
    try:
        resp = _call_with_retry(
            provider.complete,
            max_retries=max_retries,
            base_seconds=retry_base_seconds,
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
    except (ProviderPermanentError, ProviderTransientError,
            HttpResponseError, ServiceRequestError) as exc:
        # Out of retries — skip this chunk so the rest of the PR still gets
        # reviewed. Print to stderr so CI logs surface the failure.
        status = getattr(exc, "status_code", None)
        if status in (401, 403):
            print(
                f"[error] {path}: model API rejected the token "
                f"(HTTP {status}). Check that the configured provider "
                f"({provider.name}) credentials are valid and have the "
                f"required permissions (e.g. 'models: read' for "
                f"github-models).",
                file=sys.stderr,
            )
        else:
            print(f"[warn] {path}: model call failed after retries — {exc}",
                  file=sys.stderr)
        return []
    latency = time.perf_counter() - t0

    # Extract token counts once; needed for both usage_log and cost-cap accounting.
    u = resp.usage
    prompt_tokens = u.prompt_tokens if u else 0
    completion_tokens = u.completion_tokens if u else 0
    cache_read_tokens = u.cache_read_tokens if u else 0
    cache_creation_tokens = u.cache_creation_tokens if u else 0
    total_tokens = (
        prompt_tokens + completion_tokens
        + cache_read_tokens + cache_creation_tokens
    )

    if usage_log is not None:
        with log_lock:
            usage_log.append({
                "provider": provider.name,
                "model": model,
                "path": path,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cache_read_tokens": cache_read_tokens,
                "cache_creation_tokens": cache_creation_tokens,
                "total_tokens": total_tokens,
                "cost_usd": cost_usd(
                    model, prompt_tokens, completion_tokens,
                    provider=provider.name,
                    cache_read_tokens=cache_read_tokens,
                    cache_creation_tokens=cache_creation_tokens,
                ),
                "latency_seconds": latency,
            })

    if budget is not None:
        budget.add(total_tokens, path=path)

    data = _parse_json(resp.content)
    out: list[dict] = []
    for item in data.get("findings", []) or []:
        try:
            line = int(item.get("line", 0))
        except (TypeError, ValueError):
            continue
        if line not in new_lines:
            continue
        item["path"] = path
        item["line"] = line
        out.append(item)
    return out


def review_patch(
    diff_text: str,
    *,
    model: str,
    repo: str = "unknown/unknown",
    pr_title: str = "",
    pr_description: str = "",
    min_confidence: float = 0.6,
    min_severity: str | None = None,
    client=None,
    provider: Provider | None = None,
    usage_log: list[dict] | None = None,
    concurrency: int | None = None,
    max_diff_chars: int | None = None,
    max_retries: int | None = None,
    retry_base_seconds: float | None = None,
    models_by_language: dict[str, str] | None = None,
    max_files_per_pr: int | None = None,
    max_tokens_per_pr: int | None = None,
) -> list[dict]:
    """Review a unified diff and return a list of findings (dicts).

    Each finding is keyed by path and validated against the diff:
    - Unknown paths are skipped.
    - Line numbers not present in the new-file side of the diff are dropped.
    - Confidence below `min_confidence` is filtered out.
    - Duplicates keyed by (path, line, title) are removed.

    Files whose total hunks exceed ``max_diff_chars`` are split per-hunk so
    each model request stays under the prompt budget. Up to ``concurrency``
    chunks are reviewed in parallel via a thread pool.

    If `usage_log` is provided, one record per model call is appended to it
    (one record per chunk, not per file):
        {"model", "path", "prompt_tokens", "completion_tokens",
         "total_tokens", "cost_usd"}
    `cost_usd` is None when the model is not in the pricing table.
    """
    if provider is None:
        provider = (
            _LegacyClientAdapter(client) if client is not None
            else build_provider()
        )
    if concurrency is None:
        concurrency = int(os.environ.get(
            "REVIEWER_CONCURRENCY", DEFAULT_CONCURRENCY))
    if max_diff_chars is None:
        max_diff_chars = int(os.environ.get(
            "REVIEWER_MAX_DIFF_CHARS", DEFAULT_MAX_DIFF_CHARS))
    if max_retries is None:
        max_retries = int(os.environ.get(
            "REVIEWER_MAX_RETRIES", DEFAULT_MAX_RETRIES))
    if retry_base_seconds is None:
        retry_base_seconds = float(os.environ.get(
            "REVIEWER_RETRY_BASE_SECONDS", DEFAULT_RETRY_BASE_SECONDS))
    if min_severity is None:
        min_severity = os.environ.get("MIN_SEVERITY", "low")
    if max_files_per_pr is None:
        max_files_per_pr = int(os.environ.get(
            "REVIEWER_MAX_FILES_PER_PR", DEFAULT_MAX_FILES_PER_PR))
    if max_tokens_per_pr is None:
        max_tokens_per_pr = int(os.environ.get(
            "REVIEWER_MAX_TOKENS_PER_PR", DEFAULT_MAX_TOKENS_PER_PR))

    log_lock = threading.Lock()
    patch = PatchSet(diff_text)
    tasks: list[dict] = []
    extra_block = tuple(
        p.strip() for p in os.environ.get("REVIEWER_BLOCK_PATTERNS", "").split(",")
        if p.strip()
    )

    for patched_file in patch:
        if patched_file.is_binary_file:
            continue
        path = patched_file.path
        if should_skip(path):
            continue
        if is_sensitive_path(path, extra_patterns=extra_block):
            print(f"[warn] skipping sensitive path (not sent to model): {path}",
                  file=sys.stderr)
            continue
        new_lines = valid_new_lines(patched_file)
        if not new_lines:
            continue
        lang = language_for(path)
        task_model = (models_by_language or {}).get(lang)
        for _label, hunks_text in _file_chunks(patched_file, max_diff_chars):
            tasks.append({
                "path": path,
                "lang": lang,
                "hunks_text": hunks_text,
                "new_lines": new_lines,
                "task_model": task_model,
            })

    all_findings: list[dict] = []
    if not tasks:
        return []

    # Pre-flight file cap: refuse to spend money on more than N chunks.
    # We truncate deterministically (preserve the diff's natural ordering)
    # and surface the skip in stderr for CI visibility.
    if max_files_per_pr and len(tasks) > max_files_per_pr:
        skipped = len(tasks) - max_files_per_pr
        print(
            f"[warn] cost cap: {len(tasks)} chunks exceed "
            f"max_files_per_pr={max_files_per_pr}; reviewing the first "
            f"{max_files_per_pr}, skipping {skipped}",
            file=sys.stderr,
        )
        tasks = tasks[:max_files_per_pr]

    budget = _CostBudget(max_tokens_per_pr) if max_tokens_per_pr > 0 else None

    common = dict(
        provider=provider, model=model, repo=repo,
        pr_title=pr_title, pr_description=pr_description,
        usage_log=usage_log, log_lock=log_lock,
        max_retries=max_retries, retry_base_seconds=retry_base_seconds,
        budget=budget,
    )
    workers = max(1, min(concurrency, len(tasks)))

    if workers == 1 or len(tasks) == 1:
        for t in tasks:
            all_findings.extend(_review_chunk(**common, **t))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_review_chunk, **common, **t) for t in tasks]
            for fut in as_completed(futures):
                all_findings.extend(fut.result())

    findings = dedupe(all_findings)
    findings = filter_by_confidence(findings, min_confidence)
    findings = filter_by_severity(findings, min_severity)
    findings = sort_findings(findings)
    return findings
