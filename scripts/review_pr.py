"""Review a GitHub PR and post inline comments. Designed for CI (GitHub Actions)."""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

from reviewer import review_patch
from reviewer.config import effective_config, load_config
from reviewer.github_poster import (
    fetch_pr_diff, fetch_pr_labels, post_findings, upsert_summary_comment,
)
from reviewer.providers import build_provider
from reviewer.reporting import (
    render_markdown, summarize_run, write_artifact, write_step_summary,
)
from reviewer.utils import find_skip_label


def _require(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        print(f"error: required env var {name} is not set", file=sys.stderr)
        sys.exit(2)
    return v


def main() -> int:
    load_dotenv()
    token = _require("GITHUB_TOKEN")
    repo_full = _require("GITHUB_REPOSITORY")  # "owner/name"
    pr_number = int(_require("PR_NUMBER"))
    dry_run = os.environ.get("DRY_RUN", "").lower() in ("1", "true", "yes")
    artifact_path = os.environ.get(
        "REVIEWER_ARTIFACT_PATH", str(ROOT / "ai-review-run.json"))

    # Repo-level .ai-review.yml lives in the checkout root. In Actions
    # that's GITHUB_WORKSPACE; locally we just look in cwd.
    workspace = Path(os.environ.get("GITHUB_WORKSPACE", "."))
    cfg = effective_config(config_file=load_config(workspace / ".ai-review.yml"))

    print(f"[info] reviewing {repo_full} PR #{pr_number} via "
          f"{cfg.provider}/{cfg.model}")
    print(f"[info] config: min_conf={cfg.min_confidence} "
          f"min_sev={cfg.min_severity} concurrency={cfg.concurrency} "
          f"block_patterns={list(cfg.block_patterns)} "
          f"skip_labels={list(cfg.skip_labels)}")

    # Short-circuit on opt-out label BEFORE any model cost is incurred.
    if cfg.skip_labels:
        try:
            pr_labels = fetch_pr_labels(token, repo_full, pr_number)
        except Exception as exc:
            # Don't let a label fetch failure block reviews; log and continue.
            print(f"[warn] could not fetch PR labels ({exc}); "
                  f"proceeding with review", file=sys.stderr)
        else:
            matched = find_skip_label(pr_labels, cfg.skip_labels)
            if matched:
                print(f"[info] skipped: label {matched!r} present on "
                      f"PR #{pr_number} (matched {list(cfg.skip_labels)})")
                return 0

    provider = build_provider(cfg.provider)
    diff, title, body = fetch_pr_diff(token, repo_full, pr_number)
    print(f"[info] diff size: {len(diff)} bytes")

    usage_log: list[dict] = []
    t0 = time.perf_counter()
    findings = review_patch(
        diff,
        model=cfg.model,
        repo=repo_full,
        pr_title=title,
        pr_description=body,
        min_confidence=cfg.min_confidence,
        min_severity=cfg.min_severity,
        concurrency=cfg.concurrency,
        max_diff_chars=cfg.max_diff_chars,
        max_retries=cfg.max_retries,
        retry_base_seconds=cfg.retry_base_seconds,
        models_by_language=cfg.models_by_language or None,
        max_files_per_pr=cfg.max_files_per_pr,
        max_tokens_per_pr=cfg.max_tokens_per_pr,
        provider=provider,
        usage_log=usage_log,
    )
    wall = time.perf_counter() - t0
    print(f"[info] {len(findings)} finding(s) passed filters in {wall:.2f}s")

    if findings:
        report = post_findings(
            token=token,
            repo_full_name=repo_full,
            pr_number=pr_number,
            findings=findings,
            cleanup=True,
            dry_run=dry_run,
        )
        print(f"[info] posted={report['posted']} "
              f"kept={report.get('kept', 0)} "
              f"skipped={report['skipped']} "
              f"removed_stale={report['removed_stale']}")
    else:
        report = {"posted": 0, "kept": 0, "skipped": 0, "removed_stale": 0}

    summary = summarize_run(
        model=cfg.model, repo=repo_full, pr_number=pr_number,
        findings=findings, post_report=report, usage_log=usage_log,
        wall_seconds=wall,
    )
    md = render_markdown(summary)

    artifact = write_artifact(artifact_path, summary, usage_log)
    print(f"[info] wrote artifact {artifact}")

    if write_step_summary(md):
        print("[info] appended GitHub step summary")

    if cfg.post_summary and not dry_run:
        action = upsert_summary_comment(token, repo_full, pr_number, md)
        print(f"[info] PR summary comment: {action}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
