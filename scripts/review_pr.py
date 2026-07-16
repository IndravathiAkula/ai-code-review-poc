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
from reviewer.coverage import (
    coverage_gate_finding, load_coverage, new_code_coverage,
)
from reviewer.github_poster import (
    fetch_pr_diff, fetch_pr_labels, post_findings, upsert_summary_comment,
)
from reviewer.linters import run_all as run_all_linters
from reviewer.pr_review import review_pr_level
from reviewer.providers import build_provider
from reviewer.reporting import (
    render_markdown, summarize_run, write_artifact, write_step_summary,
)
from reviewer.standards import apply_rules, load_rules
from reviewer.utils import (
    dedupe, filter_by_severity, find_skip_label, sort_findings,
)


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
        prompt_extras_by_language=cfg.prompt_extras_by_language or None,
        include_maintainability_findings=cfg.include_maintainability_findings,
        max_files_per_pr=cfg.max_files_per_pr,
        max_tokens_per_pr=cfg.max_tokens_per_pr,
        provider=provider,
        usage_log=usage_log,
        repo_root=str(workspace),
        enable_repo_context=cfg.enable_repo_context,
        max_context_files=cfg.max_context_files,
        max_context_chars=cfg.max_context_chars,
        enable_test_review=cfg.enable_test_review,
    )
    wall = time.perf_counter() - t0
    print(f"[info] {len(findings)} AI finding(s) passed filters in {wall:.2f}s")

    # Deterministic tooling: linters, type-checkers, and standards rules.
    # These run on the checked-out working tree (GITHUB_WORKSPACE) and
    # only surface findings on lines the PR actually added. The extra
    # findings share the LLM's schema so they flow through the same
    # dedupe / severity-filter / post pipeline.
    det_findings: list[dict] = []
    if cfg.enable_linters or cfg.enable_type_check or cfg.enable_semgrep:
        det_findings += run_all_linters(
            diff, workspace,
            enable_lint=cfg.enable_linters,
            enable_type_check=cfg.enable_type_check,
            enable_semgrep=cfg.enable_semgrep,
        )
    if cfg.standards_rules:
        det_findings += apply_rules(diff, load_rules(cfg.standards_rules))
    if det_findings:
        print(f"[info] {len(det_findings)} deterministic finding(s) "
              f"from linters / type-checkers / standards")
        # LLM finding wins when both hit the same (path, line): the
        # model's explanation is usually richer. dedupe() keys on
        # (path, line, title) — different titles from the same line are
        # kept, so lint noise on a line the LLM already flagged only
        # gets suppressed if the titles happen to collide. To also
        # suppress lint on any line the LLM already flagged, drop them
        # here before merging.
        ai_anchors = {(f["path"], int(f["line"])) for f in findings
                      if f.get("source", "ai") == "ai"}
        det_findings = [
            f for f in det_findings
            if (f["path"], int(f["line"])) not in ai_anchors
        ]
        merged = dedupe(findings + det_findings)
        merged = filter_by_severity(merged, cfg.min_severity)
        findings = sort_findings(merged)
        print(f"[info] {len(findings)} total finding(s) after merge")

    # PR-level architectural review — the "senior engineer stepping back"
    # pass. Findings from here have ``line: 0`` and are rendered in the
    # summary comment, not as inline comments.
    pr_level_findings: list[dict] = []
    if cfg.enable_pr_level_review or cfg.enable_missing_tests_check:
        try:
            pr_level_findings = review_pr_level(
                diff,
                provider=provider,
                model=cfg.model,
                pr_title=title,
                pr_description=body,
                repo=repo_full,
                max_retries=cfg.max_retries,
                retry_base_seconds=cfg.retry_base_seconds,
                usage_log=usage_log,
                enable_missing_tests=cfg.enable_missing_tests_check,
            ) if cfg.enable_pr_level_review else []
            if not cfg.enable_pr_level_review and cfg.enable_missing_tests_check:
                # Deterministic check only — no LLM call.
                from reviewer.pr_review import build_summary, missing_tests_finding
                mt = missing_tests_finding(build_summary(diff))
                if mt is not None:
                    pr_level_findings = [mt]
        except Exception as exc:
            print(f"[warn] PR-level review skipped: {exc}", file=sys.stderr)
        # Filter PR-level findings by confidence/severity same as
        # inline. They don't go through dedupe (different keying) but
        # the LLM shouldn't repeat itself.
        pr_level_findings = [
            f for f in pr_level_findings
            if float(f.get("confidence", 0.0)) >= cfg.min_confidence
        ]
        pr_level_findings = filter_by_severity(
            pr_level_findings, cfg.min_severity)
        pr_level_findings = sort_findings(pr_level_findings)
        if pr_level_findings:
            print(f"[info] {len(pr_level_findings)} PR-level finding(s)")

    # Coverage-diff gate. Runs after the LLM/pr_level pipeline so its
    # finding sits alongside the others in the summary comment. Reads a
    # coverage report the target repo's CI is expected to have written
    # earlier in the workflow.
    if cfg.enable_coverage_gate:
        cov_path = workspace / cfg.coverage_report_path
        report = load_coverage(cov_path)
        if report is None:
            print(f"[warn] coverage gate: report not found or unparseable "
                  f"at {cov_path}; skipping gate", file=sys.stderr)
        else:
            ncc = new_code_coverage(diff, report)
            gate = coverage_gate_finding(ncc, cfg.new_code_coverage_threshold)
            if gate is not None:
                pr_level_findings.append(gate)
                print(
                    f"[info] coverage gate: {ncc.covered}/"
                    f"{ncc.total_instrumented} new lines covered "
                    f"({ncc.ratio:.0%}) — below threshold "
                    f"{cfg.new_code_coverage_threshold:.0%}"
                )
            else:
                print(
                    f"[info] coverage gate passed: {ncc.covered}/"
                    f"{ncc.total_instrumented} new lines covered "
                    f"({ncc.ratio:.0%})"
                )

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
        pr_level_findings=pr_level_findings,
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
