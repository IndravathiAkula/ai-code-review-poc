import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reviewer.reporting import (
    SUMMARY_TAG, RunSummary, render_markdown, summarize_run,
    write_artifact, write_step_summary,
)


def _usage(prompt, completion, cost=None, latency=0.5):
    return {
        "model": "openai/gpt-4o-mini",
        "path": "x.py",
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "cost_usd": cost,
        "latency_seconds": latency,
    }


def _findings(*severities):
    return [{"severity": s, "category": "security",
             "title": f"f{i}", "explanation": "...", "confidence": 0.9}
            for i, s in enumerate(severities)]


def test_summarize_run_aggregates_token_and_cost_totals():
    s = summarize_run(
        model="openai/gpt-4o-mini", repo="o/r", pr_number=42,
        findings=_findings("high", "high", "critical"),
        post_report={"posted": 2, "kept": 1, "skipped": 0, "removed_stale": 1},
        usage_log=[_usage(100, 50, cost=0.0001),
                   _usage(200, 80, cost=0.0002)],
        wall_seconds=2.5,
    )
    assert s.posted == 2
    assert s.kept == 1
    assert s.removed_stale == 1
    assert s.total_calls == 2
    assert s.prompt_tokens == 300
    assert s.completion_tokens == 130
    assert s.total_tokens == 430
    assert abs(s.cost_usd - 0.0003) < 1e-9
    assert s.severity_counts == {"high": 2, "critical": 1}
    assert s.wall_seconds == 2.5


def test_summarize_run_cost_is_none_when_no_priced_calls():
    s = summarize_run(
        model="unknown/x", repo="o/r", pr_number=1,
        findings=[], post_report={},
        usage_log=[_usage(100, 50, cost=None)],
        wall_seconds=0.1,
    )
    assert s.cost_usd is None


def test_summarize_run_handles_empty_usage_log():
    s = summarize_run(
        model="x", repo="o/r", pr_number=1,
        findings=[], post_report={},
        usage_log=[], wall_seconds=0.0,
    )
    assert s.total_calls == 0
    assert s.total_tokens == 0
    assert s.cost_usd is None


def test_render_markdown_contains_tag_and_key_numbers():
    s = RunSummary(
        model="openai/gpt-4o-mini", repo="o/r", pr_number=42,
        posted=3, kept=2, skipped=0, removed_stale=1,
        severity_counts={"critical": 1, "high": 2},
        total_calls=4, prompt_tokens=4200, completion_tokens=1232,
        total_tokens=5432, cost_usd=0.0009, wall_seconds=2.34,
    )
    md = render_markdown(s)
    assert SUMMARY_TAG in md
    assert "PR #42" in md
    assert "5,432" in md  # total tokens with thousands separator
    assert "4,200" in md  # prompt
    assert "1,232" in md  # completion
    assert "1 critical, 2 high" in md
    assert "$0.0009" in md
    assert "2.34s" in md


def test_render_markdown_does_not_leak_model_name():
    """The PR-level summary comment is user-facing; the model name stays
    in the artifact JSON + CI log instead so the comment looks neutral."""
    s = RunSummary(
        model="anthropic/claude-sonnet-4-6", repo="o/r", pr_number=1,
        posted=1, kept=0, skipped=0, removed_stale=0,
        severity_counts={"high": 1}, total_calls=1,
        prompt_tokens=10, completion_tokens=5, total_tokens=15,
        cost_usd=0.0001, wall_seconds=1.0,
    )
    md = render_markdown(s)
    assert "claude-sonnet" not in md
    assert "anthropic" not in md
    assert s.model not in md


def test_render_markdown_unpriced_run_shows_dash():
    s = RunSummary(
        model="x", repo="o/r", pr_number=1,
        posted=0, kept=0, skipped=0, removed_stale=0,
        severity_counts={}, total_calls=1,
        prompt_tokens=10, completion_tokens=5, total_tokens=15,
        cost_usd=None, wall_seconds=0.1,
    )
    md = render_markdown(s)
    assert "—" in md  # unicode em-dash for missing cost


def test_render_markdown_severity_breakdown_orders_critical_first():
    s = RunSummary(
        model="x", repo="o/r", pr_number=1,
        posted=0, kept=0, skipped=0, removed_stale=0,
        severity_counts={"low": 5, "critical": 1, "medium": 3, "high": 2},
        total_calls=0, prompt_tokens=0, completion_tokens=0,
        total_tokens=0, cost_usd=None, wall_seconds=0.0,
    )
    md = render_markdown(s)
    # critical -> high -> medium -> low ordering
    cri = md.find("1 critical")
    hi = md.find("2 high")
    me = md.find("3 medium")
    lo = md.find("5 low")
    assert cri < hi < me < lo


def test_render_markdown_severity_breakdown_none_when_empty():
    s = RunSummary(
        model="x", repo="o/r", pr_number=1,
        posted=0, kept=0, skipped=0, removed_stale=0,
        severity_counts={}, total_calls=0,
        prompt_tokens=0, completion_tokens=0, total_tokens=0,
        cost_usd=None, wall_seconds=0.0,
    )
    assert "Severity breakdown:** none" in render_markdown(s)


def test_write_step_summary_no_op_when_env_unset():
    assert write_step_summary("body", env={}) is False


def test_write_step_summary_appends_to_file(tmp_path):
    target = tmp_path / "summary.md"
    env = {"GITHUB_STEP_SUMMARY": str(target)}
    assert write_step_summary("first run", env=env) is True
    assert write_step_summary("second run", env=env) is True
    text = target.read_text(encoding="utf-8")
    assert "first run" in text
    assert "second run" in text


def test_summarize_run_aggregates_cache_tokens():
    """Cache hit/write tokens (from Anthropic) flow through to the summary."""
    log = [
        {"prompt_tokens": 200, "completion_tokens": 100,
         "cache_read_tokens": 0, "cache_creation_tokens": 800,
         "cost_usd": 0.001},  # chunk 1: writes cache
        {"prompt_tokens": 200, "completion_tokens": 100,
         "cache_read_tokens": 800, "cache_creation_tokens": 0,
         "cost_usd": 0.0005},  # chunk 2+: reads cache
        {"prompt_tokens": 200, "completion_tokens": 100,
         "cache_read_tokens": 800, "cache_creation_tokens": 0,
         "cost_usd": 0.0005},
    ]
    s = summarize_run(
        model="claude-sonnet-4-6", repo="o/r", pr_number=1,
        findings=[], post_report={},
        usage_log=log, wall_seconds=2.0,
    )
    assert s.cache_creation_tokens == 800
    assert s.cache_read_tokens == 1600
    assert s.total_calls == 3


def test_render_markdown_shows_cache_line_when_present():
    s = RunSummary(
        model="claude-sonnet-4-6", repo="o/r", pr_number=1,
        posted=0, kept=0, skipped=0, removed_stale=0,
        severity_counts={}, total_calls=3,
        prompt_tokens=600, completion_tokens=300, total_tokens=2500,
        cost_usd=0.002, wall_seconds=2.0,
        cache_read_tokens=1600, cache_creation_tokens=800,
    )
    md = render_markdown(s)
    assert "**Cache:**" in md
    assert "1,600 hits" in md
    assert "800 writes" in md


def test_render_markdown_omits_cache_line_when_no_caching():
    """Non-Anthropic runs shouldn't show a cache line at all."""
    s = RunSummary(
        model="openai/gpt-4o-mini", repo="o/r", pr_number=1,
        posted=0, kept=0, skipped=0, removed_stale=0,
        severity_counts={}, total_calls=1,
        prompt_tokens=100, completion_tokens=50, total_tokens=150,
        cost_usd=0.00005, wall_seconds=1.0,
    )
    assert "Cache:" not in render_markdown(s)


def test_summarize_run_stores_pr_level_findings():
    plf = [
        {"path": "(pull request)", "line": 0, "severity": "medium",
         "category": "maintainability",
         "title": "Missing tests", "explanation": "no test files touched",
         "confidence": 0.9, "source": "pr_review"},
    ]
    s = summarize_run(
        model="x", repo="o/r", pr_number=7,
        findings=[], post_report={},
        usage_log=[], wall_seconds=0.1,
        pr_level_findings=plf,
    )
    assert s.pr_level_findings == plf


def test_render_markdown_includes_pr_level_section():
    s = RunSummary(
        model="x", repo="o/r", pr_number=7,
        posted=0, kept=0, skipped=0, removed_stale=0,
        severity_counts={}, total_calls=0,
        prompt_tokens=0, completion_tokens=0, total_tokens=0,
        cost_usd=None, wall_seconds=0.0,
        pr_level_findings=[{
            "path": "svc/x.py", "line": 0, "severity": "high",
            "category": "maintainability",
            "title": "Breaking change: process_order removed",
            "explanation": "Callers in web/orders.tsx still reference it.",
            "confidence": 0.9, "source": "pr_review",
        }],
    )
    md = render_markdown(s)
    assert "### PR-level review" in md
    assert "Breaking change" in md
    assert "process_order removed" in md
    assert "HIGH / maintainability" in md
    # PR-level count shown in the metric table
    assert "| PR-level findings | 1 |" in md


def test_render_markdown_omits_pr_level_section_when_none():
    s = RunSummary(
        model="x", repo="o/r", pr_number=1,
        posted=0, kept=0, skipped=0, removed_stale=0,
        severity_counts={}, total_calls=0,
        prompt_tokens=0, completion_tokens=0, total_tokens=0,
        cost_usd=None, wall_seconds=0.0,
    )
    md = render_markdown(s)
    assert "### PR-level review" not in md
    assert "| PR-level findings | 0 |" in md


def test_write_artifact_writes_summary_and_usage_log_as_json(tmp_path):
    s = RunSummary(
        model="x", repo="o/r", pr_number=1,
        posted=1, kept=0, skipped=0, removed_stale=0,
        severity_counts={"high": 1}, total_calls=1,
        prompt_tokens=10, completion_tokens=5, total_tokens=15,
        cost_usd=0.0001, wall_seconds=0.1,
    )
    log = [_usage(10, 5, cost=0.0001)]
    out = write_artifact(tmp_path / "run.json", s, log)
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["summary"]["model"] == "x"
    assert payload["summary"]["total_tokens"] == 15
    assert len(payload["usage_log"]) == 1
    assert payload["usage_log"][0]["prompt_tokens"] == 10
