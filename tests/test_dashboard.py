"""Tests for the dashboard aggregator and HTML renderer.

We don't open the HTML in a browser — we just verify the aggregate
math is right and the rendered HTML contains the expected markers
and data. Plotly is loaded via CDN at view time, so test runs need
no network."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from reviewer.dashboard import (
    DashboardAggregate, aggregate_runs, load_runs, main,
    render_html,
)


def _write_artifact(tmp_path: Path, name: str, payload: dict) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _sample_artifact(**overrides):
    """A minimal artifact JSON shaped like ``reporting.write_artifact``
    produces. Tests override individual fields."""
    summary = {
        "model": "openai/gpt-4o-mini",
        "repo": "o/r",
        "pr_number": 1,
        "posted": 2,
        "kept": 0,
        "skipped": 0,
        "removed_stale": 0,
        "severity_counts": {"high": 2},
        "total_calls": 1,
        "prompt_tokens": 400,
        "completion_tokens": 100,
        "total_tokens": 500,
        "cost_usd": 0.0001,
        "wall_seconds": 2.0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
        "run_timestamp": "2026-05-15T12:00:00+00:00",
        "provider": "github-models",
    }
    summary.update(overrides)
    return {"summary": summary, "usage_log": []}


# ---------- load_runs ----------

def test_load_runs_returns_empty_for_empty_directory(tmp_path):
    assert load_runs(tmp_path) == []


def test_load_runs_skips_non_json_files(tmp_path):
    (tmp_path / "note.txt").write_text("hi", encoding="utf-8")
    _write_artifact(tmp_path, "run.json", _sample_artifact())
    runs = load_runs(tmp_path)
    assert len(runs) == 1


def test_load_runs_skips_invalid_json(tmp_path):
    (tmp_path / "broken.json").write_text("not json {{{", encoding="utf-8")
    _write_artifact(tmp_path, "good.json", _sample_artifact())
    runs = load_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0]["summary"]["pr_number"] == 1


def test_load_runs_skips_files_without_summary_block(tmp_path):
    _write_artifact(tmp_path, "weird.json", {"not_a_summary": True})
    _write_artifact(tmp_path, "good.json", _sample_artifact())
    runs = load_runs(tmp_path)
    assert len(runs) == 1


def test_load_runs_sorts_by_run_timestamp(tmp_path):
    _write_artifact(tmp_path, "b.json", _sample_artifact(
        pr_number=2, run_timestamp="2026-05-15T15:00:00+00:00"))
    _write_artifact(tmp_path, "a.json", _sample_artifact(
        pr_number=1, run_timestamp="2026-05-15T12:00:00+00:00"))
    runs = load_runs(tmp_path)
    assert [r["summary"]["pr_number"] for r in runs] == [1, 2]


def test_load_runs_falls_back_to_filename_when_no_timestamp(tmp_path):
    _write_artifact(tmp_path, "b.json",
                    {"summary": {"pr_number": 2, "posted": 0}})
    _write_artifact(tmp_path, "a.json",
                    {"summary": {"pr_number": 1, "posted": 0}})
    runs = load_runs(tmp_path)
    assert [r["summary"]["pr_number"] for r in runs] == [1, 2]


def test_load_runs_raises_on_missing_directory(tmp_path):
    with pytest.raises(RuntimeError, match="not a directory"):
        load_runs(tmp_path / "does-not-exist")


# ---------- aggregate_runs ----------

def test_aggregate_runs_empty_returns_zeros():
    agg = aggregate_runs([])
    assert agg.total_runs == 0
    assert agg.total_cost_usd == 0.0
    assert agg.total_findings_posted == 0


def test_aggregate_runs_sums_costs_and_tokens():
    runs = [
        _sample_artifact(cost_usd=0.001, total_tokens=500),
        _sample_artifact(cost_usd=0.002, total_tokens=700),
        _sample_artifact(cost_usd=0.003, total_tokens=1000),
    ]
    agg = aggregate_runs(runs)
    assert agg.total_runs == 3
    assert abs(agg.total_cost_usd - 0.006) < 1e-9
    assert agg.total_tokens == 2200


def test_aggregate_runs_handles_unpriced_runs():
    """When cost_usd is None, the run is skipped for cost but counted
    elsewhere (so total_runs and tokens are still right)."""
    runs = [
        _sample_artifact(cost_usd=None, total_tokens=500),
        _sample_artifact(cost_usd=0.001, total_tokens=300),
    ]
    agg = aggregate_runs(runs)
    assert agg.total_runs == 2
    assert abs(agg.total_cost_usd - 0.001) < 1e-9


def test_aggregate_runs_groups_by_provider():
    runs = [
        _sample_artifact(provider="github-models", cost_usd=0.0001),
        _sample_artifact(provider="anthropic", cost_usd=0.003),
        _sample_artifact(provider="anthropic", cost_usd=0.002),
    ]
    agg = aggregate_runs(runs)
    assert set(agg.provider_totals) == {"github-models", "anthropic"}
    assert abs(agg.provider_totals["anthropic"]["cost_usd"] - 0.005) < 1e-9


def test_aggregate_runs_groups_by_model():
    runs = [
        _sample_artifact(model="openai/gpt-4o-mini", cost_usd=0.0001),
        _sample_artifact(model="openai/gpt-4o", cost_usd=0.003),
    ]
    agg = aggregate_runs(runs)
    assert "openai/gpt-4o-mini" in agg.model_totals
    assert "openai/gpt-4o" in agg.model_totals


def test_aggregate_runs_sums_severity_counts():
    runs = [
        _sample_artifact(severity_counts={"critical": 1, "high": 2}),
        _sample_artifact(severity_counts={"high": 1, "medium": 3}),
    ]
    agg = aggregate_runs(runs)
    assert agg.severity_totals == {"critical": 1, "high": 3, "medium": 3}


def test_aggregate_runs_computes_cache_hit_rate():
    """A run with mostly cache reads should report a high hit rate."""
    runs = [
        _sample_artifact(
            prompt_tokens=100, cache_read_tokens=900, cache_creation_tokens=0),
        _sample_artifact(
            prompt_tokens=100, cache_read_tokens=900, cache_creation_tokens=0),
    ]
    agg = aggregate_runs(runs)
    # 1800 cache_read / (200 prompt + 1800 cache) = 0.9
    assert abs(agg.cache_hit_rate - 0.9) < 1e-9


def test_aggregate_runs_cache_hit_rate_zero_when_no_caching():
    runs = [_sample_artifact(
        prompt_tokens=500, cache_read_tokens=0, cache_creation_tokens=0)]
    agg = aggregate_runs(runs)
    assert agg.cache_hit_rate == 0.0


def test_aggregate_runs_averages_wall_seconds():
    runs = [
        _sample_artifact(wall_seconds=2.0),
        _sample_artifact(wall_seconds=4.0),
        _sample_artifact(wall_seconds=6.0),
    ]
    agg = aggregate_runs(runs)
    assert agg.avg_wall_seconds == 4.0


# ---------- render_html ----------

def test_render_html_contains_aggregate_kpis():
    agg = DashboardAggregate(
        total_runs=3, total_cost_usd=0.005,
        total_findings_posted=7, total_findings_kept=2,
    )
    html = render_html(agg, [])
    assert "AI Code Review Dashboard" in html
    assert "<div class=\"value\">3</div>" in html  # total_runs KPI
    assert "$0.0050" in html
    assert "<div class=\"value\">7</div>" in html  # findings posted KPI


def test_render_html_runs_table_lists_each_run():
    runs = [
        _sample_artifact(pr_number=1, model="openai/gpt-4o-mini",
                          run_timestamp="2026-05-15T10:00:00+00:00"),
        _sample_artifact(pr_number=2, model="claude-sonnet-4-6",
                          run_timestamp="2026-05-15T11:00:00+00:00"),
    ]
    agg = aggregate_runs(runs)
    html = render_html(agg, runs)
    assert "#1" in html
    assert "#2" in html
    assert "openai/gpt-4o-mini" in html
    assert "claude-sonnet-4-6" in html
    assert "2026-05-15 10:00" in html


def test_render_html_empty_runs_shows_empty_message():
    agg = DashboardAggregate()
    html = render_html(agg, [])
    assert "no artifacts found" in html


def test_render_html_includes_plotly_cdn():
    html = render_html(DashboardAggregate(), [])
    assert "cdn.plot.ly/plotly" in html


def test_render_html_escapes_user_supplied_strings():
    """Defense against malformed model names with HTML in them
    (paranoid: we control the inputs, but escape() is cheap insurance)."""
    runs = [_sample_artifact(model="<script>alert(1)</script>")]
    html = render_html(aggregate_runs(runs), runs)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


# ---------- CLI ----------

def test_main_writes_html_file(tmp_path):
    inputs = tmp_path / "artifacts"
    inputs.mkdir()
    _write_artifact(inputs, "run.json", _sample_artifact())
    out = tmp_path / "dashboard.html"
    exit_code = main(["--inputs", str(inputs), "--out", str(out)])
    assert exit_code == 0
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "AI Code Review Dashboard" in text


def test_main_creates_parent_dirs(tmp_path):
    inputs = tmp_path / "artifacts"
    inputs.mkdir()
    _write_artifact(inputs, "run.json", _sample_artifact())
    out = tmp_path / "nested" / "subdir" / "dash.html"
    exit_code = main(["--inputs", str(inputs), "--out", str(out)])
    assert exit_code == 0
    assert out.exists()


def test_main_errors_on_missing_inputs_dir(tmp_path):
    out = tmp_path / "x.html"
    exit_code = main([
        "--inputs", str(tmp_path / "nope"),
        "--out", str(out),
    ])
    assert exit_code == 2
    assert not out.exists()
