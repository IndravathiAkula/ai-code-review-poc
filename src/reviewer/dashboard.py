"""Dashboard generator — aggregate per-PR artifacts into one HTML report.

Every CI run writes an ``ai-review-run-{N}.json`` artifact (see
``reporting.write_artifact``). Over time those artifacts accumulate
into a real dataset about your reviewer's cost, recall, and FP rate
on your actual codebase. This module reads a directory of those
artifacts and emits a self-contained HTML dashboard with trend lines.

Use:

    python -m reviewer.dashboard --inputs ./artifacts --out dashboard.html

The dashboard uses Plotly via CDN so the output is a single HTML file
with no Python dependencies on the viewer end. Open it in any browser.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any


# ---------- loading ----------

def load_runs(directory: str | Path) -> list[dict]:
    """Read every artifact JSON in ``directory`` (non-recursive).

    Files that don't parse, or that don't carry a ``summary`` block,
    are skipped silently. Returns a list of raw artifact dicts sorted
    by ``run_timestamp`` if present, else by filename.
    """
    d = Path(directory)
    if not d.is_dir():
        raise RuntimeError(f"not a directory: {d}")
    runs: list[dict] = []
    for path in sorted(d.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        summary = payload.get("summary")
        if not isinstance(summary, dict):
            continue
        payload["_source_file"] = path.name
        runs.append(payload)
    runs.sort(key=_run_sort_key)
    return runs


def _run_sort_key(payload: dict) -> str:
    ts = payload.get("summary", {}).get("run_timestamp") or ""
    return ts or payload.get("_source_file", "")


# ---------- aggregation ----------

@dataclass
class DashboardAggregate:
    total_runs: int = 0
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    total_findings_posted: int = 0
    total_findings_kept: int = 0
    total_findings_removed_stale: int = 0
    total_chunks_skipped: int = 0
    avg_wall_seconds: float = 0.0
    severity_totals: dict[str, int] = field(default_factory=dict)
    provider_totals: dict[str, dict] = field(default_factory=dict)
    model_totals: dict[str, dict] = field(default_factory=dict)
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_hit_rate: float = 0.0  # cache_read / (cache_read + regular input)

    def to_dict(self) -> dict:
        return {
            "total_runs": self.total_runs,
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_tokens,
            "total_findings_posted": self.total_findings_posted,
            "total_findings_kept": self.total_findings_kept,
            "total_findings_removed_stale": self.total_findings_removed_stale,
            "total_chunks_skipped": self.total_chunks_skipped,
            "avg_wall_seconds": self.avg_wall_seconds,
            "severity_totals": self.severity_totals,
            "provider_totals": self.provider_totals,
            "model_totals": self.model_totals,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cache_hit_rate": self.cache_hit_rate,
        }


def aggregate_runs(runs: list[dict]) -> DashboardAggregate:
    """Roll N artifact dicts into one summary."""
    agg = DashboardAggregate()
    wall_seconds_samples: list[float] = []
    prompt_tokens_total = 0

    for r in runs:
        s = r.get("summary", {})
        agg.total_runs += 1
        cost = s.get("cost_usd")
        if cost is not None:
            agg.total_cost_usd += float(cost)
        agg.total_tokens += int(s.get("total_tokens", 0) or 0)
        agg.total_findings_posted += int(s.get("posted", 0) or 0)
        agg.total_findings_kept += int(s.get("kept", 0) or 0)
        agg.total_findings_removed_stale += int(s.get("removed_stale", 0) or 0)
        agg.total_chunks_skipped += int(s.get("skipped", 0) or 0)

        wall = s.get("wall_seconds")
        if wall is not None:
            wall_seconds_samples.append(float(wall))

        for sev, n in (s.get("severity_counts") or {}).items():
            agg.severity_totals[sev] = agg.severity_totals.get(sev, 0) + int(n)

        provider = s.get("provider") or "unknown"
        pt = int(s.get("prompt_tokens", 0) or 0)
        ct = int(s.get("completion_tokens", 0) or 0)
        cr = int(s.get("cache_read_tokens", 0) or 0)
        cw = int(s.get("cache_creation_tokens", 0) or 0)
        prompt_tokens_total += pt

        bucket = agg.provider_totals.setdefault(
            provider, {"calls": 0, "tokens": 0, "cost_usd": 0.0})
        bucket["calls"] += int(s.get("total_calls", 0) or 0)
        bucket["tokens"] += pt + ct + cr + cw
        if cost is not None:
            bucket["cost_usd"] += float(cost)

        model = s.get("model") or "unknown"
        mb = agg.model_totals.setdefault(
            model, {"calls": 0, "tokens": 0, "cost_usd": 0.0})
        mb["calls"] += int(s.get("total_calls", 0) or 0)
        mb["tokens"] += pt + ct + cr + cw
        if cost is not None:
            mb["cost_usd"] += float(cost)

        agg.cache_read_tokens += cr
        agg.cache_creation_tokens += cw

    if wall_seconds_samples:
        agg.avg_wall_seconds = statistics.mean(wall_seconds_samples)

    # Cache hit rate is meaningful only when caching was attempted.
    cached_input = agg.cache_read_tokens + agg.cache_creation_tokens
    total_input = prompt_tokens_total + cached_input
    if total_input > 0:
        agg.cache_hit_rate = agg.cache_read_tokens / total_input

    return agg


# ---------- HTML rendering ----------

_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def render_html(
    aggregate: DashboardAggregate,
    runs: list[dict],
    *,
    title: str = "AI Code Review Dashboard",
) -> str:
    """Render the aggregate + per-run timeline as a self-contained HTML doc.

    Charts are Plotly via CDN — opens in any browser, no Python on the
    consumer side. The dashboard is read-only and timeless: regenerating
    it costs nothing and the output reflects whatever artifacts you
    passed in.
    """
    timeline = _build_timeline(runs)
    provider_pie = _build_provider_pie(aggregate.provider_totals)
    severity_bar = _build_severity_bar(aggregate.severity_totals)
    runs_table = _build_runs_table(runs)
    aggregate_json = json.dumps(aggregate.to_dict(), indent=2)

    cost_fmt = (
        f"${aggregate.total_cost_usd:.4f}" if aggregate.total_cost_usd else "$0.00"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{escape(title)}</title>
<script src="{_PLOTLY_CDN}"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          margin: 2rem; max-width: 1100px; color: #1f2328; }}
  h1 {{ margin-bottom: 0.25rem; }}
  .subtitle {{ color: #6e7681; margin-top: 0; }}
  .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
           gap: 1rem; margin: 1.5rem 0; }}
  .kpi {{ background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px;
          padding: 1rem; }}
  .kpi .label {{ color: #6e7681; font-size: 0.85rem; text-transform: uppercase; }}
  .kpi .value {{ font-size: 1.6rem; font-weight: 600; margin-top: 0.25rem; }}
  .chart {{ margin: 2rem 0; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #d0d7de; }}
  th {{ background: #f6f8fa; font-weight: 600; }}
  details {{ margin-top: 2rem; }}
  pre {{ background: #f6f8fa; padding: 1rem; overflow-x: auto; font-size: 0.85rem; }}
  .empty {{ text-align: center; padding: 3rem; color: #6e7681; }}
</style>
</head>
<body>
<h1>{escape(title)}</h1>
<p class="subtitle">{aggregate.total_runs} runs aggregated from disk.</p>

<div class="kpis">
  <div class="kpi"><div class="label">Total runs</div>
    <div class="value">{aggregate.total_runs}</div></div>
  <div class="kpi"><div class="label">Total cost</div>
    <div class="value">{cost_fmt}</div></div>
  <div class="kpi"><div class="label">Findings posted</div>
    <div class="value">{aggregate.total_findings_posted}</div></div>
  <div class="kpi"><div class="label">Findings kept (re-runs)</div>
    <div class="value">{aggregate.total_findings_kept}</div></div>
  <div class="kpi"><div class="label">Stale removed</div>
    <div class="value">{aggregate.total_findings_removed_stale}</div></div>
  <div class="kpi"><div class="label">Avg latency</div>
    <div class="value">{aggregate.avg_wall_seconds:.2f}s</div></div>
  <div class="kpi"><div class="label">Total tokens</div>
    <div class="value">{aggregate.total_tokens:,}</div></div>
  <div class="kpi"><div class="label">Cache hit rate</div>
    <div class="value">{aggregate.cache_hit_rate*100:.1f}%</div></div>
</div>

<h2>Cost per run over time</h2>
<div id="chart-cost" class="chart"></div>

<h2>Findings posted per run</h2>
<div id="chart-findings" class="chart"></div>

<h2>Cost share by provider</h2>
<div id="chart-provider" class="chart"></div>

<h2>Findings by severity (across all runs)</h2>
<div id="chart-severity" class="chart"></div>

<h2>All runs</h2>
{runs_table}

<details>
<summary>Raw aggregate (JSON)</summary>
<pre>{escape(aggregate_json)}</pre>
</details>

<script>
const timeline = {json.dumps(timeline)};
const providerPie = {json.dumps(provider_pie)};
const severityBar = {json.dumps(severity_bar)};

if (timeline.timestamps.length === 0) {{
  document.getElementById("chart-cost").innerHTML =
    '<div class="empty">no runs to plot yet — open a PR and re-run this script</div>';
  document.getElementById("chart-findings").innerHTML = '';
}} else {{
  Plotly.newPlot("chart-cost", [{{
    x: timeline.timestamps,
    y: timeline.cost,
    type: "scatter",
    mode: "lines+markers",
    name: "$/run",
    line: {{ color: "#0969da" }},
  }}], {{
    margin: {{ t: 10, l: 60, r: 20, b: 60 }},
    yaxis: {{ title: "USD" }},
  }}, {{ responsive: true }});

  Plotly.newPlot("chart-findings", [
    {{ x: timeline.timestamps, y: timeline.posted, name: "posted", type: "bar",
       marker: {{ color: "#cf222e" }} }},
    {{ x: timeline.timestamps, y: timeline.kept, name: "kept", type: "bar",
       marker: {{ color: "#1f883d" }} }},
  ], {{
    barmode: "stack",
    margin: {{ t: 10, l: 60, r: 20, b: 60 }},
    yaxis: {{ title: "findings" }},
  }}, {{ responsive: true }});
}}

if (providerPie.values.length > 0 && providerPie.values.some(v => v > 0)) {{
  Plotly.newPlot("chart-provider", [{{
    labels: providerPie.labels, values: providerPie.values, type: "pie",
  }}], {{ margin: {{ t: 10, l: 20, r: 20, b: 20 }} }},
     {{ responsive: true }});
}} else {{
  document.getElementById("chart-provider").innerHTML =
    '<div class="empty">no cost data — providers without pricing tables show $0</div>';
}}

if (severityBar.x.length > 0) {{
  Plotly.newPlot("chart-severity", [{{
    x: severityBar.x, y: severityBar.y, type: "bar",
    marker: {{ color: ["#cf222e", "#fb8500", "#bf8700", "#6e7681"] }},
  }}], {{
    margin: {{ t: 10, l: 60, r: 20, b: 60 }},
    yaxis: {{ title: "count" }},
  }}, {{ responsive: true }});
}} else {{
  document.getElementById("chart-severity").innerHTML =
    '<div class="empty">no findings yet</div>';
}}
</script>
</body>
</html>
"""


def _build_timeline(runs: list[dict]) -> dict:
    timestamps: list[str] = []
    cost: list[float] = []
    posted: list[int] = []
    kept: list[int] = []
    for r in runs:
        s = r.get("summary", {})
        ts = s.get("run_timestamp") or r.get("_source_file", "")
        timestamps.append(ts)
        cost.append(float(s.get("cost_usd") or 0))
        posted.append(int(s.get("posted", 0) or 0))
        kept.append(int(s.get("kept", 0) or 0))
    return {"timestamps": timestamps, "cost": cost,
            "posted": posted, "kept": kept}


def _build_provider_pie(provider_totals: dict) -> dict:
    labels: list[str] = []
    values: list[float] = []
    for name, totals in provider_totals.items():
        labels.append(name)
        values.append(float(totals.get("cost_usd") or 0))
    return {"labels": labels, "values": values}


def _build_severity_bar(severity_totals: dict) -> dict:
    order = ["critical", "high", "medium", "low"]
    x = [s for s in order if s in severity_totals]
    extra = [s for s in severity_totals if s not in order]
    x.extend(sorted(extra))
    y = [severity_totals[s] for s in x]
    return {"x": x, "y": y}


def _build_runs_table(runs: list[dict]) -> str:
    if not runs:
        return '<div class="empty">no artifacts found</div>'

    rows = []
    rows.append(
        "<tr><th>Time</th><th>PR</th><th>Provider</th><th>Model</th>"
        "<th>Posted</th><th>Kept</th><th>Stale</th>"
        "<th>Tokens</th><th>Cost</th><th>Wall</th></tr>"
    )
    for r in runs:
        s = r.get("summary", {})
        ts = _format_timestamp(s.get("run_timestamp"))
        pr = s.get("pr_number")
        pr_cell = f"#{pr}" if pr else "—"
        cost = s.get("cost_usd")
        cost_cell = f"${float(cost):.4f}" if cost is not None else "—"
        rows.append(
            "<tr>"
            f"<td>{escape(ts)}</td>"
            f"<td>{escape(str(pr_cell))}</td>"
            f"<td>{escape(s.get('provider') or '—')}</td>"
            f"<td>{escape(s.get('model') or '—')}</td>"
            f"<td>{int(s.get('posted', 0) or 0)}</td>"
            f"<td>{int(s.get('kept', 0) or 0)}</td>"
            f"<td>{int(s.get('removed_stale', 0) or 0)}</td>"
            f"<td>{int(s.get('total_tokens', 0) or 0):,}</td>"
            f"<td>{cost_cell}</td>"
            f"<td>{float(s.get('wall_seconds') or 0):.2f}s</td>"
            "</tr>"
        )
    return "<table>" + "".join(rows) + "</table>"


def _format_timestamp(ts: str | None) -> str:
    if not ts:
        return "—"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return ts[:16]


# ---------- CLI ----------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="reviewer.dashboard",
        description="Aggregate AI-review artifact JSONs into an HTML dashboard.",
    )
    ap.add_argument("--inputs", required=True,
                    help="Directory containing ai-review-run-*.json artifacts.")
    ap.add_argument("--out", required=True,
                    help="Output HTML path (e.g. dashboard.html).")
    ap.add_argument("--title", default="AI Code Review Dashboard",
                    help="Title rendered at the top of the dashboard.")
    args = ap.parse_args(argv)

    try:
        runs = load_runs(args.inputs)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    aggregate = aggregate_runs(runs)
    html = render_html(aggregate, runs, title=args.title)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"wrote {out_path}  ({aggregate.total_runs} run(s) aggregated)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
