"""Evaluate N models against the labelled corpus and emit a leaderboard.

Case types
----------
- **Bug cases** (sample_*.diff): `labels.expected` is a non-empty list.
  Scored for recall/precision.
- **Clean cases** (clean_*.diff, or any case with `expected: []`): no real
  findings exist. Any finding the model emits here is counted as a false
  positive. Reported as `FP/clean` (mean findings per clean case).

Scoring (simple, auditable)
---------------------------
For each expected finding, a model "hits" it if it produces a finding on the
same file whose line is within `line_range` AND whose (title + explanation)
contains one of the `keywords` (case-insensitive).

- recall    = hits / total_expected            (across bug cases)
- precision = hits / findings_on_bug_cases     (across bug cases)
- FP/clean  = findings_on_clean_cases / num_clean_cases
- tokens/case, $/case, latency/case            (across ALL cases)

Usage:
  python eval/harness.py --models openai/gpt-4o-mini openai/gpt-4o
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from reviewer import review_patch
from reviewer.client import build_client

sys.path.insert(0, str(ROOT / "eval"))
from scorer import score_case_keyword, score_case_judge  # noqa: E402


def load_corpus(corpus_dir: Path) -> list[dict]:
    cases = []
    for diff_path in sorted(corpus_dir.glob("*.diff")):
        label_path = diff_path.with_suffix(".labels.json")
        if not label_path.exists():
            continue
        labels = json.loads(label_path.read_text(encoding="utf-8"))
        cases.append({
            "id": diff_path.stem,
            "diff": diff_path.read_text(encoding="utf-8"),
            "labels": labels,
            "is_clean": not labels.get("expected"),
        })
    return cases


def _score_for(scorer: str, judge_model: str | None, judge_client):
    """Return a callable ``(findings, labels) -> result_dict`` per scorer choice."""
    if scorer == "keyword":
        return score_case_keyword
    if scorer == "judge":
        if judge_client is None or not judge_model:
            raise RuntimeError(
                "judge scorer requires a judge_model and a client")
        return lambda findings, labels: score_case_judge(
            findings, labels, judge_model=judge_model, client=judge_client)
    raise ValueError(f"unknown scorer: {scorer}")


def run_model(
    model: str,
    cases: list[dict],
    min_conf: float,
    *,
    score_case,
) -> dict:
    totals = {
        "expected": 0, "hits": 0,
        "findings_bug": 0, "extras": 0,
        "clean_cases": 0, "false_positives": 0,
        "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0,
        "by_cat_hits": {}, "by_cat_expected": {},
    }
    elapsed = 0.0
    per_case: list[dict] = []
    usage_log: list[dict] = []

    for c in cases:
        t0 = time.perf_counter()
        findings = review_patch(
            c["diff"],
            model=model,
            repo="eval/corpus",
            pr_title=c["id"],
            pr_description=c["labels"].get("description", ""),
            min_confidence=min_conf,
            usage_log=usage_log,
        )
        elapsed += time.perf_counter() - t0

        if c["is_clean"]:
            totals["clean_cases"] += 1
            totals["false_positives"] += len(findings)
            per_case.append({
                "id": c["id"], "clean": True,
                "findings_emitted": len(findings),
                "false_positives": len(findings),
            })
        else:
            s = score_case(findings, c["labels"])
            per_case.append({"id": c["id"], "clean": False, **s})
            totals["expected"] += s["expected"]
            totals["hits"] += s["hits"]
            totals["findings_bug"] += s["findings_emitted"]
            totals["extras"] += s["extras"]
            for cat, n in s.get("by_category_hits", {}).items():
                totals["by_cat_hits"][cat] = totals["by_cat_hits"].get(cat, 0) + n
            for cat, n in s.get("by_category_expected", {}).items():
                totals["by_cat_expected"][cat] = (
                    totals["by_cat_expected"].get(cat, 0) + n)

    priced_calls = 0
    for u in usage_log:
        totals["prompt_tokens"] += u.get("prompt_tokens", 0)
        totals["completion_tokens"] += u.get("completion_tokens", 0)
        if u.get("cost_usd") is not None:
            totals["cost_usd"] += u["cost_usd"]
            priced_calls += 1

    ncases = len(cases) or 1
    nclean = totals["clean_cases"] or 0
    recall = totals["hits"] / totals["expected"] if totals["expected"] else 0.0
    precision = totals["hits"] / totals["findings_bug"] if totals["findings_bug"] else 0.0
    fp_per_clean = totals["false_positives"] / nclean if nclean else 0.0
    tokens_per_case = (totals["prompt_tokens"] + totals["completion_tokens"]) / ncases
    cost_per_case = totals["cost_usd"] / ncases if priced_calls else None

    return {
        "model": model,
        "cases": per_case,
        "totals": totals,
        "recall": recall,
        "precision": precision,
        "fp_per_clean_case": fp_per_clean,
        "tokens_per_case": tokens_per_case,
        "cost_per_case": cost_per_case,
        "latency_s_per_case": elapsed / ncases,
        "usage": usage_log,
    }


def _fmt_cost(c: float | None) -> str:
    if c is None:
        return "-"
    return f"${c:.4f}"


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--corpus", default=str(ROOT / "eval" / "corpus"))
    ap.add_argument("--min-confidence", type=float,
                    default=float(os.environ.get("MIN_CONFIDENCE", "0.6")))
    ap.add_argument("--out", default=str(ROOT / "eval" / "out"))
    ap.add_argument("--scorer", choices=["keyword", "judge"],
                    default=os.environ.get("REVIEWER_SCORER", "keyword"),
                    help="keyword (free, brittle) or judge (LLM, accurate, costs tokens)")
    ap.add_argument("--judge-model",
                    default=os.environ.get("REVIEWER_JUDGE_MODEL",
                                           "openai/gpt-4o"),
                    help="model used for LLM-as-judge scoring")
    args = ap.parse_args()

    cases = load_corpus(Path(args.corpus))
    if not cases:
        print(f"no corpus at {args.corpus}", file=sys.stderr)
        return 2

    n_bug = sum(1 for c in cases if not c["is_clean"])
    n_clean = sum(1 for c in cases if c["is_clean"])

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    console = Console()
    console.print(f"[bold]Corpus:[/bold] {len(cases)} cases "
                  f"({n_bug} bug, {n_clean} clean)")
    console.print(f"[bold]Scorer:[/bold] {args.scorer}"
                  + (f" (judge={args.judge_model})" if args.scorer == "judge" else ""))

    judge_client = build_client() if args.scorer == "judge" else None
    score_case = _score_for(args.scorer, args.judge_model, judge_client)

    results = []
    for m in args.models:
        console.print(f"[bold]Running[/bold] {m}...")
        r = run_model(m, cases, args.min_confidence, score_case=score_case)
        results.append(r)
        (out_dir / f"{m.replace('/', '_')}.json").write_text(
            json.dumps(r, indent=2), encoding="utf-8")

    t = Table(title="Leaderboard", show_lines=True)
    for col in ("Model", "Recall", "Precision", "FP/clean",
                "Hits", "Tokens/case", "$/case", "Latency/case"):
        t.add_column(col)
    for r in sorted(results, key=lambda x: (-x["recall"], x["fp_per_clean_case"],
                                            -x["precision"])):
        t.add_row(
            r["model"],
            f"{r['recall']:.2f}",
            f"{r['precision']:.2f}",
            f"{r['fp_per_clean_case']:.2f}",
            f"{r['totals']['hits']}/{r['totals']['expected']}",
            f"{r['tokens_per_case']:.0f}",
            _fmt_cost(r["cost_per_case"]),
            f"{r['latency_s_per_case']:.2f}s",
        )
    console.print(t)

    all_cats = sorted({
        cat for r in results for cat in r["totals"]["by_cat_expected"]
    })
    if all_cats:
        ct = Table(title="Per-category recall", show_lines=True)
        ct.add_column("Model")
        for cat in all_cats:
            ct.add_column(cat)
        for r in results:
            row = [r["model"]]
            for cat in all_cats:
                hits = r["totals"]["by_cat_hits"].get(cat, 0)
                exp = r["totals"]["by_cat_expected"].get(cat, 0)
                row.append(f"{hits}/{exp}" if exp else "-")
            ct.add_row(*row)
        console.print(ct)

    console.print(f"[dim]Detailed reports in {out_dir}[/dim]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
