"""Review a local diff file and print findings. Useful for quick iteration."""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

# Make `src` importable when run from repo root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from reviewer import review_patch
from reviewer.config import CONFIG_FILENAME, effective_config, load_config


def main() -> int:
    load_dotenv()
    cfg_default = effective_config()  # defaults+env, for arg defaults

    ap = argparse.ArgumentParser()
    ap.add_argument("--diff", required=True, help="Path to a unified diff file.")
    ap.add_argument("--config", default=str(Path.cwd() / CONFIG_FILENAME),
                    help=f"Path to {CONFIG_FILENAME} (default: ./{CONFIG_FILENAME})")
    ap.add_argument("--model", default=None,
                    help="Override the model from config / env.")
    ap.add_argument("--repo", default="local/demo")
    ap.add_argument("--title", default="local diff review")
    ap.add_argument("--description", default="")
    ap.add_argument("--min-confidence", type=float, default=None)
    ap.add_argument("--min-severity", default=None,
                    choices=["low", "medium", "high", "critical"])
    ap.add_argument("--json", action="store_true",
                    help="Emit JSON instead of a table.")
    ap.add_argument("--stream", action="store_true",
                    help="Print model output to stderr as it streams. "
                         "Useful as a 'is it doing anything?' indicator "
                         "on slow providers. Doesn't change the final "
                         "findings — JSON is parsed at end-of-stream.")
    args = ap.parse_args()

    cfg = effective_config(config_file=load_config(args.config))
    model = args.model or cfg.model
    min_confidence = args.min_confidence if args.min_confidence is not None else cfg.min_confidence
    min_severity = args.min_severity or cfg.min_severity

    diff_text = Path(args.diff).read_text(encoding="utf-8")

    stream_callback = None
    if args.stream:
        def stream_callback(path: str, delta: str) -> None:
            # Lightweight indicator: print deltas without paths to keep
            # output readable on long files.
            sys.stderr.write(delta)
            sys.stderr.flush()

    findings = review_patch(
        diff_text,
        model=model,
        repo=args.repo,
        pr_title=args.title,
        pr_description=args.description,
        min_confidence=min_confidence,
        min_severity=min_severity,
        concurrency=cfg.concurrency,
        max_diff_chars=cfg.max_diff_chars,
        max_retries=cfg.max_retries,
        retry_base_seconds=cfg.retry_base_seconds,
        models_by_language=cfg.models_by_language or None,
        prompt_extras_by_language=cfg.prompt_extras_by_language or None,
        include_maintainability_findings=cfg.include_maintainability_findings,
        stream_callback=stream_callback,
    )
    if args.stream:
        sys.stderr.write("\n")

    if args.json:
        print(json.dumps({"model": model, "findings": findings}, indent=2))
        return 0

    console = Console()
    console.print(f"[bold]Model:[/bold] {model}   "
                  f"[bold]Findings:[/bold] {len(findings)}")
    if not findings:
        return 0

    t = Table(show_lines=True)
    t.add_column("Sev"); t.add_column("Cat"); t.add_column("File:Line")
    t.add_column("Title"); t.add_column("Conf")
    for f in findings:
        t.add_row(
            f.get("severity", ""), f.get("category", ""),
            f"{f.get('path','')}:{f.get('line','')}",
            f.get("title", ""), f"{float(f.get('confidence', 0)):.2f}",
        )
    console.print(t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
