#!/usr/bin/env python3
"""Write summary.md for a run folder under experiments/results/.

Reads ``judged.jsonl`` if present (preferred — gives compliance / persistence /
contradiction columns), else falls back to ``run.jsonl``. Captures the
``analyze()`` report and writes ``summary.md`` into the same folder, prefixed
with a metadata header derived from ``config.json``.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import analyze, load_results_jsonl


def _resolve_input(run_dir: Path) -> Path:
    judged = run_dir / "judged.jsonl"
    if judged.exists():
        return judged
    run = run_dir / "run.jsonl"
    if run.exists():
        return run
    raise SystemExit(f"No run.jsonl or judged.jsonl found in {run_dir}")


def _format_config_md(cfg: dict) -> str:
    lines = ["# Run summary", ""]
    lines.append(f"- **run_stem**: `{cfg.get('run_stem', '?')}`")
    if cfg.get("run_name"):
        lines.append(f"- **run_name**: {cfg['run_name']}")
    lines.append(f"- **started_at**: {cfg.get('started_at', '?')}")
    if cfg.get("finished_at"):
        lines.append(f"- **finished_at**: {cfg['finished_at']}")
    if cfg.get("git_sha"):
        lines.append(f"- **git_sha**: `{cfg['git_sha'][:12]}`")
    lines.append(f"- **agent_model**: `{cfg.get('agent_model', '?')}`")
    atk = cfg.get("attacker") or {}
    if atk.get("kind") == "human":
        lines.append("- **attacker**: HumanAttacker (interactive)")
    else:
        lines.append(
            f"- **attacker**: `{atk.get('model', '?')}` "
            f"(temp={atk.get('temperature')})"
        )
    lines.append(f"- **max_turns**: {cfg.get('max_turns', '?')}")
    lines.append(f"- **memory_modes**: {', '.join(cfg.get('memory_modes') or [])}")
    if cfg.get("rag_backend"):
        lines.append(f"- **rag_backend**: {cfg['rag_backend']}")
    lines.append(
        f"- **instrumentation**: {'on' if cfg.get('instrumentation') else 'off'}"
    )
    seeds = cfg.get("seeds") or {}
    sha = (seeds.get("sha256") or "")[:12]
    lines.append(
        f"- **seeds**: `{seeds.get('path', '?')}` "
        f"(count={seeds.get('count')}, sha256=`{sha}`)"
    )
    if "violations" in cfg:
        lines.append(
            f"- **violations**: {cfg['violations']}/{cfg.get('total_attacks', '?')}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Write summary.md for a run folder under experiments/results/. "
            "Uses judged.jsonl if present (richer metrics), else run.jsonl."
        )
    )
    p.add_argument(
        "run_dir",
        type=Path,
        help="Run folder (contains run.jsonl and config.json)",
    )
    p.add_argument(
        "--stdout",
        action="store_true",
        help="Also print the report to stdout",
    )
    args = p.parse_args()

    run_dir: Path = args.run_dir
    if not run_dir.is_dir():
        raise SystemExit(f"{run_dir} is not a directory")

    src = _resolve_input(run_dir)
    rows = load_results_jsonl(src)

    buf = io.StringIO()
    with redirect_stdout(buf):
        analyze(rows)
    report = buf.getvalue()

    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        header = _format_config_md(cfg)
    else:
        header = "# Run summary\n\n_(no config.json — pre-migration run)_\n\n"

    md = header + f"_source: `{src.name}`_\n\n```\n" + report + "\n```\n"

    out = run_dir / "summary.md"
    out.write_text(md)
    print(f"Wrote {out}")
    if args.stdout:
        sys.stdout.write(report)


if __name__ == "__main__":
    main()
