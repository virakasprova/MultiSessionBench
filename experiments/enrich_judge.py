#!/usr/bin/env python3
"""Post-hoc LLM judge: read experiment JSONL, add per-session judge + compliance, write JSONL.

Mechanical runs (e.g. ``run_multisession.py``) omit judge fields; run this before ``analyze`` when
you need compliance, persistence (accepted), or contradiction metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_env = Path(__file__).resolve().parent / ".env"
if _env.exists():
    for ln in _env.read_text().splitlines():
        if ln.strip() and "=" in ln and not ln.startswith("#"):
            k, v = ln.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from core.judge import SessionJudge, enrich_experiment_dict

MODEL = "openrouter/google/gemini-2.5-flash"


def main() -> None:
    p = argparse.ArgumentParser(description="Enrich experiment JSONL with SessionJudge scores")
    p.add_argument("input", type=Path, help="Input JSONL (e.g. multi_*.jsonl)")
    p.add_argument("-o", "--output", type=Path, required=True, help="Output JSONL path")
    p.add_argument(
        "--model",
        default=MODEL,
        help="LiteLLM model id (include provider prefix, e.g. openrouter/...)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run judge even when compliance_score is already set",
    )
    args = p.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Set OPENROUTER_API_KEY or create experiments/.env")
        sys.exit(1)

    judge = SessionJudge(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_in, n_out = 0, 0
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            row = json.loads(line)
            if not args.force and row.get("compliance_score") is not None:
                fout.write(json.dumps(row, default=str) + "\n")
                n_out += 1
                continue
            enriched = enrich_experiment_dict(row, judge, force=args.force)
            fout.write(json.dumps(enriched, default=str) + "\n")
            n_out += 1

    print(f"Wrote {n_out} rows to {args.output} (read {n_in} from {args.input})")


if __name__ == "__main__":
    main()
