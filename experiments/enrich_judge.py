#!/usr/bin/env python3
"""Post-hoc LLM judge: read experiment JSONL, add per-session judge + compliance, write JSONL.

Mechanical runs (``run_craft_multisession.py``) omit judge fields; run this before ``analyze`` when
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
    p.add_argument(
        "input",
        type=Path,
        help=(
            "Run folder under experiments/results/ (reads run.jsonl, writes "
            "judged.jsonl into the same folder) or a legacy flat JSONL path."
        ),
    )
    p.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help=(
            "Output JSONL path. Required when input is a flat JSONL file; "
            "defaults to <run_folder>/judged.jsonl when input is a folder."
        ),
    )
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

    if args.input.is_dir():
        src = args.input / "run.jsonl"
        if not src.exists():
            raise SystemExit(f"No run.jsonl found in {args.input}")
        out_path = args.output or (args.input / "judged.jsonl")
    else:
        src = args.input
        if args.output is None:
            raise SystemExit(
                "-o OUTPUT is required when input is a flat JSONL file"
            )
        out_path = args.output

    # Model-aware key check: ``openrouter/...`` -> OPENROUTER_API_KEY,
    # ``openai/...`` (or no prefix) -> OPENAI_API_KEY, ``anthropic/...`` ->
    # ANTHROPIC_API_KEY. Other providers fall through to litellm's own error.
    if args.model.startswith("openrouter/"):
        env, prov = "OPENROUTER_API_KEY", "OpenRouter"
    elif args.model.startswith("anthropic/"):
        env, prov = "ANTHROPIC_API_KEY", "Anthropic"
    else:
        env, prov = "OPENAI_API_KEY", "OpenAI"
    if not os.environ.get(env):
        print(f"Set {env} (required for {prov} model {args.model!r}; "
              "or create experiments/.env)")
        sys.exit(1)

    judge = SessionJudge(args.model)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_in, n_out = 0, 0
    with open(src) as fin, open(out_path, "w") as fout:
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

    print(f"Wrote {n_out} rows to {out_path} (read {n_in} from {src})")


if __name__ == "__main__":
    main()
