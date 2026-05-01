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
        "run_dir",
        type=Path,
        help=(
            "Run folder under experiments/results/ — reads run.jsonl, "
            "writes judged.jsonl into the same folder."
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
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "If judged.jsonl already exists, append to it and skip rows "
            "whose attack_id is already present. Use this after a crashed "
            "judge run; without it, judged.jsonl is overwritten."
        ),
    )
    args = p.parse_args()

    if not args.run_dir.is_dir():
        raise SystemExit(f"{args.run_dir} is not a directory")
    src = args.run_dir / "run.jsonl"
    if not src.exists():
        raise SystemExit(f"No run.jsonl found in {args.run_dir}")
    out_path = args.run_dir / "judged.jsonl"

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

    # Resume: pre-populate the dedup set with attack_ids already in the
    # output and append. Without --resume the previous behavior is preserved
    # (open in 'w' mode, dedup set empty).
    already_judged: set[str] = set()
    out_mode = "w"
    if args.resume and out_path.exists():
        with open(out_path) as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    aid = json.loads(ln).get("attack_id")
                except json.JSONDecodeError:
                    continue
                if aid:
                    already_judged.add(aid)
        out_mode = "a"
        print(f"[resume] {len(already_judged)} attack_ids already in {out_path}; will skip those.")

    n_in, n_out, n_skipped, n_failed = 0, 0, 0, 0
    with open(src) as fin, open(out_path, out_mode) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[skip line {n_in}: malformed JSON ({e})]")
                n_failed += 1
                continue
            if row.get("attack_id") in already_judged:
                n_skipped += 1
                continue
            if not args.force and row.get("compliance_score") is not None:
                fout.write(json.dumps(row, default=str) + "\n")
                n_out += 1
                continue
            try:
                enriched = enrich_experiment_dict(row, judge, force=args.force)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                # Don't drop the row — pass the un-judged input through with an
                # error annotation so analyze() / downstream still see it, and
                # a future --resume run can retry just the failed ones.
                print(f"[judge failed for {row.get('attack_id')!r}: "
                      f"{type(e).__name__}: {e}; passing through unjudged]")
                row.setdefault("judge_errors", []).append(
                    {"error_type": type(e).__name__, "error": str(e)}
                )
                fout.write(json.dumps(row, default=str) + "\n")
                n_failed += 1
                continue
            fout.write(json.dumps(enriched, default=str) + "\n")
            n_out += 1

    summary = f"Wrote {n_out} rows to {out_path} (read {n_in} from {src}"
    if n_skipped:
        summary += f"; skipped {n_skipped} already-judged via --resume"
    if n_failed:
        summary += f"; {n_failed} rows failed and were passed through unjudged"
    summary += ")"
    print(summary)


if __name__ == "__main__":
    main()
