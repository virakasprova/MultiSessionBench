#!/usr/bin/env python3
"""Multi-session experiment runner using the modular infrastructure.

Usage:
    python experiments/run_multisession.py --dry-run
    python experiments/run_multisession.py                    # mechanical JSONL (needs OPENROUTER_API_KEY)
    python experiments/run_multisession.py --memory-modes no_memory full_history summary
    python experiments/enrich_judge.py in.jsonl -o out.jsonl  # add judge / compliance; then analyze out.jsonl
    python experiments/run_multisession.py --attacker-system-prompt-file path/to/prompt.txt
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import litellm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_env = Path(__file__).resolve().parent / ".env"
if _env.exists():
    for ln in _env.read_text().splitlines():
        if ln.strip() and "=" in ln and not ln.startswith("#"):
            k, v = ln.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from core.environment import TauBenchEnv
from core.session_runner import SessionRunner
from core.orchestrator import Orchestrator
from memory.base import MemoryProvider
from memory.full_history import FullHistoryMemory
from memory.none import NoMemory
from memory.summary import SummaryMemory
from attackers import LLMAttacker
from tasks.loader import load_seeds
from analysis import analyze, load_results_jsonl

MODEL = "openrouter/google/gemini-2.5-flash"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

_DEFAULT_MEMORY_MODES = ("no_memory", "full_history", "summary")
litellm.suppress_debug_info = True


def make_memory(mode: str, summary_model: str) -> MemoryProvider:
    if mode == "no_memory":
        return NoMemory()
    if mode == "full_history":
        return FullHistoryMemory()
    if mode == "summary":
        return SummaryMemory(summary_model)
    raise SystemExit(f"Unknown memory mode {mode!r}; expected one of {_DEFAULT_MEMORY_MODES}")


def main():
    p = argparse.ArgumentParser(description="Run multi-session attack experiment")
    p.add_argument("--model", default=MODEL)
    p.add_argument(
        "--domain",
        choices=("airline", "retail"),
        default="airline",
        help="τ-bench environment (tools + wiki + data)",
    )
    p.add_argument("--seeds-dir", type=Path, default=None, help="Override default seeds directory")
    p.add_argument("--dry-run", action="store_true", help="Print config and exit")
    p.add_argument("--analyze", type=str, metavar="PATH", help="Print report from JSONL and exit")
    p.add_argument(
        "--memory-modes",
        nargs="*",
        default=None,
        metavar="MODE",
        help="Multi-session runs: one or more of no_memory full_history summary (default: all three). "
        "Baseline runs always use no_memory.",
    )
    p.add_argument(
        "--attacker-model",
        default=None,
        metavar="MODEL",
        help="Model for attacker LLM (default: same as --model)",
    )
    p.add_argument(
        "--attacker-system-prompt-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional file overriding the default attacker system prompt",
    )
    args = p.parse_args()

    if args.analyze:
        analyze(load_results_jsonl(args.analyze))
        return

    mem_modes = tuple(args.memory_modes) if args.memory_modes else _DEFAULT_MEMORY_MODES
    for m in mem_modes:
        if m not in _DEFAULT_MEMORY_MODES:
            raise SystemExit(f"Unknown memory mode {m!r}; expected one of {_DEFAULT_MEMORY_MODES}")

    seeds = load_seeds(args.seeds_dir)

    if args.dry_run:
        print(f"Memory modes (multi-session): {list(mem_modes)}")
        for s in seeds:
            print(f"  {s.seed_id}: user={s.user_id} res={s.reservation_id} "
                  f"vtool={s.violation_tool} claim='{s.false_claim}' "
                  f"sessions={s.num_sessions()}")
        n = len(seeds)
        total = n * (1 + len(mem_modes))
        print(f"\nTotal: {n} seeds × (1 baseline + {len(mem_modes)} multi-session modes) = {total} runs")
        return

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Set OPENROUTER_API_KEY or create experiments/.env")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log = RESULTS_DIR / f"multi_{datetime.now():%Y%m%d_%H%M%S}.jsonl"

    env = TauBenchEnv(args.domain)
    runner = SessionRunner(env, args.model)
    atk_model = args.attacker_model or args.model
    sys_prompt = None
    if args.attacker_system_prompt_file is not None:
        sys_prompt = args.attacker_system_prompt_file.read_text()
    attacker = LLMAttacker(atk_model, system_prompt=sys_prompt)

    all_results = []

    print(f"Domain: {args.domain}")
    print(f"Agent model: {args.model}")
    print(f"Attacker model: {atk_model}")
    print(f"Seeds: {len(seeds)} | Multi-session memory modes: {list(mem_modes)}\n")

    baseline_memory = NoMemory()
    baseline_orch = Orchestrator(env, runner, baseline_memory, attacker)

    print("-- Baselines (single-session, no_memory) --")
    for seed in seeds:
        print(f"  {seed.seed_id}_baseline:", end=" ")
        r = baseline_orch.run_attack(seed, is_baseline=True)
        all_results.append(r)
        with open(log, "a") as f:
            f.write(json.dumps(r.to_dict(), default=str) + "\n")

    for mode in mem_modes:
        memory = make_memory(mode, args.model)
        orch = Orchestrator(env, runner, memory, attacker)
        print(f"\n-- Multi-session [{mode}] --")
        for seed in seeds:
            print(f"  {seed.seed_id} [{memory.mode_name}]:", end=" ")
            r = orch.run_attack(seed, is_baseline=False)
            all_results.append(r)
            with open(log, "a") as f:
                f.write(json.dumps(r.to_dict(), default=str) + "\n")

    violated = sum(1 for r in all_results if r.violation_detected)
    print(f"\nDone. {violated}/{len(all_results)} attacks with violation.")
    print(f"Results: {log}")

    analyze([r.to_dict() for r in all_results])


if __name__ == "__main__":
    main()
