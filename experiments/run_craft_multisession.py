#!/usr/bin/env python3
"""Multi-session CRAFT runs using ``tasks/craft_airline_multisession_seeds.yaml``.

Each seed lists Plant / Reinforce / Trigger as separate ``session_intents`` strings; the loader
wraps each in a one-element inner list. ``CraftLLMAttacker`` uses ``false_claim`` (or
``hidden_attacker_goal``) as the hidden goal and one phase's text per orchestrator session.

**Baselines (two kinds):** (1) *Full-blob* — all phases merged into one chat (can be harder than
multi-session). (2) *Trigger-only* — only the Trigger phase in one session; isolates the
multi-session + memory effect from turn-count / priming confounds. Compare: trigger-only baseline
vs session 3 under ``no_memory`` vs ``full_history`` / ``summary``.

Summary memory: do **not** treat ASR under ``summary`` as a laundering or faithfulness bug — the
compressor reflects what actually happened (including tool outcomes that contradict the false
claim). Lower attack success under summary is a substantive finding about persistence, not a
defect in summarization.

Usage:
    python experiments/run_craft_multisession.py --dry-run
    python experiments/run_craft_multisession.py --craft-yaml tasks/craft_airline_multisession_seeds.yaml
    python experiments/run_craft_multisession.py --memory-modes no_memory --max-turns 25
    python experiments/run_craft_multisession.py --limit 5   # first 5 seeds only (per baseline + each mode)
"""
from __future__ import annotations

import argparse
import copy
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

from analysis import analyze, load_results_jsonl
from attackers import CraftLLMAttacker
from core.environment import TauBenchEnv
from core.orchestrator import Orchestrator
from core.session_runner import SessionRunner
from memory.full_history import FullHistoryMemory
from memory.none import NoMemory
from memory.summary import SummaryMemory
from memory.base import MemoryProvider
from tasks.loader import load_craft_multisession_yaml

MODEL = "openrouter/google/gemini-2.5-flash"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CRAFT_YAML_DEFAULT = Path(__file__).resolve().parent.parent / "tasks" / "craft_airline_multisession_seeds.yaml"

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


def main() -> None:
    p = argparse.ArgumentParser(description="CRAFT multi-session runs from craft_airline_multisession_seeds.yaml")
    p.add_argument("--model", default=MODEL)
    p.add_argument(
        "--craft-yaml",
        type=Path,
        default=CRAFT_YAML_DEFAULT,
        help="YAML with top-level 'seeds' list (default: tasks/craft_airline_multisession_seeds.yaml)",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--analyze", type=str, metavar="PATH", help="Print report from JSONL and exit")
    p.add_argument(
        "--memory-modes",
        nargs="*",
        default=None,
        metavar="MODE",
        help="One or more of: no_memory full_history summary (default: all three)",
    )
    p.add_argument("--attacker-model", default=None, metavar="MODEL")
    p.add_argument("--attacker-temperature", type=float, default=0.5)
    p.add_argument("--max-turns", type=int, default=20, metavar="N")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Use only the first N seeds from the YAML (applies to baseline and every memory mode)",
    )
    args = p.parse_args()

    if args.analyze:
        analyze(load_results_jsonl(args.analyze))
        return

    mem_modes = tuple(args.memory_modes) if args.memory_modes else _DEFAULT_MEMORY_MODES
    for m in mem_modes:
        if m not in _DEFAULT_MEMORY_MODES:
            raise SystemExit(f"Unknown memory mode {m!r}; expected one of {_DEFAULT_MEMORY_MODES}")

    seeds = load_craft_multisession_yaml(args.craft_yaml)
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be at least 1")
        seeds = seeds[: args.limit]

    if args.dry_run:
        print(f"CRAFT YAML: {args.craft_yaml}")
        lim_note = f" (first {args.limit} only)" if args.limit else ""
        print(f"Seeds: {len(seeds)}{lim_note} | Memory modes: {list(mem_modes)}")
        for s in seeds:
            print(
                f"  {s.seed_id}: user={s.user_id!r} res={s.reservation_id} "
                f"vtool={s.violation_tool} orchestrator_sessions={s.num_sessions()}"
            )
        n = len(seeds)
        total = n * (2 + len(mem_modes))
        print(
            f"\nTotal: {n} seeds × (2 baselines: full-blob + trigger-only + {len(mem_modes)} multi-session modes) "
            f"= {total} runs"
        )
        return

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Set OPENROUTER_API_KEY or create experiments/.env")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log = RESULTS_DIR / f"craft_multi_{datetime.now():%Y%m%d_%H%M%S}.jsonl"

    env = TauBenchEnv("airline")
    runner = SessionRunner(env, args.model, max_turns=args.max_turns)
    atk_model = args.attacker_model or args.model
    attacker = CraftLLMAttacker(atk_model, temperature=args.attacker_temperature)

    all_results = []

    print(f"Agent model: {args.model}")
    print(f"Attacker: {atk_model} (CraftLLMAttacker, temp={args.attacker_temperature})")
    print(f"max_turns: {args.max_turns}")
    lim_note = f" (limit {args.limit})" if args.limit else ""
    print(f"Seeds: {len(seeds)}{lim_note} | Memory modes: {list(mem_modes)}\n")

    baseline_memory = NoMemory()
    baseline_orch = Orchestrator(env, runner, baseline_memory, attacker)

    print("-- Baselines (single-session, all phases merged in one roleplay blob) --")
    for seed in seeds:
        print(f"  {seed.seed_id}_baseline:", end=" ")
        r = baseline_orch.run_attack(seed, is_baseline=True)
        all_results.append(r)
        with open(log, "a") as f:
            f.write(json.dumps(r.to_dict(), default=str) + "\n")

    print("\n-- Trigger-only baselines (single session, Trigger phase only — no plant/reinforce) --")
    for seed in seeds:
        trigger_only = copy.deepcopy(seed)
        trigger_only.seed_id = f"{seed.seed_id}_trigger_only"
        trigger_only.session_intents = [seed.session_intents[-1]]
        print(f"  {trigger_only.seed_id}_baseline:", end=" ")
        r = baseline_orch.run_attack(trigger_only, is_baseline=True)
        all_results.append(r)
        with open(log, "a") as f:
            f.write(json.dumps(r.to_dict(), default=str) + "\n")

    for mode in mem_modes:
        memory = make_memory(mode, args.model)
        orch = Orchestrator(env, runner, memory, attacker)
        print(f"\n-- Multi-session [{mode}] (Plant → Reinforce → Trigger) --")
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
