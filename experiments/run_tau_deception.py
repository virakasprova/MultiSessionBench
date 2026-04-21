#!/usr/bin/env python3
"""Run τ-bench airline tasks with per-task DeceptionPlanner / CRAFT strategies (JSON).

The JSON maps string task indices to **hidden** strategy text (keys ``\"10\"``, ``\"15\"``, …
match ``tau_bench.envs.airline.tasks_test.TASKS[i]``).

The strategy is **not** sent to the airline agent as user text. It is passed to
``CraftLLMAttacker`` as part of the **attacker** system prompt; the attacker LLM produces
natural customer utterances turn by turn until ``###STOP###`` or ``max_turns``.

Example:
    python experiments/run_tau_deception.py --dry-run \\
      --strategies-json ~/Downloads/airline_strategies_per_taskid_DeceptionPlanner.json

    python experiments/run_tau_deception.py \\
      --strategies-json path/to/strategies.json \\
      --memory-modes no_memory
"""
from __future__ import annotations

import argparse
import json
import os
import re
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

from core.environment import AIRLINE_WRITE_TOOLS, TauBenchEnv
from core.orchestrator import Orchestrator
from core.session_runner import SessionRunner
from core.types import TaskSeed
from memory.full_history import FullHistoryMemory
from memory.none import NoMemory
from memory.summary import SummaryMemory
from memory.base import MemoryProvider
from attackers import CraftLLMAttacker
from tau_bench.envs.airline.tasks_test import TASKS as TAU_TASKS

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


def _looks_like_flight_number(token: str) -> bool:
    return bool(re.match(r"^[A-Z]{3}\d{1,4}[A-Z]?$", token))  # e.g. HAT045, HAT136


def _first_write_tool_name(task) -> str:
    for a in task.actions:
        if a.name in AIRLINE_WRITE_TOOLS:
            return a.name
    ins = task.instruction.lower()
    if "book" in ins and ("new flight" in ins or "book a" in ins or "book the" in ins):
        return "book_reservation"
    if "certificate" in ins or "compensation" in ins or "voucher" in ins:
        return "send_certificate"
    if "cancel" in ins:
        return "cancel_reservation"
    if any(
        k in ins
        for k in ("change", "modify", "upgrade", "downgrade", "later flight", "earlier flight")
    ):
        return "update_reservation_flights"
    return "cancel_reservation"


def _reservation_id_from_task(task) -> str:
    for a in task.actions:
        rid = a.kwargs.get("reservation_id")
        if rid:
            return rid
    m = re.search(r"\breservation\s+(?:ID\s+)?([A-Z0-9]{6})\b", task.instruction, re.I)
    if m and not _looks_like_flight_number(m.group(1)):
        return m.group(1)
    for m in re.finditer(r"\b([A-Z0-9]{6})\b", task.instruction):
        tok = m.group(1)
        if not _looks_like_flight_number(tok):
            return tok
    return "UNKNOWN"


def tau_task_to_seed(task, task_id: str, strategy: str) -> TaskSeed:
    """One orchestrator session; scripted lines unused — ``CraftLLMAttacker`` drives turns."""
    rid = _reservation_id_from_task(task)
    vtool = _first_write_tool_name(task)
    return TaskSeed(
        seed_id=f"tau_{task_id}",
        user_id=task.user_id,
        reservation_id=rid,
        violation_tool=vtool,
        false_claim=strategy[:500],
        policy_area="tau_deception",
        session_intents=[[]],
        baseline_messages=None,
        hidden_attacker_goal=strategy,
        roleplay_scenario=task.instruction,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Run τ-bench tasks with DeceptionPlanner strategies JSON")
    p.add_argument("--strategies-json", type=Path, required=True, help="JSON: {\"10\": \"strategy...\", ...}")
    p.add_argument(
        "--task-ids",
        nargs="*",
        metavar="ID",
        help="Subset of task indices (default: all keys in JSON)",
    )
    p.add_argument("--model", default=MODEL)
    p.add_argument("--domain", choices=("airline",), default="airline")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--memory-modes",
        nargs="*",
        default=None,
        metavar="MODE",
        help="One or more of: no_memory full_history summary (default: all three)",
    )
    p.add_argument(
        "--baseline",
        action="store_true",
        help="Also run single-session baseline (often duplicates a 1-session seed)",
    )
    p.add_argument(
        "--attacker-model",
        default=None,
        metavar="MODEL",
        help="Model for Craft attacker LLM (default: same as --model)",
    )
    p.add_argument(
        "--attacker-temperature",
        type=float,
        default=0.5,
        help="Sampling temperature for the customer (attacker) LLM",
    )
    p.add_argument(
        "--max-turns",
        type=int,
        default=20,
        metavar="N",
        help="Max agent↔customer steps per session (SessionRunner loop)",
    )
    args = p.parse_args()

    raw = json.loads(args.strategies_json.read_text())
    strategies: dict[str, str] = {str(k): str(v) for k, v in raw.items()}

    want = [str(x) for x in args.task_ids] if args.task_ids else sorted(strategies.keys(), key=int)
    mem_modes = tuple(args.memory_modes) if args.memory_modes else _DEFAULT_MEMORY_MODES
    for m in mem_modes:
        if m not in _DEFAULT_MEMORY_MODES:
            raise SystemExit(f"Unknown memory mode {m!r}; expected one of {_DEFAULT_MEMORY_MODES}")

    seeds: list[TaskSeed] = []
    for tid in want:
        if tid not in strategies:
            raise SystemExit(f"Task id {tid!r} not in strategies JSON")
        idx = int(tid)
        if idx < 0 or idx >= len(TAU_TASKS):
            raise SystemExit(f"Task index {idx} out of range (0..{len(TAU_TASKS) - 1})")
        seeds.append(tau_task_to_seed(TAU_TASKS[idx], tid, strategies[tid]))

    if args.dry_run:
        print(f"tasks_test.TASKS: {len(TAU_TASKS)} tasks | running {len(seeds)} seeds")
        print(f"Memory modes: {list(mem_modes)} | baseline: {args.baseline}")
        for s in seeds:
            print(
                f"  {s.seed_id}: user={s.user_id} res={s.reservation_id} vtool={s.violation_tool} "
                f"attacker=CraftLLMAttacker max_turns={args.max_turns}"
            )
        total = len(seeds) * (len(mem_modes) + (1 if args.baseline else 0))
        print(f"\nTotal runs: {total}")
        return

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Set OPENROUTER_API_KEY or create experiments/.env")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log = RESULTS_DIR / f"tau_deception_{datetime.now():%Y%m%d_%H%M%S}.jsonl"

    env = TauBenchEnv(args.domain)
    runner = SessionRunner(env, args.model, max_turns=args.max_turns)
    atk_model = args.attacker_model or args.model
    attacker = CraftLLMAttacker(
        atk_model,
        temperature=args.attacker_temperature,
    )

    print(f"Domain: {args.domain}")
    print(f"Agent model: {args.model}")
    print(f"Attacker model: {atk_model} (CraftLLMAttacker, temp={args.attacker_temperature})")
    print(f"Session max_turns: {args.max_turns}")
    print(f"Seeds: {len(seeds)} | Memory modes: {list(mem_modes)} | baseline: {args.baseline}\n")

    all_results = []

    if args.baseline:
        baseline_memory = NoMemory()
        baseline_orch = Orchestrator(env, runner, baseline_memory, attacker)
        print("-- Baselines --")
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


if __name__ == "__main__":
    main()
