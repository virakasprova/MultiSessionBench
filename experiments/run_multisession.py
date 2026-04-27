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
from core.instrumentation import build_instruments, derive_planted_claims
from core.session_runner import SessionRunner
from core.orchestrator import Orchestrator
from memory import BaseMemoryProvider, make_memory_provider
from memory.litellm_summariser import make_litellm_summariser
from attackers import LLMAttacker
from tasks.loader import load_seeds
from analysis import analyze, load_results_jsonl

MODEL = "openrouter/google/gemini-2.5-flash"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

_DEFAULT_MEMORY_MODES = ("no_memory", "full_history", "summary")
litellm.suppress_debug_info = True


# CLI memory-mode -> provider builder. The legacy CLI exposes only three
# modes (``no_memory`` / ``full_history`` / ``summary``); aliases map to the
# structured-turn factory's canonical ids so prior JSONL ``memory_mode``
# values stay comparable. Adding a new mode is a single new line here.
_MEMORY_DISPATCH = {
    "no_memory":    lambda model: make_memory_provider("no_memory"),
    "full_history": lambda model: make_memory_provider("full_context"),
    "summary":      lambda model: make_memory_provider(
        "summary_rolling", summariser=make_litellm_summariser(model)),
}


def make_memory(mode: str, summary_model: str) -> BaseMemoryProvider:
    """Map CLI memory-mode strings to ``BaseMemoryProvider`` instances."""
    if mode not in _MEMORY_DISPATCH:
        raise SystemExit(
            f"Unknown memory mode {mode!r}; expected one of {sorted(_MEMORY_DISPATCH)}"
        )
    return _MEMORY_DISPATCH[mode](summary_model)


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
    p.add_argument(
        "--instrument",
        action="store_true",
        help=(
            "Enable per-attack ClaimTracker / MemoryAuditLog / "
            "ConfidenceDriftTracker instrumentation. Surfaces mar, "
            "distortion_rate, first_laundering_session, and "
            "context_size_curve onto each ExperimentResult, and writes a "
            "consolidated instruments JSON per attack into "
            "<results_dir>/<run_stem>_audits/. Default OFF."
        ),
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
        if args.instrument:
            print("Instrumentation: ON (claim tracker + audit log + drift tracker)")
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
    run_stem = f"multi_{datetime.now():%Y%m%d_%H%M%S}"
    log = RESULTS_DIR / f"{run_stem}.jsonl"
    audit_dir = (RESULTS_DIR / f"{run_stem}_audits") if args.instrument else None
    if audit_dir is not None:
        audit_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"Seeds: {len(seeds)} | Multi-session memory modes: {list(mem_modes)}")
    if args.instrument:
        print(f"Instrumentation: ON — audits in {audit_dir}/")
    print()

    def _run_one(orch_obj: Orchestrator, the_seed, *, is_baseline: bool):
        if not args.instrument:
            return orch_obj.run_attack(the_seed, is_baseline=is_baseline)
        assert audit_dir is not None  # invariant: instrument => audit_dir set
        claims = derive_planted_claims(the_seed)
        instruments = build_instruments(orch_obj.memory, claims)
        attack_id_local = the_seed.seed_id + ("_baseline" if is_baseline else "")
        path = audit_dir / f"audit_{attack_id_local}__{orch_obj.mode_name}.json"
        return orch_obj.run_attack(
            the_seed,
            is_baseline=is_baseline,
            instruments=instruments,
            audit_path=path,
        )

    baseline_memory = make_memory("no_memory", args.model)
    baseline_orch = Orchestrator(
        env, runner, baseline_memory, attacker, mode_name="no_memory"
    )

    print("-- Baselines (single-session, no_memory) --")
    for seed in seeds:
        print(f"  {seed.seed_id}_baseline:", end=" ")
        r = _run_one(baseline_orch, seed, is_baseline=True)
        all_results.append(r)
        with open(log, "a") as f:
            f.write(json.dumps(r.to_dict(), default=str) + "\n")

    for mode in mem_modes:
        memory = make_memory(mode, args.model)
        orch = Orchestrator(env, runner, memory, attacker, mode_name=mode)
        print(f"\n-- Multi-session [{mode}] --")
        for seed in seeds:
            print(f"  {seed.seed_id} [{mode}]:", end=" ")
            r = _run_one(orch, seed, is_baseline=False)
            all_results.append(r)
            with open(log, "a") as f:
                f.write(json.dumps(r.to_dict(), default=str) + "\n")

    violated = sum(1 for r in all_results if r.violation_detected)
    print(f"\nDone. {violated}/{len(all_results)} attacks with violation.")
    print(f"Results: {log}")

    analyze([r.to_dict() for r in all_results])


if __name__ == "__main__":
    main()
