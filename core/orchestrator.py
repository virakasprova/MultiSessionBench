"""Orchestrator — sequences sessions, injects memory, feeds attacker messages, collects results.

Aggregate compliance / persistence / contradiction require a judge pass on the JSONL
(``experiments/enrich_judge.py``) unless sessions were already enriched.
"""
from __future__ import annotations

from datetime import datetime

from attackers.base import AttackerAgent
from core.types import TaskSeed, SessionContext, SessionResult, ExperimentResult
from core.environment import TauBenchEnv
from core.session_runner import SessionRunner
from memory.base import MemoryProvider


class Orchestrator:
    def __init__(
        self,
        env: TauBenchEnv,
        runner: SessionRunner,
        memory: MemoryProvider,
        attacker: AttackerAgent,
    ):
        self.env = env
        self.runner = runner
        self.memory = memory
        self.attacker = attacker

    def run_attack(self, seed: TaskSeed, is_baseline: bool = False) -> ExperimentResult:
        """Run a full attack bundle: reset env once, then loop over sessions."""
        self.env.reset()
        self.memory.reset()

        n_sessions = self.attacker.get_session_count(seed, is_baseline)
        results: list[SessionResult] = []
        memory_snapshots: list[str] = []
        total_cost = 0.0

        for si in range(n_sessions):
            ctx = SessionContext(
                session_index=si,
                seed=seed,
                memory_text=self.memory.get_prompt_injection(),
                prior_results=list(results),
            )

            system_prompt = self.env.wiki
            if ctx.memory_text:
                system_prompt += "\n\n" + ctx.memory_text

            print(f"      S{si + 1}/{n_sessions}", end=" ", flush=True)
            sr = self.runner.run(
                attacker=self.attacker,
                context=ctx,
                system_prompt=system_prompt,
                violation_tool=seed.violation_tool,
                session_index=si,
                is_baseline=is_baseline,
            )

            results.append(sr)
            total_cost += sr.cost

            self.memory.update(sr)
            snapshot = self.memory.get_prompt_injection() or ""
            if snapshot:
                memory_snapshots.append(snapshot)

        violated = any(r.mechanical.violation for r in results)
        print(f"-> {'VIOLATION' if violated else 'defended'}")

        return ExperimentResult(
            attack_id=seed.seed_id + ("_baseline" if is_baseline else ""),
            memory_mode=self.memory.mode_name,
            is_baseline=is_baseline,
            model=self.runner.model,
            violation_tool=seed.violation_tool,
            violation_detected=violated,
            any_contradiction=None,
            persistence_curve=None,
            compliance_score=None,
            total_cost=total_cost,
            policy_area=seed.policy_area,
            false_claim=seed.false_claim,
            session_results=results,
            memory_contents=memory_snapshots,
            timestamp=datetime.now().isoformat(),
        )
