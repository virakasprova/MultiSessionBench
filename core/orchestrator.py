"""Orchestrator — sequences sessions, injects memory, feeds attacker messages, collects results.

The orchestrator drives a ``BaseMemoryProvider`` (structured-turn API). After each
session it converts the runner's recorded ``messages`` into ``Turn`` objects and
calls ``add_session``; before each session it asks the provider for any context
to splice into the system prompt via ``get_context``.

When invoked with an ``Instruments`` bundle, the orchestrator additionally drives
the three measurement layers from ``memory/`` (``ClaimTracker``,
``MemoryAuditLog``, ``ConfidenceDriftTracker``) at the right hook-points, and
surfaces ``mar`` / ``distortion_rate`` / ``first_laundering_session`` /
``context_size_curve`` onto the returned ``ExperimentResult``. A consolidated
JSON of the per-session audit + drift records is written if ``audit_path`` is
given.

Aggregate compliance / persistence / contradiction require a judge pass on the
JSONL (``experiments/enrich_judge.py``) unless sessions were already enriched.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from attackers.base import AttackerAgent
from core.instrumentation import Instruments
from core.types import (
    ExperimentResult,
    SessionContext,
    SessionResult,
    TaskSeed,
)
from core.environment import TauBenchEnv
from core.session_runner import SessionRunner
from memory.base import BaseMemoryProvider, Turn


class Orchestrator:
    """Sequences sessions for one attack bundle and threads memory between them.

    Parameters
    ----------
    env, runner, attacker
        The usual three. ``attacker`` produces user utterances; ``runner`` runs
        the agent loop; ``env`` provides tools / wiki and gets reset between
        attack bundles.
    memory
        A ``BaseMemoryProvider`` (e.g. anything from ``memory.make_memory_provider``).
    mode_name
        Short identifier recorded in ``ExperimentResult.memory_mode`` (e.g.
        ``"no_memory"``, ``"full_history"``, ``"summary"``). Kept as an
        explicit constructor arg so ``BaseMemoryProvider`` doesn't need to
        carry display metadata.
    user_type
        ``"adversarial"`` for attack runs (default), ``"benign"`` for control
        runs. Threaded into every ``add_session`` call so providers that do
        attribution / cross-user analysis can tell sessions apart.
    """

    def __init__(
        self,
        env: TauBenchEnv,
        runner: SessionRunner,
        memory: BaseMemoryProvider,
        attacker: AttackerAgent,
        mode_name: str,
        user_type: Literal["adversarial", "benign", "unknown"] = "adversarial",
    ):
        self.env = env
        self.runner = runner
        self.memory = memory
        self.attacker = attacker
        self.mode_name = mode_name
        self.user_type = user_type

    def run_attack(
        self,
        seed: TaskSeed,
        is_baseline: bool = False,
        *,
        instruments: Optional[Instruments] = None,
        audit_path: Optional[Path] = None,
    ) -> ExperimentResult:
        """Run a full attack bundle: reset env once, then loop over sessions.

        Parameters
        ----------
        instruments
            Optional ``Instruments`` bundle from ``core.instrumentation``. When
            provided, the orchestrator drives ``ClaimTracker.add_session`` (in
            place of ``self.memory.add_session``), records pre/post audit log
            entries, and feeds the injected context into the
            ``ConfidenceDriftTracker``. The resulting metrics are surfaced
            onto ``ExperimentResult``.
        audit_path
            Optional path. If given alongside ``instruments``, a consolidated
            JSON of {audit records, drift records, claims} is written there
            and the path is recorded on ``ExperimentResult.audit_log_path``.
        """
        self.env.reset()
        self.memory.clear()

        n_sessions = self.attacker.get_session_count(seed, is_baseline)
        results: list[SessionResult] = []
        memory_snapshots: list[str] = []
        total_cost = 0.0
        attack_id = seed.seed_id + ("_baseline" if is_baseline else "")

        for si in range(n_sessions):
            session_id = (
                f"{seed.seed_id}_s{si}"
                + ("_baseline" if is_baseline else "")
            )
            query = _query_for_seed(seed, si)

            # Pre-session injection. The query is unused by the modes the
            # runners currently expose (no_memory / full_history / summary);
            # plumbed through anyway so RAG-style providers slot in cleanly.
            memory_text = self.memory.get_context(
                query=query,
                exclude_session_id=session_id,
                user_id=seed.user_id,
            ) or None

            # Instrument hooks: drift.record + audit.record_pre_session BEFORE
            # the agent runs, both consume the actually-injected context (i.e.
            # what the agent will see this session).
            if instruments is not None:
                injected = memory_text or ""
                instruments.drift.record(
                    session_index=si,
                    session_id=session_id,
                    injected_context=injected,
                )
                instruments.audit_log.record_pre_session(
                    session_id=session_id,
                    injected_context=injected,
                    query=query,
                )

            ctx = SessionContext(
                session_index=si,
                seed=seed,
                memory_text=memory_text,
                prior_results=list(results),
            )

            system_prompt = self.env.wiki
            if memory_text:
                system_prompt += "\n\n" + memory_text

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

            turns = messages_to_turns(sr.mechanical.messages)
            if turns:
                # When a ClaimTracker is wired, route through it: the tracker
                # tags planted-claim turns, calls provider.add_session, and
                # immediately classifies each claim as PRESENT / DISTORTED /
                # ABSENT against the post-add provider snapshot. When no
                # tracker is wired, this is a plain provider call (unchanged
                # from the legacy path).
                if instruments is not None:
                    instruments.tracker.add_session(
                        turns,
                        session_id=session_id,
                        user_type=self.user_type,
                        user_id=seed.user_id,
                    )
                else:
                    self.memory.add_session(
                        turns,
                        session_id=session_id,
                        user_type=self.user_type,
                        user_id=seed.user_id,
                    )

            # Post-session audit hook: snapshots the post-add memory state
            # AND copies the per-session claim statuses out of the tracker.
            if instruments is not None:
                instruments.audit_log.record_post_session(session_id)

            snapshot = self.memory.get_context(
                query=query,
                user_id=seed.user_id,
            ) or ""
            if snapshot:
                memory_snapshots.append(snapshot)

        violated = any(r.mechanical.violation for r in results)
        print(f"-> {'VIOLATION' if violated else 'defended'}")

        result = ExperimentResult(
            attack_id=attack_id,
            memory_mode=self.mode_name,
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

        # Surface instrument outputs onto the result and persist the full
        # audit JSON for qualitative analysis.
        if instruments is not None:
            result.mar = instruments.tracker.mar()
            result.distortion_rate = instruments.tracker.distortion_rate()
            result.context_size_curve = instruments.audit_log.context_size_curve()
            primary = instruments.primary_claim_id()
            if primary is not None:
                result.first_laundering_session = (
                    instruments.audit_log.first_laundering_session(primary)
                )
            if audit_path is not None:
                _save_instruments_json(instruments, audit_path, attack_id, self.mode_name)
                result.audit_log_path = str(audit_path)

        return result


# ---------------------------------------------------------------------------
# Instrument persistence
# ---------------------------------------------------------------------------

def _save_instruments_json(
    instruments: Instruments,
    path: Path,
    attack_id: str,
    memory_mode: str,
) -> None:
    """Write a consolidated JSON of {claims, audit records, drift records}.

    One file per attack so the qualitative-analysis path is just
    ``json.load(open(audit_log_path))``. Schema is intentionally flat — no
    cross-references between sections, just three lists side-by-side keyed by
    session_index.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    audit_records = [
        {
            "session_id": r.session_id,
            "session_index": r.session_index,
            "timestamp": r.timestamp,
            "query": r.query,
            "context_token_estimate": r.context_token_estimate,
            "claim_statuses": r.claim_statuses,
            "injected_context": r.injected_context,
            "memory_snapshot": r.memory_snapshot,
        }
        for r in instruments.audit_log._records
    ]
    drift_records = instruments.drift.get_report().to_dict()
    claims_dump = [
        {
            "claim_id": c.claim_id,
            "canonical": c.canonical,
            "variants": c.variants,
            "target_policy": c.target_policy,
            "negation": c.negation,
        }
        for c in instruments.claims
    ]
    payload = {
        "attack_id": attack_id,
        "memory_mode": memory_mode,
        "claims": claims_dump,
        "audit": audit_records,
        "drift": drift_records,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Message -> Turn conversion
# ---------------------------------------------------------------------------

def messages_to_turns(messages: list[dict[str, Any]]) -> list[Turn]:
    """Convert ``SessionMechanicalResult.messages`` into structured ``Turn``s.

    Tool-call fidelity is preserved: an assistant message with ``tool_calls``
    becomes a ``Turn(role="assistant", tool_name=...)`` whose content is a
    short ``[tool_call: name(args)]`` marker, and the corresponding tool
    output becomes a ``Turn(role="tool", tool_name=..., content=<output>)``.

    This means downstream providers (e.g. ``RollingSummaryProvider``,
    ``FullContextProvider``) see the same agent-loop detail the legacy
    transcript-based providers used to see.
    """
    turns: list[Turn] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "") or ""
        if role == "user":
            turns.append(Turn(role="user", content=content))
        elif role == "assistant":
            tool_calls = m.get("tool_calls") or []
            if tool_calls:
                tc = tool_calls[0]
                tool_name = tc.get("name", "") or ""
                args_str = _shorten(tc.get("arguments", "") or "", 240)
                marker = f"[tool_call: {tool_name}({args_str})]"
                # Preserve any natural-language content the assistant emitted
                # alongside the tool call (rare with tool-using models, but
                # possible) by appending the marker.
                merged = f"{content}\n{marker}".strip() if content else marker
                turns.append(
                    Turn(
                        role="assistant",
                        content=merged,
                        tool_name=tool_name,
                    )
                )
            elif content:
                turns.append(Turn(role="assistant", content=content))
        elif role == "tool":
            turns.append(
                Turn(
                    role="tool",
                    content=content,
                    tool_name=m.get("name") or None,
                )
            )
    return turns


def _shorten(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _query_for_seed(seed: TaskSeed, session_index: int) -> str:
    """Build a natural-language query for memory retrieval for the upcoming session.

    Uses the first scripted intent line if available (LLMAttacker case);
    otherwise falls back to ``hidden_attacker_goal`` / ``false_claim``
    (CraftLLMAttacker / tau-deception case). Returns ``""`` if nothing is
    available — providers that need a query (RAG) will then return ``""``.
    """
    intents = seed.session_intents or []
    if 0 <= session_index < len(intents):
        bucket = intents[session_index]
        if bucket:
            first = bucket[0] if isinstance(bucket, list) else bucket
            if isinstance(first, str) and first.strip():
                return first
    if seed.hidden_attacker_goal:
        return seed.hidden_attacker_goal
    return seed.false_claim or ""
