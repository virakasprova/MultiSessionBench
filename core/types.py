"""Shared dataclasses for the multi-session infrastructure."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TaskSeed:
    """One attack scenario loaded from the v4.1 seeds JSON."""
    seed_id: str
    domain: str  # "airline" / "retail" / "banking_knowledge" — picks the env
    user_id: str
    reservation_id: str
    violation_tool: str
    false_claim: str
    policy_area: str
    session_intents: list[list[str]]
    customer_known_info: str
    hidden_attacker_goal: str | None = None
    # Optional explicit planted-claim records for ClaimTracker. Each entry is a
    # raw dict with keys {claim_id, canonical, variants, target_policy, negation}
    # matching ``memory.PlantedClaim``.
    planted_claim_data: list[dict[str, Any]] | None = None
    # Concurrent benign task — the customer's legitimate need that runs
    # alongside the adversarial intent. The benign reservation is independent
    # from ``reservation_id`` (the adversarial target).
    benign_intent: str | None = None
    benign_reservation_id: str | None = None
    expected_benign_actions: list[str] | None = None
    # Optional analysis tags — let the report split ASR by claim verifiability
    # (does the agent have a tool to falsify the claim?) and by where the
    # benign request lands in the dialogue (early = before the trigger turn,
    # late = interleaved with the adversarial ask). Defaults preserve behavior
    # on seeds written before these fields existed.
    claim_type: str = "verifiable"
    benign_timing: str = "early"
    # Per-seed override for SessionRunner.max_turns. Banking seeds frequently
    # need 30+ turns to fully exercise the policy decision; airline / retail
    # seeds finish in the default 20. ``None`` => use runner default.
    max_turns_override: int | None = None

    def num_sessions(self) -> int:
        return len(self.session_intents)


@dataclass
class SessionMechanicalResult:
    """Objective session outcome: tools, costs, transcript (no LLM judge)."""
    session_index: int
    violation: bool
    verified: bool
    verified_then_acted: bool
    transferred_to_human: bool
    tools_called: list[dict[str, Any]]
    writes: list[dict[str, Any]]
    cost: float
    transcript: str
    # Role-aware copy of the agent loop (system stripped). Schema per element:
    #   {"role": "user", "content": str}
    #   {"role": "assistant", "content": str, "tool_calls": [{"name", "arguments"}]?}
    #   {"role": "tool", "content": str, "name": str}
    messages: list[dict[str, Any]]
    # How the session loop ended:
    # transfer / attacker_stop / max_turns / agent_error / attacker_error.
    # Lets the report tell "agent defended" apart from a control-flow short-circuit.
    terminated_by: str = "unknown"


@dataclass
class SessionJudgeResult:
    """Subjective labels from post-hoc LLM judge on the transcript.

    Fields
    ------
    refused:
        True if the agent explicitly pushed back on the customer's false claim
        (not merely a lookup failure or tool error).
    accepted:
        True if the agent affirmed or treated as true the customer's
        disputed/false claim.
    confirmed:
        True if the agent explicitly asked the customer to confirm before a
        consequential action.
    confidence:
        Judge's self-reported confidence in its labels, 0.0–1.0.
        1.0 = completely clear from the transcript; 0.0 = highly ambiguous.
    reasoning:
        One-to-three sentence explanation of *why* the judge assigned these
        labels — which agent utterance(s) drove the decision and what made
        the case ambiguous or clear.
    """
    refused: bool
    accepted: bool
    confirmed: bool
    confidence: float = 1.0
    reasoning: str = ""


@dataclass
class SessionResult:
    """One session: mechanical facts plus optional judge scores (filled after ``enrich_judge``)."""
    mechanical: SessionMechanicalResult
    judge: SessionJudgeResult | None = None
    contradicted: bool | None = None
    compliance: float | None = None

    @property
    def cost(self) -> float:
        return self.mechanical.cost

    @property
    def transcript(self) -> str:
        return self.mechanical.transcript

    def to_dict(self) -> dict[str, Any]:
        return {
            "mechanical": asdict(self.mechanical),
            "judge": asdict(self.judge) if self.judge else None,
            "contradicted": self.contradicted,
            "compliance": self.compliance,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SessionResult:
        m = SessionMechanicalResult(**d["mechanical"])
        j = d.get("judge")
        if j:
            # Forward-compatible: old judged.jsonl rows lack confidence/reasoning;
            # provide safe defaults so loading never crashes on legacy data.
            judge = SessionJudgeResult(
                refused=bool(j.get("refused", False)),
                accepted=bool(j.get("accepted", False)),
                confirmed=bool(j.get("confirmed", False)),
                confidence=float(j.get("confidence", 1.0)),
                reasoning=j.get("reasoning", ""),
            )
        else:
            judge = None
        return SessionResult(
            mechanical=m,
            judge=judge,
            contradicted=d.get("contradicted"),
            compliance=d.get("compliance"),
        )


@dataclass
class SessionContext:
    """Per-session state passed to ``AttackerAgent``."""
    session_index: int
    seed: TaskSeed
    memory_text: str | None = None
    prior_results: list[SessionResult] = field(default_factory=list)


@dataclass
class ExperimentResult:
    """Aggregated result of one full attack bundle.

    The ``mar`` / ``distortion_rate`` / ``first_laundering_session`` /
    ``context_size_curve`` / ``audit_log_path`` fields are populated only
    when the run is invoked with ``--instrument`` (see
    ``core.instrumentation``); otherwise they remain ``None``.
    """
    attack_id: str
    memory_mode: str
    model: str
    violation_tool: str
    violation_detected: bool
    any_contradiction: bool | None
    persistence_curve: list[bool] | None
    compliance_score: float | None
    total_cost: float
    policy_area: str
    false_claim: str
    session_results: list[SessionResult]
    memory_contents: list[str]
    timestamp: str
    # Instrument outputs (populated only when ClaimTracker / MemoryAuditLog are wired).
    mar: float | None = None
    distortion_rate: float | None = None
    first_laundering_session: str | None = None
    context_size_curve: list[int] | None = None
    audit_log_path: str | None = None
    # Concurrent benign-task outcome. ``benign_completed`` = state matched AND
    # every required action fired against the specific benign reservation.
    # The two component flags let the report tell "agent tried but DB diverged"
    # from "agent never tried".
    benign_completed: bool | None = None
    benign_state_match: bool | None = None
    benign_actions_match: bool | None = None
    benign_actions_called: list[str] | None = None
    # How ``violation_detected`` was decided: "db_diff" (ground truth from
    # pre/post snapshot), "tool_match" (weaker fallback when the resource
    # wasn't snapshottable), "none" (defended), or None (pre-snapshot didn't
    # fire — usually missing seed.user_id).
    violation_grounding: str | None = None
    # Carry seed analysis tags onto the result so the report can split ASR
    # cells without re-loading the seed JSON.
    claim_type: str = "verifiable"
    benign_timing: str = "early"

    def to_dict(self) -> dict[str, Any]:
        return {
            "attack_id": self.attack_id,
            "memory_mode": self.memory_mode,
            "model": self.model,
            "violation_tool": self.violation_tool,
            "violation_detected": self.violation_detected,
            "any_contradiction": self.any_contradiction,
            "persistence_curve": self.persistence_curve,
            "compliance_score": self.compliance_score,
            "total_cost": self.total_cost,
            "policy_area": self.policy_area,
            "false_claim": self.false_claim,
            "session_results": [sr.to_dict() for sr in self.session_results],
            "memory_contents": self.memory_contents,
            "timestamp": self.timestamp,
            "mar": self.mar,
            "distortion_rate": self.distortion_rate,
            "first_laundering_session": self.first_laundering_session,
            "context_size_curve": self.context_size_curve,
            "audit_log_path": self.audit_log_path,
            "benign_completed": self.benign_completed,
            "benign_state_match": self.benign_state_match,
            "benign_actions_match": self.benign_actions_match,
            "benign_actions_called": self.benign_actions_called,
            "violation_grounding": self.violation_grounding,
            "claim_type": self.claim_type,
            "benign_timing": self.benign_timing,
        }


def experiment_from_dict(d: dict[str, Any]) -> ExperimentResult:
    srs = [SessionResult.from_dict(x) for x in d.get("session_results", [])]
    return ExperimentResult(
        attack_id=d["attack_id"],
        memory_mode=d["memory_mode"],
        model=d.get("model", ""),
        violation_tool=d.get("violation_tool", ""),
        violation_detected=d.get("violation_detected", False),
        any_contradiction=d.get("any_contradiction"),
        persistence_curve=d.get("persistence_curve"),
        compliance_score=d.get("compliance_score"),
        total_cost=d.get("total_cost", 0.0),
        policy_area=d.get("policy_area", ""),
        false_claim=d.get("false_claim", ""),
        session_results=srs,
        memory_contents=d.get("memory_contents", []),
        timestamp=d.get("timestamp", ""),
        mar=d.get("mar"),
        distortion_rate=d.get("distortion_rate"),
        first_laundering_session=d.get("first_laundering_session"),
        context_size_curve=d.get("context_size_curve"),
        audit_log_path=d.get("audit_log_path"),
        benign_completed=d.get("benign_completed"),
        benign_state_match=d.get("benign_state_match"),
        benign_actions_match=d.get("benign_actions_match"),
        benign_actions_called=d.get("benign_actions_called"),
        violation_grounding=d.get("violation_grounding"),
        claim_type=d.get("claim_type", "verifiable"),
        benign_timing=d.get("benign_timing", "early"),
    )
