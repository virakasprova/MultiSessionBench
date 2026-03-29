"""Shared dataclasses for the multi-session infrastructure."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TaskSeed:
    """One attack scenario loaded from YAML."""
    seed_id: str
    user_id: str
    reservation_id: str
    violation_tool: str
    false_claim: str
    policy_area: str
    session_intents: list[list[str]]
    baseline_messages: list[str] | None = None

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
    messages: list[dict[str, str]]


@dataclass
class SessionJudgeResult:
    """Subjective labels from post-hoc LLM judge on the transcript."""
    refused: bool
    accepted: bool
    confirmed: bool


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
        judge = SessionJudgeResult(**j) if j else None
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
    """Aggregated result of one full attack bundle."""
    attack_id: str
    memory_mode: str
    is_baseline: bool
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "attack_id": self.attack_id,
            "memory_mode": self.memory_mode,
            "is_baseline": self.is_baseline,
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
        }


def experiment_from_dict(d: dict[str, Any]) -> ExperimentResult:
    srs = [SessionResult.from_dict(x) for x in d.get("session_results", [])]
    return ExperimentResult(
        attack_id=d["attack_id"],
        memory_mode=d["memory_mode"],
        is_baseline=d.get("is_baseline", False),
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
    )
