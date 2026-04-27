"""memory.base — core data structures and abstract interfaces.

Hosts the structured-turn API used by every memory provider:

- ``Turn`` / ``Session``  — dataclasses for one utterance / one completed session.
- ``BaseMemoryProvider``  — ABC with ``add_session`` / ``get_context`` /
  ``get_memory_contents`` / optional ``snapshot``.

``Turn.role`` is one of ``"user" | "assistant" | "tool"``. The ``"tool"`` role
preserves tool-call fidelity from agent-loop transcripts: the orchestrator
emits a ``Turn(role="assistant", tool_name="X", content="[tool_call: X(...)]")``
followed by a ``Turn(role="tool", tool_name="X", content="<tool output>")`` for
every tool invocation. Providers that summarise / inject prior context will
therefore see the full agent loop rather than just the dialogue.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# Structured turn / session datatypes
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    """A single utterance in a conversation.

    ``role="tool"`` represents a tool invocation result. For an assistant turn
    that triggered a tool call, set ``tool_name`` and put a short marker like
    ``"[tool_call: get_reservation_details(...)]"`` in ``content``.
    """
    role: Literal["user", "assistant", "tool"]
    content: str
    session_id: str = ""
    turn_index: int = 0
    user_type: Literal["adversarial", "benign", "unknown"] = "unknown"
    # Set by ClaimTracker at ingestion time for survival analysis.
    is_planted_claim: bool = False
    claim_id: Optional[str] = None
    # Tool-call metadata. Populated for role="tool" (the executed tool's name)
    # and for role="assistant" turns whose content is a tool-call marker.
    tool_name: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "user_type": self.user_type,
            "is_planted_claim": self.is_planted_claim,
            "claim_id": self.claim_id,
            "tool_name": self.tool_name,
        }


@dataclass
class Session:
    """A completed conversation session."""
    session_id: str
    turns: list[Turn]
    user_type: Literal["adversarial", "benign", "unknown"] = "unknown"
    user_id: str = "default"              # for cross-user contamination experiments
    metadata: dict = field(default_factory=dict)

    def as_text(self, include_roles: bool = True) -> str:
        if include_roles:
            return "\n".join(f"[{t.role}]: {t.content}" for t in self.turns)
        return "\n".join(t.content for t in self.turns)

    def planted_claims(self) -> list[Turn]:
        return [t for t in self.turns if t.is_planted_claim]


# ---------------------------------------------------------------------------
# BaseMemoryProvider — structured-turn interface used by every provider
# ---------------------------------------------------------------------------

class BaseMemoryProvider(ABC):
    """
    Common interface for all structured-turn memory providers.

    Required methods
    ----------------
    add_session(turns, session_id, user_type, user_id) -> str
    get_context(query, exclude_session_id, user_id)    -> str
    get_memory_contents()                               -> list[dict]

    Optional override
    -----------------
    snapshot()  -> str   (raw memory state; used by MemoryAuditLog)
    """

    def __init__(self, use_attribution: bool = False):
        self.use_attribution = use_attribution
        self._sessions: list[Session] = []

    @abstractmethod
    def add_session(
        self,
        turns: list[Turn],
        session_id: Optional[str] = None,
        user_type: Literal["adversarial", "benign", "unknown"] = "unknown",
        user_id: str = "default",
    ) -> str: ...

    @abstractmethod
    def get_context(
        self,
        query: str = "",
        exclude_session_id: Optional[str] = None,
        user_id: str = "default",
    ) -> str: ...

    @abstractmethod
    def get_memory_contents(self) -> list[dict]: ...

    def snapshot(self) -> str:
        """Human-readable dump of current memory state (for AuditLog)."""
        return json.dumps(self.get_memory_contents(), indent=2)

    def clear(self) -> None:
        self._sessions.clear()

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.get_memory_contents(), f, indent=2)

    def _attribution_prefix(self, role: str, user_type: str) -> str:
        if not self.use_attribution:
            return ""
        if role == "user":
            tag = "Adversarial user claimed" if user_type == "adversarial" else "User stated"
        elif role == "tool":
            # Tool outputs are observed facts, not claims — no attribution prefix.
            return ""
        else:
            tag = "Assistant stated"
        return f"{tag}: "

    def __len__(self) -> int:
        return len(self._sessions)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"sessions={len(self._sessions)}, "
            f"attribution={self.use_attribution})"
        )
