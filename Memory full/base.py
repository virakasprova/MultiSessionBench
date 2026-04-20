"""
memory/base.py
==============
Core data structures and the abstract BaseMemoryProvider interface.

New in this version
-------------------
- Turn carries an optional `is_planted_claim` flag + `claim_id` for ClaimTracker
- Session carries a `user_id` for cross-user contamination experiments
- BaseMemoryProvider exposes a snapshot() method for MemoryAuditLog
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    """A single utterance in a conversation."""
    role: Literal["user", "assistant"]
    content: str
    session_id: str = ""
    turn_index: int = 0
    user_type: Literal["adversarial", "benign", "unknown"] = "unknown"
    # Claim tracking — set by ClaimTracker at ingestion time
    is_planted_claim: bool = False
    claim_id: Optional[str] = None        # links back to a PlantedClaim record

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "user_type": self.user_type,
            "is_planted_claim": self.is_planted_claim,
            "claim_id": self.claim_id,
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
# Abstract base
# ---------------------------------------------------------------------------

class BaseMemoryProvider(ABC):
    """
    Common interface for all memory providers.

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
