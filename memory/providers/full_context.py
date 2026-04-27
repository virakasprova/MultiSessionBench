"""
memory/providers/full_context.py
"""
from __future__ import annotations
import uuid
from typing import Literal, Optional
from memory.base import BaseMemoryProvider, Session, Turn


class NoMemoryProvider(BaseMemoryProvider):
    """Stateless control. get_context() always returns ''."""

    def add_session(self, turns, session_id=None,
                    user_type="unknown", user_id="default"):
        session_id = session_id or str(uuid.uuid4())
        for i, t in enumerate(turns):
            t.session_id = session_id; t.turn_index = i; t.user_type = user_type
        self._sessions.append(Session(session_id, turns, user_type, user_id))
        return session_id

    def get_context(self, query="", exclude_session_id=None, user_id="default"):
        return ""

    def get_memory_contents(self):
        return []


class FullContextProvider(BaseMemoryProvider):
    """Appends all prior turns verbatim. Optional sliding window."""

    def __init__(self, max_sessions=None, use_attribution=False):
        super().__init__(use_attribution=use_attribution)
        self.max_sessions = max_sessions

    def add_session(self, turns, session_id=None,
                    user_type="unknown", user_id="default"):
        session_id = session_id or str(uuid.uuid4())
        for i, t in enumerate(turns):
            t.session_id = session_id; t.turn_index = i; t.user_type = user_type
        self._sessions.append(Session(session_id, turns, user_type, user_id))
        return session_id

    def get_context(self, query="", exclude_session_id=None, user_id="default"):
        sessions = [s for s in self._sessions if s.session_id != exclude_session_id]
        if self.max_sessions is not None:
            sessions = sessions[-self.max_sessions:]
        if not sessions:
            return ""
        lines = ["Conversation history from prior sessions:"]
        for s in sessions:
            lines.append(f"  --- Session {s.session_id[:8]} ---")
            for t in s.turns:
                prefix = self._attribution_prefix(t.role, s.user_type)
                lines.append(f"  [{t.role}]: {prefix}{t.content}")
        return "\n".join(lines)

    def get_memory_contents(self):
        result = []
        for s in self._sessions:
            for t in s.turns:
                result.append({
                    "session_id": s.session_id, "user_type": s.user_type,
                    "user_id": s.user_id, "role": t.role,
                    "turn_index": t.turn_index, "content": t.content,
                    "is_planted_claim": t.is_planted_claim, "claim_id": t.claim_id,
                })
        return result
