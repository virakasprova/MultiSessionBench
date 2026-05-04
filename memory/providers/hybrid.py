"""
memory/providers/hybrid.py
"""
from __future__ import annotations

import uuid
from typing import Callable, Literal, Optional

from memory.base import BaseMemoryProvider, Session, Turn
from memory.providers.full_context import FullContextProvider
from memory.providers.summary import RollingSummaryProvider
from memory.providers.rag import RAGMemoryProvider, EmbeddingBackend


class RecentFullOldSummaryProvider(BaseMemoryProvider):
    """Recent N sessions verbatim + older sessions as rolling summaries."""

    def __init__(self, recent_n=2, summariser=None, use_attribution=False):
        super().__init__(use_attribution=use_attribution)
        self.recent_n = recent_n
        self._full = FullContextProvider(use_attribution=use_attribution)
        self._summary = RollingSummaryProvider(summariser=summariser,
                                               use_attribution=use_attribution)

    def add_session(self, turns, session_id=None,
                    user_type="unknown", user_id="default"):
        session_id = session_id or str(uuid.uuid4())
        for i, t in enumerate(turns):
            t.session_id = session_id; t.turn_index = i; t.user_type = user_type
        self._sessions.append(Session(session_id, turns, user_type, user_id))
        self._full.add_session(turns, session_id=session_id, user_type=user_type, user_id=user_id)
        self._summary.add_session(turns, session_id=session_id, user_type=user_type, user_id=user_id)
        return session_id

    def get_context(self, query="", exclude_session_id=None, user_id="default"):
        all_sessions = [s for s in self._sessions if s.session_id != exclude_session_id]
        if not all_sessions:
            return ""
        # ``all_sessions[-0:]`` is the entire list (not empty), which would
        # invert the "recent vs old" partition when ``recent_n=0``. Guard the
        # slice explicitly so 0 means "no recent sessions" as intended.
        if self.recent_n <= 0:
            recent_ids: set[str] = set()
            old_ids = {s.session_id for s in all_sessions}
        else:
            recent_ids = {s.session_id for s in all_sessions[-self.recent_n:]}
            old_ids = {s.session_id for s in all_sessions[:-self.recent_n]}
        parts = []

        old_summaries = [e for e in self._summary._summary_log
                         if e.session_id in old_ids]
        if old_summaries:
            lines = ["Older session summaries:"]
            for e in old_summaries:
                tag = (f"[{'Adversarial' if e.user_type=='adversarial' else e.user_type.capitalize()} session] "
                       if self.use_attribution else "- ")
                lines.append(f"  {tag}{e.summary_text}")
            parts.append("\n".join(lines))

        recent_sessions = [s for s in all_sessions if s.session_id in recent_ids]
        if recent_sessions:
            lines = ["Recent session history (verbatim):"]
            for s in recent_sessions:
                lines.append(f"  --- Session {s.session_id[:8]} ---")
                for t in s.turns:
                    prefix = self._attribution_prefix(t.role, s.user_type)
                    lines.append(f"  [{t.role}]: {prefix}{t.content}")
            parts.append("\n".join(lines))

        return "\n\n".join(parts)

    def get_memory_contents(self):
        return self._full.get_memory_contents()

    def clear(self):
        super().clear()
        self._full.clear()
        self._summary.clear()


class SummaryPlusRAGProvider(BaseMemoryProvider):
    """Rolling summaries (always-present) + RAG chunks (query-specific)."""

    def __init__(self, summariser=None, embedding_backend=None,
                 top_k=3, use_attribution=False):
        super().__init__(use_attribution=use_attribution)
        self._summary = RollingSummaryProvider(summariser=summariser,
                                               use_attribution=use_attribution)
        self._rag = RAGMemoryProvider(embedding_backend=embedding_backend,
                                      top_k=top_k, use_attribution=use_attribution)

    def add_session(self, turns, session_id=None,
                    user_type="unknown", user_id="default"):
        session_id = session_id or str(uuid.uuid4())
        for i, t in enumerate(turns):
            t.session_id = session_id; t.turn_index = i; t.user_type = user_type
        self._sessions.append(Session(session_id, turns, user_type, user_id))
        self._summary.add_session(turns, session_id=session_id, user_type=user_type, user_id=user_id)
        self._rag.add_session(turns, session_id=session_id, user_type=user_type, user_id=user_id)
        return session_id

    def get_context(self, query="", exclude_session_id=None, user_id="default"):
        parts = []
        s = self._summary.get_context(exclude_session_id=exclude_session_id, user_id=user_id)
        if s:
            parts.append(s)
        if query:
            r = self._rag.get_context(query=query, exclude_session_id=exclude_session_id, user_id=user_id)
            if r:
                parts.append(r)
        return "\n\n".join(parts)

    def get_memory_contents(self):
        return self._summary.get_memory_contents()

    def clear(self):
        super().clear()
        self._summary.clear()
        self._rag.clear()
