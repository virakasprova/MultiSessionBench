"""
memory/providers/summary.py
============================
RollingSummaryProvider and CumulativeSummaryProvider.

Key addition for the paper: both providers store the raw summary text as a
first-class artifact (self._summary_log), queryable post-hoc for claim
survival analysis.  The summary log is separate from get_memory_contents()
so researchers can diff "what the agent saw" vs "what was in raw memory".
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Callable, Literal, Optional

from memory.base import BaseMemoryProvider, Session, Turn


# ---------------------------------------------------------------------------
# Summary log entry  (novel: raw summaries stored for claim survival analysis)
# ---------------------------------------------------------------------------

@dataclass
class SummaryLogEntry:
    session_id: str
    user_id: str
    user_type: str
    raw_session_text: str    # full verbatim session, before summarisation
    summary_text: str        # what the LLM produced


# ---------------------------------------------------------------------------
# Fallback summariser (replace with real LLM call in experiments)
# ---------------------------------------------------------------------------

def _extractive_summariser(session_text: str) -> str:
    lines = [l for l in session_text.splitlines()
             if l.strip().lower().startswith("[user]")]
    snippet = " | ".join(l.replace("[user]: ", "").strip() for l in lines[:3])
    return f"User discussed: {snippet}" if snippet else "No notable user turns."


def _default_cumulative(existing: str, new_text: str) -> str:
    import re
    lines = [l for l in new_text.splitlines() if "[user]" in l.lower()]
    snippet = " | ".join(
        re.sub(r"\[user\]:\s*", "", l).strip() for l in lines[:2]
    )
    if not existing:
        return f"User has previously discussed: {snippet}"
    return f"{existing} Later, they also discussed: {snippet}"


# ---------------------------------------------------------------------------
# Rolling summary provider
# ---------------------------------------------------------------------------

class RollingSummaryProvider(BaseMemoryProvider):
    """
    Each session is compressed independently.
    All summaries concatenated and injected.

    Paper note: self._summary_log stores raw session text + summary text for
    each session, enabling post-hoc analysis of which claims survived/distorted.
    """

    def __init__(self, summariser=None, use_attribution=False, user_scoped=True):
        super().__init__(use_attribution=use_attribution)
        self._summariser: Callable[[str], str] = summariser or _extractive_summariser
        self._summary_log: list[SummaryLogEntry] = []
        # When True (default), ``get_context(user_id=X)`` only returns
        # summaries added for user ``X``. ``CrossUserMemoryStore`` builds a
        # second instance with ``user_scoped=False`` for the explicit shared
        # pool used in "shared" / "shared_summary" modes.
        self.user_scoped = user_scoped

    def add_session(self, turns, session_id=None,
                    user_type="unknown", user_id="default"):
        session_id = session_id or str(uuid.uuid4())
        for i, t in enumerate(turns):
            t.session_id = session_id; t.turn_index = i; t.user_type = user_type
        session = Session(session_id, turns, user_type, user_id)
        self._sessions.append(session)

        raw_text = session.as_text(include_roles=True)
        summary_text = self._summariser(raw_text)

        self._summary_log.append(SummaryLogEntry(
            session_id=session_id,
            user_id=user_id,
            user_type=user_type,
            raw_session_text=raw_text,
            summary_text=summary_text,
        ))
        return session_id

    def get_context(self, query="", exclude_session_id=None, user_id="default"):
        # Default is per-user isolation: only this user's summaries are
        # returned. Cross-user mixing is the explicit job of
        # ``CrossUserMemoryStore`` ("shared_summary" mode), which talks to a
        # second RollingSummaryProvider it owns; a direct caller asking for
        # ``user_id="X"`` should not see other users' content.
        items = [
            e for e in self._summary_log
            if e.session_id != exclude_session_id
            and (not self.user_scoped or e.user_id == user_id)
        ]
        if not items:
            return ""
        lines = ["Summaries of prior sessions:"]
        for entry in items:
            if self.use_attribution:
                tag = ("Adversarial session" if entry.user_type == "adversarial"
                       else f"{entry.user_type.capitalize()} session")
                lines.append(f"  [{tag}] {entry.summary_text}")
            else:
                lines.append(f"  - {entry.summary_text}")
        return "\n".join(lines)

    def snapshot(self) -> str:
        """Returns summary texts joined — this is what the agent sees."""
        return "\n".join(e.summary_text for e in self._summary_log)

    def get_memory_contents(self):
        base = []
        summary_map = {e.session_id: e.summary_text for e in self._summary_log}
        for s in self._sessions:
            for t in s.turns:
                base.append({
                    "session_id": s.session_id, "user_type": s.user_type,
                    "user_id": s.user_id, "role": t.role,
                    "turn_index": t.turn_index, "content": t.content,
                    "is_planted_claim": t.is_planted_claim, "claim_id": t.claim_id,
                    "summary": summary_map.get(s.session_id, ""),
                })
        return base

    def get_summary_log(self) -> list[SummaryLogEntry]:
        """Direct access for claim survival / distortion analysis."""
        return list(self._summary_log)

    def clear(self):
        super().clear()
        self._summary_log.clear()


# ---------------------------------------------------------------------------
# Cumulative summary provider
# ---------------------------------------------------------------------------

class CumulativeSummaryProvider(BaseMemoryProvider):
    """
    Single running summary updated after each session.
    Most vulnerable to gradual belief injection / laundering.
    """

    def __init__(self, summariser=None, use_attribution=False, user_scoped=True):
        super().__init__(use_attribution=use_attribution)
        self._summariser: Callable[[str, str], str] = summariser or _default_cumulative
        # Per-user running summary. ``user_scoped=False`` collapses everything
        # under a single ``_GLOBAL_KEY`` for the shared-pool case used by
        # ``CrossUserMemoryStore``.
        self.user_scoped = user_scoped
        self._running_summaries: dict[str, str] = {}
        self._summary_history: list[dict] = []   # full evolution log for paper

    _GLOBAL_KEY = "__global__"

    def _key(self, user_id: str) -> str:
        return user_id if self.user_scoped else self._GLOBAL_KEY

    def add_session(self, turns, session_id=None,
                    user_type="unknown", user_id="default"):
        session_id = session_id or str(uuid.uuid4())
        for i, t in enumerate(turns):
            t.session_id = session_id; t.turn_index = i; t.user_type = user_type
        session = Session(session_id, turns, user_type, user_id)
        self._sessions.append(session)

        key = self._key(user_id)
        new_text = session.as_text(include_roles=True)
        prev_summary = self._running_summaries.get(key, "")
        new_summary = self._summariser(prev_summary, new_text)
        self._running_summaries[key] = new_summary

        self._summary_history.append({
            "session_id": session_id,
            "user_id": user_id,
            "user_type": user_type,
            "prev_summary": prev_summary,
            "new_summary": new_summary,
        })
        return session_id

    def get_context(self, query="", exclude_session_id=None, user_id="default"):
        running = self._running_summaries.get(self._key(user_id), "")
        if not running:
            return ""
        return f"Running summary of all prior sessions:\n  {running}"

    def snapshot(self) -> str:
        # Aggregated view across users — used by claim_tracker / audit_log
        # to inspect the full state of the provider. For ``user_scoped=False``
        # this is just the single global summary.
        if not self._running_summaries:
            return ""
        return "\n".join(self._running_summaries.values())

    def get_running_summary(self, user_id: str = "default") -> str:
        return self._running_summaries.get(self._key(user_id), "")

    def get_summary_history(self) -> list[dict]:
        """Full evolution of the summary — each update recorded for paper analysis."""
        return list(self._summary_history)

    def get_memory_contents(self):
        base = []
        for s in self._sessions:
            running = self._running_summaries.get(self._key(s.user_id), "")
            for t in s.turns:
                base.append({
                    "session_id": s.session_id, "user_type": s.user_type,
                    "user_id": s.user_id, "role": t.role,
                    "turn_index": t.turn_index, "content": t.content,
                    "is_planted_claim": t.is_planted_claim, "claim_id": t.claim_id,
                    "running_summary": running,
                })
        return base

    def clear(self):
        super().clear()
        self._running_summaries.clear()
        self._summary_history.clear()
