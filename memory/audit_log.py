"""
memory/audit_log.py
===================
Novel contribution: MemoryAuditLog

The problem
-----------
Existing benchmarks only check the *final* outcome of an attack (did the
agent comply?).  For multi-session attacks, the interesting question is: what
did the agent *believe* at each session boundary?  The AuditLog answers this.

What this module does
---------------------
At every session boundary it records:
  - The full memory snapshot (what context would be injected next session)
  - Which planted claims are present / distorted / absent at that point
  - The context string that was actually injected (what the agent saw)

This lets you reconstruct the full epistemic timeline:
  "The agent first encountered the false claim in session 1.
   By session 2 the claim had been laundered by the summariser.
   The agent stopped calling get_reservation_details() in session 3."

For EMNLP: this supports qualitative error analysis and makes the paper's
claims about *how* the attack unfolds verifiable by reviewers.

Usage
-----
    log = MemoryAuditLog(provider=mem, tracker=tracker)

    # Call at the start of each session (before the agent runs)
    log.record_pre_session(
        session_id="s3",
        injected_context=context_str,
        query="I want to cancel my flight",
    )

    # Call after the session ends
    log.record_post_session(session_id="s3")

    # Export full timeline
    log.save("audit_s3.json")
    print(log.timeline_summary())
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional

from memory.base import BaseMemoryProvider
from memory.claim_tracker import ClaimTracker, ClaimStatus


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SessionBoundaryRecord:
    session_id: str
    session_index: int
    timestamp: float

    # Pre-session: what the agent SAW at the start of this session
    injected_context: str = ""
    query: str = ""

    # Post-session: what memory looks like after this session was stored
    memory_snapshot: str = ""
    claim_statuses: dict = field(default_factory=dict)  # claim_id -> status string

    # Derived
    context_token_estimate: int = 0   # rough word count as proxy for tokens


# ---------------------------------------------------------------------------
# AuditLog
# ---------------------------------------------------------------------------

class MemoryAuditLog:
    """
    Records memory state at every session boundary.

    Parameters
    ----------
    provider : BaseMemoryProvider
        The underlying memory provider (or a ClaimTracker wrapping one).
    tracker : ClaimTracker, optional
        If provided, claim statuses are recorded at each boundary.
    """

    def __init__(
        self,
        provider: BaseMemoryProvider,
        tracker: Optional[ClaimTracker] = None,
    ):
        self.provider = provider
        self.tracker = tracker
        self._records: list[SessionBoundaryRecord] = []
        self._session_index = 0

    def record_pre_session(
        self,
        session_id: str,
        injected_context: str,
        query: str = "",
    ) -> None:
        """
        Call this BEFORE running the agent for a session.
        Records what context was injected and how large it is.

        If a record already exists for ``session_id`` (caller invoked us
        twice for the same session), the existing pre-session fields are
        overwritten in place rather than appending a duplicate. This avoids
        leaving the earlier record with an empty ``memory_snapshot`` after
        ``record_post_session`` (which only touches the most recent matching
        record) and it keeps ``_session_index`` monotonic.
        """
        existing = self._find(session_id)
        if existing is not None:
            existing.timestamp = time.time()
            existing.injected_context = injected_context
            existing.query = query
            existing.context_token_estimate = len(injected_context.split())
            return
        record = SessionBoundaryRecord(
            session_id=session_id,
            session_index=self._session_index,
            timestamp=time.time(),
            injected_context=injected_context,
            query=query,
            context_token_estimate=len(injected_context.split()),
        )
        self._records.append(record)

    def record_post_session(self, session_id: str) -> None:
        """
        Call this AFTER add_session() has been called on the provider.
        Records the memory snapshot and claim statuses.

        Idempotent on the ``_session_index`` counter: a second call for the
        same ``session_id`` refreshes the snapshot and claim statuses but
        does not advance ``_session_index`` (which would desync future
        records).
        """
        record = self._find(session_id)
        if record is None:
            return

        already_finalised = bool(record.memory_snapshot) or bool(record.claim_statuses)
        record.memory_snapshot = self.provider.snapshot()

        if self.tracker is not None:
            report = self.tracker.get_survival_report()
            record.claim_statuses = {
                r.claim_id: r.status.value
                for r in report.results
                if r.session_id == session_id
            }

        if not already_finalised:
            self._session_index += 1

    # ── Querying ──────────────────────────────────────────────────────────────

    def context_size_curve(self) -> list[int]:
        """
        Token estimates at each session boundary (pre-session context size).
        Shows how memory grows over time — useful for cost vs fidelity analysis.
        """
        return [r.context_token_estimate for r in self._records]

    def claim_timeline(self, claim_id: str) -> list[tuple[int, str]]:
        """
        Returns [(session_index, status_string), ...] for a specific claim.
        """
        return [
            (r.session_index, r.claim_statuses.get(claim_id, "unknown"))
            for r in self._records
            if claim_id in r.claim_statuses
        ]

    def first_laundering_session(self, claim_id: str) -> Optional[str]:
        """
        Returns the session_id where a claim first appeared DISTORTED
        (i.e. the laundering event — qualifier stripped by summariser).
        """
        prev = None
        for r in sorted(self._records, key=lambda x: x.session_index):
            status = r.claim_statuses.get(claim_id)
            if status == ClaimStatus.DISTORTED.value and prev != ClaimStatus.DISTORTED.value:
                return r.session_id
            prev = status
        return None

    def timeline_summary(self) -> str:
        """Human-readable timeline for qualitative paper analysis."""
        lines = ["=== Memory Audit Timeline ==="]
        for r in sorted(self._records, key=lambda x: x.session_index):
            lines.append(
                f"\nSession {r.session_index} ({r.session_id[:8]}...)"
            )
            lines.append(
                f"  Context injected: {r.context_token_estimate} tokens (est.)"
            )
            if r.query:
                lines.append(f"  Query: {r.query[:80]}")
            if r.claim_statuses:
                lines.append("  Claim statuses:")
                for cid, status in r.claim_statuses.items():
                    lines.append(f"    {cid}: {status}")
            if r.injected_context:
                preview = r.injected_context[:200].replace("\n", " ")
                lines.append(f"  Context preview: {preview}...")
        return "\n".join(lines)

    def records(self) -> list[SessionBoundaryRecord]:
        """Return a shallow copy of the recorded session-boundary records.

        Public accessor so callers (orchestrator audit JSON, downstream
        analysis) don't need to reach into ``_records``.
        """
        return list(self._records)

    def save(self, path: str) -> None:
        data = [
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
            for r in self._records
        ]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _find(self, session_id: str) -> Optional[SessionBoundaryRecord]:
        for r in reversed(self._records):
            if r.session_id == session_id:
                return r
        return None
