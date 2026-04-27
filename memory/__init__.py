"""
memory/__init__.py
==================
Public API for the memory module.

Single, structured-turn API: ``Turn``, ``Session``, ``BaseMemoryProvider``,
the provider zoo (``RollingSummaryProvider``, ``RAGMemoryProvider``, …), and
the four novel measurement layers (``ClaimTracker``, ``MemoryAuditLog``,
``ConfidenceDriftTracker``, ``CrossUserMemoryStore``).

Quick start:

    from memory import make_memory_provider, Turn, ClaimTracker, PlantedClaim
    from memory import MemoryAuditLog, ConfidenceDriftTracker, CrossUserMemoryStore

    mem = make_memory_provider("summary_rolling")
    mem.add_session(turns, session_id="s1", user_type="adversarial")
    print(mem.get_context(query="cancel my flight"))

For a LiteLLM-backed summariser, import explicitly to keep ``import memory``
free of the ``litellm`` dependency::

    from memory.litellm_summariser import make_litellm_summariser
    mem = make_memory_provider("summary_rolling",
                               summariser=make_litellm_summariser(model))
"""
from __future__ import annotations

# ── Structured-turn core ──────────────────────────────────────────────────
from memory.base import BaseMemoryProvider, Session, Turn

# ── Providers ─────────────────────────────────────────────────────────────
from memory.providers.full_context import FullContextProvider, NoMemoryProvider
from memory.providers.summary import (
    CumulativeSummaryProvider,
    RollingSummaryProvider,
    SummaryLogEntry,
)
from memory.providers.rag import (
    EmbeddingBackend,
    MemoryChunk,
    OpenAIEmbeddingBackend,
    RAGMemoryProvider,
    SentenceTransformerBackend,
)
from memory.providers.hybrid import (
    RecentFullOldSummaryProvider,
    SummaryPlusRAGProvider,
)

# ── Novel measurement layers ──────────────────────────────────────────────
from memory.claim_tracker import (
    ClaimStatus,
    ClaimSurvivalReport,
    ClaimSurvivalResult,
    ClaimTracker,
    PlantedClaim,
)
from memory.audit_log import MemoryAuditLog, SessionBoundaryRecord
from memory.confidence_drift import (
    ConfidenceDriftTracker,
    DriftRecord,
    DriftReport,
)
from memory.cross_user_store import CrossUserMemoryStore


__all__ = [
    # Structured-turn core
    "Turn", "Session", "BaseMemoryProvider",
    # Providers
    "NoMemoryProvider", "FullContextProvider",
    "RollingSummaryProvider", "CumulativeSummaryProvider", "SummaryLogEntry",
    "RAGMemoryProvider", "SentenceTransformerBackend", "OpenAIEmbeddingBackend",
    "EmbeddingBackend", "MemoryChunk",
    "RecentFullOldSummaryProvider", "SummaryPlusRAGProvider",
    # Novel measurement layers
    "ClaimTracker", "PlantedClaim", "ClaimStatus",
    "ClaimSurvivalResult", "ClaimSurvivalReport",
    "MemoryAuditLog", "SessionBoundaryRecord",
    "ConfidenceDriftTracker", "DriftRecord", "DriftReport",
    "CrossUserMemoryStore",
    # Factory
    "make_memory_provider",
]


def make_memory_provider(config: str, **kwargs) -> BaseMemoryProvider:
    """
    Factory returning a configured structured-turn memory provider.

    Configs
    -------
    no_memory               stateless control
    full_context            all turns verbatim
    full_context_recent2    sliding window, last 2 sessions
    summary_rolling         per-session summaries concatenated
    summary_cumulative      single growing paragraph
    rag_baseline            RAG, no attribution
    rag_attribution         RAG + claim-attribution defense
    rag_shared              RAG with shared index (cross-user)
    hybrid_recent_summary   recent verbatim + older summaries
    hybrid_summary_rag      rolling summaries + RAG chunks
    """
    registry = {
        "no_memory":             lambda kw: NoMemoryProvider(**kw),
        "full_context":          lambda kw: FullContextProvider(**kw),
        "full_context_recent2":  lambda kw: FullContextProvider(max_sessions=2, **kw),
        "summary_rolling":       lambda kw: RollingSummaryProvider(**kw),
        "summary_cumulative":    lambda kw: CumulativeSummaryProvider(**kw),
        "rag_baseline":          lambda kw: RAGMemoryProvider(**kw),
        "rag_attribution":       lambda kw: RAGMemoryProvider(use_attribution=True, **kw),
        "rag_shared":            lambda kw: RAGMemoryProvider(user_scoped=False, **kw),
        "hybrid_recent_summary": lambda kw: RecentFullOldSummaryProvider(**kw),
        "hybrid_summary_rag":    lambda kw: SummaryPlusRAGProvider(**kw),
    }
    if config not in registry:
        raise ValueError(f"Unknown config {config!r}. Available: {sorted(registry)}")
    return registry[config](kwargs)
