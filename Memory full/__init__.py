"""
memory/__init__.py
==================
Public API.  Import everything from here.

    from memory import make_memory_provider, Turn, ClaimTracker, PlantedClaim
    from memory import MemoryAuditLog, ConfidenceDriftTracker, CrossUserMemoryStore
"""

from memory2.base import Turn, Session, BaseMemoryProvider
from memory2.providers.full_context import NoMemoryProvider, FullContextProvider
from memory2.providers.summary import (
    RollingSummaryProvider, CumulativeSummaryProvider, SummaryLogEntry
)
from memory2.providers.rag import (
    RAGMemoryProvider, SentenceTransformerBackend, OpenAIEmbeddingBackend, MemoryChunk
)
from memory2.providers.hybrid import RecentFullOldSummaryProvider, SummaryPlusRAGProvider
from memory2.claim_tracker import (
    ClaimTracker, PlantedClaim, ClaimStatus,
    ClaimSurvivalResult, ClaimSurvivalReport
)
from memory2.audit_log import MemoryAuditLog, SessionBoundaryRecord
from memory2.confidence_drift import ConfidenceDriftTracker, DriftRecord, DriftReport
from memory2.cross_user_store import CrossUserMemoryStore

__all__ = [
    # Core
    "Turn", "Session", "BaseMemoryProvider",
    # Providers
    "NoMemoryProvider", "FullContextProvider",
    "RollingSummaryProvider", "CumulativeSummaryProvider", "SummaryLogEntry",
    "RAGMemoryProvider", "SentenceTransformerBackend", "OpenAIEmbeddingBackend", "MemoryChunk",
    "RecentFullOldSummaryProvider", "SummaryPlusRAGProvider",
    # Novel contributions
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
    Factory returning a configured memory provider.

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
