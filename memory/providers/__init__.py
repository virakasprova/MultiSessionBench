"""memory.providers — concrete BaseMemoryProvider implementations."""
from __future__ import annotations

from memory.providers.full_context import FullContextProvider, NoMemoryProvider
from memory.providers.hybrid import (
    RecentFullOldSummaryProvider,
    SummaryPlusRAGProvider,
)
from memory.providers.rag import (
    EmbeddingBackend,
    MemoryChunk,
    OpenAIEmbeddingBackend,
    RAGMemoryProvider,
    SentenceTransformerBackend,
)
from memory.providers.summary import (
    CumulativeSummaryProvider,
    RollingSummaryProvider,
    SummaryLogEntry,
)

__all__ = [
    "NoMemoryProvider",
    "FullContextProvider",
    "RollingSummaryProvider",
    "CumulativeSummaryProvider",
    "SummaryLogEntry",
    "RAGMemoryProvider",
    "EmbeddingBackend",
    "SentenceTransformerBackend",
    "OpenAIEmbeddingBackend",
    "MemoryChunk",
    "RecentFullOldSummaryProvider",
    "SummaryPlusRAGProvider",
]
