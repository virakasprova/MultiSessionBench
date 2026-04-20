"""
memory/providers/rag.py
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

from memory2.base import BaseMemoryProvider, Session, Turn


@dataclass
class MemoryChunk:
    chunk_id: str
    session_id: str
    text: str
    speaker: str
    user_type: str
    user_id: str
    turn_index: int
    is_planted_claim: bool = False
    claim_id: Optional[str] = None
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


class EmbeddingBackend:
    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerBackend(EmbeddingBackend):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("pip install sentence-transformers")
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(texts, normalize_embeddings=True)


class OpenAIEmbeddingBackend(EmbeddingBackend):
    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def encode(self, texts):
        response = self.client.embeddings.create(input=texts, model=self.model)
        arr = np.array([item.embedding for item in response.data], dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms == 0, 1, norms)


class RAGMemoryProvider(BaseMemoryProvider):
    """
    Embedding-based retrieval. Supports user_id namespacing for
    cross-user contamination experiments.

    Parameters
    ----------
    user_scoped : bool
        If True, retrieval is restricted to chunks from the same user_id.
        If False, all chunks are searched (simulates shared RAG index).
    """

    def __init__(self, embedding_backend=None, top_k=5,
                 chunk_by: Literal["turn", "session"] = "turn",
                 use_attribution=False, similarity_threshold=0.0,
                 user_scoped=True):
        super().__init__(use_attribution=use_attribution)
        self.backend = embedding_backend or SentenceTransformerBackend()
        self.top_k = top_k
        self.chunk_by = chunk_by
        self.similarity_threshold = similarity_threshold
        self.user_scoped = user_scoped
        self._chunks: list[MemoryChunk] = []

    def add_session(self, turns, session_id=None,
                    user_type="unknown", user_id="default"):
        session_id = session_id or str(uuid.uuid4())
        for i, t in enumerate(turns):
            t.session_id = session_id; t.turn_index = i; t.user_type = user_type
        self._sessions.append(Session(session_id, turns, user_type, user_id))

        new_chunks = (self._chunk_by_turn(turns, session_id, user_type, user_id)
                      if self.chunk_by == "turn"
                      else self._chunk_by_session(turns, session_id, user_type, user_id))
        if new_chunks:
            embeddings = self.backend.encode([c.text for c in new_chunks])
            for chunk, emb in zip(new_chunks, embeddings):
                chunk.embedding = emb
        self._chunks.extend(new_chunks)
        return session_id

    def _chunk_by_turn(self, turns, session_id, user_type, user_id):
        return [
            MemoryChunk(
                chunk_id=str(uuid.uuid4()),
                session_id=session_id, text=t.content,
                speaker=t.role, user_type=user_type, user_id=user_id,
                turn_index=t.turn_index,
                is_planted_claim=t.is_planted_claim, claim_id=t.claim_id,
            )
            for t in turns
        ]

    def _chunk_by_session(self, turns, session_id, user_type, user_id):
        text = "\n".join(f"[{t.role}]: {t.content}" for t in turns)
        has_claim = any(t.is_planted_claim for t in turns)
        claim_ids = list({t.claim_id for t in turns if t.claim_id})
        return [MemoryChunk(
            chunk_id=str(uuid.uuid4()),
            session_id=session_id, text=text,
            speaker="mixed", user_type=user_type, user_id=user_id,
            turn_index=0, is_planted_claim=has_claim,
            claim_id=claim_ids[0] if claim_ids else None,
        )]

    def retrieve(self, query, exclude_session_id=None, user_id="default"):
        candidates = [
            c for c in self._chunks
            if c.embedding is not None
            and (exclude_session_id is None or c.session_id != exclude_session_id)
            and (not self.user_scoped or c.user_id == user_id)
        ]
        if not candidates:
            return []
        query_emb = self.backend.encode([query])[0]
        scores = np.stack([c.embedding for c in candidates]) @ query_emb
        results = []
        for idx in np.argsort(scores)[::-1]:
            if len(results) >= self.top_k:
                break
            if scores[idx] >= self.similarity_threshold:
                results.append(candidates[idx])
        return results

    def get_context(self, query="", exclude_session_id=None, user_id="default"):
        if not query:
            return ""
        chunks = self.retrieve(query, exclude_session_id=exclude_session_id, user_id=user_id)
        if not chunks:
            return ""
        lines = ["Relevant information from prior sessions:"]
        for chunk in chunks:
            prefix = self._attribution_prefix(chunk.speaker, chunk.user_type)
            lines.append(f"  - {prefix}{chunk.text}")
        return "\n".join(lines)

    def get_memory_contents(self):
        return [
            {
                "chunk_id": c.chunk_id, "session_id": c.session_id,
                "speaker": c.speaker, "user_type": c.user_type,
                "user_id": c.user_id, "turn_index": c.turn_index,
                "text": c.text, "is_planted_claim": c.is_planted_claim,
                "claim_id": c.claim_id,
            }
            for c in self._chunks
        ]

    def clear(self):
        super().clear()
        self._chunks.clear()
