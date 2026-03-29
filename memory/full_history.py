"""FullHistory — append prior session transcripts to the system prompt."""
from __future__ import annotations

from core.types import SessionResult
from memory.base import MemoryProvider


class FullHistoryMemory(MemoryProvider):
    def __init__(self) -> None:
        self._transcripts: list[str] = []

    @property
    def mode_name(self) -> str:
        return "full_history"

    def reset(self) -> None:
        self._transcripts.clear()

    def update(self, result: SessionResult) -> None:
        self._transcripts.append(result.mechanical.transcript[:2000])

    def get_prompt_injection(self) -> str | None:
        if not self._transcripts:
            return None
        return "## Previous Sessions\n" + "\n---\n".join(self._transcripts)
