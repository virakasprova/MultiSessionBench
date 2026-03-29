"""NoMemory — reference ``MemoryProvider``: no cross-session prompt injection."""
from __future__ import annotations

from core.types import SessionResult
from memory.base import MemoryProvider


class NoMemory(MemoryProvider):

    @property
    def mode_name(self) -> str:
        return "no_memory"

    def update(self, result: SessionResult) -> None:
        pass

    def get_prompt_injection(self) -> str | None:
        return None

    def reset(self) -> None:
        pass
