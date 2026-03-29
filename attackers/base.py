"""AttackerAgent ABC — same pattern as ``memory.base.MemoryProvider``."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.types import TaskSeed, SessionContext


class AttackerAgent(ABC):

    @abstractmethod
    def start_session(self, context: SessionContext, is_baseline: bool = False) -> str:
        """First customer message before any agent reply.

        ``is_baseline`` is True for the single-session control (same class, one session).
        """
        ...

    @abstractmethod
    def next_user_message(
        self,
        context: SessionContext,
        dialogue_without_system: list[dict[str, Any]],
        is_baseline: bool = False,
    ) -> str | None:
        """After a non-tool assistant turn, return the next customer message or ``None`` to stop.

        ``dialogue_without_system`` is OpenAI-style messages for this session only
        (``user`` / ``assistant`` / ``tool``), excluding ``system``, chronological order.
        """
        ...

    @abstractmethod
    def get_session_count(self, seed: TaskSeed, is_baseline: bool = False) -> int:
        """How many sessions this attacker needs for the given seed (1 when ``is_baseline``)."""
        ...
