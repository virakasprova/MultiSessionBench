"""AttackerAgent ABC — same pattern as ``memory.base.BaseMemoryProvider``."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.types import TaskSeed, SessionContext


class AttackerAgent(ABC):

    @abstractmethod
    def start_session(self, context: SessionContext) -> str:
        """First customer message before any agent reply."""
        ...

    @abstractmethod
    def next_user_message(
        self,
        context: SessionContext,
        dialogue_without_system: list[dict[str, Any]],
    ) -> str | None:
        """After a non-tool assistant turn, return the next customer message or ``None`` to stop.

        ``dialogue_without_system`` is OpenAI-style messages for this session only
        (``user`` / ``assistant`` / ``tool``), excluding ``system``, chronological order.
        """
        ...

    @abstractmethod
    def get_session_count(self, seed: TaskSeed) -> int:
        """How many sessions this attacker needs for the given seed."""
        ...
