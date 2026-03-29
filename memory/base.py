"""MemoryProvider ABC

Each provider controls what cross-session context the agent sees.
The orchestrator calls update() after each session and get_prompt_injection()
before each session to build the system prompt.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from core.types import SessionResult


class MemoryProvider(ABC):

    @property
    @abstractmethod
    def mode_name(self) -> str:
        """Short identifier, e.g. 'no_memory', 'full_history', 'summary'."""
        ...

    @abstractmethod
    def update(self, result: SessionResult) -> None:
        """Ingest a completed session's result into memory state."""
        ...

    @abstractmethod
    def get_prompt_injection(self) -> str | None:
        """Return text to append to the system prompt, or None if nothing to add."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all memory state. Called between attack bundles."""
        ...
