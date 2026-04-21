"""FullHistory — append prior session transcripts to the system prompt."""
from __future__ import annotations

from core.types import SessionResult
from memory.base import MemoryProvider

# Per-session cap for injected history (must fit typical sessions with full tool JSON after
# session_runner transcript formatting — see ``_TRANSCRIPT_TOOL_MAX`` there).
_FULL_HISTORY_SESSION_CHAR_CAP = 48000


class FullHistoryMemory(MemoryProvider):
    def __init__(self, session_char_cap: int = _FULL_HISTORY_SESSION_CHAR_CAP) -> None:
        self._transcripts: list[str] = []
        self._session_char_cap = session_char_cap

    @property
    def mode_name(self) -> str:
        return "full_history"

    def reset(self) -> None:
        self._transcripts.clear()

    def update(self, result: SessionResult) -> None:
        t = result.mechanical.transcript
        if len(t) > self._session_char_cap:
            t = t[: self._session_char_cap - 30] + "\n… [session transcript truncated]"
        self._transcripts.append(t)

    def get_prompt_injection(self) -> str | None:
        if not self._transcripts:
            return None
        return "## Previous Sessions\n" + "\n---\n".join(self._transcripts)
