"""LLM-backed attacker — Beyza implements ``start_session`` / ``next_user_message``.

Baseline vs multi-session: same ``LLMAttacker``. Orchestrator sets ``is_baseline`` for the
single-session control.

Scripted fallback: drains ``session_intents`` (or ``baseline_messages``) turn-by-turn so the
runner still works without an attacker LLM.

Contract: ``attackers.base.AttackerAgent``.
"""
from __future__ import annotations

from typing import Any

from attackers.base import AttackerAgent
from core.types import TaskSeed, SessionContext

DEFAULT_ATTACKER_SYSTEM = ""


def _scripted_turns(context: SessionContext, is_baseline: bool) -> list[str]:
    seed = context.seed
    if is_baseline:
        if seed.baseline_messages:
            return list(seed.baseline_messages)
        return list(seed.session_intents[-1])
    return list(seed.session_intents[context.session_index])


class LLMAttacker(AttackerAgent):
    """Wire attacker LLM in ``start_session`` / ``next_user_message``."""

    def __init__(
        self,
        model: str,
        provider: str,
        system_prompt: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 1024,
        seed: int | None = None,
    ):
        self.model = model
        self.provider = provider
        self.system_prompt = system_prompt if system_prompt is not None else DEFAULT_ATTACKER_SYSTEM
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self._remaining_script: list[str] = []

    def get_session_count(self, seed: TaskSeed, is_baseline: bool = False) -> int:
        return 1 if is_baseline else seed.num_sessions()

    def start_session(self, context: SessionContext, is_baseline: bool = False) -> str:
        turns = _scripted_turns(context, is_baseline)
        if not turns:
            return ""
        self._remaining_script = turns[1:]
        return turns[0]

    def next_user_message(
        self,
        context: SessionContext,
        dialogue_without_system: list[dict[str, Any]],
        is_baseline: bool = False,
    ) -> str | None:
        # Beyza: use dialogue_without_system + context to call an LLM; return next line or None.
        if not self._remaining_script:
            return None
        nxt = self._remaining_script.pop(0)
        return nxt
