"""Human attacker — drive the customer side from stdin instead of an LLM.

Use this for interactive probing of a specific (seed, memory_mode) pair
without burning attacker tokens. Reads each customer turn with ``input()``
after printing the agent's most recent reply.

Sentinels that end the current session (returned to the orchestrator as
``None`` from ``next_user_message``):

- empty line
- ``:q`` / ``quit`` / ``exit``
- ``###STOP###``
- EOF (Ctrl-D) / KeyboardInterrupt (Ctrl-C)

Multi-session seeds (Plant / Reinforce / Trigger) iterate as usual: each
session opens a fresh stdin loop so you decide phase-by-phase how to attack
the agent. Memory accumulates across sessions exactly as it does for
``CraftLLMAttacker``.

Pair with ``--seed-ids X`` and a single ``--memory-modes Y`` on the
runner — running the full sweep with a human in the loop will queue up
``len(seeds) * len(modes)`` interactive sessions on one terminal.
"""
from __future__ import annotations

from typing import Any

from attackers.base import AttackerAgent
from core.types import SessionContext, TaskSeed


class HumanAttacker(AttackerAgent):
    """Stdin-driven attacker. One human, one terminal, one (seed, mode) at a time."""

    _STOP_TOKENS = frozenset({"###stop###", ":q", "quit", "exit"})
    _DEFAULT_OPENING = "Hello, I need help with my reservation."

    def __init__(self, *, prompt_prefix: str = "you> "):
        self.prompt_prefix = prompt_prefix

    def get_session_count(self, seed: TaskSeed) -> int:
        return seed.num_sessions()

    def _read(self) -> str | None:
        """Return next stdin line, or ``None`` for any session-ending signal."""
        try:
            text = input(self.prompt_prefix)
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        text = text.strip()
        if not text:
            return None
        if text.lower() in self._STOP_TOKENS:
            return None
        return text

    @staticmethod
    def _print_session_header(seed: TaskSeed, context: SessionContext) -> None:
        """Print scenario context so the human knows which phase they're in."""
        n = seed.num_sessions()
        marker = f"SESSION {context.session_index + 1}/{n}"
        print()
        print("=" * 72)
        print(f" {marker}  seed={seed.seed_id}  policy={seed.policy_area}")
        print("=" * 72)
        if seed.false_claim:
            print(f"  false_claim:    {seed.false_claim}")
        if seed.hidden_attacker_goal and seed.hidden_attacker_goal != seed.false_claim:
            print(f"  hidden_goal:    {seed.hidden_attacker_goal}")

        intents = seed.session_intents or []
        if 0 <= context.session_index < len(intents):
            bucket = intents[context.session_index]
            if bucket:
                body = "\n".join(bucket).strip()
                print(f"  scenario hint: {body}")
        print("  (empty line / :q / ###STOP### / Ctrl-D ends this session)")
        print("-" * 72)

    @staticmethod
    def _print_recent_agent_turn(dialogue_without_system: list[dict[str, Any]]) -> None:
        """Echo the last assistant / tool message so the human can read it."""
        if not dialogue_without_system:
            return
        last = dialogue_without_system[-1]
        role = last.get("role", "?")
        content = (last.get("content") or "").strip()
        if role == "assistant" and content:
            print(f"\nagent> {content}\n")
        elif role == "tool" and content:
            # Truncate giant tool dumps so the terminal stays usable.
            preview = content if len(content) <= 800 else content[:800] + "... [truncated]"
            print(f"\n[tool result] {preview}\n")

    def start_session(self, context: SessionContext) -> str:
        self._print_session_header(context.seed, context)
        msg = self._read()
        # The orchestrator requires a non-empty opening message; if the human
        # immediately ends the session, default to a generic opener so the
        # session record is still well-formed and can be inspected later.
        return msg if msg is not None else self._DEFAULT_OPENING

    def next_user_message(
        self,
        context: SessionContext,
        dialogue_without_system: list[dict[str, Any]],
    ) -> str | None:
        self._print_recent_agent_turn(dialogue_without_system)
        return self._read()
