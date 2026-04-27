"""LiteLLM-backed summariser factories for the structured-turn memory providers.

This module is intentionally NOT imported from ``memory/__init__.py``: it depends
on ``litellm``, and we want ``import memory`` to stay lightweight for code paths
that don't actually run summary memory. Import explicitly:

    from memory.litellm_summariser import (
        make_litellm_summariser,
        make_litellm_cumulative_summariser,
    )
    from memory import make_memory_provider

    rolling = make_memory_provider(
        "summary_rolling",
        summariser=make_litellm_summariser("openrouter/google/gemini-2.5-flash"),
    )
    cumulative = make_memory_provider(
        "summary_cumulative",
        summariser=make_litellm_cumulative_summariser("openrouter/google/gemini-2.5-flash"),
    )
"""
from __future__ import annotations

from typing import Callable


_DEFAULT_SYSTEM_PROMPT = (
    "Summarize this customer service session in 2-3 sentences. "
    "Include the customer ID, reservation IDs, what was discussed, "
    "tool calls and tool results that established or refuted any "
    "factual claims, and the outcome. Be concise and to the point."
)

_DEFAULT_CUMULATIVE_SYSTEM_PROMPT = (
    "You maintain a single running summary of a customer's history across "
    "multiple sessions. Given the existing running summary and the transcript "
    "of the newest session, return an updated running summary that integrates "
    "the new session's facts. Keep customer/reservation IDs, retain tool-call "
    "outcomes that established or refuted factual claims, and stay concise "
    "(at most 5-6 sentences total). Prefer rewriting over appending."
)


def make_litellm_summariser(
    model: str,
    max_tokens: int = 600,
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.0,
) -> Callable[[str], str]:
    """Return a ``Callable[[str], str]`` suitable for ``RollingSummaryProvider``
    and ``RecentFullOldSummaryProvider``.

    The returned closure accepts the structured ``Session.as_text()`` rendering
    (one ``[role]: content`` line per turn, including ``[tool]: ...`` lines
    when the orchestrator preserved tool calls) and returns a short textual
    summary. Failures fall back to ``"[summary failed]"`` rather than raising,
    matching the behaviour of the original ``SummaryMemory`` adapter.
    """
    # Imported lazily so importing this module doesn't require litellm at
    # module load time of any caller that just happens to ``from memory import ...``.
    from litellm import completion

    def _summarise(session_text: str) -> str:
        try:
            response = completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": session_text},
                ],
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception:
            return "[summary failed]"

    return _summarise


def make_litellm_cumulative_summariser(
    model: str,
    max_tokens: int = 800,
    system_prompt: str = _DEFAULT_CUMULATIVE_SYSTEM_PROMPT,
    temperature: float = 0.0,
) -> Callable[[str, str], str]:
    """Return a ``Callable[[str, str], str]`` suitable for ``CumulativeSummaryProvider``.

    The provider passes ``(existing_running_summary, new_session_text)`` and
    expects the updated running summary. On the first session ``existing`` is
    the empty string. Failures preserve the previous summary rather than
    raising, so a transient API hiccup never erases history.
    """
    from litellm import completion

    def _summarise(existing: str, new_text: str) -> str:
        prior_block = existing.strip() or "(no prior summary — this is the first session)"
        user_msg = (
            f"Existing running summary:\n{prior_block}\n\n"
            f"Newest session transcript:\n{new_text}\n\n"
            "Return the updated running summary."
        )
        try:
            response = completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            updated = (response.choices[0].message.content or "").strip()
            return updated or existing
        except Exception:
            return existing

    return _summarise
