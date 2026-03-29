"""Summary — LLM-compressed notes from prior sessions (LiteLLM)."""
from __future__ import annotations

from litellm import completion

from core.types import SessionResult
from memory.base import MemoryProvider


class SummaryMemory(MemoryProvider):
    def __init__(self, model: str, max_tokens: int = 200) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._bullets: list[str] = []

    @property
    def mode_name(self) -> str:
        return "summary"

    def reset(self) -> None:
        self._bullets.clear()

    def update(self, result: SessionResult) -> None:
        transcript = result.mechanical.transcript
        try:
            r = completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Summarize this customer service conversation in 2-3 sentences. "
                            "Include customer ID, claims made, key facts."
                        ),
                    },
                    {"role": "user", "content": transcript},
                ],
                model=self.model,
                temperature=0.0,
                max_tokens=self.max_tokens,
            )
            text = (r.choices[0].message.content or "").strip()
        except Exception:
            text = "[summary failed]"
        self._bullets.append(text)

    def get_prompt_injection(self) -> str | None:
        if not self._bullets:
            return None
        return "## Notes from Previous Interactions\n" + "\n".join(f"- {m}" for m in self._bullets)
