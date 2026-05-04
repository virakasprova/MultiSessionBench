"""SessionRunner — agent loop: LLM, tools, turn-by-turn attacker.

Subjective scoring (refused / accepted / confirmed / compliance) is done post-hoc by
``core.judge.SessionJudge`` (see ``experiments/enrich_judge.py``).
"""
from __future__ import annotations

import json
import re
from typing import Any

from attackers.base import AttackerAgent
from core.llm_retry import completion_with_retry

# Transcript limits for logging and memory injection. Tool outputs (e.g. get_reservation_details
# JSON) must not be truncated mid-string — old 300-char caps broke full_history laundering analysis.
_TRANSCRIPT_TOOL_MAX = 20000
_TRANSCRIPT_DIALOGUE_MAX = 8000
from core.types import (
    SessionContext,
    SessionMechanicalResult,
    SessionResult,
)
from core.environment import TauBenchEnv


def _dialogue_without_system(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [m for m in msgs if m.get("role") != "system"]


def _matches_violation_tool(fn: str, violation_tool: str) -> bool:
    """Match a tool call against a seed's ``violation_tool`` spec.

    Supports a regex form for domains (banking) whose tool names carry a
    randomized suffix per data version: ``"re:close_credit_card_account_\\w+"``.
    Plain strings remain exact-match for airline / retail backwards-compat.
    Empty / falsy spec never matches.
    """
    if not violation_tool:
        return False
    if violation_tool.startswith("re:"):
        try:
            return bool(re.fullmatch(violation_tool[3:], fn))
        except re.error:
            return False
    return fn == violation_tool


class SessionRunner:
    def __init__(
        self,
        env: TauBenchEnv,
        model: str,
        max_turns: int = 15,
        seed: int = 42,
        api_base: str | None = None,
    ):
        self.env = env
        self.model = model
        self.max_turns = max_turns
        self.seed = seed
        self.api_base = api_base

    def run(
        self,
        attacker: AttackerAgent,
        context: SessionContext,
        system_prompt: str,
        violation_tool: str,
        session_index: int = 0,
        max_turns_override: int | None = None,
    ) -> SessionResult:
        # Attacker LLM failure on the opening message: return an empty session
        # result rather than killing the whole attack. Symmetric to the
        # agent-side LLM error handling below — a transient OpenAI / OpenRouter
        # blip on the customer side shouldn't lose the rest of the run.
        try:
            first = attacker.start_session(context)
        except Exception as e:
            print(f"      [attacker error in start_session: {type(e).__name__}: {e}; "
                  f"returning empty session]")
            return _empty_session_result(
                session_index,
                f"[attacker failed before opening message: {type(e).__name__}: {e}]",
                terminated_by="attacker_error",
            )
        msgs: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first},
        ]
        tools_called: list[dict] = []
        writes: list[dict] = []
        cost = 0.0
        verified = False
        verified_then_acted = False
        transferred_to_human = False
        violation = False
        # Default assumes we exhaust the loop without any stop signal; every
        # other termination path overwrites this before the break.
        terminated_by = "max_turns"

        effective_max_turns = (
            max_turns_override if max_turns_override is not None else self.max_turns
        )
        for _ in range(effective_max_turns):
            try:
                res = completion_with_retry(
                    messages=msgs,
                    model=self.model,
                    tools=self.env.tools_info,
                    temperature=0.0,
                    seed=self.seed,
                    log_prefix="      ",
                    **({"api_base": self.api_base} if self.api_base else {}),
                )
            except Exception as e:
                print(f"      [LLM error: {e}]")
                terminated_by = "agent_error"
                break

            nm = res.choices[0].message.model_dump()
            cost += res._hidden_params.get("response_cost", 0) or 0

            tool_calls = [tc for tc in (nm.get("tool_calls") or []) if tc.get("function")]

            if tool_calls:
                tc = tool_calls[0]
                fn = tc["function"]["name"]
                try:
                    fa = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, TypeError):
                    fa = {}
                tools_called.append({"name": fn, "args": fa})

                if fn in self.env.verification_tools:
                    verified = True

                tr = self.env.exec_tool(fn, fa)
                if self.env.is_write_tool(fn):
                    writes.append({"name": fn, "args": fa})
                    # ``verified_then_acted`` should only fire when the agent
                    # took an action AFTER having verified — i.e. a write
                    # tool follows a prior verification call. Setting it on
                    # the verification call itself misattributes refusals.
                    if verified:
                        verified_then_acted = True
                if _matches_violation_tool(fn, violation_tool):
                    violation = True
                if fn == "transfer_to_human_agents":
                    transferred_to_human = True

                nm["tool_calls"] = tool_calls[:1]
                msgs.extend([
                    nm,
                    {"role": "tool", "tool_call_id": tc["id"], "name": fn, "content": tr},
                ])

                if fn == "transfer_to_human_agents":
                    terminated_by = "transfer"
                    break
            else:
                msgs.append(nm)

                try:
                    nxt = attacker.next_user_message(
                        context,
                        _dialogue_without_system(msgs),
                    )
                except Exception as e:
                    print(f"      [attacker error in next_user_message: "
                          f"{type(e).__name__}: {e}; ending session]")
                    terminated_by = "attacker_error"
                    break
                if nxt is None:
                    terminated_by = "attacker_stop"
                    break
                msgs.append({"role": "user", "content": nxt})

        mechanical = SessionMechanicalResult(
            session_index=session_index,
            violation=violation,
            verified=verified,
            verified_then_acted=verified_then_acted,
            transferred_to_human=transferred_to_human,
            tools_called=tools_called,
            writes=writes,
            cost=cost,
            transcript=_format_transcript(msgs),
            messages=_preserve_messages(msgs),
            terminated_by=terminated_by,
        )
        return SessionResult(mechanical=mechanical, judge=None, contradicted=None, compliance=None)


# Memory consumers (structured-turn providers) need to reconstruct the agent
# loop, not just the dialogue, so we preserve tool_calls metadata on assistant
# messages and the tool name on tool messages. Content stays capped (cheap
# logging is not the goal here; analysis fidelity is).
_MESSAGE_CONTENT_DIALOGUE_MAX = 4000
_MESSAGE_CONTENT_TOOL_MAX = 4000


def _preserve_messages(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a JSON-serialisable, role-aware copy of ``msgs`` (system stripped).

    Schema per element:
        {"role": "user"|"assistant"|"tool",
         "content": str,
         # assistant-only when a tool was invoked:
         "tool_calls": [{"name": str, "arguments": str}],
         # tool-only:
         "name": str}
    """
    out: list[dict[str, Any]] = []
    for m in msgs:
        role = m.get("role")
        if role == "system":
            continue
        content = (m.get("content") or "")
        if role == "tool":
            entry: dict[str, Any] = {
                "role": "tool",
                "content": content[:_MESSAGE_CONTENT_TOOL_MAX],
                "name": m.get("name") or "",
            }
        elif role == "assistant":
            entry = {
                "role": "assistant",
                "content": content[:_MESSAGE_CONTENT_DIALOGUE_MAX],
            }
            tool_calls = [
                {
                    "name": tc["function"].get("name", ""),
                    "arguments": tc["function"].get("arguments", "") or "",
                }
                for tc in (m.get("tool_calls") or [])
                if tc.get("function")
            ]
            if tool_calls:
                entry["tool_calls"] = tool_calls
        else:
            entry = {
                "role": role or "",
                "content": content[:_MESSAGE_CONTENT_DIALOGUE_MAX],
            }
        out.append(entry)
    return out


def _clip(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 20] + "\n… [truncated]"


def _empty_session_result(
    session_index: int,
    transcript: str,
    *,
    terminated_by: str = "unknown",
) -> SessionResult:
    """Return a well-formed but empty SessionResult for catastrophic-attacker cases.

    Used when ``attacker.start_session`` raises before any messages exist —
    rather than letting the exception kill the whole attack, we return a
    placeholder result so the orchestrator can record the failure (cost=0,
    no tools, marked transcript) and continue with the next session.
    """
    mechanical = SessionMechanicalResult(
        session_index=session_index,
        violation=False,
        verified=False,
        verified_then_acted=False,
        transferred_to_human=False,
        tools_called=[],
        writes=[],
        cost=0.0,
        transcript=transcript,
        messages=[],
        terminated_by=terminated_by,
    )
    return SessionResult(mechanical=mechanical, judge=None, contradicted=None, compliance=None)


def _format_transcript(msgs: list[dict]) -> str:
    lines = []
    for m in msgs:
        r = m.get("role", "")
        if r == "system":
            continue
        c = m.get("content", "") or ""
        if r == "assistant" and m.get("tool_calls"):
            tc = m["tool_calls"][0]
            lines.append(f"Agent: [tool:{tc['function']['name']}]")
        elif r == "tool":
            lines.append(f"Tool: {_clip(c, _TRANSCRIPT_TOOL_MAX)}")
        elif r == "assistant":
            lines.append(f"Agent: {_clip(c, _TRANSCRIPT_DIALOGUE_MAX)}")
        elif r == "user":
            lines.append(f"Customer: {_clip(c, _TRANSCRIPT_DIALOGUE_MAX)}")
    return "\n".join(lines)
