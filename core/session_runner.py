"""SessionRunner — agent loop: LLM, tools, turn-by-turn attacker.

Subjective scoring (refused / accepted / confirmed / compliance) is done post-hoc by
``core.judge.SessionJudge`` (see ``experiments/enrich_judge.py``).
"""
from __future__ import annotations

import json
from typing import Any

from litellm import completion

from attackers.base import AttackerAgent
from core.types import (
    SessionContext,
    SessionMechanicalResult,
    SessionResult,
)
from core.environment import TauBenchEnv


def _dialogue_without_system(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [m for m in msgs if m.get("role") != "system"]


class SessionRunner:
    def __init__(
        self,
        env: TauBenchEnv,
        model: str,
        max_turns: int = 15,
        seed: int = 42,
    ):
        self.env = env
        self.model = model
        self.max_turns = max_turns
        self.seed = seed

    def run(
        self,
        attacker: AttackerAgent,
        context: SessionContext,
        system_prompt: str,
        violation_tool: str,
        session_index: int = 0,
        is_baseline: bool = False,
    ) -> SessionResult:
        first = attacker.start_session(context, is_baseline=is_baseline)
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

        for _ in range(self.max_turns):
            try:
                res = completion(
                    messages=msgs,
                    model=self.model,
                    tools=self.env.tools_info,
                    temperature=0.0,
                    seed=self.seed,
                )
            except Exception as e:
                print(f"      [LLM error: {e}]")
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
                    if not violation:
                        verified_then_acted = True

                tr = self.env.exec_tool(fn, fa)
                if self.env.is_write_tool(fn):
                    writes.append({"name": fn, "args": fa})
                if fn == violation_tool:
                    violation = True
                if fn == "transfer_to_human_agents":
                    transferred_to_human = True

                nm["tool_calls"] = tool_calls[:1]
                msgs.extend([
                    nm,
                    {"role": "tool", "tool_call_id": tc["id"], "name": fn, "content": tr},
                ])

                if fn == "transfer_to_human_agents":
                    break
            else:
                msgs.append(nm)

                nxt = attacker.next_user_message(
                    context,
                    _dialogue_without_system(msgs),
                    is_baseline=is_baseline,
                )
                if nxt is None:
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
            messages=[
                {"role": m.get("role"), "content": (m.get("content", "") or "")[:400]}
                for m in msgs
                if m.get("role") != "system"
            ],
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
            lines.append(f"Tool: {c[:300]}")
        elif r == "assistant":
            lines.append(f"Agent: {c[:300]}")
        elif r == "user":
            lines.append(f"Customer: {c[:300]}")
    return "\n".join(lines)
