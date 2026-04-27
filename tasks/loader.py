"""YAML seed loader.

The active path is ``tasks/craft_airline_multisession_seeds.yaml`` — a
top-level ``seeds:`` list whose entries have flat ``session_intents`` (one
string per session). Use ``load_craft_multisession_yaml`` to read it.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from core.types import TaskSeed


def _normalize_craft_seed_entry(raw: dict) -> dict:
    """Build ``TaskSeed`` kwargs from one entry under ``seeds:``.

    Optional ``planted_claims:`` YAML block — a list of dicts each with
    ``claim_id`` + ``canonical`` (required) and ``variants`` /
    ``target_policy`` / ``negation`` (optional) — is passed through verbatim
    as ``planted_claim_data``. When absent, ``core.instrumentation
    .derive_planted_claims`` will auto-derive a single primary claim from
    ``false_claim`` + ``policy_area`` at instrument-build time, so existing
    YAMLs keep working with no edits.
    """
    si = raw.get("session_intents") or []
    if si and isinstance(si[0], str):
        session_intents = [[s.strip()] for s in si]
    else:
        session_intents = si
    hidden = raw.get("hidden_attacker_goal")
    if hidden is None:
        hidden = raw.get("false_claim")
    if hidden is None:
        raise ValueError(f"seed {raw.get('seed_id')!r} needs false_claim or hidden_attacker_goal")
    planted = raw.get("planted_claims")
    if planted is not None and not isinstance(planted, list):
        raise ValueError(
            f"seed {raw.get('seed_id')!r}: planted_claims must be a list of dicts"
        )
    return {
        "seed_id": raw["seed_id"],
        "user_id": raw.get("user_id") or "",
        "reservation_id": raw.get("reservation_id") or "UNKNOWN",
        "violation_tool": raw["violation_tool"],
        "false_claim": raw["false_claim"],
        "policy_area": raw["policy_area"],
        "session_intents": session_intents,
        "baseline_messages": raw.get("baseline_messages"),
        "hidden_attacker_goal": hidden,
        "roleplay_scenario": raw.get("roleplay_scenario"),
        "planted_claim_data": planted,
    }


def load_craft_multisession_yaml(path: Path | str) -> list[TaskSeed]:
    """Load ``tasks/craft_airline_multisession_seeds.yaml``-style file (``seeds:`` list)."""
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)
    block = data.get("seeds")
    if not isinstance(block, list):
        raise ValueError(f"Expected top-level 'seeds' list in {path}")
    return [TaskSeed(**_normalize_craft_seed_entry(entry)) for entry in block]
