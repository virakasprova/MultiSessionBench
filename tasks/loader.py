"""YAML seed loader.

Each .yaml file in tasks/seeds/ defines one attack scenario (TaskSeed).
Requires ``session_intents`` in YAML (see ``tasks/seeds/*.yaml``).

``craft_airline_multisession_seeds.yaml`` uses a top-level ``seeds:`` list and may use a *flat*
``session_intents`` (one string per session); use ``load_craft_multisession_yaml``.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from core.types import TaskSeed

SEEDS_DIR = Path(__file__).resolve().parent / "seeds"


def load_seed(path: Path) -> TaskSeed:
    with open(path) as f:
        d = yaml.safe_load(f)
    return TaskSeed(**d)


def load_seeds(directory: Path | None = None) -> list[TaskSeed]:
    d = directory or SEEDS_DIR
    return [load_seed(p) for p in sorted(d.glob("*.yaml"))]


def _normalize_craft_seed_entry(raw: dict) -> dict:
    """Build ``TaskSeed`` kwargs from one entry under ``seeds:``."""
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
