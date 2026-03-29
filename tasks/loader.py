"""YAML seed loader.

Each .yaml file in tasks/seeds/ defines one attack scenario (TaskSeed).
Requires ``session_intents`` in YAML (see ``tasks/seeds/*.yaml``).
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
