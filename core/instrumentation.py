"""Instrumentation bundle for the orchestrator.

This module is the single integration point between the three measurement
layers in ``memory/`` (``ClaimTracker``, ``MemoryAuditLog``,
``ConfidenceDriftTracker``) and the multi-session orchestrator. It exposes:

* ``derive_planted_claims(seed)`` — returns a list of ``PlantedClaim`` for the
  seed. If ``seed.planted_claim_data`` is set (from a richer YAML schema), uses
  those entries verbatim. Otherwise auto-derives a single primary claim from
  ``seed.false_claim`` + ``seed.policy_area``, which is enough for the
  laundering / MAR experiments without requiring YAML edits.
* ``Instruments`` — a small dataclass bundle of the three trackers + the
  list of claims they were built for. Created per ``(seed, mode)`` pair by the
  runner and handed to ``Orchestrator.run_attack(...)``.
* ``build_instruments(memory, claims)`` — convenience constructor.

The orchestrator itself stays unaware of *how* instruments are configured —
it just calls the hook methods on whatever it's given.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.types import TaskSeed
    from memory.base import BaseMemoryProvider


@dataclass
class Instruments:
    """Bundle of per-attack measurement objects.

    Designed to be created once per ``(seed, mode)`` pair and discarded after
    the attack completes. ``ClaimTracker`` and ``MemoryAuditLog`` both hold
    refs to the *same* underlying memory provider; the orchestrator clears the
    provider between attacks, so the corresponding instruments are short-lived
    by construction.
    """
    claims: list  # list[PlantedClaim] — typed as list to avoid forced import
    tracker: object  # ClaimTracker
    audit_log: object  # MemoryAuditLog
    drift: object  # ConfidenceDriftTracker

    def primary_claim_id(self) -> Optional[str]:
        """Return the first claim's id, or None if no claims are configured."""
        if not self.claims:
            return None
        return self.claims[0].claim_id


def derive_planted_claims(seed: "TaskSeed") -> list:
    """Return ``list[PlantedClaim]`` for a seed.

    Two sources, in priority order:

    1. ``seed.planted_claim_data`` (optional explicit YAML block). Each entry
       must have ``claim_id`` and ``canonical``; ``variants``, ``target_policy``,
       and ``negation`` are optional.
    2. Auto-derived single primary claim from ``seed.false_claim`` and
       ``seed.policy_area``. ``claim_id`` is set to ``f"{seed.seed_id}_primary"``
       so it's unique per seed and stable across re-runs.

    This split lets you ship the instrumentation now (auto-derive) and later
    upgrade individual seeds with richer claim records by editing only the
    YAML, without touching code.
    """
    from memory.claim_tracker import PlantedClaim

    if seed.planted_claim_data:
        out = []
        for entry in seed.planted_claim_data:
            if "claim_id" not in entry or "canonical" not in entry:
                raise ValueError(
                    f"seed {seed.seed_id!r}: planted_claim_data entry missing "
                    "required keys 'claim_id' and 'canonical'"
                )
            out.append(PlantedClaim(
                claim_id=entry["claim_id"],
                canonical=entry["canonical"],
                variants=entry.get("variants") or [],
                target_policy=entry.get("target_policy") or seed.policy_area,
                negation=entry.get("negation") or "",
            ))
        return out

    if not seed.false_claim:
        return []

    return [PlantedClaim(
        claim_id=f"{seed.seed_id}_primary",
        canonical=seed.false_claim,
        variants=[],
        target_policy=seed.policy_area,
        negation="",
    )]


def build_instruments(
    memory: "BaseMemoryProvider",
    claims: list,
) -> Instruments:
    """Build a fresh ``Instruments`` bundle for one attack run.

    The ``ClaimTracker`` wraps ``memory`` and tags incoming turns; the
    ``MemoryAuditLog`` is told about the same provider so its ``snapshot()``
    calls go to the right place; the ``ConfidenceDriftTracker`` is provider-
    independent (it just consumes the injected-context string at each
    boundary).
    """
    from memory.claim_tracker import ClaimTracker
    from memory.audit_log import MemoryAuditLog
    from memory.confidence_drift import ConfidenceDriftTracker

    tracker = ClaimTracker(memory, claims)
    audit_log = MemoryAuditLog(memory, tracker=tracker)
    drift = ConfidenceDriftTracker(claims)
    return Instruments(claims=claims, tracker=tracker, audit_log=audit_log, drift=drift)
