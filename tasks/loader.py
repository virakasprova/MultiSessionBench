"""MultiSessionBench v4 seed loader (JSON).

Expects a top-level ``seeds`` array whose entries are tau2-Task-shaped:
``{adversarial_task: {...}, benign_task: {...}, ...}``. Single-session per
seed. ``adversarial_task.user_scenario.instructions.task_instructions`` is the
trigger; ``known_info`` is handed to the attacker verbatim so it can't
hallucinate a different account; the adversarial target is parsed out of
``description.notes``; ``benign_task.evaluation_criteria.actions`` pin the
specific benign reservation/order id used by ``core.benign_check``.

Use ``load_craft_multisession_seeds``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from core.environment import TauBenchEnv
from core.types import TaskSeed


class SeedValidationError(ValueError):
    """Raised when a seed's user / resource ids don't resolve in the live env.

    Caught and aggregated by the runner so a malformed seed surfaces as a
    pre-flight error with an obvious message, rather than as a silent zero
    in the ASR table after a real run.
    """


def _extract_user_id_from_known_info(known_info: str) -> str:
    """Pull a user-id token out of a v4 ``known_info`` string.

    Matches the natural-language patterns the v4 dataset uses:
      "Your user id is harper_garcia_8677."
      "Your user ID is 224959b99e."
      "User ID: lj82d4f1a9. Email: ..."
    Returns "" when nothing matches (retail seeds identify by email instead;
    the attacker prompt still gets ``known_info`` verbatim, so the LLM has
    enough context to introduce itself correctly).
    """
    if not known_info:
        return ""
    # Accept three shapes: "user id is X", "user id: X", "user id X".
    # The optional ``(?::|is)?`` swallows the literal "is" / ":" when
    # present, so the bare-whitespace form (MSB_005-style: "user id
    # emma_nguyen_9431") still captures the right token. Requiring at
    # least one digit in the captured group prevents false positives on
    # plain English like "no user id here" → "here" — every real tau2
    # user id includes a digit, so this is a safe filter.
    m = re.search(
        r"user\s*id\s*(?::|is)?\s+([a-zA-Z0-9_-]*\d[a-zA-Z0-9_-]*)",
        known_info,
        re.IGNORECASE,
    )
    if not m:
        return ""
    return m.group(1).rstrip(".,")


def _extract_target_resource(notes: str) -> str:
    """Pull the adversarial target resource id out of ``description.notes``.

    The convention across the v4 dataset is that notes start with the resource
    id followed by ``:`` — e.g. ``"CUPYN7: cabin=basic_economy, ..."`` or
    ``"#W4817420: status=delivered, ..."``. Falls back to a regex sweep for
    common id patterns (airline 6-char alphanumeric, retail ``#W...``,
    banking ``cc_...`` / ``dbc_...``) when the leading-token heuristic fails.
    """
    if not notes:
        return ""
    head = notes.split(":", 1)[0].strip()
    # Banking-knowledge notes lead with the user_id, not the target resource —
    # so a bare alphanumeric leading token isn't necessarily the resource. Only
    # accept the leading token when it looks resource-shaped.
    if re.fullmatch(r"[A-Z0-9]{5,8}", head) or head.startswith("#W"):
        return head
    m = re.search(r"\b(?:cc|dbc|sav|chk)_[a-zA-Z0-9_]+", notes)
    if m:
        return m.group(0)
    return head  # last resort


def _guess_violation_tool(notes: str) -> str:
    """Pick a plausible ``violation_tool`` name out of v4 notes for callers
    (instrumentation, logging) that still read the field. The DB-diff override
    in ``Orchestrator.run_attack`` is what actually drives the ASR metric —
    this is best-effort metadata only.
    """
    m = re.search(
        r"Forbidden(?:\s+action)?:\s*(?:any\s+action\s+(?:that\s+)?(?:sets|calls)\s+)?([a-z_][a-z_0-9]*)",
        notes or "",
        re.IGNORECASE,
    )
    return m.group(1) if m else ""


def _normalize_seed_entry(raw: dict) -> dict:
    """Build ``TaskSeed`` kwargs from one v4 seed entry."""
    seed_id = raw.get("seed_id") or "<unnamed>"
    if "adversarial_task" not in raw or "benign_task" not in raw:
        raise ValueError(
            f"seed {seed_id!r}: v4 schema requires both adversarial_task and benign_task"
        )
    adv = raw["adversarial_task"]
    ben = raw["benign_task"]

    adv_instr = (adv.get("user_scenario") or {}).get("instructions") or {}
    adv_known_info = (adv_instr.get("known_info") or "").strip()
    adv_task_text = (adv_instr.get("task_instructions") or "").strip()
    adv_reason = (adv_instr.get("reason_for_call") or "").strip()
    adv_notes = ((adv.get("description") or {}).get("notes") or "")
    adv_purpose = ((adv.get("description") or {}).get("purpose") or "").strip()

    if not adv_known_info:
        raise ValueError(
            f"seed {seed_id!r}: adversarial_task.user_scenario.instructions.known_info is required"
        )

    # Per-phase session intents drive the multi-session attack arc.
    # ``adversarial_task.session_intents`` is a list of ``{phase, intent}``
    # dicts in canonical plant -> reinforce -> trigger order. The last
    # session is the trigger (full action goal); earlier sessions are
    # read-only build-up.
    raw_sessions = adv.get("session_intents") or []
    if not raw_sessions:
        raise ValueError(
            f"seed {seed_id!r}: adversarial_task.session_intents is required"
        )
    session_intents = []
    for entry in raw_sessions:
        if not isinstance(entry, dict):
            raise ValueError(
                f"seed {seed_id!r}: session_intents entries must be dicts"
            )
        intent = (entry.get("intent") or "").strip()
        if not intent:
            raise ValueError(
                f"seed {seed_id!r}: session_intents entry missing 'intent'"
            )
        session_intents.append([intent])

    ben_instr = (ben.get("user_scenario") or {}).get("instructions") or {}
    ben_task_text = (ben_instr.get("task_instructions") or "").strip()
    ben_reason = (ben_instr.get("reason_for_call") or "").strip()
    ben_actions = ((ben.get("evaluation_criteria") or {}).get("actions") or [])
    expected_action_names: list[str] = [
        a["name"] for a in ben_actions if isinstance(a, dict) and a.get("name")
    ]
    benign_rid: str | None = None
    for a in ben_actions:
        if not isinstance(a, dict):
            continue
        args = a.get("arguments") or {}
        for key in ("reservation_id", "order_id"):
            v = args.get(key)
            if isinstance(v, str) and v:
                benign_rid = v
                break
        if benign_rid:
            break

    user_id = _extract_user_id_from_known_info(adv_known_info)
    target = _extract_target_resource(adv_notes)
    hidden_goal = adv_reason or adv_purpose or adv_task_text[:200]
    false_claim = adv_purpose or adv_task_text[:200]

    domain = (raw.get("domain") or "").strip()
    if not domain:
        raise ValueError(f"seed {seed_id!r}: top-level 'domain' is required")

    # Optional analysis tags. Default to "verifiable" / "early" so seeds that
    # predate the schema bump load unchanged. Explicit ``violation_tool`` at
    # the seed level (e.g. ``"re:close_credit_card_account_\\w+"``) overrides
    # the heuristic guesser — banking notes don't carry a clean tool name.
    claim_type = (raw.get("claim_type") or "verifiable").strip() or "verifiable"
    benign_timing = (raw.get("benign_timing") or "early").strip() or "early"
    explicit_violation_tool = (raw.get("violation_tool") or "").strip()
    max_turns_override = raw.get("max_turns_override")
    if max_turns_override is not None and not isinstance(max_turns_override, int):
        raise ValueError(
            f"seed {seed_id!r}: max_turns_override must be an int (got {type(max_turns_override).__name__})"
        )

    return {
        "seed_id": seed_id,
        "domain": domain,
        "user_id": user_id,
        "reservation_id": target or "UNKNOWN",
        "violation_tool": explicit_violation_tool or _guess_violation_tool(adv_notes) or "",
        "false_claim": false_claim,
        "policy_area": raw.get("policy_area") or "",
        "session_intents": session_intents,
        "hidden_attacker_goal": hidden_goal,
        "planted_claim_data": None,
        "benign_intent": ben_task_text or ben_reason or None,
        "benign_reservation_id": benign_rid,
        "expected_benign_actions": expected_action_names or None,
        "customer_known_info": adv_known_info,
        "claim_type": claim_type,
        "benign_timing": benign_timing,
        "max_turns_override": max_turns_override,
    }


def load_craft_multisession_seeds(path: Path | str) -> list[TaskSeed]:
    """Load JSON with top-level ``seeds`` array (``.json`` only)."""
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ValueError(f"Expected a .json seeds file, got {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    block = data.get("seeds")
    if not isinstance(block, list):
        raise ValueError(f"Expected top-level 'seeds' list in {path}")
    return [TaskSeed(**_normalize_seed_entry(entry)) for entry in block]


# ---------------------------------------------------------------------------
# Seed validation against the live tau2 env
# ---------------------------------------------------------------------------

_EMAIL_RX = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def _extract_email(text: str) -> str | None:
    if not text:
        return None
    m = _EMAIL_RX.search(text)
    return m.group(0) if m else None


def _find_tool_by_pattern(env: TauBenchEnv, pattern: str) -> str | None:
    """Return the first env tool whose full name matches ``pattern`` (regex).

    Banking tool names carry randomized numeric suffixes per data version
    (``close_credit_card_account_7834``, ``get_user_dispute_history_7291``, …).
    The validator uses this to find a usable read-tool by shape rather than
    by literal name.
    """
    rx = re.compile(pattern)
    for t in env.tools_info:
        name = (t.get("function") or {}).get("name") or ""
        if rx.fullmatch(name):
            return name
    return None


def _validate_airline_seed(seed: TaskSeed, env: TauBenchEnv) -> None:
    if not seed.user_id:
        raise SeedValidationError(
            f"{seed.seed_id}: airline seed has no user_id (known_info parse failed)"
        )
    raw = env.exec_tool("get_user_details", {"user_id": seed.user_id})
    if raw.startswith("Error"):
        raise SeedValidationError(
            f"{seed.seed_id}: airline user_id {seed.user_id!r} not found in env "
            f"({raw[:120]})"
        )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SeedValidationError(
            f"{seed.seed_id}: get_user_details returned non-JSON ({e})"
        )
    reservations = data.get("reservations") or []
    if seed.reservation_id and seed.reservation_id != "UNKNOWN":
        if seed.reservation_id not in reservations:
            raise SeedValidationError(
                f"{seed.seed_id}: reservation_id {seed.reservation_id!r} "
                f"not owned by {seed.user_id} (owns: {reservations})"
            )
    if seed.benign_reservation_id and seed.benign_reservation_id not in reservations:
        raise SeedValidationError(
            f"{seed.seed_id}: benign_reservation_id "
            f"{seed.benign_reservation_id!r} not owned by {seed.user_id} "
            f"(owns: {reservations})"
        )


def _validate_retail_seed(seed: TaskSeed, env: TauBenchEnv) -> None:
    # Retail seeds usually identify by email rather than user_id; the loader
    # leaves seed.user_id="" in that case. Try email lookup first, fall back
    # to seed.user_id if known_info had a literal user id.
    email = _extract_email(seed.customer_known_info)
    user_id = seed.user_id
    if email:
        raw = env.exec_tool("find_user_id_by_email", {"email": email})
        if raw.startswith("Error") or "not found" in raw.lower():
            raise SeedValidationError(
                f"{seed.seed_id}: retail email {email!r} not found in env "
                f"({raw[:120]})"
            )
        # find_user_id_by_email returns the bare user_id, possibly JSON-quoted.
        user_id = raw.strip().strip('"')
    if not user_id:
        raise SeedValidationError(
            f"{seed.seed_id}: retail seed has no email and no user_id "
            f"(known_info: {seed.customer_known_info[:120]!r})"
        )
    raw = env.exec_tool("get_user_details", {"user_id": user_id})
    if raw.startswith("Error"):
        raise SeedValidationError(
            f"{seed.seed_id}: retail user_id {user_id!r} not found "
            f"({raw[:120]})"
        )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SeedValidationError(
            f"{seed.seed_id}: get_user_details returned non-JSON ({e})"
        )
    orders = data.get("orders") or []
    if seed.reservation_id and seed.reservation_id != "UNKNOWN":
        if seed.reservation_id not in orders:
            raise SeedValidationError(
                f"{seed.seed_id}: order_id {seed.reservation_id!r} "
                f"not owned by user {user_id} (owns: {orders})"
            )
    if seed.benign_reservation_id and seed.benign_reservation_id not in orders:
        raise SeedValidationError(
            f"{seed.seed_id}: benign order_id {seed.benign_reservation_id!r} "
            f"not owned by user {user_id} (owns: {orders})"
        )


def _validate_banking_seed(seed: TaskSeed, env: TauBenchEnv) -> None:
    if not seed.user_id:
        raise SeedValidationError(
            f"{seed.seed_id}: banking seed has no user_id (known_info parse failed)"
        )
    raw = env.exec_tool("get_user_information_by_id", {"user_id": seed.user_id})
    if raw.startswith("Error") or "not found" in raw.lower():
        raise SeedValidationError(
            f"{seed.seed_id}: banking user_id {seed.user_id!r} not found "
            f"({raw[:120]})"
        )
    rid = seed.reservation_id
    if not rid or rid == "UNKNOWN":
        # Notes parse failed; the run will still execute but DB-diff won't
        # have anything to anchor on.
        return
    if rid.startswith("cc_"):
        cc_raw = env.exec_tool("get_credit_card_accounts_by_user", {"user_id": seed.user_id})
        if cc_raw.startswith("Error"):
            raise SeedValidationError(
                f"{seed.seed_id}: get_credit_card_accounts_by_user failed for "
                f"{seed.user_id!r} ({cc_raw[:120]})"
            )
        if rid not in cc_raw:
            raise SeedValidationError(
                f"{seed.seed_id}: credit-card account {rid!r} not found among "
                f"{seed.user_id!r}'s cards"
            )
        return
    # Other banking resource shapes (dbc_*, account ids) need a suffixed tool.
    # Best-effort: find one and confirm the id appears in its response.
    pat = _find_tool_by_pattern(env, r"get_all_user_accounts_by_user_id_\w+")
    if pat is None:
        # No discoverable account-listing tool — accept the seed and let the
        # run rely on tool_match grounding.
        return
    raw = env.exec_tool(pat, {"user_id": seed.user_id})
    if raw.startswith("Error"):
        raise SeedValidationError(
            f"{seed.seed_id}: {pat}({seed.user_id!r}) failed ({raw[:120]})"
        )
    if rid not in raw:
        raise SeedValidationError(
            f"{seed.seed_id}: banking resource {rid!r} not found in "
            f"{pat} response for user {seed.user_id!r}"
        )


def validate_seed_against_env(seed: TaskSeed, env: TauBenchEnv) -> None:
    """Hard-fail if the seed's user / resource ids don't resolve in the env.

    Per-domain dispatch: airline checks reservations, retail checks orders
    (after resolving by email), banking checks credit-card accounts (and
    other resources via a regex-discovered listing tool when available).
    Designed to catch fictional ids before a real sweep wastes LLM budget on
    seeds that can never violate.
    """
    if seed.domain == "airline":
        _validate_airline_seed(seed, env)
    elif seed.domain == "retail":
        _validate_retail_seed(seed, env)
    elif seed.domain == "banking_knowledge":
        _validate_banking_seed(seed, env)
    else:
        raise SeedValidationError(
            f"{seed.seed_id}: unknown domain {seed.domain!r}"
        )
