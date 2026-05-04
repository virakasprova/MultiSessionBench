"""
seed_loader_validate.py
-----------------------
Validates every seed in a MultiSessionBench JSON file against the live
tau2-bench domain APIs before any sessions are run.

Usage:
    tau2 play --domain airline     # start the environment, then in a
                                   # separate terminal:
    python seed_loader_validate.py --seeds multisessionbench_v5_full.json \
                                   --airline-url http://localhost:8000 \
                                   --retail-url  http://localhost:8001 \
                                   --banking-url http://localhost:8002

Exit codes:
    0  all seeds pass
    1  one or more seeds fail (details printed to stdout)

The script DOES NOT run any sessions. It only calls read-only lookup
endpoints to verify that users and resources referenced in each seed
actually exist in the live DB.
"""

import argparse
import json
import sys
import re
from typing import Optional
import requests


# ── helpers ────────────────────────────────────────────────────────────────

class LoadError(Exception):
    pass


def _get(url: str, path: str, params: dict | None = None) -> dict:
    r = requests.get(f"{url}{path}", params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def extract_email(known_info: str) -> Optional[str]:
    """Pull the first email-like token out of a known_info string."""
    m = re.search(r"[\w.+-]+@[\w.-]+\.\w+", known_info or "")
    return m.group(0) if m else None


# ── per-domain validators ──────────────────────────────────────────────────

def validate_airline(seed: dict, base_url: str) -> list[str]:
    errors = []
    user_id = None

    # 1. user identity
    adv_instr  = seed["adversarial_task"]["user_scenario"]["instructions"]
    ben_instr  = seed["benign_task"]["user_scenario"]["instructions"]

    for instr in (adv_instr, ben_instr):
        uid_match = re.search(r"user id is (\S+)", instr.get("known_info", ""), re.I)
        if uid_match:
            user_id = uid_match.group(1).rstrip(".")
            break

    if not user_id:
        errors.append("Could not extract user_id from known_info")
        return errors

    try:
        user = _get(base_url, f"/get_user_details", {"user_id": user_id})
    except Exception as e:
        errors.append(f"get_user_details({user_id}) failed: {e}")
        return errors

    user_reservations = [r["reservation_id"] for r in user.get("reservations", [])]

    # 2. adversarial reservation. The free-text ``reason_for_call`` may
    # mention several 6-char uppercase tokens (``GOLD24``, ``REFUND``,
    # ``FLIGHT``, etc.) before the real reservation id, so picking the
    # *first* regex hit produces false negatives. Prefer any token that is
    # actually one of the user's reservations; only fall back to the first
    # 6-char token (and report it as missing) when no candidate matches —
    # which keeps the historical "no reservation in text" diagnostic.
    reason = adv_instr.get("reason_for_call", "")
    candidates = re.findall(r"\b([A-Z0-9]{6})\b", reason)
    matched = next((c for c in candidates if c in user_reservations), None)
    if matched is not None:
        # Real reservation present — nothing to flag.
        pass
    elif candidates:
        errors.append(
            f"No candidate token in reason_for_call ({candidates}) "
            f"matches user {user_id} reservations: {user_reservations}"
        )
    else:
        errors.append("Could not extract adversarial reservation ID from reason_for_call")

    # 3. benign reservation
    for action in seed["benign_task"]["evaluation_criteria"].get("actions", []):
        res_id = action.get("arguments", {}).get("reservation_id")
        if res_id and res_id not in user_reservations:
            errors.append(f"Benign reservation {res_id} not in user {user_id} reservations: {user_reservations}")

    return errors


def validate_retail(seed: dict, base_url: str) -> list[str]:
    errors = []

    ben_instr = seed["benign_task"]["user_scenario"]["instructions"]
    email = extract_email(ben_instr.get("known_info", ""))
    if not email:
        errors.append("Could not extract email from benign known_info")
        return errors

    try:
        result = _get(base_url, "/find_user_id_by_email", {"email": email})
        user_id = result.get("user_id")
    except Exception as e:
        errors.append(f"find_user_id_by_email({email}) failed: {e}")
        return errors

    if not user_id:
        errors.append(f"find_user_id_by_email({email}) returned no user_id — fictional user")
        return errors

    try:
        user = _get(base_url, "/get_user_details", {"user_id": user_id})
    except Exception as e:
        errors.append(f"get_user_details({user_id}) failed: {e}")
        return errors

    order_ids = [o["order_id"] for o in user.get("orders", [])]

    # benign order
    for action in seed["benign_task"]["evaluation_criteria"].get("actions", []):
        oid = action.get("arguments", {}).get("order_id")
        if oid and oid not in order_ids:
            errors.append(f"Benign order {oid} not in user {user_id} orders: {order_ids}")

    # adversarial order
    adv_instr = seed["adversarial_task"]["user_scenario"]["instructions"]
    adv_order_match = re.search(r"#W\d+", adv_instr.get("reason_for_call", ""))
    if adv_order_match:
        adv_oid = adv_order_match.group(0)
        if adv_oid not in order_ids:
            errors.append(f"Adversarial order {adv_oid} not in user {user_id} orders: {order_ids}")

    return errors


def validate_banking(seed: dict, base_url: str) -> list[str]:
    errors = []

    adv_instr = seed["adversarial_task"]["user_scenario"]["instructions"]
    uid_match = re.search(r"(?:user id|user ID) (?:is )?(\S+)", adv_instr.get("known_info", ""), re.I)
    if not uid_match:
        errors.append("Could not extract user_id from banking known_info")
        return errors

    user_id = uid_match.group(1).rstrip(".")

    try:
        _get(base_url, "/get_user_information_by_id", {"user_id": user_id})
    except Exception as e:
        errors.append(f"get_user_information_by_id({user_id}) failed: {e}")

    return errors


# ── main ───────────────────────────────────────────────────────────────────

VALIDATORS = {
    "airline":          validate_airline,
    "retail":           validate_retail,
    "banking_knowledge": validate_banking,
}

BLOCKED_STATUSES = {"BROKEN_USER", "BROKEN_SURFACE"}


def main():
    parser = argparse.ArgumentParser(description="Validate MultiSessionBench seeds against live tau2-bench DBs")
    parser.add_argument("--seeds", required=True, help="Path to seeds JSON file")
    parser.add_argument("--airline-url",  default="http://localhost:8000")
    parser.add_argument("--retail-url",   default="http://localhost:8001")
    parser.add_argument("--banking-url",  default="http://localhost:8002")
    parser.add_argument("--skip-blocked", action="store_true",
                        help="Warn on BROKEN seeds instead of failing (useful for inventory runs)")
    args = parser.parse_args()

    with open(args.seeds) as f:
        data = json.load(f)

    domain_urls = {
        "airline":           args.airline_url,
        "retail":            args.retail_url,
        "banking_knowledge": args.banking_url,
    }

    all_pass = True

    for seed in data["seeds"]:
        sid    = seed["seed_id"]
        domain = seed["domain"]
        status = seed.get("validation_status", "OK")

        # Hard-fail (or warn) on pre-known broken seeds
        if status in BLOCKED_STATUSES:
            msg = f"[BLOCKED] {sid}: validation_status={status} — {seed.get('validation_note', '')}"
            if args.skip_blocked:
                print(f"WARN  {msg}")
            else:
                print(f"FAIL  {msg}")
                all_pass = False
            continue

        validator = VALIDATORS.get(domain)
        if validator is None:
            print(f"SKIP  {sid}: no validator for domain '{domain}'")
            continue

        base_url = domain_urls[domain]
        errors = validator(seed, base_url)

        if errors:
            all_pass = False
            print(f"FAIL  {sid}:")
            for e in errors:
                print(f"        {e}")
        else:
            print(f"OK    {sid}")

    print()
    if all_pass:
        print("All seeds passed validation.")
        sys.exit(0)
    else:
        print("One or more seeds FAILED validation. Fix or tag them before running sessions.")
        sys.exit(1)


if __name__ == "__main__":
    main()
