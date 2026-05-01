#!/usr/bin/env python3
"""Multi-session CRAFT runs using ``tasks/craft_airline_multisession_seeds.json``.

Each seed lists Plant / Reinforce / Trigger as separate ``session_intents`` strings; the loader
wraps each in a one-element inner list. ``CraftLLMAttacker`` uses ``false_claim`` (or
``hidden_attacker_goal``) as the hidden goal and one phase's text per orchestrator session.

**Baselines (two kinds):** (1) *Full-blob* — all phases merged into one chat (can be harder than
multi-session). (2) *Trigger-only* — only the Trigger phase in one session; isolates the
multi-session + memory effect from turn-count / priming confounds. Compare: trigger-only baseline
vs session 3 under ``no_memory`` vs ``full_history`` / ``summary``.

Summary memory: do **not** treat ASR under ``summary`` as a laundering or faithfulness bug — the
compressor reflects what actually happened (including tool outcomes that contradict the false
claim). Lower attack success under summary is a substantive finding about persistence, not a
defect in summarization.

Usage:
    python experiments/run_craft_multisession.py --dry-run
    python experiments/run_craft_multisession.py --craft-seeds tasks/craft_airline_multisession_seeds.json
    python experiments/run_craft_multisession.py --memory-modes no_memory --max-turns 25
    python experiments/run_craft_multisession.py --limit 5   # first 5 seeds only (per baseline + each mode)
    python experiments/run_craft_multisession.py --all-non-rag --limit 1   # 6 non-RAG configs on 1 seed
    python experiments/run_craft_multisession.py --all-modes --limit 1     # all 10 configs (incl. RAG)
    python experiments/run_craft_multisession.py --memory-modes rag_baseline rag_shared --rag-backend openai
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import litellm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_env = Path(__file__).resolve().parent / ".env"
if _env.exists():
    for ln in _env.read_text().splitlines():
        if ln.strip() and "=" in ln and not ln.startswith("#"):
            k, v = ln.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# `huggingface-cli login` writes the token here; litellm reads HF_TOKEN /
# HUGGINGFACE_API_KEY from env. Bridge the two so a logged-in user doesn't
# need to also export the token.
_hf_token_file = Path.home() / ".cache" / "huggingface" / "token"
if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGINGFACE_API_KEY") and _hf_token_file.exists():
    tok = _hf_token_file.read_text().strip()
    if tok:
        os.environ["HF_TOKEN"] = tok

from analysis import analyze, load_results_jsonl
from attackers import CraftLLMAttacker, HumanAttacker
from core.environment import TauBenchEnv
from core.instrumentation import build_instruments, derive_planted_claims
from core.orchestrator import Orchestrator
from core.session_runner import SessionRunner
from memory import BaseMemoryProvider, make_memory_provider
from memory.litellm_summariser import (
    make_litellm_cumulative_summariser,
    make_litellm_summariser,
)
from tasks.loader import load_craft_multisession_seeds

MODEL = "openrouter/google/gemini-2.5-flash"
ATTACKER_MODEL = "openai/gpt-4.1-mini"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CRAFT_SEEDS_DEFAULT = Path(__file__).resolve().parent.parent / "tasks" / "craft_airline_multisession_seeds.json"

_DEFAULT_MEMORY_MODES = ("no_memory", "full_context", "summary_rolling")

# Every non-RAG configuration the structured-turn factory exposes.
_NON_RAG_MEMORY_MODES = (
    "no_memory",              # NoMemoryProvider
    "full_context",           # FullContextProvider
    "full_context_recent2",   # FullContextProvider(max_sessions=2)
    "summary_rolling",        # RollingSummaryProvider
    "summary_cumulative",     # CumulativeSummaryProvider
    "hybrid_recent_summary",  # RecentFullOldSummaryProvider
)
_RAG_MEMORY_MODES = (
    "rag_baseline",           # RAGMemoryProvider (default)
    "rag_attribution",        # RAGMemoryProvider(use_attribution=True)
    "rag_shared",             # RAGMemoryProvider(user_scoped=False)
    "hybrid_summary_rag",     # SummaryPlusRAGProvider
)
_ALL_MEMORY_MODES = _NON_RAG_MEMORY_MODES + _RAG_MEMORY_MODES

_RAG_BACKEND_CHOICES = ("sentence_transformers", "openai")
litellm.suppress_debug_info = True


def _build_rag_backend(kind: str, api_base: str | None = None):
    """Construct the embedding backend once and let RAG providers share it.

    Building the SentenceTransformer model is the slow path (~80MB download
    on first run, several seconds each call). Sharing the backend instance
    across rag_baseline / rag_attribution / rag_shared / hybrid_summary_rag
    avoids paying that cost N times. ``encode()`` is stateless w.r.t. the
    backend, so this is safe.

    For ``kind == "openai"`` we prefer ``OPENAI_API_KEY`` (direct OpenAI),
    and fall back to ``OPENROUTER_API_KEY`` (OpenRouter is OpenAI-compatible
    and exposes ``openai/text-embedding-3-small`` at /embeddings). This
    means a single OpenRouter key is enough to run the entire sweep,
    including RAG modes, without a separate OpenAI account.
    """
    if kind == "sentence_transformers":
        from memory.providers.rag import SentenceTransformerBackend
        return SentenceTransformerBackend()
    if kind == "openai":
        from memory.providers.rag import OpenAIEmbeddingBackend
        if os.environ.get("OPENAI_API_KEY"):
            return OpenAIEmbeddingBackend(
                **({"base_url": api_base} if api_base else {}),
            )
        if os.environ.get("OPENROUTER_API_KEY"):
            return OpenAIEmbeddingBackend(
                model="openai/text-embedding-3-small",
                api_key=os.environ["OPENROUTER_API_KEY"],
                base_url="https://openrouter.ai/api/v1",
            )
        raise SystemExit(
            "--rag-backend openai requires OPENAI_API_KEY or OPENROUTER_API_KEY."
        )
    raise SystemExit(
        f"Unknown --rag-backend {kind!r}; expected one of {_RAG_BACKEND_CHOICES}"
    )


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str | None:
    """Return current HEAD sha, or None if not in a git repo / git missing."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return out or None
    except Exception:
        return None


def _slugify_run_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")


def _load_done_keys(run_log: Path, skip_log: Path) -> set[tuple[str, str]]:
    """Return ``{(attack_id, memory_mode)}`` already recorded in either log.

    Used by --resume so a partially completed sweep can be re-invoked and
    only run the missing combinations. Malformed lines are tolerated (a
    partially-flushed JSONL row from a hard kill won't poison the resume).
    """
    keys: set[tuple[str, str]] = set()
    for path in (run_log, skip_log):
        if not path.exists():
            continue
        with open(path) as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    row = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                aid = row.get("attack_id")
                mode = row.get("memory_mode")
                if aid and mode:
                    keys.add((aid, mode))
    return keys


def _record_skip(
    skip_log: Path,
    *,
    attack_id: str,
    memory_mode: str,
    is_baseline: bool,
    error: str,
    error_type: str,
    tb: str,
) -> None:
    """Append a structured record of one failed attack to ``skipped.jsonl``."""
    rec = {
        "skipped": True,
        "attack_id": attack_id,
        "memory_mode": memory_mode,
        "is_baseline": is_baseline,
        "error_type": error_type,
        "error": error,
        "traceback": tb,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(skip_log, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def _count_skipped(skip_log: Path) -> int:
    if not skip_log.exists():
        return 0
    n = 0
    with open(skip_log) as f:
        for ln in f:
            if ln.strip():
                n += 1
    return n


def _provider_key_for_model(model_id: str) -> tuple[str, str] | None:
    """Map a litellm model id to ``(env_var, human_provider_name)``.

    Used by the startup gate so the runner accepts whatever provider the
    requested ``--model`` / ``--attacker-model`` actually targets, instead
    of always demanding ``OPENROUTER_API_KEY``. Returns ``None`` for
    providers we don't have a runtime check for — litellm will surface its
    own error if the matching env var is missing.

    litellm's prefix conventions (the ones we exercise):
      - ``openrouter/...``   -> OPENROUTER_API_KEY
      - ``openai/...`` or no provider prefix -> OPENAI_API_KEY
      - ``anthropic/...``    -> ANTHROPIC_API_KEY
      - ``ollama/...`` / ``ollama_chat/...`` -> no key (local server)
    """
    if model_id.startswith("openrouter/"):
        return ("OPENROUTER_API_KEY", "OpenRouter")
    if model_id.startswith("ollama/") or model_id.startswith("ollama_chat/"):
        return None
    if model_id.startswith("huggingface/"):
        return ("HF_TOKEN", "Hugging Face")
    if model_id.startswith("openai/") or "/" not in model_id:
        return ("OPENAI_API_KEY", "OpenAI")
    if model_id.startswith("anthropic/"):
        return ("ANTHROPIC_API_KEY", "Anthropic")
    return None


def _check_provider_keys(*model_ids: str, api_base: str | None = None) -> None:
    """Exit if any required provider env var is missing for the given models.

    Deduplicates: an agent + attacker that both target OpenRouter only
    needs one ``OPENROUTER_API_KEY`` mention in the error message.

    When ``api_base`` is set the user is targeting a local OpenAI-compatible
    server (vLLM / LM Studio / llama.cpp) — skip the key check, and populate
    ``OPENAI_API_KEY`` with a placeholder so litellm's required-arg validator
    doesn't complain about a model id like ``openai/llama-3.1-8b``.
    """
    if api_base:
        os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
        return
    required: dict[str, str] = {}
    for m in model_ids:
        pair = _provider_key_for_model(m)
        if pair is not None:
            env, prov = pair
            required.setdefault(env, prov)
    # Hugging Face accepts either HF_TOKEN or HUGGINGFACE_API_KEY (litellm
    # reads both); treat the gate as satisfied if either is set.
    def _have(env: str) -> bool:
        if os.environ.get(env):
            return True
        if env == "HF_TOKEN" and os.environ.get("HUGGINGFACE_API_KEY"):
            return True
        return False

    missing = [(env, prov) for env, prov in required.items() if not _have(env)]
    if missing:
        for env, prov in missing:
            print(f"Set {env} (required for {prov} model in this run; "
                  f"or create experiments/.env)")
        sys.exit(1)


def _openai_backend_route() -> str:
    """Return a short label describing which key/endpoint the openai backend
    will use, for human-readable logging."""
    if os.environ.get("OPENAI_API_KEY"):
        return "OpenAI direct (OPENAI_API_KEY, model=text-embedding-3-small)"
    if os.environ.get("OPENROUTER_API_KEY"):
        return ("OpenRouter (OPENROUTER_API_KEY, "
                "base_url=openrouter.ai/api/v1, model=openai/text-embedding-3-small)")
    return "<no key found>"


# CLI memory-mode -> provider builder. Each entry takes ``(summary_model,
# rag_backend)``; lambdas ignore arguments they don't need. Adding a new mode
# is exactly one new line here plus one entry in ``_NON_RAG_MEMORY_MODES`` or
# ``_RAG_MEMORY_MODES``.
_MEMORY_DISPATCH = {
    "no_memory":              lambda model, rag: make_memory_provider("no_memory"),
    "full_context":           lambda model, rag: make_memory_provider("full_context"),
    "full_context_recent2":   lambda model, rag: make_memory_provider("full_context_recent2"),
    "summary_rolling":        lambda model, rag: make_memory_provider(
        "summary_rolling",    summariser=make_litellm_summariser(model)),
    "summary_cumulative":     lambda model, rag: make_memory_provider(
        "summary_cumulative", summariser=make_litellm_cumulative_summariser(model)),
    "hybrid_recent_summary":  lambda model, rag: make_memory_provider(
        "hybrid_recent_summary", summariser=make_litellm_summariser(model)),
    "rag_baseline":           lambda model, rag: make_memory_provider(
        "rag_baseline",       embedding_backend=rag),
    "rag_attribution":        lambda model, rag: make_memory_provider(
        "rag_attribution",    embedding_backend=rag),
    "rag_shared":             lambda model, rag: make_memory_provider(
        "rag_shared",         embedding_backend=rag),
    "hybrid_summary_rag":     lambda model, rag: make_memory_provider(
        "hybrid_summary_rag", summariser=make_litellm_summariser(model),
        embedding_backend=rag),
}

# Belt-and-braces: catch at import time the bug of "added a new mode to
# _ALL_MEMORY_MODES but forgot a dispatch entry" (and vice versa). Cheap.
assert set(_ALL_MEMORY_MODES) == set(_MEMORY_DISPATCH), (
    "memory-mode tuples and _MEMORY_DISPATCH are out of sync: "
    f"missing from dispatch: {sorted(set(_ALL_MEMORY_MODES) - set(_MEMORY_DISPATCH))}; "
    f"missing from _ALL_MEMORY_MODES: {sorted(set(_MEMORY_DISPATCH) - set(_ALL_MEMORY_MODES))}"
)


def make_memory(
    mode: str,
    summary_model: str,
    *,
    rag_backend=None,
) -> BaseMemoryProvider:
    """Map CLI memory-mode strings to ``BaseMemoryProvider`` instances.

    ``rag_backend`` (an ``EmbeddingBackend`` instance) must be pre-built by
    the caller via ``_build_rag_backend(...)`` for any RAG-using mode so that
    rag_baseline / rag_attribution / rag_shared / hybrid_summary_rag share a
    single model load.
    """
    if mode not in _MEMORY_DISPATCH:
        raise SystemExit(
            f"Unknown memory mode {mode!r}; expected one of {sorted(_MEMORY_DISPATCH)}"
        )
    if mode in _RAG_MEMORY_MODES and rag_backend is None:
        raise SystemExit(
            f"make_memory({mode!r}) requires rag_backend; "
            "pass one built via _build_rag_backend(...)"
        )
    return _MEMORY_DISPATCH[mode](summary_model, rag_backend)


def main() -> None:
    p = argparse.ArgumentParser(
        description="CRAFT multi-session runs from tasks/craft_airline_multisession_seeds.json"
    )
    p.add_argument("--model", default=MODEL)
    p.add_argument(
        "--craft-seeds",
        type=Path,
        default=CRAFT_SEEDS_DEFAULT,
        dest="craft_seeds",
        help=(
            "JSON file with top-level 'seeds' array "
            "(default: tasks/craft_airline_multisession_seeds.json)"
        ),
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--analyze",
        type=Path,
        metavar="RUN_DIR",
        help=(
            "Print report and exit. RUN_DIR is a run folder under "
            "experiments/results/ (uses judged.jsonl if present, else "
            "run.jsonl)."
        ),
    )
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        metavar="LABEL",
        help=(
            "Optional label appended to the run folder name "
            "(e.g. 'no-memory-baseline' -> "
            "results/craft_multi_<timestamp>_no-memory-baseline/)."
        ),
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="RUN_DIR",
        help=(
            "Resume an existing run folder. Skips any (attack_id, "
            "memory_mode) pair already recorded in run.jsonl or "
            "skipped.jsonl. --run-name is ignored when resuming."
        ),
    )
    # The three mode-selection flags are mutually exclusive. argparse enforces
    # that natively (via add_mutually_exclusive_group) with a clean error
    # message; no manual ``sum(sweep_flags) > 1`` post-parse check needed.
    sweep = p.add_mutually_exclusive_group()
    sweep.add_argument(
        "--memory-modes",
        nargs="*",
        default=None,
        metavar="MODE",
        help=(
            "One or more of: " + " ".join(_ALL_MEMORY_MODES) +
            " (default: " + " ".join(_DEFAULT_MEMORY_MODES) + ")"
        ),
    )
    sweep.add_argument(
        "--all-non-rag",
        action="store_true",
        help=(
            "Shortcut: run every non-RAG memory configuration ("
            + ", ".join(_NON_RAG_MEMORY_MODES) + ")."
        ),
    )
    sweep.add_argument(
        "--all-modes",
        action="store_true",
        help=(
            "Shortcut: run every memory configuration the factory exposes "
            "(non-RAG + RAG, " + str(len(_ALL_MEMORY_MODES)) + " total)."
        ),
    )
    p.add_argument(
        "--rag-backend",
        choices=_RAG_BACKEND_CHOICES,
        default="openai",
        help=(
            "Embedding backend for RAG modes. 'openai' (default) uses "
            "text-embedding-3-small via OPENAI_API_KEY (direct OpenAI) or "
            "OPENROUTER_API_KEY (OpenRouter's OpenAI-compatible /embeddings; "
            "model id 'openai/text-embedding-3-small'). "
            "'sentence_transformers' runs all-MiniLM-L6-v2 locally (no extra "
            "API key, but requires the sentence-transformers package). "
            "Ignored if no RAG mode is selected."
        ),
    )
    p.add_argument("--attacker-model", default=None, metavar="MODEL")
    p.add_argument("--attacker-temperature", type=float, default=0.5)
    p.add_argument(
        "--api-base",
        default=None,
        metavar="URL",
        help="Base URL for OpenAI-compatible API calls (agent, attacker, and RAG backend).",
    )
    p.add_argument("--max-turns", type=int, default=20, metavar="N")
    p.add_argument(
        "--human-attacker",
        action="store_true",
        help=(
            "Drive the customer side from stdin instead of CraftLLMAttacker. "
            "Useful for manual probing of one (seed, memory_mode) pair. "
            "Skips the auto full-blob and trigger-only baselines (you don't "
            "want to manually replay the same arc three times). Pair with "
            "--seed-ids and a single --memory-modes value; sweeps with many "
            "(seed, mode) combinations are rejected since each combination "
            "queues another interactive session on the same terminal."
        ),
    )
    p.add_argument(
        "--instrument",
        action="store_true",
        help=(
            "Enable per-attack ClaimTracker / MemoryAuditLog / "
            "ConfidenceDriftTracker instrumentation. Surfaces mar, "
            "distortion_rate, first_laundering_session, and "
            "context_size_curve onto each ExperimentResult, and writes a "
            "consolidated instruments JSON per attack into "
            "<run_dir>/audits/. Default OFF so baseline "
            "reproducibility against pre-instrumentation JSONLs is preserved."
        ),
    )
    p.add_argument(
        "--seed-ids",
        nargs="*",
        default=None,
        metavar="SEED_ID",
        help="Run only these exact seed_ids from the seeds file (overrides --limit)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Use only the first N seeds from the seeds file (applies to baseline and every memory mode)",
    )
    args = p.parse_args()

    if args.analyze:
        if not args.analyze.is_dir():
            raise SystemExit(f"{args.analyze} is not a run folder")
        cand = args.analyze / "judged.jsonl"
        if not cand.exists():
            cand = args.analyze / "run.jsonl"
        if not cand.exists():
            raise SystemExit(
                f"No run.jsonl or judged.jsonl found in {args.analyze}"
            )
        analyze(load_results_jsonl(cand))
        return

    # Mutual exclusion is enforced by the argparse group above.
    if args.all_modes:
        mem_modes = _ALL_MEMORY_MODES
    elif args.all_non_rag:
        mem_modes = _NON_RAG_MEMORY_MODES
    elif args.memory_modes:
        mem_modes = tuple(args.memory_modes)
    else:
        mem_modes = _DEFAULT_MEMORY_MODES
    for m in mem_modes:
        if m not in _ALL_MEMORY_MODES:
            raise SystemExit(
                f"Unknown memory mode {m!r}; expected one of {_ALL_MEMORY_MODES}"
            )

    needs_rag = any(m in _RAG_MEMORY_MODES for m in mem_modes)
    if needs_rag and args.rag_backend == "openai":
        if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")):
            raise SystemExit(
                "--rag-backend openai requires OPENAI_API_KEY (direct OpenAI) "
                "or OPENROUTER_API_KEY (OpenRouter). Either set one of those, "
                "or pass --rag-backend sentence_transformers (requires the "
                "sentence-transformers package installed locally)."
            )

    seeds = load_craft_multisession_seeds(args.craft_seeds)
    if args.seed_ids:
        wanted = set(args.seed_ids)
        seeds = [s for s in seeds if s.seed_id in wanted]
        found = {s.seed_id for s in seeds}
        missing = sorted(wanted - found)
        if missing:
            raise SystemExit(f"Unknown --seed-ids: {', '.join(missing)}")
    elif args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be at least 1")
        seeds = seeds[: args.limit]

    if args.dry_run:
        print(f"CRAFT seeds: {args.craft_seeds}")
        lim_note = f" (first {args.limit} only)" if args.limit else ""
        print(f"Seeds: {len(seeds)}{lim_note} | Memory modes: {list(mem_modes)}")
        if needs_rag:
            print(f"RAG backend: {args.rag_backend}")
            if args.rag_backend == "openai":
                print(f"  route: {_openai_backend_route()}")
        if args.instrument:
            print("Instrumentation: ON (claim tracker + audit log + drift tracker)")
        for s in seeds:
            print(
                f"  {s.seed_id}: user={s.user_id!r} res={s.reservation_id} "
                f"vtool={s.violation_tool} orchestrator_sessions={s.num_sessions()}"
            )
        n = len(seeds)
        total = n * (2 + len(mem_modes))
        print(
            f"\nTotal: {n} seeds × (2 baselines: full-blob + trigger-only + {len(mem_modes)} multi-session modes) "
            f"= {total} runs"
        )
        return

    # Gate is model-aware: only require the env var matching the litellm
    # prefix on the actual ``--model`` and ``--attacker-model`` (so e.g.
    # ``--model openai/gpt-4o-mini`` runs against ``OPENAI_API_KEY``
    # alone). Attacker model resolves to ATTACKER_MODEL when not passed.
    _check_provider_keys(args.model, args.attacker_model or ATTACKER_MODEL, api_base=args.api_base)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.resume:
        run_dir = args.resume
        if not run_dir.is_dir():
            raise SystemExit(f"--resume {run_dir} is not a directory")
        if args.run_name:
            print(f"[resume] ignoring --run-name {args.run_name!r} (using existing folder)")
        run_stem = run_dir.name
    else:
        run_stem = f"craft_multi_{datetime.now():%Y%m%d_%H%M%S}"
        if args.run_name:
            slug = _slugify_run_name(args.run_name)
            if slug:
                run_stem = f"{run_stem}_{slug}"
        run_dir = RESULTS_DIR / run_stem
        run_dir.mkdir(parents=True, exist_ok=True)
    log = run_dir / "run.jsonl"
    skip_log = run_dir / "skipped.jsonl"
    audit_dir = (run_dir / "audits") if args.instrument else None
    if audit_dir is not None:
        audit_dir.mkdir(parents=True, exist_ok=True)

    done_keys = _load_done_keys(log, skip_log)
    if args.resume:
        print(f"[resume] {run_dir}: {len(done_keys)} (attack_id, memory_mode) pairs already recorded; will skip those.")

    env = TauBenchEnv("airline")
    runner = SessionRunner(env, args.model, max_turns=args.max_turns, api_base=args.api_base)
    atk_model = args.attacker_model or ATTACKER_MODEL

    started_at = datetime.now().isoformat(timespec="seconds")
    config_path = run_dir / "config.json"
    # Status starts as "running" and is updated to "completed" / "interrupted"
    # / "failed" in the finally block at the end of main(). Lets resume callers
    # tell whether a folder is mid-run or finished cleanly.
    config: dict = {
        "run_stem": run_stem,
        "run_name": args.run_name,
        "started_at": started_at,
        "status": "running",
        "argv": sys.argv,
        "git_sha": _git_sha(),
        "agent_model": args.model,
        "attacker": {
            "kind": "human" if args.human_attacker else "craft_llm",
            "model": None if args.human_attacker else atk_model,
            "temperature": None if args.human_attacker else args.attacker_temperature,
        },
        "max_turns": args.max_turns,
        "memory_modes": list(mem_modes),
        "rag_backend": args.rag_backend if needs_rag else None,
        "instrumentation": bool(args.instrument),
        "api_base": args.api_base,
        "seeds": {
            "path": str(args.craft_seeds),
            "sha256": _file_sha256(args.craft_seeds),
            "count": len(seeds),
            "seed_ids": [s.seed_id for s in seeds],
        },
    }
    if args.resume and config_path.exists():
        # Preserve the original config; only refresh the status + last-seen argv
        # so reruns leave a trail. Don't overwrite started_at.
        try:
            with open(config_path) as f:
                existing = json.load(f)
        except Exception:
            existing = {}
        existing["status"] = "running"
        existing.setdefault("resumed_argv", []).append(sys.argv)
        existing["last_resumed_at"] = started_at
        config = existing
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    if args.human_attacker:
        # One human, one terminal: refuse to queue many interactive sessions.
        n_combinations = len(seeds) * len(mem_modes)
        if n_combinations != 1:
            raise SystemExit(
                f"--human-attacker requires exactly one (seed, mode) pair "
                f"but {len(seeds)} seed(s) x {len(mem_modes)} mode(s) "
                f"= {n_combinations} combinations were selected. "
                "Pair with --seed-ids X and a single --memory-modes Y."
            )
        attacker = HumanAttacker()
    else:
        attacker = CraftLLMAttacker(
            atk_model,
            temperature=args.attacker_temperature,
            api_base=args.api_base,
        )

    all_results = []

    print(f"Agent model: {args.model}")
    if args.human_attacker:
        print("Attacker: HumanAttacker (you drive; baselines skipped)")
    else:
        print(f"Attacker: {atk_model} (CraftLLMAttacker, temp={args.attacker_temperature})")
    print(f"max_turns: {args.max_turns}")
    lim_note = f" (limit {args.limit})" if args.limit else ""
    print(f"Seeds: {len(seeds)}{lim_note} | Memory modes: {list(mem_modes)}")
    if needs_rag:
        print(f"RAG backend: {args.rag_backend} (loading once, shared across RAG modes)")
        if args.rag_backend == "openai":
            print(f"  route: {_openai_backend_route()}")
    if args.instrument:
        print(f"Instrumentation: ON — audits in {audit_dir}/")
    print()

    rag_backend = _build_rag_backend(args.rag_backend, api_base=args.api_base) if needs_rag else None

    def _run_one(orch_obj: Orchestrator, the_seed, *, is_baseline: bool):
        """Build per-attack instruments (if enabled) and dispatch to orchestrator.

        Each (seed, mode) attack gets its own ClaimTracker / MemoryAuditLog /
        ConfidenceDriftTracker because the orchestrator clears the underlying
        memory provider at the start of every ``run_attack``. Reusing a tracker
        across attacks would break the per-session classification timeline.

        If instrument construction fails (e.g. malformed planted_claim_data on
        one seed), we still run the attack — losing the qualitative metrics
        for that seed is preferable to losing the attack record entirely.
        """
        if not args.instrument:
            return orch_obj.run_attack(the_seed, is_baseline=is_baseline)
        assert audit_dir is not None  # invariant: instrument => audit_dir set
        try:
            claims = derive_planted_claims(the_seed)
            instruments = build_instruments(orch_obj.memory, claims)
        except Exception as e:
            print(f"      [instrument build failed: {type(e).__name__}: {e}; "
                  f"running attack without instruments]")
            return orch_obj.run_attack(the_seed, is_baseline=is_baseline)
        attack_id_local = the_seed.seed_id + ("_baseline" if is_baseline else "")
        path = audit_dir / f"audit_{attack_id_local}__{orch_obj.mode_name}.json"
        return orch_obj.run_attack(
            the_seed,
            is_baseline=is_baseline,
            instruments=instruments,
            audit_path=path,
        )

    def _attempt(
        orch_obj: Orchestrator,
        the_seed,
        *,
        is_baseline: bool,
        label_attack_id: str,
    ) -> None:
        """Run one (seed, mode) attack with skip-on-error + resume dedup.

        On unhandled exception, write a record to ``skipped.jsonl`` and continue
        the sweep. On resume, dedup by ``(attack_id, memory_mode)`` against
        already-recorded rows in run.jsonl + skipped.jsonl.
        """
        key = (label_attack_id, orch_obj.mode_name)
        if key in done_keys:
            print(f"[skip — already recorded]")
            return
        try:
            r = _run_one(orch_obj, the_seed, is_baseline=is_baseline)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[FAILED: {type(e).__name__}: {e}]")
            _record_skip(
                skip_log,
                attack_id=label_attack_id,
                memory_mode=orch_obj.mode_name,
                is_baseline=is_baseline,
                error=str(e),
                error_type=type(e).__name__,
                tb=tb,
            )
            done_keys.add(key)
            return
        all_results.append(r)
        with open(log, "a") as f:
            f.write(json.dumps(r.to_dict(), default=str) + "\n")
        done_keys.add(key)

    final_status = "failed"
    try:
        # Skip auto-baselines for human-driven runs: replaying the full arc plus
        # trigger-only as separate single sessions interactively is tedious and
        # not the point of dropping into the REPL. Multi-session run still fires
        # so memory accumulation across phases is exercised.
        if not args.human_attacker:
            baseline_memory = make_memory("no_memory", args.model)
            baseline_orch = Orchestrator(
                env, runner, baseline_memory, attacker, mode_name="no_memory"
            )

            print("-- Baselines (single-session, all phases merged in one roleplay blob) --")
            for seed in seeds:
                aid = f"{seed.seed_id}_baseline"
                print(f"  {aid}:", end=" ")
                _attempt(baseline_orch, seed, is_baseline=True, label_attack_id=aid)

            print("\n-- Trigger-only baselines (single session, Trigger phase only — no plant/reinforce) --")
            for seed in seeds:
                trigger_only = copy.deepcopy(seed)
                trigger_only.seed_id = f"{seed.seed_id}_trigger_only"
                trigger_only.session_intents = [seed.session_intents[-1]]
                aid = f"{trigger_only.seed_id}_baseline"
                print(f"  {aid}:", end=" ")
                _attempt(baseline_orch, trigger_only, is_baseline=True, label_attack_id=aid)

        for mode in mem_modes:
            memory = make_memory(mode, args.model, rag_backend=rag_backend)
            orch = Orchestrator(env, runner, memory, attacker, mode_name=mode)
            print(f"\n-- Multi-session [{mode}] (Plant → Reinforce → Trigger) --")
            for seed in seeds:
                print(f"  {seed.seed_id} [{mode}]:", end=" ")
                _attempt(orch, seed, is_baseline=False, label_attack_id=seed.seed_id)
        final_status = "completed"
    except KeyboardInterrupt:
        final_status = "interrupted"
        print("\n[interrupted by user]")
    finally:
        # Final analyze + config update read from disk so they cover BOTH new
        # attacks from this invocation and any prior attacks loaded via
        # --resume (which never enter all_results).
        all_rows = load_results_jsonl(log) if log.exists() else []
        violated = sum(1 for r in all_rows if r.get("violation_detected"))
        n_skipped = _count_skipped(skip_log)
        print(f"\n{final_status.title()}. {violated}/{len(all_rows)} attacks with violation.")
        if n_skipped:
            print(f"Skipped (errors): {n_skipped} — see {skip_log.name}")
        print(f"Run folder: {run_dir}")

        config["status"] = final_status
        config["finished_at"] = datetime.now().isoformat(timespec="seconds")
        config["violations"] = violated
        config["total_attacks"] = len(all_rows)
        config["skipped"] = n_skipped
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        if all_rows and final_status != "interrupted":
            analyze(all_rows)


if __name__ == "__main__":
    main()
