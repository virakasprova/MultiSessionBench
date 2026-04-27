# MultiSessionBench

Multi-session τ-bench experiments for studying memory-mediated attacks on agentic
LLMs: an attacker agent that runs over N sessions, ten pluggable memory provider
configurations, three novel measurement layers (claim tracking, memory audit log,
confidence drift), YAML seeds, an optional post-hoc LLM judge, and JSONL
analysis.

## Setup

From the repo root:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set an API key for real runs (OpenRouter is the default provider):
```bash
export OPENROUTER_API_KEY=your_key_here
```

Or place `OPENROUTER_API_KEY=...` in `experiments/.env`.

**Model strings** use LiteLLM's provider-prefix format (e.g.
`openrouter/google/gemini-2.5-flash`). No separate `--provider` flag.

**RAG modes** additionally need either:
- An embeddings key — same `OPENROUTER_API_KEY` works for the OpenRouter
  embeddings endpoint, or set `OPENAI_API_KEY` to use OpenAI directly. This is
  the default (`--rag-backend openai`).
- Or `pip install sentence-transformers` and pass `--rag-backend
  sentence_transformers` for fully offline embeddings.

## Layout

| Path | Role |
|------|------|
| `core/types.py` | `TaskSeed`, `Session`, `ExperimentResult`, `SessionMechanicalResult` |
| `core/environment.py` | `TauBenchEnv` (airline / retail tools + wiki + data) |
| `core/session_runner.py` | Single-session agent loop |
| `core/orchestrator.py` | Multi-session attack driver, wires the optional instrument bundle |
| `core/instrumentation.py` | `Instruments` bundle + `derive_planted_claims` factory |
| `core/judge.py` | Post-hoc LLM judge for compliance scoring |
| `attackers/` | `AttackerAgent` ABC, `LLMAttacker`, `CraftLLMAttacker` (Plant / Reinforce / Trigger) |
| `memory/` | `BaseMemoryProvider` ABC + `make_memory_provider` factory (10 configs) + the four novel measurement modules. **See [`memory/README.md`](memory/README.md) for in-depth docs.** |
| `tasks/seeds/` | Per-task YAML `TaskSeed` definitions for the generic runner |
| `tasks/craft_airline_multisession_seeds.yaml` | Plant/Reinforce/Trigger seeds for the CRAFT runner |
| `tasks/loader.py` | Reads both YAML formats (incl. optional `planted_claims:` blocks) |
| `analysis/report.py` | `analyze()` on JSONL — per-mode tables, laundering heuristic, summary-degradation comparison |
| `experiments/run_craft_multisession.py` | **Primary runner.** CRAFT-style Plant/Reinforce/Trigger attacks across all 10 memory configs |
| `experiments/run_multisession.py` | Generic τ-bench runner (no_memory / full_history / summary only) |
| `experiments/enrich_judge.py` | Post-hoc judge + compliance enrichment on a JSONL run |
| `experiments/run_experiment.py` | Original single-file PoC monolith (kept for reference) |

## Running experiments

### CRAFT multi-session runner (primary)

`run_craft_multisession.py` reads `tasks/craft_airline_multisession_seeds.yaml`
(Plant / Reinforce / Trigger session intents per seed) and runs the CRAFT
attacker over every selected memory config.

```bash
# Dry-run: print the run plan and exit
python experiments/run_craft_multisession.py --dry-run

# Default sweep: no_memory, full_history, summary (matches legacy JSONLs)
python experiments/run_craft_multisession.py

# All six non-RAG configs on the first seed (no API key for embeddings needed)
python experiments/run_craft_multisession.py --all-non-rag --limit 1

# Every config the factory exposes (10 total: non-RAG + RAG)
python experiments/run_craft_multisession.py --all-modes --limit 1

# Specific configs only
python experiments/run_craft_multisession.py --memory-modes summary_cumulative hybrid_recent_summary

# Pick the embedding backend for any RAG mode (openai is the default)
python experiments/run_craft_multisession.py --memory-modes rag_baseline rag_attribution \
    --rag-backend openai

# Enable the novel measurement layers (writes audit JSONs alongside the run JSONL)
python experiments/run_craft_multisession.py --all-modes --limit 5 --instrument
```

`--memory-modes`, `--all-non-rag`, and `--all-modes` are mutually exclusive
(enforced by argparse).

### Available memory modes

Six non-RAG configs and four RAG configs, all built by the same
`make_memory_provider` factory in `memory/__init__.py`:

| Mode | Provider | Notes |
|------|----------|-------|
| `no_memory` | `NoMemoryProvider` | Stateless ASR baseline |
| `full_history` | `FullContextProvider` | All prior turns verbatim |
| `full_context_recent2` | `FullContextProvider(max_sessions=2)` | Sliding window |
| `summary` | `RollingSummaryProvider` | One LLM-written summary per session |
| `summary_cumulative` | `CumulativeSummaryProvider` | Single re-absorbed running paragraph (most laundering-prone) |
| `hybrid_recent_summary` | `RecentFullOldSummaryProvider` | Recent verbatim + older summarised |
| `rag_baseline` | `RAGMemoryProvider` | Top-k retrieval, no attribution |
| `rag_attribution` | `RAGMemoryProvider(use_attribution=True)` | Retrieval + epistemic source tags (defense) |
| `rag_shared` | `RAGMemoryProvider(user_scoped=False)` | Cross-user shared index |
| `hybrid_summary_rag` | `SummaryPlusRAGProvider` | Rolling summaries + RAG chunks |

Per-config details, attack-surface analysis, and code samples live in
[`memory/README.md`](memory/README.md).

### Instrumentation (`--instrument`)

Opt-in flag that wires three measurement layers into every attack:

- **`ClaimTracker`** — classifies each planted claim's status in memory after
  every session as `PRESENT` (qualifier intact), `DISTORTED` (qualifier
  stripped — the **laundering event**), or `ABSENT`. Surfaces `mar` and
  `distortion_rate` onto each `ExperimentResult`.
- **`MemoryAuditLog`** — records the full injected context, query, snapshot,
  and per-claim status at every session boundary. Surfaces
  `first_laundering_session` and `context_size_curve`. The full timeline
  is written to `<results_dir>/<run_stem>_audits/<attack_id>.json`.
- **`ConfidenceDriftTracker`** — measures claim density and qualifier
  fraction in the injected context per session, for the multi-session
  drift hypothesis.

Off by default so JSONLs remain comparable to pre-instrumentation runs.

`PlantedClaim` data can be supplied explicitly via a `planted_claims:` block
on each YAML seed (recommended — lets you specify `variants` that match
expected laundered phrasings); when omitted, a single primary claim is
auto-derived from `false_claim` + `policy_area`.

### Generic runner

`run_multisession.py` is the lighter, original runner that exposes only
`no_memory`, `full_history`, and `summary` and reads per-task YAMLs from
`tasks/seeds/`. Useful for τ-bench retail or non-CRAFT tasks. Also accepts
`--instrument`.

```bash
python experiments/run_multisession.py
python experiments/run_multisession.py --domain retail
python experiments/run_multisession.py --memory-modes no_memory full_history
```

### Post-hoc judge

The runners write the **mechanical** session record only (transcripts +
tool calls). Compliance, persistence curves, and contradiction detection
require the judge:

```bash
python experiments/enrich_judge.py experiments/results/craft_multi_*.jsonl \
    -o experiments/results/craft_multi_judged.jsonl
```

### Analysis

```bash
python experiments/run_craft_multisession.py --analyze \
    experiments/results/craft_multi_judged.jsonl
```

The report emits per-mode tables, a per-summary-mode false-claim laundering
heuristic, and a summary-vs-`full_history` compliance-degradation comparison
that runs separately for each pure-summary mode (`summary`,
`summary_rolling`, `summary_cumulative`).

## Original PoC

```bash
python experiments/run_experiment.py --mock
python experiments/run_experiment.py
```
