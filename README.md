# MultiSessionBench

Multi-session τ-bench runs for CRAFT-style, memory-mediated attacks. Internal repo.

## Setup

Python ≥3.12, from repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Linux only: `pyaudio` needs `sudo apt-get install -y portaudio19-dev`.
`requirements.txt` pins `tau2[voice,knowledge]` from git; domain data ships with it
unless you set `TAU2_DATA_DIR=/path/to/data`.

API key — pick the one that matches the model id you pass (LiteLLM
`provider/model` form: `openrouter/...` → `OPENROUTER_API_KEY`,
`openai/...` → `OPENAI_API_KEY`, `anthropic/...` → `ANTHROPIC_API_KEY`).
Keys can also live in `experiments/.env`.

```bash
export OPENROUTER_API_KEY=your_key_here   # matches script defaults
```

## How to run

```bash
# Plan only, no API calls
python experiments/run_craft_multisession.py --dry-run --limit 1

# Default sweep (no_memory, full_history, summary)
python experiments/run_craft_multisession.py --limit 1

# All six non-RAG modes on one seed (no embedding API needed)
python experiments/run_craft_multisession.py --all-non-rag --limit 1

# Tag the run folder
python experiments/run_craft_multisession.py --memory-modes no_memory --run-name no-mem-baseline
```

RAG modes need embeddings — default `--rag-backend openai` reuses
`OPENROUTER_API_KEY` / `OPENAI_API_KEY`, or install `sentence-transformers`
and pass `--rag-backend sentence_transformers`.

All flags (`--all-modes`, `--instrument`, `--craft-seeds PATH`, OpenAI-direct
models, etc.): `python experiments/run_craft_multisession.py --help`.

### Where results go

```
experiments/results/craft_multi_<timestamp>[_<run-name>]/
    config.json
    run.jsonl              # mechanical ExperimentResult rows
    audits/               # only with --instrument
    judged.jsonl          # after enrich_judge.py
    summary.md           # after analysis/summarize_run.py
```

### After a run: judge + report

The runner writes **mechanical** traces only; compliance-style scoring uses the judge.

```bash
RUN=experiments/results/craft_multi_<timestamp>_<optional-label>

python experiments/enrich_judge.py "$RUN"
python experiments/run_craft_multisession.py --analyze "$RUN"
# or: python analysis/summarize_run.py "$RUN"
```

Legacy flat `.jsonl` paths still work for `enrich_judge.py -o OUT.jsonl` and `--analyze PATH.jsonl`.

## Layout

| Path | Role |
|------|------|
| `experiments/run_craft_multisession.py` | Main CRAFT multi-session runner |
| `tasks/craft_airline_multisession_seeds.json` | Default Plant / Reinforce / Trigger seeds (`--craft-seeds` overrides) |
| `memory/` | Memory providers + measurement modules — **[`memory/README.md`](memory/README.md)** |
| `analysis/report.py`, `analysis/summarize_run.py` | Tables / `summary.md` from a run folder |
| `experiments/enrich_judge.py` | Post-hoc judge → `judged.jsonl` |
| `core/` | Types, τ env, orchestrator, session loop, instrumentation, judge |

## More

**Memory mode names** (10 configs) — table and semantics: [`memory/README.md`](memory/README.md).

**`--instrument`** — enables ClaimTracker, MemoryAuditLog, and ConfidenceDriftTracker; audits under `<run_dir>/audits/`. Seeds can set a `planted_claims` array in JSON for finer claim instrumentation (see instrumentation docs there).

**`--human-attacker`** — stdin-driven user for one `(seed_id, memory_mode)`; ends a session with empty line / Ctrl-D (also `:q` / `###STOP###`).
