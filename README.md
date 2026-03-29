# MultiSessionBench

Multi-session τ-bench experiments: attacker + memory providers + orchestration, YAML seeds, optional post-hoc LLM judge, and JSONL analysis.

## Setup

From the repo root:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set an API key for real runs (e.g. OpenRouter):
```bash
export OPENROUTER_API_KEY=your_key_here
```

Or place `OPENROUTER_API_KEY=...` in `experiments/.env`.

**Model strings** use LiteLLM's provider-prefix format (e.g. `openrouter/google/gemini-2.5-flash`). No separate `--provider` flag.

## Layout

| Path | Role |
|------|------|
| `core/` | Types, `TauBenchEnv` (airline / retail), `SessionRunner`, `Orchestrator`, `SessionJudge` |
| `attackers/` | `AttackerAgent` ABC + `LLMAttacker` |
| `memory/` | `MemoryProvider` ABC + `NoMemory`, `FullHistoryMemory`, `SummaryMemory` |
| `tasks/seeds/` | YAML `TaskSeed` definitions (`session_intents`, …) |
| `analysis/report.py` | `analyze()` on JSONL |
| `experiments/run_multisession.py` | Modular runner (writes mechanical JSONL) |
| `experiments/enrich_judge.py` | Post-hoc judge + compliance on JSONL |
| `experiments/run_experiment.py` | Original PoC monolith (kept for reference) |

## Modular pipeline

1. **Run experiments** — writes one JSONL line per attack bundle (`mechanical` per session; judge fields empty until step 2):
```bash
python experiments/run_multisession.py
python experiments/run_multisession.py --model openrouter/google/gemini-2.5-flash
python experiments/run_multisession.py --domain airline   # or retail
python experiments/run_multisession.py --memory-modes no_memory full_history summary   # default: all three
python experiments/run_multisession.py --memory-modes no_memory   # single mode
python experiments/run_multisession.py --dry-run
```

Baselines always use `no_memory`; each `--memory-modes` value runs the full multi-session arc per seed.

2. **Enrich with LLM judge** (needed for compliance, persistence curves, contradictions):
```bash
python experiments/enrich_judge.py experiments/results/multi_*.jsonl -o experiments/results/multi_judged.jsonl
```

3. **Analyze**:
```bash
python experiments/run_multisession.py --analyze experiments/results/multi_judged.jsonl
```

## Original PoC
```bash
python experiments/run_experiment.py --mock
python experiments/run_experiment.py
```