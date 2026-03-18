# MultiSessionBench

## Run The Experiment

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run a simulated experiment without API calls:

```bash
python3 experiments/run_experiment.py --mock
```

Run a real experiment with OpenRouter:

```bash
export OPENROUTER_API_KEY=your_key_here
python3 experiments/run_experiment.py
```

You can also place `OPENROUTER_API_KEY=...` in `experiments/.env`.

Useful options:

```bash
python3 experiments/run_experiment.py --dry-run
python3 experiments/run_experiment.py --analyze experiments/results/experiment_YYYYMMDD_HHMMSS.jsonl
```