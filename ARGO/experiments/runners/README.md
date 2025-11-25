# Experiment Runners & Entry Points

This directory is the canonical place to track scripts that *launch* experiments:

- Python drivers (`Exp_real_cost_impact_v2.py`, `Exp_RAG_benchmark.py`, `Exp_7B_optimized.py`, etc.).
- Shell wrappers (`run_exp1_full.sh`, `run_multi_gpu.sh`, `start_experiment.sh`).
- Helper prompts like `START_HERE.sh` or `quickstart_prompts_v2.sh`.

For each runner, document:
1. Expected datasets (e.g., `ORAN-Bench-13K`, `results/multi_gpu` cache).
2. GPU/memory requirements (link to `docs/strategy/ACCELERATION_PLAN.md`).
3. Output location (point to `artifacts/results`).

When moving scripts from the root, keep their original names and add per-script READMEs or docstrings summarizing CLI arguments.
