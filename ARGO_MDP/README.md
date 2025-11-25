# ARGO: Optimal Policy-Based Adaptive RAG

Implementation of "ARGO: Optimal Policy-Based Adaptive RAG for O-RAN QA" - A Markov Decision Process framework for adaptive retrieval-augmented generation.

## Overview

ARGO models adaptive RAG as a discrete-time MDP with:
- **State**: Information progress U ∈ [0, U_max]
- **Actions**: {Retrieve, Reason, Terminate}
- **Objective**: Maximize quality while minimizing cost

## Project Structure

```
ARGO_MDP/
├── configs/
│   └── base.yaml           # Default configuration
├── src/
│   ├── mdp_solver.py       # Value iteration solver
│   ├── env_argo.py         # MDP environment
│   └── policy.py           # Policies (ARGO + baselines)
├── scripts/
│   └── run_single.py       # Main experiment runner
├── draw_figs/
│   ├── plot_value_function.py
│   └── plot_comparison.py
├── results/                # Output directory
├── figs/                   # Visualization output
└── requirements.txt
```

## Installation

### 1. Create Virtual Environment

```bash
conda create -n ARGO python=3.9 -y
conda activate ARGO
```

### 2. Install Dependencies

```bash
cd /data/user/huangxiaolin/ARGO2/ARGO_MDP
pip install -r requirements.txt
```

## Quick Start

### Run Complete Experiment

```bash
python scripts/run_single.py --config configs/base.yaml
```

This will:
1. Solve MDP using value iteration
2. Compute optimal thresholds (θ_cont, θ_star)
3. Evaluate ARGO and baseline policies
4. Save results to `results/`

### Run with Sensitivity Analysis

```bash
python scripts/run_single.py --config configs/base.yaml --sensitivity
```

### Generate Visualizations

```bash
# Plot value function and thresholds
python draw_figs/plot_value_function.py --results results --output figs

# Plot policy comparison
python draw_figs/plot_comparison.py --results results --output figs
```

## Configuration

Edit `configs/base.yaml` to modify parameters:

```yaml
mdp:
  U_max: 1.0          # Maximum information progress
  delta_r: 0.15       # Retrieve gain
  delta_p: 0.08       # Reason gain
  p_s: 0.7            # Retrieve success probability
  c_r: 0.2            # Retrieve cost
  c_p: 0.1            # Reason cost
  mu: 0.6             # Cost weight
  gamma: 1.0          # Discount factor
```

## Key Algorithms

### 1. MDP Solver (Value Iteration)

Solves Bellman equation:
```
V(U) = max_a [ R(U,a) + γ E[V(U')|U,a] ]
```

Computes:
- θ_cont: Retrieve/Reason threshold
- θ_star: Termination threshold

### 2. Threshold Policy

```python
if U >= θ_star:
    return Terminate
elif U < θ_cont:
    return Retrieve
else:
    return Reason
```

### 3. Baseline Policies

- **AlwaysRetrieve**: Always retrieve until U_max
- **AlwaysReason**: Always reason until U_max
- **FixedK**: Retrieve K times, then reason
- **Random**: Random action selection
- **SingleThreshold**: Only one threshold (ablation)

## Results

After running experiments, check:

### Output Files

- `results/value_function.csv` - V*(U) and Q*(U,a)
- `results/thresholds.txt` - Optimal thresholds
- `results/policy_comparison.csv` - Performance metrics
- `results/ARGO_episodes.csv` - Detailed episode data
- `results/sensitivity_analysis.csv` - Parameter sensitivity

### Visualizations

- `figs/value_function.png` - V*(U) and Q*(U,a) plots
- `figs/threshold_diagram.png` - Policy regions
- `figs/policy_comparison.png` - Performance comparison
- `figs/cost_quality_tradeoff.png` - Pareto frontier

## Expected Results

ARGO should achieve:
- **Higher reward** than baselines (quality - μ*cost)
- **Balanced** retrieve/reason actions
- **Lower cost** than AlwaysRetrieve
- **Higher quality** than AlwaysReason

## Advanced Usage

### Custom Configuration

```bash
# Create custom config
cp configs/base.yaml configs/my_config.yaml
# Edit my_config.yaml
python scripts/run_single.py --config configs/my_config.yaml
```

### Skip MDP Solving (Use Existing Results)

```bash
python scripts/run_single.py --no-solver
```

### Run Only Baselines

```bash
python scripts/run_single.py --no-baselines
```

## Parameters

### MDP Parameters

- **U_max**: Maximum information state (default: 1.0)
- **δ_r**: Information gain from retrieval (default: 0.15)
- **δ_p**: Information gain from reasoning (default: 0.08)
- **p_s**: Probability of retrieval success (default: 0.7)
- **c_r**: Cost of retrieval action (default: 0.2)
- **c_p**: Cost of reasoning action (default: 0.1)
- **μ**: Cost weight in reward (default: 0.6)
- **γ**: Discount factor (default: 1.0)

### Experiment Parameters

- **num_episodes**: Episodes per policy (default: 100)
- **max_steps_per_episode**: Max steps (default: 50)
- **seed**: Random seed (default: 42)

## Troubleshooting

### Import Errors

```bash
# Ensure ARGO environment is activated
conda activate ARGO

# Check installations
pip list | grep -E "numpy|pandas|matplotlib"
```

### Memory Issues

Reduce grid size in config:
```yaml
mdp:
  U_grid_size: 50  # Default: 100
```

### Convergence Issues

Increase max iterations:
```yaml
solver:
  max_iterations: 20000  # Default: 10000
  convergence_threshold: 1.0e-6
```

## Citation

```bibtex
@article{argo2024,
  title={ARGO: Optimal Policy-Based Adaptive RAG for O-RAN QA},
  author={Your Name},
  year={2024}
}
```

## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub.
