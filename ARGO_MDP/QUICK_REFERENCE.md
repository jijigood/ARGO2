# ARGO MDP - Quick Reference Guide

## 🚀 Quick Start Commands

### Using Shell Script (Recommended)
```bash
cd /data/user/huangxiaolin/ARGO2/ARGO_MDP

# Run tests
bash run_experiments.sh test

# Run full pipeline
bash run_experiments.sh full

# Run sensitivity analysis
bash run_experiments.sh sensitivity

# Generate visualizations only
bash run_experiments.sh visualize

# Clean results
bash run_experiments.sh clean
```

### Using Python Directly
```bash
# Activate environment
conda activate ARGO
# OR use direct path
PYTHON=/root/miniconda/envs/ARGO/bin/python

# Run tests
$PYTHON scripts/test_basic.py

# Run main experiment
$PYTHON scripts/run_single.py --config configs/base.yaml

# Run with sensitivity analysis
$PYTHON scripts/run_single.py --config configs/base.yaml --sensitivity

# Generate plots
$PYTHON draw_figs/plot_value_function.py
$PYTHON draw_figs/plot_comparison.py
```

## 📁 Key Files

### Input
- `configs/base.yaml` - Main configuration file

### Source Code
- `src/mdp_solver.py` - MDP solver
- `src/env_argo.py` - Environment
- `src/policy.py` - Policies

### Output
- `results/policy_comparison.csv` - Performance summary
- `results/value_function.csv` - V* and Q* values
- `results/thresholds.txt` - Optimal thresholds
- `results/sensitivity_analysis.csv` - Parameter sensitivity
- `figs/*.png` - All visualizations

## ⚙️ Configuration Quick Edit

Edit `configs/base.yaml`:

```yaml
# Increase retrieval efficiency
mdp:
  delta_r: 0.25    # Increase from 0.15
  p_s: 0.85        # Increase from 0.7

# Change cost weight
mdp:
  mu: 0.8          # Increase from 0.6

# Finer grid
mdp:
  U_grid_size: 200  # Increase from 100
```

## 📊 Understanding Results

### Thresholds (`results/thresholds.txt`)
```
theta_cont: 0.000000
theta_star: 1.000000
```
- **θ_cont**: Switch from Retrieve to Reason
- **θ_star**: Terminate

### Policy Comparison (`results/policy_comparison.csv`)
| Policy | Avg_Reward | Avg_Quality | Avg_Cost |
|--------|-----------|-------------|----------|
| ARGO   | Best      | High        | Low      |

### Key Metrics
- **Avg_Reward**: Quality - μ * Cost (higher is better)
- **Avg_Quality**: Final information quality (0-1)
- **Avg_Cost**: Total accumulated cost
- **Avg_Steps**: Episode length

## 🎨 Generated Visualizations

1. **value_function.png**
   - Left: V*(U) optimal value function
   - Right: Q*(U,a) for all actions

2. **action_selection.png**
   - Shows optimal action for each state

3. **threshold_diagram.png**
   - Policy regions visualization

4. **policy_comparison.png**
   - Bar charts comparing all policies

5. **cost_quality_tradeoff.png**
   - Scatter plot showing Pareto frontier

6. **sensitivity_analysis.png**
   - Parameter sensitivity plots

## 🔧 Common Tasks

### Change Parameter and Re-run
```bash
# Edit config
nano configs/base.yaml

# Re-run
bash run_experiments.sh full
```

### Run Only MDP Solver
```python
from src.mdp_solver import MDPSolver
import yaml

config = yaml.safe_load(open('configs/base.yaml'))
solver = MDPSolver(config)
results = solver.solve()
print(f"θ_cont={results['theta_cont']:.4f}, θ*={results['theta_star']:.4f}")
```

### Evaluate Custom Policy
```python
from src.env_argo import ARGOEnv, MultiEpisodeRunner
from src.policy import ThresholdPolicy

mdp_config = {...}  # Your config
policy = ThresholdPolicy(theta_cont=0.3, theta_star=0.7)
env = ARGOEnv(mdp_config, seed=42)
runner = MultiEpisodeRunner(env, policy, num_episodes=100)
runner.run()
print(runner.get_summary())
```

### Add New Baseline
```python
# In src/policy.py, add:
class MyCustomPolicy:
    def act(self, U):
        # Your logic here
        return action
```

## 📈 Expected Behavior

### Normal Operation
- Solver converges in <10 iterations
- All policies achieve 100% termination rate
- ARGO reward ≥ baseline rewards
- Visualizations generate without errors

### Warning Signs
- Solver doesn't converge → Increase max_iterations
- Thresholds at 0 and 1 → Adjust delta_r/delta_p ratio
- Episodes timeout → Increase max_steps

## 🐛 Troubleshooting

### "Cannot find module"
```bash
# Check you're in ARGO environment
conda activate ARGO
python -c "import numpy, pandas, matplotlib; print('OK')"
```

### "Gym deprecated warning"
```bash
# Safe to ignore, or upgrade to gymnasium:
pip install gymnasium
# Then replace 'import gym' with 'import gymnasium as gym'
```

### Plots not showing
```bash
# Check matplotlib backend
python -c "import matplotlib; print(matplotlib.get_backend())"
# For headless: export MPLBACKEND=Agg
```

## 🔬 Experiment Ideas

### 1. High-Efficiency Retrieval
```yaml
delta_r: 0.30
delta_p: 0.08
p_s: 0.9
```
**Expected**: θ_cont > 0, earlier retrieval

### 2. Expensive Retrieval
```yaml
c_r: 0.5
c_p: 0.1
```
**Expected**: Less retrieval, more reasoning

### 3. Quality-Focused
```yaml
mu: 0.2
```
**Expected**: Higher quality acceptance threshold

## 📂 File Locations

- **Project Root**: `/data/user/huangxiaolin/ARGO2/ARGO_MDP/`
- **Python Environment**: `/root/miniconda/envs/ARGO/bin/python`
- **Results**: `./results/`
- **Figures**: `./figs/`
- **Logs**: Terminal output (redirect with `> log.txt`)

## 💡 Tips

1. **Always use the ARGO environment** - different numpy versions can cause issues
2. **Check results/ directory** before re-running to save old results
3. **Visualizations are PNG** - high DPI (300) for paper quality
4. **Sensitivity analysis takes time** - ~2-3 minutes for full run
5. **Seed is fixed (42)** - results are reproducible

## 📞 Need Help?

1. Run tests first: `bash run_experiments.sh test`
2. Check PROJECT_SUMMARY.md for detailed info
3. Review generated CSVs in results/
4. Check terminal output for errors

---

**Last Updated**: 2025-10-28
**Version**: 1.0.0
**Status**: ✅ Operational
