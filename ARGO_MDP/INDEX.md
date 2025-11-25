# ARGO MDP - Complete Project Index

## 📍 Project Location
**Path**: `/data/user/huangxiaolin/ARGO2/ARGO_MDP/`  
**Environment**: `/root/miniconda/envs/ARGO/`  
**Python**: `/root/miniconda/envs/ARGO/bin/python`

---

## 📂 Complete File Structure

```
ARGO_MDP/
├── configs/
│   └── base.yaml                    # MDP configuration (U_max, δ_r, δ_p, etc.)
│
├── src/                             # Core implementation (1,823 lines)
│   ├── __init__.py                  # Package initialization
│   ├── mdp_solver.py                # Value iteration solver (273 lines)
│   ├── env_argo.py                  # MDP environment (310 lines)
│   └── policy.py                    # ARGO + 6 baselines (214 lines)
│
├── scripts/                         # Experiment runners
│   ├── run_single.py                # Main experiment (387 lines)
│   ├── test_basic.py                # Unit tests (232 lines)
│   └── project_overview.py          # Statistics generator
│
├── draw_figs/                       # Visualization (407 lines)
│   ├── plot_value_function.py       # V*(U), Q*(U,a), thresholds (172 lines)
│   └── plot_comparison.py           # Policy comparison (235 lines)
│
├── results/                         # Experimental outputs (13 files, 71KB)
│   ├── value_function.csv           # V* and Q* for all states
│   ├── thresholds.txt               # θ_cont, θ_star
│   ├── policy_comparison.csv        # Summary table
│   ├── sensitivity_analysis.csv     # Parameter sensitivity
│   └── [Policy]_episodes.csv        # Detailed episode data (9 policies)
│
├── figs/                            # Generated visualizations (6 images, 1.5MB)
│   ├── value_function.png           # V*(U) and Q*(U,a)
│   ├── action_selection.png         # Optimal action by state
│   ├── threshold_diagram.png        # Policy regions
│   ├── policy_comparison.png        # Performance bar charts
│   ├── cost_quality_tradeoff.png    # Pareto frontier
│   └── sensitivity_analysis.png     # Parameter sensitivity
│
├── data/                            # (Reserved for future use)
├── models/                          # (Reserved for LLM integration)
│
├── requirements.txt                 # Python dependencies
├── run_experiments.sh               # Convenience script
│
├── README.md                        # Main documentation (240 lines)
├── PROJECT_SUMMARY.md               # Implementation details (262 lines)
└── QUICK_REFERENCE.md               # Command reference (254 lines)
```

**Total**: ~3,600 lines of code + documentation

---

## 🎯 Core Components

### 1. MDP Solver (`src/mdp_solver.py`)
- **Bellman Equation**: V(U) = max_a [R(U,a) + γ E[V(U')|U,a]]
- **Value Iteration**: Converges in 2-10 iterations
- **Threshold Detection**: Automatic θ_cont and θ_star computation
- **Quality Functions**: Sigmoid and linear modes

### 2. Environment (`src/env_argo.py`)
- **State**: U ∈ [0, U_max] (information progress)
- **Actions**: {0: Retrieve, 1: Reason, 2: Terminate}
- **Transitions**:
  - Retrieve: Stochastic (p_s success → U += δ_r)
  - Reason: Deterministic (U += δ_p)
  - Terminate: Absorbing
- **Rewards**: Step costs + terminal quality

### 3. Policies (`src/policy.py`)
1. **ThresholdPolicy** (ARGO optimal)
2. **AlwaysRetrievePolicy**
3. **AlwaysReasonPolicy**
4. **FixedKRetrieveThenReasonPolicy** (K ∈ {1,2,3,5})
5. **RandomPolicy**
6. **SingleThresholdPolicy** (ablation)

### 4. Experiments (`scripts/run_single.py`)
- Policy comparison (9 policies × 100 episodes)
- Sensitivity analysis (μ, p_s, δ_r/δ_p)
- CSV output with statistics
- Configurable via YAML

### 5. Visualization (`draw_figs/`)
- Value and Q-function plots
- Action selection regions
- Policy performance comparison
- Cost-quality tradeoff
- Sensitivity curves

---

## 🚀 Usage Patterns

### Pattern 1: Quick Test
```bash
cd /data/user/huangxiaolin/ARGO2/ARGO_MDP
bash run_experiments.sh test
```

### Pattern 2: Full Pipeline
```bash
bash run_experiments.sh full
# Runs: solver → baselines → sensitivity → visualizations
```

### Pattern 3: Custom Configuration
```bash
# Edit config
nano configs/base.yaml

# Run experiment
/root/miniconda/envs/ARGO/bin/python scripts/run_single.py \
    --config configs/base.yaml \
    --sensitivity
```

### Pattern 4: Programmatic Use
```python
from src.mdp_solver import MDPSolver
from src.env_argo import ARGOEnv
from src.policy import ThresholdPolicy

# Solve MDP
config = {...}
solver = MDPSolver(config)
results = solver.solve()

# Create policy
policy = ThresholdPolicy(
    theta_cont=results['theta_cont'],
    theta_star=results['theta_star']
)

# Run episodes
env = ARGOEnv(mdp_config=config['mdp'])
stats = env.run_episode(max_steps=50, policy=policy)
```

---

## 📊 Key Results (Default Config)

### Optimal Thresholds
- **θ_cont**: 0.0000 (Always reason, never retrieve alone)
- **θ_star**: 1.0000 (Terminate at maximum U)

### Policy Performance
| Policy | Reward | Quality | Cost | Steps |
|--------|--------|---------|------|-------|
| ARGO | **-1.156** | 0.924 | **1.300** | 14.00 |
| AlwaysRetrieve | -2.308 | 0.924 | 2.020 | 11.10 |
| AlwaysReason | -1.156 | 0.924 | 1.300 | 14.00 |
| Random | -1.517 | **0.842** | 1.474 | 10.90 |

**Insight**: With default params, Reason is optimal due to:
- Lower cost (c_p=0.1 < c_r=0.2)
- Deterministic progress vs. stochastic Retrieve
- Cost weight μ=0.6 penalizes expensive retrieval

### Sensitivity Highlights
- **μ**: Linear impact on reward (-0.26 per 0.2 increase)
- **p_s**: Minimal impact in tested range (δ ratio too low)
- **δ_r/δ_p**: Critical threshold at ~2.5 enables retrieval

---

## 🔧 Configuration Parameters

### Default (`configs/base.yaml`)
```yaml
mdp:
  U_max: 1.0
  delta_r: 0.15      # Retrieve gain
  delta_p: 0.08      # Reason gain
  p_s: 0.7           # Retrieve success prob
  c_r: 0.2           # Retrieve cost
  c_p: 0.1           # Reason cost
  mu: 0.6            # Cost weight
  gamma: 1.0         # Discount
  U_grid_size: 100   # State discretization
```

### Recommended Modifications

**To Enable Retrieval:**
```yaml
delta_r: 0.25  # Increase efficiency
c_r: 0.15      # Decrease cost
```

**To Test Cost Sensitivity:**
```yaml
mu: [0.2, 0.4, 0.6, 0.8, 1.0]
```

**Higher Resolution:**
```yaml
U_grid_size: 200
```

---

## 📈 Output Files Reference

### Results (`results/`)

1. **value_function.csv**
   - Columns: U, V, Q_retrieve, Q_reason, Q_terminate
   - Rows: 100 (grid_size)

2. **thresholds.txt**
   - theta_cont: [value]
   - theta_star: [value]

3. **policy_comparison.csv**
   - Columns: Policy, Avg_Reward, Std_Reward, Avg_Quality, etc.
   - Rows: 9 (number of policies)

4. **sensitivity_analysis.csv**
   - Columns: parameter, value, theta_cont, theta_star, avg_reward, avg_quality, avg_cost
   - Rows: 15 (5 μ + 5 p_s + 5 δ_ratio)

5. **[Policy]_episodes.csv**
   - Columns: episode, total_reward, final_U, final_quality, total_cost, num_steps, etc.
   - Rows: 100 (num_episodes)

### Figures (`figs/`)

All PNG format, 300 DPI, publication quality:
- **value_function.png**: Dual plot (V* and Q*)
- **action_selection.png**: Scatter plot
- **threshold_diagram.png**: Policy regions bar chart
- **policy_comparison.png**: 2×2 subplot comparison
- **cost_quality_tradeoff.png**: Scatter with annotations
- **sensitivity_analysis.png**: Multi-panel line plots

---

## ✅ Validation Checklist

- [x] MDP solver converges (2-10 iterations)
- [x] Thresholds satisfy 0 ≤ θ_cont ≤ θ_star ≤ U_max
- [x] All policies execute without errors
- [x] Episodes terminate correctly (100% rate)
- [x] Results are reproducible (seed=42)
- [x] Visualizations generate successfully
- [x] Tests pass (scripts/test_basic.py)
- [x] Code is modular and documented
- [x] Configuration is flexible (YAML)
- [x] Output is interpretable (CSV + PNG)

---

## 🔮 Future Extensions

### Phase 1: LLM Integration
- [ ] `src/llm_interface.py` - Qwen2.5-14B wrapper
- [ ] Multi-GPU support (device_map="auto")
- [ ] Actual retrieval + reasoning actions
- [ ] Real quality metrics (BLEU, ROUGE, etc.)

### Phase 2: Advanced Features
- [ ] Function approximation (neural networks)
- [ ] POMDP formulation
- [ ] Online learning (Q-learning, policy gradient)
- [ ] Batch processing
- [ ] Domain-specific evaluation (O-RAN QA)

### Phase 3: Optimization
- [ ] Parallel episode execution
- [ ] GPU-accelerated solver
- [ ] Continuous state space
- [ ] Model-free RL comparison

---

## 📞 Contact Information

**Project**: ARGO MDP Simulation  
**Location**: `/data/user/huangxiaolin/ARGO2/ARGO_MDP/`  
**Environment**: `conda activate ARGO`  
**Status**: ✅ Complete and Operational  
**Version**: 1.0.0  
**Date**: 2025-10-28

---

## 🎓 Reference Documents

1. **README.md** - User guide and installation
2. **PROJECT_SUMMARY.md** - Detailed implementation notes
3. **QUICK_REFERENCE.md** - Command cheat sheet
4. **INDEX.md** - This file (navigation)

---

**Next Steps**:
1. Review generated figures in `figs/`
2. Analyze results in `results/policy_comparison.csv`
3. Experiment with different configurations
4. Integrate with actual LLM (Qwen2.5-14B)
5. Evaluate on real O-RAN QA tasks

**End of Index**
