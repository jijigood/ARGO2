#!/usr/bin/env python3
"""直接测试MDP求解器"""

import yaml
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(parent_dir, 'ARGO_MDP', 'src'))

from mdp_solver import MDPSolver

# 加载配置
with open('configs/multi_gpu.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 构造完整配置
full_config = {
    'mdp': {
        'U_max': config['mdp']['U_max'],
        'delta_r': config['mdp']['delta_r'],
        'delta_p': config['mdp']['delta_p'],
        'p_s': config['mdp']['p_s'],
        'c_r': config['mdp']['c_r'],
        'c_p': config['mdp']['c_p'],
        'mu': 0.5,  # 测试μ=0.5
        'gamma': config['mdp']['gamma'],
        'U_grid_size': config['mdp']['grid_size']
    },
    'quality': {
        'mode': config['mdp']['quality_function'],
        'k': config['mdp']['quality_k']
    },
    'solver': {
        'max_iterations': 1000,
        'convergence_threshold': 1e-6,
        'verbose': True  # 开启详细输出
    },
    'reward_shaping': config['mdp'].get('reward_shaping', {'enabled': False, 'k': 1.0})
}

print("=" * 80)
print("测试MDP求解器 (μ=0.5)")
print("=" * 80)
print(f"参数:")
print(f"  c_r={full_config['mdp']['c_r']}, c_p={full_config['mdp']['c_p']}")
print(f"  delta_r={full_config['mdp']['delta_r']}, delta_p={full_config['mdp']['delta_p']}")
print(f"  quality_k={full_config['quality']['k']}")
print(f"  μ={full_config['mdp']['mu']}")
print()

solver = MDPSolver(full_config)
theta_cont, theta_star = solver.compute_thresholds()

print(f"\n结果:")
print(f"  θ_cont = {theta_cont:.4f}")
print(f"  θ* = {theta_star:.4f}")
