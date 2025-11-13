#!/usr/bin/env python
"""
测试μ参数是否正确影响MDP求解结果
"""
import sys
import yaml
import numpy as np
sys.path.insert(0, '../ARGO_MDP/src')
from mdp_solver import MDPSolver

# 加载配置
with open('configs/multi_gpu.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("测试μ参数对MDP阈值的影响")
print("="*60)

# 测试不同的μ值
mu_values = [0.0, 1.0, 3.0, 5.0, 10.0]

for mu in mu_values:
    # 创建MDP配置
    mdp_config = config['mdp'].copy()
    mdp_config['mu'] = mu
    
    # 添加 U_grid_size (兼容性)
    if 'U_grid_size' not in mdp_config and 'grid_size' in mdp_config:
        mdp_config['U_grid_size'] = mdp_config['grid_size']
    
    solver_config = {
        'mdp': mdp_config,
        'quality': config.get('quality', {'mode': 'linear', 'k': 5.0}),
        'solver': {
            'max_iterations': 1000,
            'convergence_threshold': 1e-6,
            'verbose': False
        }
    }
    
    # 求解MDP
    mdp_solver = MDPSolver(solver_config)
    mdp_solver.solve()
    
    print(f"μ = {mu:5.2f}: θ_cont = {mdp_solver.theta_cont:.4f}, θ* = {mdp_solver.theta_star:.4f}")

print("="*60)
print("\n预期行为:")
print("- μ=0 (优先质量): 阈值应该较低,倾向于检索和推理")
print("- μ增大: 阈值应该升高,更倾向于提前终止以节省成本")
