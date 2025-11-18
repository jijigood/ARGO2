#!/usr/bin/env python
"""
精细诊断 - 在μ的过渡区域进行密集采样
"""

import sys
import os
import yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '../ARGO_MDP/src')

from mdp_solver import MDPSolver


def test_fine_grained():
    """在μ的过渡区进行精细测试"""
    
    print("=" * 80)
    print("精细MDP诊断: 找到θ*的过渡区")
    print("=" * 80)
    
    # 加载配置
    with open("configs/multi_gpu.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # 在0到5之间密集采样
    mu_values = np.linspace(0.0, 5.0, 21)  # 每0.25一个点
    
    print(f"\nμ范围: 0.0 ~ 5.0 (21个点)\n")
    print(f"{'μ':<10} {'θ_cont':<10} {'θ*':<10}")
    print("-" * 40)
    
    for mu in mu_values:
        mdp_config = config['mdp'].copy()
        mdp_config['mu'] = mu
        mdp_config['U_grid_size'] = mdp_config.get('grid_size', 101)
        
        solver_config = {
            'mdp': mdp_config,
            'quality': {'mode': 'linear', 'k': config['mdp']['quality_k']},
            'solver': {
                'max_iterations': 1000,
                'convergence_threshold': 1e-6,
                'verbose': False
            }
        }
        
        try:
            solver = MDPSolver(solver_config)
            solver.solve()
            
            print(f"{mu:<10.2f} {solver.theta_cont:<10.3f} {solver.theta_star:<10.3f}")
            
        except Exception as e:
            print(f"{mu:<10.2f} ERROR: {e}")


if __name__ == "__main__":
    test_fine_grained()
