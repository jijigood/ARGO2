#!/usr/bin/env python3
"""
测试MDP策略对参数的敏感性
证明：不执行Retrieve是参数问题，不是查询复杂度问题
"""

import sys
import os

# 添加ARGO_MDP路径
argo_mdp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ARGO_MDP', 'src'))
sys.path.insert(0, argo_mdp_path)

from mdp_solver import MDPSolver
import numpy as np

def test_parameter_sensitivity():
    """测试相同查询下，不同参数对MDP决策的影响"""
    
    print("=" * 80)
    print("MDP参数敏感性测试 - 证明Retrieve不执行是参数问题，非查询复杂度")
    print("=" * 80)
    
    # 固定的查询状态（模拟一个中等复杂度的问题）
    U_test = 0.3  # 当前信息度30%
    
    # 测试配置1: 当前配置（Reason更优）
    print("\n【配置1: 当前规范参数】")
    config1 = {
        'mdp': {
            'delta_r': 0.15,
            'delta_p': 0.08,
            'c_r': 0.05,
            'c_p': 0.02,
            'p_s': 0.8,
            'gamma': 0.98,
            'U_max': 1.0,
            'mu': 0.6,
            'U_grid_size': 101
        },
        'quality': {
            'mode': 'linear',
            'k': 1.0
        },
        'solver': {
            'max_iterations': 1000,
            'convergence_threshold': 1e-6,
            'verbose': False
        }
    }
    
    solver1 = MDPSolver(config1)
    solver1.solve()
    
    print(f"成本效益比:")
    print(f"  Retrieve: {config1['mdp']['c_r']}/{config1['mdp']['delta_r']} = {config1['mdp']['c_r']/config1['mdp']['delta_r']:.3f}")
    print(f"  Reason:   {config1['mdp']['c_p']}/{config1['mdp']['delta_p']} = {config1['mdp']['c_p']/config1['mdp']['delta_p']:.3f}")
    print(f"\n阈值结果:")
    print(f"  θ_cont = {solver1.theta_cont:.4f}")
    print(f"  θ* = {solver1.theta_star:.4f}")
    
    # 在U=0.3时的决策
    if U_test <= solver1.theta_cont:
        action1 = "Retrieve"
    elif U_test <= solver1.theta_star:
        action1 = "Reason"
    else:
        action1 = "Terminate"
    print(f"\n在U={U_test}时的决策: {action1}")
    
    # 测试配置2: 调整delta_r使Retrieve更有吸引力
    print("\n" + "-" * 80)
    print("【配置2: 提升Retrieve收益 (delta_r=0.25)】")
    config2 = {
        'mdp': {
            'delta_r': 0.25,  # 提高retrieve收益
            'delta_p': 0.08,
            'c_r': 0.05,
            'c_p': 0.02,
            'p_s': 0.8,
            'gamma': 0.98,
            'U_max': 1.0,
            'mu': 0.6,
            'U_grid_size': 101
        },
        'quality': {
            'mode': 'linear',
            'k': 1.0
        },
        'solver': {
            'max_iterations': 1000,
            'convergence_threshold': 1e-6,
            'verbose': False
        }
    }
    
    solver2 = MDPSolver(config2)
    solver2.solve()
    
    print(f"成本效益比:")
    print(f"  Retrieve: {config2['mdp']['c_r']}/{config2['mdp']['delta_r']} = {config2['mdp']['c_r']/config2['mdp']['delta_r']:.3f}")
    print(f"  Reason:   {config2['mdp']['c_p']}/{config2['mdp']['delta_p']} = {config2['mdp']['c_p']/config2['mdp']['delta_p']:.3f}")
    print(f"\n阈值结果:")
    print(f"  θ_cont = {solver2.theta_cont:.4f}")
    print(f"  θ* = {solver2.theta_star:.4f}")
    
    # 在U=0.3时的决策
    if U_test <= solver2.theta_cont:
        action2 = "Retrieve"
    elif U_test <= solver2.theta_star:
        action2 = "Reason"
    else:
        action2 = "Terminate"
    print(f"\n在U={U_test}时的决策: {action2}")
    
    # 测试配置3: 降低Retrieve成本
    print("\n" + "-" * 80)
    print("【配置3: 降低Retrieve成本 (c_r=0.02)】")
    config3 = {
        'mdp': {
            'delta_r': 0.15,
            'delta_p': 0.08,
            'c_r': 0.02,  # 降低retrieve成本
            'c_p': 0.02,
            'p_s': 0.8,
            'gamma': 0.98,
            'U_max': 1.0,
            'mu': 0.6,
            'U_grid_size': 101
        },
        'quality': {
            'mode': 'linear',
            'k': 1.0
        },
        'solver': {
            'max_iterations': 1000,
            'convergence_threshold': 1e-6,
            'verbose': False
        }
    }
    
    solver3 = MDPSolver(config3)
    solver3.solve()
    
    print(f"成本效益比:")
    print(f"  Retrieve: {config3['mdp']['c_r']}/{config3['mdp']['delta_r']} = {config3['mdp']['c_r']/config3['mdp']['delta_r']:.3f}")
    print(f"  Reason:   {config3['mdp']['c_p']}/{config3['mdp']['delta_p']} = {config3['mdp']['c_p']/config3['mdp']['delta_p']:.3f}")
    print(f"\n阈值结果:")
    print(f"  θ_cont = {solver3.theta_cont:.4f}")
    print(f"  θ* = {solver3.theta_star:.4f}")
    
    # 在U=0.3时的决策
    if U_test <= solver3.theta_cont:
        action3 = "Retrieve"
    elif U_test <= solver3.theta_star:
        action3 = "Reason"
    else:
        action3 = "Terminate"
    print(f"\n在U={U_test}时的决策: {action3}")
    
    # 总结
    print("\n" + "=" * 80)
    print("📊 结论总结")
    print("=" * 80)
    print(f"\n相同查询状态 (U={U_test}) 下的决策变化:")
    print(f"  配置1 (当前): {action1}")
    print(f"  配置2 (↑收益): {action2}")
    print(f"  配置3 (↓成本): {action3}")
    
    if action1 == action2 == action3:
        print("\n❌ 决策未改变 - 可能是查询复杂度问题")
    else:
        print("\n✅ 决策随参数改变 - 证明是参数设置问题，非查询复杂度")
    
    print("\n关键洞察:")
    print("- 成本效益比是决定性因素")
    print("- Retrieve成本/收益 vs Reason成本/收益的相对大小决定策略")
    print("- 查询本身的复杂度不影响这个基本权衡")
    print("- MDP求解器基于参数自动找到最优阈值")
    
    return solver1, solver2, solver3

if __name__ == "__main__":
    test_parameter_sensitivity()
