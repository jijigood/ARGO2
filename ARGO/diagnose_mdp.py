#!/usr/bin/env python
"""
MDP诊断脚本 - 测试不同μ值是否产生不同的阈值
"""

import sys
import os
import yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '../ARGO_MDP/src')

from mdp_solver import MDPSolver


def test_mdp_sensitivity():
    """测试MDP对μ的敏感性"""
    
    print("=" * 80)
    print("MDP诊断: 测试不同μ值下的阈值变化")
    print("=" * 80)
    
    # 加载配置
    with open("configs/multi_gpu.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # 测试不同的μ值
    mu_values = [0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0]
    
    print(f"\n配置参数:")
    print(f"  c_r = {config['mdp']['c_r']}")
    print(f"  c_p = {config['mdp']['c_p']}")
    print(f"  δ_r = {config['mdp']['delta_r']}")
    print(f"  δ_p = {config['mdp']['delta_p']}")
    print(f"  quality_k = {config['mdp']['quality_k']}")
    print(f"  U_max = {config['mdp']['U_max']}")
    
    print(f"\n{'μ':<10} {'θ_cont':<10} {'θ*':<10} {'V(0)':<10} {'V(0.5)':<10} {'V(1.0)':<10}")
    print("-" * 70)
    
    results = []
    
    for mu in mu_values:
        # 创建MDP配置
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
        
        # 求解
        try:
            solver = MDPSolver(solver_config)
            solver.solve()
            
            # 提取值
            n_states = len(solver.U_grid)
            V_0 = solver.V[0]
            V_half = solver.V[n_states // 2]
            V_max = solver.V[-1]
            
            print(f"{mu:<10.1f} {solver.theta_cont:<10.3f} {solver.theta_star:<10.3f} "
                  f"{V_0:<10.3f} {V_half:<10.3f} {V_max:<10.3f}")
            
            results.append({
                'mu': mu,
                'theta_cont': solver.theta_cont,
                'theta_star': solver.theta_star,
                'V_0': V_0,
                'V_half': V_half,
                'V_max': V_max
            })
            
        except Exception as e:
            print(f"{mu:<10.1f} ERROR: {e}")
    
    # 分析结果
    print("\n" + "=" * 80)
    print("分析:")
    print("=" * 80)
    
    if len(results) > 0:
        theta_stars = [r['theta_star'] for r in results]
        
        if all(abs(t - 1.0) < 0.01 for t in theta_stars):
            print("❌ 问题: θ* 对所有μ都是1.0 - 没有提前终止!")
            print("   建议:")
            print("   1. 进一步增大 c_r 和 c_p (例如 c_r=2.0, c_p=0.8)")
            print("   2. 降低 quality_k (从1.0降到0.5)")
            print("   3. 增大 μ_max (从100增到500)")
        elif max(theta_stars) - min(theta_stars) < 0.1:
            print("⚠️  问题: θ* 变化太小 (范围 < 0.1)")
            print(f"   当前范围: {min(theta_stars):.3f} ~ {max(theta_stars):.3f}")
            print("   建议: 调整成本参数使阈值变化更明显")
        else:
            print("✅ 良好: θ* 随μ变化显著")
            print(f"   θ*(μ=0) = {results[0]['theta_star']:.3f}")
            print(f"   θ*(μ={results[-1]['mu']}) = {results[-1]['theta_star']:.3f}")
            print(f"   变化幅度: {results[0]['theta_star'] - results[-1]['theta_star']:.3f}")
    
    # 计算关键比率
    print("\n关键比率分析:")
    k = config['mdp']['quality_k']
    c_r = config['mdp']['c_r']
    delta_r = config['mdp']['delta_r']
    
    reward_per_retrieve = k * delta_r  # 每次检索的质量增益
    print(f"  质量增益/检索: {reward_per_retrieve:.3f}")
    print(f"  检索成本: {c_r:.3f}")
    print(f"  比率 (reward/cost): {reward_per_retrieve / c_r:.3f}")
    print(f"\n  当 μ × c_r > reward_per_retrieve 时，应该提前终止")
    print(f"  即 μ > {reward_per_retrieve / c_r:.1f} 时应看到θ*下降")


if __name__ == "__main__":
    test_mdp_sensitivity()
