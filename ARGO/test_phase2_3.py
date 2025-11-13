#!/usr/bin/env python3
"""
Phase 2.3 验证脚本：测试扩展的质量函数
测试内容：
1. 验证4种质量函数的实现
2. 对比不同质量函数的阈值
3. 可视化质量函数曲线
4. 分析对MDP策略的影响
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 添加ARGO_MDP路径
argo_mdp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ARGO_MDP', 'src'))
sys.path.insert(0, argo_mdp_path)

from mdp_solver import MDPSolver


def create_config(quality_mode: str, quality_k: float = 5.0):
    """创建测试配置"""
    return {
        'mdp': {
            'delta_r': 0.25,
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
            'mode': quality_mode,
            'k': quality_k
        },
        'reward_shaping': {
            'enabled': False,
            'k': 1.0
        },
        'solver': {
            'max_iterations': 1000,
            'convergence_threshold': 1e-6,
            'verbose': False
        }
    }


def test_quality_functions():
    """主测试函数"""
    print("=" * 80)
    print("Phase 2.3: 质量函数扩展验证测试")
    print("=" * 80)
    
    # 测试所有质量函数
    quality_modes = [
        ("linear", 1.0, "线性: σ(x) = x"),
        ("sqrt", 1.0, "平方根: σ(x) = √x"),
        ("saturating", 3.0, "饱和: σ(x) = 1 - e^(-3x)"),
        ("sigmoid", 5.0, "Sigmoid: σ(x) = 1/(1+e^(-5(x-0.5)))")
    ]
    
    results = {}
    solvers = {}
    
    # 1. 测试每个质量函数
    for mode, k, desc in quality_modes:
        print(f"\n【测试: {desc}】")
        config = create_config(mode, k)
        solver = MDPSolver(config)
        solver.solve()
        
        print(f"  迭代次数: {solver.iterations if hasattr(solver, 'iterations') else 'N/A'}")
        print(f"  θ_cont = {solver.theta_cont:.4f}")
        print(f"  θ* = {solver.theta_star:.4f}")
        
        # 计算一些关键点的质量值
        U_samples = [0.0, 0.25, 0.5, 0.75, 1.0]
        quality_values = [solver.quality_function(u) for u in U_samples]
        
        print(f"  质量函数值:")
        for u, q in zip(U_samples, quality_values):
            print(f"    σ({u:.2f}) = {q:.4f}")
        
        results[mode] = {
            'theta_cont': solver.theta_cont,
            'theta_star': solver.theta_star,
            'description': desc,
            'k': k
        }
        solvers[mode] = solver
    
    # 2. 可视化
    print("\n" + "=" * 80)
    print("生成可视化图表...")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 2.1 质量函数曲线对比
    ax1 = axes[0, 0]
    U_range = np.linspace(0, 1, 1000)
    
    for mode, solver in solvers.items():
        quality_curve = [solver.quality_function(u) for u in U_range]
        ax1.plot(U_range, quality_curve, label=results[mode]['description'], linewidth=2)
    
    ax1.set_xlabel('Information Progress U')
    ax1.set_ylabel('Quality σ(U)')
    ax1.set_title('Quality Function Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.1])
    
    # 2.2 质量函数导数（边际效用）
    ax2 = axes[0, 1]
    
    for mode, solver in solvers.items():
        # 数值导数
        delta = 0.001
        derivative = []
        for u in U_range[:-1]:
            dq = (solver.quality_function(u + delta) - solver.quality_function(u)) / delta
            derivative.append(dq)
        
        ax2.plot(U_range[:-1], derivative, label=results[mode]['description'], linewidth=2)
    
    ax2.set_xlabel('Information Progress U')
    ax2.set_ylabel("σ'(U) (Marginal Utility)")
    ax2.set_title('Marginal Quality (Derivative)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    
    # 2.3 阈值对比
    ax3 = axes[1, 0]
    modes_list = list(results.keys())
    theta_conts = [results[m]['theta_cont'] for m in modes_list]
    theta_stars = [results[m]['theta_star'] for m in modes_list]
    
    x = np.arange(len(modes_list))
    width = 0.35
    
    ax3.bar(x - width/2, theta_conts, width, label='θ_cont', alpha=0.7)
    ax3.bar(x + width/2, theta_stars, width, label='θ*', alpha=0.7)
    ax3.set_ylabel('Threshold Value')
    ax3.set_title('Optimal Thresholds by Quality Function')
    ax3.set_xticks(x)
    ax3.set_xticklabels(modes_list)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 2.4 Value函数对比（在几个关键点）
    ax4 = axes[1, 1]
    
    U_test_points = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    for mode, solver in solvers.items():
        V_values = []
        for u in U_test_points:
            idx = solver.get_state_index(u)
            V_values.append(solver.V[idx])
        
        ax4.plot(U_test_points, V_values, marker='o', label=results[mode]['description'], linewidth=2)
    
    ax4.set_xlabel('Information Progress U')
    ax4.set_ylabel('Value Function V(U)')
    ax4.set_title('Value Function Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    os.makedirs('figs', exist_ok=True)
    output_path = 'figs/phase2_3_quality_functions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 图表已保存: {output_path}")
    
    # 3. 详细分析
    print("\n" + "=" * 80)
    print("📊 Phase 2.3 验证总结")
    print("=" * 80)
    
    print("\n1. 质量函数特性:")
    print("-" * 80)
    
    for mode in modes_list:
        desc = results[mode]['description']
        print(f"\n{desc}")
        
        solver = solvers[mode]
        
        # 计算凹凸性（二阶导数）
        U_mid = 0.5
        delta = 0.01
        q_minus = solver.quality_function(U_mid - delta)
        q_mid = solver.quality_function(U_mid)
        q_plus = solver.quality_function(U_mid + delta)
        
        second_derivative = (q_plus - 2*q_mid + q_minus) / (delta**2)
        
        if second_derivative < -0.01:
            concavity = "凹函数 (边际效用递减)"
        elif second_derivative > 0.01:
            concavity = "凸函数 (边际效用递增)"
        else:
            concavity = "线性 (边际效用不变)"
        
        print(f"  - 凹凸性: {concavity}")
        print(f"  - σ(0) = {solver.quality_function(0.0):.4f}")
        print(f"  - σ(0.5) = {solver.quality_function(0.5):.4f}")
        print(f"  - σ(1) = {solver.quality_function(1.0):.4f}")
    
    print("\n2. 阈值对比:")
    print("-" * 80)
    print(f"{'模式':<15} {'θ_cont':>10} {'θ*':>10} {'含义'}")
    print("-" * 80)
    
    for mode in modes_list:
        tc = results[mode]['theta_cont']
        ts = results[mode]['theta_star']
        
        if tc < 0.05:
            meaning = "几乎总是Reason"
        elif tc > 0.5:
            meaning = "更倾向Retrieve"
        else:
            meaning = "平衡策略"
        
        print(f"{mode:<15} {tc:>10.4f} {ts:>10.4f} {meaning}")
    
    print("\n3. 关键发现:")
    print("-" * 80)
    
    # 找出最倾向Retrieve的函数
    max_retrieve_mode = max(modes_list, key=lambda m: results[m]['theta_cont'])
    min_retrieve_mode = min(modes_list, key=lambda m: results[m]['theta_cont'])
    
    print(f"  • 最倾向Retrieve: {max_retrieve_mode} (θ_cont={results[max_retrieve_mode]['theta_cont']:.4f})")
    print(f"  • 最倾向Reason:   {min_retrieve_mode} (θ_cont={results[min_retrieve_mode]['theta_cont']:.4f})")
    
    print("\n4. 理论解释:")
    print("-" * 80)
    print("""
  Linear (σ(x) = x):
    - 边际效用恒定
    - 基线策略
  
  Sqrt (σ(x) = √x):
    - 凹函数，边际效用递减
    - 早期信息获取更有价值
    - 可能导致更积极的早期检索
  
  Saturating (σ(x) = 1 - e^(-αx)):
    - 凹函数，边际效用递减
    - 接近1时增长缓慢（饱和）
    - 可能导致更早终止
  
  Sigmoid (σ(x) = 1/(1+e^(-k(x-0.5)))):
    - S型曲线
    - 中间区域增长最快
    - 两端增长缓慢
    """)
    
    print("\n" + "=" * 80)
    print("🎉 Phase 2.3 测试完成！")
    print("=" * 80)
    
    return results, solvers


if __name__ == "__main__":
    test_quality_functions()
