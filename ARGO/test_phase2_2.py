#!/usr/bin/env python3
"""
Phase 2.2 验证脚本：对比有无 Reward Shaping 的 MDP 求解器性能
测试内容：
1. 收敛速度对比
2. 阈值变化
3. Q函数差异
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# 添加ARGO_MDP路径
argo_mdp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ARGO_MDP', 'src'))
sys.path.insert(0, argo_mdp_path)

from mdp_solver import MDPSolver


def create_config(use_shaping: bool, shaping_k: float = 1.0):
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
            'mode': 'linear',
            'k': 1.0
        },
        'reward_shaping': {
            'enabled': use_shaping,
            'k': shaping_k
        },
        'solver': {
            'max_iterations': 1000,
            'convergence_threshold': 1e-6,
            'verbose': False
        }
    }


def track_convergence(solver: MDPSolver, config: dict):
    """跟踪收敛过程"""
    grid_size = config['mdp']['U_grid_size']
    max_iterations = config['solver']['max_iterations']
    convergence_threshold = config['solver']['convergence_threshold']
    gamma = config['mdp']['gamma']
    
    V = np.zeros(grid_size)
    Q = np.zeros((grid_size, 3))
    convergence_history = []
    iteration_count = 0
    
    for iteration in range(max_iterations):
        V_old = V.copy()
        
        # 更新每个状态（复制value_iteration的逻辑）
        for i, U in enumerate(solver.U_grid):
            
            for action in range(3):
                if action == 2:  # Terminate
                    Q[i, action] = solver.quality_function(U)
                else:
                    next_states, probs = solver.transition(U, action)
                    immediate_reward = solver.reward(U, action)
                    
                    expected_value = 0.0
                    expected_shaping = 0.0
                    
                    for U_next, prob in zip(next_states, probs):
                        idx_next = solver.get_state_index(U_next)
                        expected_value += prob * V_old[idx_next]
                        
                        if solver.use_reward_shaping:
                            expected_shaping += prob * solver.shaping_reward(U, U_next)
                    
                    Q[i, action] = immediate_reward + expected_shaping + gamma * expected_value
            
            V[i] = np.max(Q[i, :])
        
        # 检查收敛
        max_diff = np.max(np.abs(V - V_old))
        convergence_history.append(max_diff)
        iteration_count = iteration + 1
        
        if max_diff < convergence_threshold:
            break
    
    return iteration_count, convergence_history, V, Q


def test_reward_shaping():
    """主测试函数"""
    print("=" * 80)
    print("Phase 2.2: Reward Shaping 验证测试")
    print("=" * 80)
    
    # 测试参数
    shaping_k_values = [0.5, 1.0, 2.0]
    
    results = {}
    
    # 1. 无 Reward Shaping（基线）
    print("\n【测试1: 无 Reward Shaping (基线)】")
    config_baseline = create_config(use_shaping=False)
    solver_baseline = MDPSolver(config_baseline)
    
    start_time = time.time()
    iterations_baseline, convergence_baseline, V_baseline, Q_baseline = track_convergence(
        solver_baseline, config_baseline
    )
    time_baseline = time.time() - start_time
    
    solver_baseline.V = V_baseline
    solver_baseline.Q = Q_baseline
    solver_baseline.compute_thresholds()
    
    print(f"  迭代次数: {iterations_baseline}")
    print(f"  收敛时间: {time_baseline:.4f}s")
    print(f"  θ_cont = {solver_baseline.theta_cont:.4f}")
    print(f"  θ* = {solver_baseline.theta_star:.4f}")
    
    results['baseline'] = {
        'iterations': iterations_baseline,
        'time': time_baseline,
        'convergence': convergence_baseline,
        'theta_cont': solver_baseline.theta_cont,
        'theta_star': solver_baseline.theta_star,
        'V': V_baseline
    }
    
    # 2. 有 Reward Shaping（不同k值）
    for k in shaping_k_values:
        print(f"\n【测试2: Reward Shaping (k={k})】")
        config_shaping = create_config(use_shaping=True, shaping_k=k)
        solver_shaping = MDPSolver(config_shaping)
        
        start_time = time.time()
        iterations_shaping, convergence_shaping, V_shaping, Q_shaping = track_convergence(
            solver_shaping, config_shaping
        )
        time_shaping = time.time() - start_time
        
        solver_shaping.V = V_shaping
        solver_shaping.Q = Q_shaping
        solver_shaping.compute_thresholds()
        
        print(f"  迭代次数: {iterations_shaping}")
        print(f"  收敛时间: {time_shaping:.4f}s")
        print(f"  θ_cont = {solver_shaping.theta_cont:.4f}")
        print(f"  θ* = {solver_shaping.theta_star:.4f}")
        
        # 对比
        speedup = iterations_baseline / iterations_shaping
        print(f"  收敛加速: {speedup:.2f}x")
        
        results[f'shaping_k{k}'] = {
            'iterations': iterations_shaping,
            'time': time_shaping,
            'convergence': convergence_shaping,
            'theta_cont': solver_shaping.theta_cont,
            'theta_star': solver_shaping.theta_star,
            'V': V_shaping,
            'speedup': speedup
        }
    
    # 3. 可视化对比
    print("\n" + "=" * 80)
    print("生成对比图表...")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 3.1 收敛曲线
    ax1 = axes[0, 0]
    ax1.semilogy(convergence_baseline, label='No Shaping', linewidth=2)
    for k in shaping_k_values:
        conv = results[f'shaping_k{k}']['convergence']
        ax1.semilogy(conv, label=f'Shaping (k={k})', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Max |V - V_old|')
    ax1.set_title('Convergence Speed Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 3.2 迭代次数对比
    ax2 = axes[0, 1]
    labels = ['No Shaping'] + [f'k={k}' for k in shaping_k_values]
    iterations = [results['baseline']['iterations']] + \
                 [results[f'shaping_k{k}']['iterations'] for k in shaping_k_values]
    colors = ['red'] + ['green'] * len(shaping_k_values)
    ax2.bar(labels, iterations, color=colors, alpha=0.7)
    ax2.set_ylabel('Iterations to Converge')
    ax2.set_title('Convergence Iterations')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3.3 Value Function 对比
    ax3 = axes[1, 0]
    ax3.plot(solver_baseline.U_grid, V_baseline, label='No Shaping', linewidth=2)
    for k in shaping_k_values:
        V = results[f'shaping_k{k}']['V']
        ax3.plot(solver_baseline.U_grid, V, label=f'Shaping (k={k})', 
                linewidth=2, alpha=0.7, linestyle='--')
    ax3.set_xlabel('Information Progress U')
    ax3.set_ylabel('Value Function V(U)')
    ax3.set_title('Value Function Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 3.4 阈值对比
    ax4 = axes[1, 1]
    theta_conts = [results['baseline']['theta_cont']] + \
                  [results[f'shaping_k{k}']['theta_cont'] for k in shaping_k_values]
    theta_stars = [results['baseline']['theta_star']] + \
                  [results[f'shaping_k{k}']['theta_star'] for k in shaping_k_values]
    
    x = np.arange(len(labels))
    width = 0.35
    ax4.bar(x - width/2, theta_conts, width, label='θ_cont', alpha=0.7)
    ax4.bar(x + width/2, theta_stars, width, label='θ*', alpha=0.7)
    ax4.set_ylabel('Threshold Value')
    ax4.set_title('Optimal Thresholds Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存图表
    os.makedirs('figs', exist_ok=True)
    output_path = 'figs/phase2_2_reward_shaping_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 图表已保存: {output_path}")
    
    # 4. 总结
    print("\n" + "=" * 80)
    print("📊 Phase 2.2 验证总结")
    print("=" * 80)
    
    print("\n1. 收敛速度对比:")
    print(f"   基线（无shaping）: {iterations_baseline} 迭代")
    for k in shaping_k_values:
        iters = results[f'shaping_k{k}']['iterations']
        speedup = results[f'shaping_k{k}']['speedup']
        print(f"   Shaping (k={k}):   {iters} 迭代 ({speedup:.2f}x 加速)")
    
    print("\n2. 阈值一致性:")
    print(f"   基线: θ_cont={results['baseline']['theta_cont']:.4f}, "
          f"θ*={results['baseline']['theta_star']:.4f}")
    for k in shaping_k_values:
        tc = results[f'shaping_k{k}']['theta_cont']
        ts = results[f'shaping_k{k}']['theta_star']
        print(f"   k={k}:  θ_cont={tc:.4f}, θ*={ts:.4f}")
    
    # 检查阈值是否相同（应该相同，因为reward shaping保持最优策略不变）
    threshold_consistent = all(
        abs(results[f'shaping_k{k}']['theta_cont'] - results['baseline']['theta_cont']) < 1e-3
        for k in shaping_k_values
    )
    
    print("\n3. 理论验证:")
    if threshold_consistent:
        print("   ✅ 阈值保持一致 - Reward shaping 保持最优策略不变（理论正确）")
    else:
        print("   ⚠️  阈值有差异 - 可能需要检查实现")
    
    # 找出最佳k值
    best_k = min(shaping_k_values, 
                 key=lambda k: results[f'shaping_k{k}']['iterations'])
    best_speedup = results[f'shaping_k{best_k}']['speedup']
    
    print(f"\n4. 最佳配置:")
    print(f"   k = {best_k} (加速 {best_speedup:.2f}x)")
    
    print("\n" + "=" * 80)
    print("🎉 Phase 2.2 测试完成！")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    test_reward_shaping()
