#!/usr/bin/env python3
"""
深入分析 Reward Shaping 对 MDP 的影响
检查理论与实践的一致性
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ARGO_MDP', 'src')))
from mdp_solver import MDPSolver


def analyze_shaping_effect():
    """分析 reward shaping 的影响"""
    
    print("=" * 80)
    print("Reward Shaping 理论分析")
    print("=" * 80)
    
    # 配置
    config_base = {
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
            'enabled': False,
            'k': 1.0
        },
        'solver': {
            'max_iterations': 1000,
            'convergence_threshold': 1e-6,
            'verbose': False
        }
    }
    
    # 测试不同状态下的 Q 值
    test_U = 0.0  # 初始状态
    
    print(f"\n分析状态 U = {test_U}")
    print("-" * 80)
    
    # 无 shaping
    print("\n【无 Reward Shaping】")
    solver_no_shaping = MDPSolver(config_base)
    
    # 计算 Q值
    actions = ["Retrieve", "Reason", "Terminate"]
    
    for action_idx, action_name in enumerate(actions):
        if action_idx == 2:  # Terminate
            q_val = solver_no_shaping.quality_function(test_U)
            print(f"  Q({test_U}, {action_name}) = {q_val:.4f}")
        else:
            next_states, probs = solver_no_shaping.transition(test_U, action_idx)
            immediate_reward = solver_no_shaping.reward(test_U, action_idx)
            
            print(f"  {action_name}:")
            print(f"    即时奖励: {immediate_reward:.4f}")
            print(f"    下一状态: {next_states} (概率: {probs})")
            print(f"    δU: {next_states - test_U}")
    
    # 有 shaping (k=1.0)
    print("\n【有 Reward Shaping (k=1.0)】")
    config_with_shaping = config_base.copy()
    config_with_shaping['reward_shaping'] = {'enabled': True, 'k': 1.0}
    solver_with_shaping = MDPSolver(config_with_shaping)
    
    for action_idx, action_name in enumerate(actions):
        if action_idx == 2:  # Terminate
            q_val = solver_with_shaping.quality_function(test_U)
            print(f"  Q({test_U}, {action_name}) = {q_val:.4f}")
        else:
            next_states, probs = solver_with_shaping.transition(test_U, action_idx)
            immediate_reward = solver_with_shaping.reward(test_U, action_idx)
            
            # 计算 shaping reward
            shaping_rewards = [solver_with_shaping.shaping_reward(test_U, u_next) 
                             for u_next in next_states]
            expected_shaping = sum(p * s for p, s in zip(probs, shaping_rewards))
            
            print(f"  {action_name}:")
            print(f"    即时奖励: {immediate_reward:.4f}")
            print(f"    Shaping奖励: {shaping_rewards} (期望: {expected_shaping:.4f})")
            print(f"    总奖励: {immediate_reward + expected_shaping:.4f}")
            print(f"    下一状态: {next_states} (概率: {probs})")
    
    # 理论分析
    print("\n" + "=" * 80)
    print("理论分析")
    print("=" * 80)
    
    print("\n1. Retrieve 动作:")
    print("   成功 (p=0.8): U' = 0.25")
    print("   失败 (p=0.2): U' = 0.00")
    print("   ")
    print("   无shaping:")
    print("     R = -0.05")
    print("   ")
    print("   有shaping (k=1.0, Φ(U)=U):")
    print("     F_success = 0.98 * 0.25 - 1.0 * 0.0 = 0.245")
    print("     F_fail = 0.98 * 0.0 - 1.0 * 0.0 = 0.0")
    print("     E[F] = 0.8 * 0.245 + 0.2 * 0.0 = 0.196")
    print("     R' = R + E[F] = -0.05 + 0.196 = 0.146")
    
    print("\n2. Reason 动作:")
    print("   确定性: U' = 0.08")
    print("   ")
    print("   无shaping:")
    print("     R = -0.02")
    print("   ")
    print("   有shaping (k=1.0):")
    print("     F = 0.98 * 0.08 - 1.0 * 0.0 = 0.0784")
    print("     R' = R + F = -0.02 + 0.0784 = 0.0584")
    
    print("\n3. 关键洞察:")
    print("   Shaping使得Retrieve的有效奖励从 -0.05 变为 +0.146")
    print("   Shaping使得Reason的有效奖励从 -0.02 变为 +0.0584")
    print("   ")
    print("   这改变了动作的相对吸引力:")
    print("   - 无shaping: Reason更优 (-0.02 > -0.05)")
    print("   - 有shaping: Retrieve更优 (0.146 > 0.0584)")
    print("   ")
    print("   ⚠️  这解释了为什么阈值会改变!")
    print("   虽然理论上potential-based shaping应该保持最优策略，")
    print("   但这里的Φ(U)=kU与状态转移Δr, Δp交互，")
    print("   实际上改变了不同动作的相对价值。")
    
    print("\n4. 理论验证:")
    print("   标准的potential-based shaping理论假设:")
    print("   - F(s,a,s') = γΦ(s') - Φ(s)")
    print("   - 这保证了Q*(s,a)的最优动作不变")
    print("   ")
    print("   但在我们的情况下:")
    print("   - Φ(U) = kU 是线性的")
    print("   - 不同动作导致不同的ΔU (δ_r=0.25 vs δ_p=0.08)")
    print("   - 因此shaping会偏向产生更大ΔU的动作")
    print("   - 这实际上改变了策略！")
    
    print("\n5. 结论:")
    print("   在我们的ARGO MDP中，使用Φ(U)=kU的shaping会:")
    print("   ✅ 可能加速收敛 (虽然当前测试中没有明显效果)")
    print("   ⚠️  改变最优策略 (偏向于高信息增益的动作)")
    print("   ")
    print("   这可能是:")
    print("   - 🔴 Bug: 如果我们希望保持策略不变")
    print("   - 🟢 Feature: 如果我们希望鼓励高信息增益的动作")
    print("   ")
    print("   建议: 如果要保持策略不变，应该使用:")
    print("   - Φ(U) = σ(U) (质量函数本身)")
    print("   - 或者 disabled reward shaping")


if __name__ == "__main__":
    analyze_shaping_effect()
