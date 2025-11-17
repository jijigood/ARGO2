#!/usr/bin/env python
"""
快速测试 Experiment 0 - 单个参数集验证
======================================

这个脚本运行一个快速的单参数集验证，用于测试代码是否正常工作。
完整实验请运行 run_exp0.py

运行时间: ~30秒
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../ARGO_MDP/src'))

import numpy as np
import matplotlib.pyplot as plt
from mdp_solver import MDPSolver
from scipy import stats

print("=" * 80)
print("快速测试: 阈值结构验证")
print("=" * 80)
print()

# 创建简单的 MDP 配置
config = {
    'mdp': {
        'U_max': 1.0,
        'delta_r': 0.25,
        'delta_p': 0.08,
        'p_s': 0.8,
        'c_r': 0.05,
        'c_p': 0.02,
        'mu': 1.0,
        'gamma': 0.95,
        'U_grid_size': 101  # 较小的网格用于快速测试
    },
    'quality': {
        'mode': 'sigmoid',
        'k': 10.0
    },
    'solver': {
        'max_iterations': 1000,
        'convergence_threshold': 1e-6,
        'verbose': True
    },
    'reward_shaping': {
        'enabled': False,
        'k': 1.0
    }
}

print("配置参数:")
print(f"  c_r = {config['mdp']['c_r']}, c_p = {config['mdp']['c_p']}")
print(f"  δ_r = {config['mdp']['delta_r']}, δ_p = {config['mdp']['delta_p']}")
print(f"  p_s = {config['mdp']['p_s']}")
print()

# 求解 MDP
print("求解 MDP...")
solver = MDPSolver(config)
result = solver.solve()

print()
print("=" * 80)
print("结果:")
print("=" * 80)
print(f"终止阈值 Θ* = {result['theta_star']:.4f}")
print(f"继续阈值 Θ_cont = {result['theta_cont']:.4f}")
print()

# 验证策略结构
print("验证策略结构...")
U_grid = result['U_grid']
Q = result['Q']

# 统计每个动作的使用
action_counts = {'Retrieve': 0, 'Reason': 0, 'Terminate': 0}
for idx, U in enumerate(U_grid):
    action = np.argmax(Q[idx, :])
    if action == 0:
        action_counts['Retrieve'] += 1
    elif action == 1:
        action_counts['Reason'] += 1
    else:
        action_counts['Terminate'] += 1

print(f"动作分布 (在 {len(U_grid)} 个状态中):")
print(f"  Retrieve:  {action_counts['Retrieve']:3d} ({action_counts['Retrieve']/len(U_grid)*100:.1f}%)")
print(f"  Reason:    {action_counts['Reason']:3d} ({action_counts['Reason']/len(U_grid)*100:.1f}%)")
print(f"  Terminate: {action_counts['Terminate']:3d} ({action_counts['Terminate']/len(U_grid)*100:.1f}%)")
print()

# 验证单调性
V = result['V']
is_monotonic = np.all(np.diff(V) >= -1e-6)
print(f"价值函数单调性: {'✓ PASS' if is_monotonic else '✗ FAIL'}")

# 统计检验
if is_monotonic:
    correlation, p_value = stats.spearmanr(U_grid, V)
    print(f"  Spearman ρ = {correlation:.6f} (p = {p_value:.4e})")
    if correlation > 0.99 and p_value < 0.01:
        print(f"  ✓ Statistical validation PASS")
    else:
        print(f"  ⚠ Statistical validation WARNING")

# 验证阈值范围
theta_cont = result['theta_cont']
theta_star = result['theta_star']
threshold_valid = (0 <= theta_cont <= 1.0 and 
                   0 <= theta_star <= 1.0 and 
                   theta_cont <= theta_star)
print(f"阈值范围验证: {'✓ PASS' if threshold_valid else '✗ FAIL'}")
if threshold_valid:
    print(f"  Θ_cont ∈ [0,1]: {theta_cont:.4f}")
    print(f"  Θ* ∈ [0,1]: {theta_star:.4f}")
    print(f"  Θ_cont ≤ Θ*: {theta_cont:.4f} ≤ {theta_star:.4f}")

# 验证优势函数
advantages = []
for idx in range(len(U_grid)):
    Q_cont = max(Q[idx, 0], Q[idx, 1])
    Q_term = Q[idx, 2]
    advantages.append(Q_cont - Q_term)

# 统计零交叉
sign_changes = 0
for i in range(len(advantages) - 1):
    if advantages[i] * advantages[i+1] < 0:
        sign_changes += 1

print(f"优势函数零交叉次数: {sign_changes} (期望: 1)")
print(f"单交叉性质: {'✓ PASS' if sign_changes == 1 else ('⚠ WARNING' if sign_changes == 0 else '✗ FAIL')}")
print()

# 创建简单的可视化
print("生成可视化...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('快速测试: 阈值结构验证', fontsize=14, fontweight='bold')

# 1. 最优策略
ax1 = axes[0, 0]
optimal_actions = [np.argmax(Q[idx, :]) for idx in range(len(U_grid))]
colors = ['blue' if a == 0 else 'green' if a == 1 else 'red' for a in optimal_actions]
ax1.scatter(U_grid, optimal_actions, c=colors, alpha=0.6, s=15)
ax1.axvline(x=result['theta_star'], color='red', linestyle='--', linewidth=2, label=f"Θ* = {result['theta_star']:.3f}")
if result['theta_cont'] > 0 and result['theta_cont'] < result['theta_star']:
    ax1.axvline(x=result['theta_cont'], color='blue', linestyle='--', linewidth=2, label=f"Θ_cont = {result['theta_cont']:.3f}")
ax1.set_xlabel('Progress U')
ax1.set_ylabel('Optimal Action')
ax1.set_yticks([0, 1, 2])
ax1.set_yticklabels(['Retrieve', 'Reason', 'Terminate'])
ax1.set_title('Optimal Policy π*(U)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Q 函数
ax2 = axes[0, 1]
ax2.plot(U_grid, Q[:, 0], 'b-', label='Q(U, Retrieve)', linewidth=2, alpha=0.7)
ax2.plot(U_grid, Q[:, 1], 'g-', label='Q(U, Reason)', linewidth=2, alpha=0.7)
ax2.plot(U_grid, Q[:, 2], 'r-', label='Q(U, Terminate)', linewidth=2, alpha=0.7)
ax2.set_xlabel('Progress U')
ax2.set_ylabel('Q-value')
ax2.set_title('Q-functions')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 价值函数
ax3 = axes[1, 0]
ax3.plot(U_grid, V, 'k-', linewidth=2.5)
ax3.set_xlabel('Progress U')
ax3.set_ylabel('V*(U)')
ax3.set_title('Optimal Value Function')
ax3.grid(True, alpha=0.3)
if is_monotonic:
    ax3.text(0.05, 0.95, '✓ Monotonic', transform=ax3.transAxes, 
             color='green', fontweight='bold', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# 4. 优势函数
ax4 = axes[1, 1]
ax4.plot(U_grid, advantages, 'm-', linewidth=2.5)
ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax4.set_xlabel('Progress U')
ax4.set_ylabel('A(U)')
ax4.set_title('Advantage Function')
ax4.grid(True, alpha=0.3)
if sign_changes == 1:
    ax4.text(0.05, 0.95, '✓ Single Crossing', transform=ax4.transAxes,
             color='green', fontweight='bold', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()

# 保存图像
output_file = 'figs/exp0_quick_test.png'
os.makedirs('figs', exist_ok=True)
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"✓ 保存图像: {output_file}")
plt.close()

print()
print("=" * 80)
print("快速测试完成!")
print("=" * 80)
print()

# 综合判断
all_pass = (is_monotonic and sign_changes == 1 and threshold_valid)
if correlation > 0.99 and p_value < 0.01:
    all_pass = all_pass and True

if all_pass:
    print("✓ 所有验证通过! 代码工作正常。")
    print()
    print("现在可以运行完整实验:")
    print("  python run_exp0.py")
    print()
else:
    print("⚠ 部分验证未通过，请检查参数设置。")
    if not is_monotonic:
        print("  - 价值函数单调性: FAIL")
    if sign_changes != 1:
        print(f"  - 优势函数单交叉: FAIL (发现 {sign_changes} 个交叉点)")
    if not threshold_valid:
        print("  - 阈值范围验证: FAIL")
    print()

print("=" * 80)
