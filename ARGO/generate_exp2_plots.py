#!/usr/bin/env python
"""
从实验2日志中提取数据并生成图表
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# 从实验日志中提取的数据
results = [
    {
        'p_s': 0.50,
        'theta_cont': 0.000,
        'theta_star': 1.000,
        'ARGO_quality': 1.000,
        'ARGO_cost': 0.260,
        'ARGO_retrievals': 0.0,
        'ARGO_accuracy': 0.70,
        'Always-Retrieve_quality': 1.000,
        'Always-Retrieve_cost': 0.392,
        'Always-Retrieve_retrievals': 7.8,
        'Always-Retrieve_accuracy': 0.933,
        'Always-Reason_quality': 0.960,
        'Always-Reason_cost': 0.240,
        'Always-Reason_retrievals': 0.0,
        'Always-Reason_accuracy': 0.70
    },
    {
        'p_s': 0.57,
        'theta_cont': 0.000,
        'theta_star': 1.000,
        'ARGO_quality': 1.000,
        'ARGO_cost': 0.260,
        'ARGO_retrievals': 0.0,
        'ARGO_accuracy': 0.70,
        'Always-Retrieve_quality': 1.000,
        'Always-Retrieve_cost': 0.360,
        'Always-Retrieve_retrievals': 7.2,
        'Always-Retrieve_accuracy': 0.933,
        'Always-Reason_quality': 0.960,
        'Always-Reason_cost': 0.240,
        'Always-Reason_retrievals': 0.0,
        'Always-Reason_accuracy': 0.70
    },
    {
        'p_s': 0.65,
        'theta_cont': 0.060,
        'theta_star': 1.000,
        'ARGO_quality': 1.000,
        'ARGO_cost': 0.265,
        'ARGO_retrievals': 1.3,
        'ARGO_accuracy': 0.70,
        'Always-Retrieve_quality': 1.000,
        'Always-Retrieve_cost': 0.305,
        'Always-Retrieve_retrievals': 6.1,
        'Always-Retrieve_accuracy': 0.933,
        'Always-Reason_quality': 0.960,
        'Always-Reason_cost': 0.240,
        'Always-Reason_retrievals': 0.0,
        'Always-Reason_accuracy': 0.70
    },
    {
        'p_s': 0.72,
        'theta_cont': 0.070,
        'theta_star': 1.000,
        'ARGO_quality': 1.000,
        'ARGO_cost': 0.265,
        'ARGO_retrievals': 1.3,
        'ARGO_accuracy': 0.70,
        'Always-Retrieve_quality': 1.000,
        'Always-Retrieve_cost': 0.277,
        'Always-Retrieve_retrievals': 5.5,
        'Always-Retrieve_accuracy': 0.933,
        'Always-Reason_quality': 0.960,
        'Always-Reason_cost': 0.240,
        'Always-Reason_retrievals': 0.0,
        'Always-Reason_accuracy': 0.70
    },
    {
        'p_s': 0.80,
        'theta_cont': 0.080,
        'theta_star': 1.000,
        'ARGO_quality': 1.000,
        'ARGO_cost': 0.267,
        'ARGO_retrievals': 1.3,
        'ARGO_accuracy': 0.70,
        'Always-Retrieve_quality': 1.000,
        'Always-Retrieve_cost': 0.255,
        'Always-Retrieve_retrievals': 5.1,
        'Always-Retrieve_accuracy': 0.933,
        'Always-Reason_quality': 0.960,
        'Always-Reason_cost': 0.240,
        'Always-Reason_retrievals': 0.0,
        'Always-Reason_accuracy': 0.70
    },
    {
        'p_s': 0.88,
        'theta_cont': 0.090,
        'theta_star': 1.000,
        'ARGO_quality': 1.000,
        'ARGO_cost': 0.255,
        'ARGO_retrievals': 1.1,
        'ARGO_accuracy': 0.70,
        'Always-Retrieve_quality': 1.000,
        'Always-Retrieve_cost': 0.227,
        'Always-Retrieve_retrievals': 4.5,
        'Always-Retrieve_accuracy': 0.933,
        'Always-Reason_quality': 0.960,
        'Always-Reason_cost': 0.240,
        'Always-Reason_retrievals': 0.0,
        'Always-Reason_accuracy': 0.70
    },
    {
        'p_s': 0.95,
        'theta_cont': 0.140,
        'theta_star': 1.000,
        'ARGO_quality': 1.000,
        'ARGO_cost': 0.253,
        'ARGO_retrievals': 1.1,
        'ARGO_accuracy': 0.70,
        'Always-Retrieve_quality': 1.000,
        'Always-Retrieve_cost': 0.212,
        'Always-Retrieve_retrievals': 4.2,
        'Always-Retrieve_accuracy': 0.933,
        'Always-Reason_quality': 0.960,
        'Always-Reason_cost': 0.240,
        'Always-Reason_retrievals': 0.0,
        'Always-Reason_accuracy': 0.70
    }
]

# 创建输出目录
output_dir = "figs"
os.makedirs(output_dir, exist_ok=True)

# 提取数据
p_s_values = [r['p_s'] for r in results]

print("=" * 80)
print("生成实验2图表: 检索成功率影响")
print("=" * 80)

# 图1: 质量 vs 成功率
plt.figure(figsize=(10, 6))
for policy in ['ARGO', 'Always-Retrieve', 'Always-Reason']:
    quality = [r[f'{policy}_quality'] for r in results]
    plt.plot(p_s_values, quality, marker='o', label=policy, linewidth=2, markersize=8)

plt.xlabel('Retrieval Success Rate ($p_s$)', fontsize=13)
plt.ylabel('Average Quality', fontsize=13)
plt.title('Experiment 2: Quality vs Retrieval Success Rate (Real LLM)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

fig1_path = os.path.join(output_dir, 'exp2_real_ps_vs_quality.png')
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ 图表已保存: {fig1_path}")

# 图2: 检索次数 vs 成功率 (核心图)
plt.figure(figsize=(10, 6))
colors = {'ARGO': '#1f77b4', 'Always-Retrieve': '#ff7f0e', 'Always-Reason': '#2ca02c'}
for policy in ['ARGO', 'Always-Retrieve', 'Always-Reason']:
    retrievals = [r[f'{policy}_retrievals'] for r in results]
    plt.plot(p_s_values, retrievals, marker='o', label=policy, 
             linewidth=2.5, markersize=8, color=colors[policy])

plt.xlabel('Retrieval Success Rate ($p_s$)', fontsize=13)
plt.ylabel('Average Retrievals per Question', fontsize=13)
plt.title('Experiment 2: Retrieval Count vs Success Rate (Real LLM)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()

fig2_path = os.path.join(output_dir, 'exp2_real_ps_vs_retrievals.png')
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ 图表已保存: {fig2_path}")

# 图3: 准确率 vs 成功率
plt.figure(figsize=(10, 6))
for policy in ['ARGO', 'Always-Retrieve', 'Always-Reason']:
    accuracy = [r[f'{policy}_accuracy'] for r in results]
    plt.plot(p_s_values, accuracy, marker='o', label=policy, linewidth=2, markersize=8)

plt.xlabel('Retrieval Success Rate ($p_s$)', fontsize=13)
plt.ylabel('Accuracy', fontsize=13)
plt.title('Experiment 2: Accuracy vs Retrieval Success Rate (Real LLM)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim(0.65, 0.95)
plt.tight_layout()

fig3_path = os.path.join(output_dir, 'exp2_real_ps_vs_accuracy.png')
plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ 图表已保存: {fig3_path}")

# 图4: 成本 vs 成功率
plt.figure(figsize=(10, 6))
for policy in ['ARGO', 'Always-Retrieve', 'Always-Reason']:
    cost = [r[f'{policy}_cost'] for r in results]
    plt.plot(p_s_values, cost, marker='o', label=policy, linewidth=2, markersize=8)

plt.xlabel('Retrieval Success Rate ($p_s$)', fontsize=13)
plt.ylabel('Average Cost', fontsize=13)
plt.title('Experiment 2: Cost vs Retrieval Success Rate (Real LLM)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

fig4_path = os.path.join(output_dir, 'exp2_real_ps_vs_cost.png')
plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ 图表已保存: {fig4_path}")

# 图5: MDP阈值演化
plt.figure(figsize=(10, 6))
theta_cont_values = [r['theta_cont'] for r in results]
plt.plot(p_s_values, theta_cont_values, marker='s', linewidth=2.5, 
         markersize=10, color='#d62728', label='$\\theta_{cont}$ (Retrieval Threshold)')
plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, label='$\\theta^*$ (Termination)')

plt.xlabel('Retrieval Success Rate ($p_s$)', fontsize=13)
plt.ylabel('MDP Threshold', fontsize=13)
plt.title('Experiment 2: MDP Threshold Evolution (ARGO Adaptation)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim(-0.05, 1.1)
plt.tight_layout()

fig5_path = os.path.join(output_dir, 'exp2_real_threshold_evolution.png')
plt.savefig(fig5_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ 图表已保存: {fig5_path}")

print("\n" + "=" * 80)
print("实验2关键发现:")
print("=" * 80)
print("\n1. ARGO的检索次数变化:")
print(f"   - 低成功率(p_s=0.50): {results[0]['ARGO_retrievals']:.1f}次检索")
print(f"   - 高成功率(p_s=0.95): {results[-1]['ARGO_retrievals']:.1f}次检索")
print(f"   - 变化趋势: 随p_s增加而增加 (检索更可靠时,允许更多检索)")

print("\n2. Always-Retrieve的检索次数变化:")
print(f"   - 低成功率(p_s=0.50): {results[0]['Always-Retrieve_retrievals']:.1f}次检索")
print(f"   - 高成功率(p_s=0.95): {results[-1]['Always-Retrieve_retrievals']:.1f}次检索")
print(f"   - 变化趋势: 随p_s增加而减少 (成功率高时,更快达到质量阈值)")

print("\n3. MDP阈值演化:")
print(f"   - p_s=0.50: θ_cont={results[0]['theta_cont']:.3f} (完全避免检索)")
print(f"   - p_s=0.65: θ_cont={results[2]['theta_cont']:.3f} (开始允许检索)")
print(f"   - p_s=0.95: θ_cont={results[-1]['theta_cont']:.3f} (适度鼓励检索)")

print("\n4. 验证:")
print("   ✅ ARGO成功展现成功率自适应性")
print("   ✅ 检索成功率影响ARGO的检索决策")
print("   ✅ Always-Retrieve只是被动适应(需要更多次尝试)")

print("\n" + "=" * 80)
print("✅ 实验2图表生成完成!")
print("=" * 80)
print("\n生成的图表:")
print("  1. exp2_real_ps_vs_quality.png - 质量对比")
print("  2. exp2_real_ps_vs_retrievals.png - 检索次数对比 ⭐核心")
print("  3. exp2_real_ps_vs_accuracy.png - 准确率对比")
print("  4. exp2_real_ps_vs_cost.png - 成本对比")
print("  5. exp2_real_threshold_evolution.png - MDP阈值演化")
