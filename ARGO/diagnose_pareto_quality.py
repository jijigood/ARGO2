#!/usr/bin/env python3
"""
快速诊断：检查Pareto曲线质量
验证是否满足：
1. 至少10个不同的操作点
2. Quality范围: [0.55, 1.0] (不是[0, 0.8, 0.97, 1.0])
3. Cost范围: [1.5, 4.5] 平滑过渡
4. Retrieval次数: [0, 1, 2, 3, 4, 5] 分布在不同μ
"""

import yaml
import numpy as np
import sys
import os

# 添加ARGO_MDP路径（相对于ARGO2目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # ARGO2目录
sys.path.insert(0, os.path.join(parent_dir, 'ARGO_MDP', 'src'))

from mdp_solver import MDPSolver

# 加载配置
with open('configs/multi_gpu.yaml', 'r') as f:
    config = yaml.safe_load(f)

mdp_config = config['mdp']

# 提取参数
c_r = mdp_config['c_r']
c_p = mdp_config['c_p']
delta_r = mdp_config['delta_r']
delta_p = mdp_config['delta_p']
U_max = mdp_config['U_max']
quality_k = mdp_config['quality_k']

print("=" * 80)
print("Pareto曲线质量诊断")
print("=" * 80)
print(f"\n当前参数:")
print(f"  c_r = {c_r}")
print(f"  c_p = {c_p}")
print(f"  delta_r = {delta_r}")
print(f"  delta_p = {delta_p}")
print(f"  quality_k = {quality_k}")
print(f"  U_max = {U_max}")

# 计算关键比率
reward_per_retrieve = quality_k * delta_r
reward_cost_ratio = reward_per_retrieve / c_r
print(f"\n关键比率:")
print(f"  reward_per_retrieve = quality_k × delta_r = {quality_k} × {delta_r} = {reward_per_retrieve:.3f}")
print(f"  reward/cost ratio = {reward_per_retrieve:.3f} / {c_r} = {reward_cost_ratio:.2f}")

# 扫描μ值 (更密集，30个点)
mu_values = np.linspace(0, 3, 30)
results = []

print(f"\n扫描30个μ点 (μ∈[0,3])...")
print("-" * 80)

for mu in mu_values:
    # 创建临时配置
    temp_config = {
        'mdp': {
            'U_max': U_max,
            'delta_r': delta_r,
            'delta_p': delta_p,
            'p_s': mdp_config['p_s'],
            'c_r': c_r,
            'c_p': c_p,
            'mu': mu,
            'gamma': mdp_config['gamma'],
            'U_grid_size': mdp_config['grid_size']
        },
        'quality': {
            'mode': mdp_config['quality_function'],
            'k': quality_k
        },
        'solver': {
            'max_iterations': 1000,
            'convergence_threshold': 1e-6,
            'verbose': False
        },
        'reward_shaping': mdp_config.get('reward_shaping', {'enabled': False, 'k': 1.0})
    }
    
    solver = MDPSolver(temp_config)
    
    theta_cont, theta_star = solver.compute_thresholds()
    
    # 模拟：计算平均质量和成本
    # 假设：从U=0开始，达到theta_star时停止
    # 简化模型：总是成功检索
    U = 0.0
    cost = 0.0
    retrievals = 0
    steps = 0
    max_steps = 20
    
    while U < theta_star and steps < max_steps:
        if U < theta_cont:
            # Retrieve
            U = min(U + delta_r, U_max)
            cost += c_r
            retrievals += 1
        else:
            # Reason
            U = min(U + delta_p, U_max)
            cost += c_p
        steps += 1
    
    quality = U / U_max
    
    results.append({
        'mu': mu,
        'theta_cont': theta_cont,
        'theta_star': theta_star,
        'quality': quality,
        'cost': cost,
        'retrievals': retrievals
    })

# 分析结果
qualities = [r['quality'] for r in results]
costs = [r['cost'] for r in results]
retrievals_list = [r['retrievals'] for r in results]
theta_stars = [r['theta_star'] for r in results]

# 统计不同的操作点
unique_qualities = len(set(np.round(qualities, 2)))
unique_costs = len(set(np.round(costs, 2)))
unique_retrievals = len(set(retrievals_list))

print("\n" + "=" * 80)
print("诊断结果")
print("=" * 80)

# 检查1: 至少10个不同的操作点
print(f"\n✓ 检查1: 操作点多样性")
print(f"  不同的Quality值: {unique_qualities} (目标: ≥10)")
print(f"  不同的Cost值: {unique_costs} (目标: ≥10)")
print(f"  不同的Retrieval次数: {unique_retrievals}")
if unique_qualities >= 10:
    print(f"  ✅ 通过: {unique_qualities}个不同的quality值")
else:
    print(f"  ❌ 未通过: 只有{unique_qualities}个不同的quality值 (需要≥10)")

# 检查2: Quality范围
quality_min = min(qualities)
quality_max = max(qualities)
print(f"\n✓ 检查2: Quality范围")
print(f"  实际范围: [{quality_min:.2f}, {quality_max:.2f}]")
print(f"  目标范围: [0.55, 1.0]")
if quality_min <= 0.60 and quality_max >= 0.95:
    print(f"  ✅ 通过: 覆盖了大部分目标范围")
else:
    print(f"  ❌ 未通过: 范围不足")
    if quality_min > 0.60:
        print(f"     问题: 最小quality={quality_min:.2f}太高 (应该<0.60)")
    if quality_max < 0.95:
        print(f"     问题: 最大quality={quality_max:.2f}太低 (应该>0.95)")

# 检查3: Cost范围
cost_min = min(costs)
cost_max = max(costs)
print(f"\n✓ 检查3: Cost范围")
print(f"  实际范围: [{cost_min:.2f}, {cost_max:.2f}]")
print(f"  目标范围: [1.5, 4.5]")
if cost_min >= 1.0 and cost_max >= 3.5:
    print(f"  ✅ 通过: 成本范围合理")
else:
    print(f"  ⚠️  警告: 成本范围可能需要调整")

# 检查4: Retrieval分布
retrieval_counts = {}
for r in retrievals_list:
    retrieval_counts[r] = retrieval_counts.get(r, 0) + 1

print(f"\n✓ 检查4: Retrieval次数分布")
print(f"  观察到的次数: {sorted(set(retrievals_list))}")
print(f"  目标: [0, 1, 2, 3, 4, 5]")
print(f"  分布:")
for count in sorted(retrieval_counts.keys()):
    pct = 100 * retrieval_counts[count] / len(retrievals_list)
    bar = '█' * int(pct / 2)
    print(f"    {count}次: {retrieval_counts[count]:2d} ({pct:4.1f}%) {bar}")

if len(retrieval_counts) >= 4:
    print(f"  ✅ 通过: {len(retrieval_counts)}种不同的retrieval次数")
else:
    print(f"  ❌ 未通过: 只有{len(retrieval_counts)}种不同的retrieval次数")

# 检查5: 平滑过渡
print(f"\n✓ 检查5: 平滑过渡检查")
quality_jumps = []
for i in range(1, len(qualities)):
    jump = abs(qualities[i] - qualities[i-1])
    if jump > 0.01:  # 只记录明显的跳跃
        quality_jumps.append(jump)

if quality_jumps:
    avg_jump = np.mean(quality_jumps)
    max_jump = max(quality_jumps)
    print(f"  平均跳跃: {avg_jump:.3f}")
    print(f"  最大跳跃: {max_jump:.3f}")
    if max_jump < 0.30:
        print(f"  ✅ 通过: 过渡平滑 (最大跳跃<0.30)")
    else:
        print(f"  ❌ 未通过: 存在陡峭跳跃 (最大跳跃={max_jump:.3f})")
else:
    print(f"  ⚠️  警告: 几乎没有quality变化")

# 详细数据表
print("\n" + "=" * 80)
print("详细Pareto点 (前15个)")
print("=" * 80)
print(f"{'μ':>6} {'θ_cont':>8} {'θ*':>8} {'Quality':>8} {'Cost':>8} {'Retr':>5}")
print("-" * 80)
for i, r in enumerate(results[:15]):
    print(f"{r['mu']:6.2f} {r['theta_cont']:8.3f} {r['theta_star']:8.3f} "
          f"{r['quality']:8.3f} {r['cost']:8.3f} {r['retrievals']:5d}")

# 关键过渡点
print("\n" + "=" * 80)
print("关键过渡点")
print("=" * 80)

# 找到θ*从高到低的转折点
transition_indices = []
for i in range(1, len(theta_stars)):
    if abs(theta_stars[i] - theta_stars[i-1]) > 0.05:
        transition_indices.append(i)

if transition_indices:
    print(f"检测到{len(transition_indices)}个主要转折点:")
    for idx in transition_indices[:5]:  # 只显示前5个
        r = results[idx]
        print(f"  μ={r['mu']:.2f}: θ*={r['theta_star']:.3f}, Quality={r['quality']:.3f}, "
              f"Cost={r['cost']:.2f}, Retrievals={r['retrievals']}")
else:
    print("未检测到明显的转折点 (平滑过渡)")

# 总体评估
print("\n" + "=" * 80)
print("总体评估")
print("=" * 80)

score = 0
total = 5

if unique_qualities >= 10: score += 1
if quality_min <= 0.60 and quality_max >= 0.95: score += 1
if cost_min >= 1.0 and cost_max >= 3.5: score += 1
if len(retrieval_counts) >= 4: score += 1
if quality_jumps and max(quality_jumps) < 0.30: score += 1

print(f"通过检查: {score}/{total}")
if score == total:
    print("✅ 所有检查通过！参数配置良好，可以运行实验。")
elif score >= 3:
    print("⚠️  大部分检查通过，参数配置基本可用，但可能需要微调。")
else:
    print("❌ 多项检查未通过，建议调整参数后再运行实验。")

# 参数建议
if score < total:
    print("\n参数调整建议:")
    if unique_qualities < 10:
        print("  - 增加μ扫描范围或增加grid_size以获得更多操作点")
    if quality_min > 0.60:
        print(f"  - 降低quality_k或增加c_r以允许更低的quality (当前最低={quality_min:.2f})")
    if quality_max < 0.95:
        print(f"  - 增加quality_k或降低c_r以允许更高的quality (当前最高={quality_max:.2f})")
    if len(retrieval_counts) < 4:
        print("  - 调整delta_r或c_r以产生更多样的retrieval策略")
    if quality_jumps and max(quality_jumps) >= 0.30:
        print("  - 降低成本参数或增加μ采样密度以获得更平滑的过渡")

print("\n" + "=" * 80)
