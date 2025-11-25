#!/usr/bin/env python
"""
完整实验1: 检索成本影响 (真实LLM)
====================================
目标: 验证ARGO在不同检索成本下的自适应行为

实验设置:
- 模型: Qwen2.5-3B-Instruct (更快)
- 问题: 30道Hard题
- c_r扫描: 0.02 → 0.20 (7个点)
- GPU: 8张
- 策略对比: 4种
  1. ARGO: 动态选择动作 (Retrieve/Reason/Terminate)
  2. Always-Retrieve: 固定检索直到达到质量阈值
  3. Always-Reason: 固定推理直到达到质量阈值  
  4. Random: 随机50-50选择Retrieve或Reason

关键验证点:
- ARGO在c_r增加时应减少检索次数
- 其他策略保持固定行为
- ARGO没有预设步数限制,根据MDP阈值动态决策
"""

import sys
print("=" * 80)
print("完整实验1: 检索成本影响 (真实LLM)")
print("=" * 80)
print("模型: Qwen2.5-3B-Instruct")
print("参数: 30道Hard题, 7个c_r点, 8张GPU")
print("策略: ARGO (动态) vs 3种基线 (固定)")
print("预计时间: ~40-50分钟")
print("=" * 80)
print()

from Exp_real_cost_impact import RealCostImpactExperiment

# 创建实验
experiment = RealCostImpactExperiment(
    llm_model_path="/data/user/huangxiaolin/ARGO/RAG_Models/models/Qwen2.5-3B-Instruct",
    embedding_model_path="/data/user/huangxiaolin/ARGO/models/all-MiniLM-L6-v2",
    chroma_db_path="/data/user/huangxiaolin/ARGO2/ARGO/Environments/chroma_store",
    n_test_questions=30,  # 30道Hard题
    difficulty="hard",
    seed=42,
    gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7]  # 使用全部8张GPU
)

print("\n策略说明:")
print("-" * 80)
print("1. ARGO策略 (动态自适应):")
print("   - 根据MDP阈值 θ_cont 和 θ* 动态决策")
print("   - U < θ_cont: 执行Retrieve")
print("   - θ_cont ≤ U < θ*: 执行Reason")
print("   - U ≥ θ*: Terminate")
print("   - 没有固定步数限制!")
print()
print("2. Always-Retrieve (固定检索):")
print("   - 固定执行Retrieve直到 U ≥ θ* (0.9)")
print("   - 不会切换到Reason")
print()
print("3. Always-Reason (固定推理):")
print("   - 固定执行Reason直到 U ≥ θ* (0.9)")
print("   - 不会执行Retrieve")
print()
print("4. Random (随机策略):")
print("   - 50%概率Retrieve, 50%概率Reason")
print("   - 直到 U ≥ θ*")
print("-" * 80)
print()

# 运行完整实验
# c_r从1x c_p到10x c_p, 扫描7个点
results = experiment.run_experiment(
    c_r_min_multiplier=1.0,    # c_r_min = 0.02 (等于c_p)
    c_r_max_multiplier=10.0,   # c_r_max = 0.20 (10倍c_p)
    n_steps=7                  # 7个采样点
)

# 扫描的c_r值: [0.02, 0.05, 0.08, 0.11, 0.14, 0.17, 0.20]

# 保存结果
filepath = experiment.save_results()

# 绘图
experiment.plot_results()

# 详细分析结果
print("\n" + "=" * 80)
print("实验结果详细分析")
print("=" * 80)
print()

print("策略对比 (各c_r值下的平均检索次数):")
print("-" * 80)
print(f"{'c_r':>6s}  {'ARGO':>8s}  {'Always-R':>8s}  {'Always-P':>8s}  {'Random':>8s}  {'ARGO优势':>10s}")
print("-" * 80)

for r in results:
    c_r = r['c_r']
    argo_ret = r['ARGO_retrievals']
    alw_ret = r['Always-Retrieve_retrievals']
    alw_rea = r['Always-Reason_retrievals']
    rand_ret = r.get('Random_retrievals', 0)
    
    # 计算ARGO相比Always-Retrieve的优势
    if alw_ret > 0:
        advantage = (alw_ret - argo_ret) / alw_ret * 100
    else:
        advantage = 0
    
    print(f"{c_r:6.3f}  {argo_ret:8.1f}  {alw_ret:8.1f}  {alw_rea:8.1f}  {rand_ret:8.1f}  {advantage:9.1f}%")

print()
print("准确率对比:")
print("-" * 80)
print(f"{'c_r':>6s}  {'ARGO':>8s}  {'Always-R':>8s}  {'Always-P':>8s}")
print("-" * 80)

for r in results:
    c_r = r['c_r']
    argo_acc = r['ARGO_accuracy'] * 100
    alw_r_acc = r['Always-Retrieve_accuracy'] * 100
    alw_p_acc = r['Always-Reason_accuracy'] * 100
    
    print(f"{c_r:6.3f}  {argo_acc:7.1f}%  {alw_r_acc:7.1f}%  {alw_p_acc:7.1f}%")

print()
print("MDP阈值演化 (ARGO如何自适应):")
print("-" * 80)
print(f"{'c_r':>6s}  {'θ_cont':>8s}  {'θ*':>8s}  {'解读':>40s}")
print("-" * 80)

for r in results:
    c_r = r['c_r']
    theta_cont = r['theta_cont']
    theta_star = r['theta_star']
    
    if theta_cont < 0.3:
        interpretation = "高成本→避免检索,优先推理"
    elif theta_cont < 0.7:
        interpretation = "中等成本→平衡检索与推理"
    else:
        interpretation = "低成本→鼓励检索"
    
    print(f"{c_r:6.3f}  {theta_cont:8.3f}  {theta_star:8.3f}  {interpretation:>40s}")

print()
print("=" * 80)
print("关键发现:")
print("=" * 80)

# 计算检索次数的变化
first_argo_ret = results[0]['ARGO_retrievals']
last_argo_ret = results[-1]['ARGO_retrievals']
reduction = first_argo_ret - last_argo_ret

print(f"1. ARGO检索次数变化:")
print(f"   - 低成本(c_r={results[0]['c_r']:.3f}): {first_argo_ret:.1f}次检索")
print(f"   - 高成本(c_r={results[-1]['c_r']:.3f}): {last_argo_ret:.1f}次检索")
print(f"   - 减少: {reduction:.1f}次 ({reduction/first_argo_ret*100:.1f}%)")
print()

# Always-Retrieve应该保持恒定
first_alw_ret = results[0]['Always-Retrieve_retrievals']
last_alw_ret = results[-1]['Always-Retrieve_retrievals']
print(f"2. Always-Retrieve检索次数 (应保持恒定):")
print(f"   - 低成本: {first_alw_ret:.1f}次")
print(f"   - 高成本: {last_alw_ret:.1f}次")
print(f"   - 变化: {abs(last_alw_ret - first_alw_ret):.1f}次 (接近0说明正确)")
print()

print(f"3. 验证:")
if reduction > 0:
    print(f"   ✅ ARGO成功展现成本自适应性")
else:
    print(f"   ❌ ARGO未展现预期的自适应行为")

if abs(last_alw_ret - first_alw_ret) < 1.0:
    print(f"   ✅ Always-Retrieve保持固定行为")
else:
    print(f"   ⚠ Always-Retrieve有意外变化")

print()
print("=" * 80)
print("✅ 完整实验1完成!")
print("=" * 80)
print(f"结果已保存: {filepath}")
print(f"图表已保存: figs/exp1_real_*.png")
print()
print("查看图表以可视化:")
print("  1. figs/exp1_real_cost_vs_quality.png - 质量对比")
print("  2. figs/exp1_real_cost_vs_retrievals.png - 检索次数对比 ⭐核心")
print("  3. figs/exp1_real_cost_vs_accuracy.png - 准确率对比")
print()
