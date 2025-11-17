#!/usr/bin/env python
"""
运行实验0: 阈值结构验证
========================

这是第一个基础实验，用于验证 ARGO MDP 解的理论基础。

目标:
1. 验证 Theorem 1 的两级阈值结构
2. 确认阈值存在性和唯一性
3. 验证策略单调性: Retrieve → Reason → Terminate
4. 测试阈值对参数变化的自适应性

运行时间: ~2-3 分钟 (纯数值计算，无需 LLM)

输出:
- 4个参数集的策略结构可视化图
- 阈值敏感性分析图
- 验证结果摘要表格
"""

import sys
print("=" * 80)
print("实验0: 阈值结构验证 - 验证理论基础")
print("=" * 80)
print("这是 ARGO 的第一个实验，验证 MDP 解的理论正确性")
print("预计时间: ~2-3 分钟")
print("=" * 80)
print()

from Exp0_threshold_structure_validation import ThresholdStructureValidation

# 创建验证器
print("初始化阈值结构验证器...")
validator = ThresholdStructureValidation()

print("\n开始验证...")
print("-" * 80)

# 运行完整验证
results = validator.run_full_validation()

print("\n" + "=" * 80)
print("实验0 完成!")
print("=" * 80)
print()
print("生成的文件:")
print("  📊 Figures:")
print("     - figs/exp0_threshold_structure_0_baseline.png")
print("     - figs/exp0_threshold_structure_1_high_c_r.png")
print("     - figs/exp0_threshold_structure_2_low_p_s.png")
print("     - figs/exp0_threshold_structure_3_high_p_s.png")
print("     - figs/exp0_threshold_sensitivity.png")
print()
print("  📄 Results:")
print("     - results/exp0_threshold_validation/threshold_validation_summary.csv")
print("     - results/exp0_threshold_validation/threshold_sensitivity_analysis.csv")
print()
print("=" * 80)
print()
print("下一步:")
print("  1. 检查生成的图表，确认阈值结构清晰可见")
print("  2. 验证 V*(U) 是单调递增的")
print("  3. 确认优势函数 A(U) 只有一个零点")
print("  4. 这些结果将作为论文的 Figure 1 (理论验证)")
print()
print("如果验证通过，可以继续运行:")
print("  - Experiment 1: python run_exp1_full.py (检索成本影响)")
print("  - Experiment 2: python run_exp2_full.py (检索成功率影响)")
print("=" * 80)
