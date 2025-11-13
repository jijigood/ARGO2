#!/usr/bin/env python
"""
快速测试: 实验1 (真实LLM)
- 使用 Qwen2.5-3B 模型 (更快)
- 5道Hard题
- 2个c_r点
- 8张GPU
"""

print("=" * 80)
print("实验1: 检索成本影响 (真实LLM) - 快速测试")
print("=" * 80)
print("模型: Qwen2.5-3B-Instruct")
print("参数: 5道Hard题, 2个c_r点, 8张GPU")
print("预计时间: ~5-10分钟")
print("=" * 80)
print()

from Exp_real_cost_impact import RealCostImpactExperiment

# 使用3B模型 + 8张GPU
experiment = RealCostImpactExperiment(
    llm_model_path="/data/user/huangxiaolin/ARGO/RAG_Models/models/qwen2.5-7b-instruct",  # 实际上您有7B模型
    embedding_model_path="/data/user/huangxiaolin/ARGO/models/all-MiniLM-L6-v2",
    chroma_db_path="/data/user/huangxiaolin/ARGO2/ARGO/Environments/chroma_store",
    n_test_questions=5,  # 5道Hard题
    difficulty="hard",
    seed=42,
    gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7]  # 使用全部8张GPU
)

# 运行实验  
results = experiment.run_experiment(
    c_r_min_multiplier=1.0,
    c_r_max_multiplier=5.0,
    n_steps=2  # 2个点: c_r=0.02, 0.10
)

# 保存结果
filepath = experiment.save_results()

# 绘图
experiment.plot_results()

print("\n" + "=" * 80)
print("✅ 实验1快速测试完成!")
print("=" * 80)
print(f"结果已保存: {filepath}")
print("图表保存在: figs/")
print()

# 显示关键结果
print("关键结果:")
for i, r in enumerate(results, 1):
    print(f"\n[{i}] c_r = {r['c_r']:.3f}")
    print(f"  ARGO:")
    print(f"    - 检索次数: {r['ARGO_retrievals']:.1f}")
    print(f"    - 准确率: {r['ARGO_accuracy']:.1%}")
    print(f"  Always-Retrieve:")
    print(f"    - 检索次数: {r['Always-Retrieve_retrievals']:.1f}")
    print(f"    - 准确率: {r['Always-Retrieve_accuracy']:.1%}")
