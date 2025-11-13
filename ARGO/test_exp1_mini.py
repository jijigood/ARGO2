#!/usr/bin/env python
"""
实验1迷你测试 - 5道题快速验证
"""

print("=" * 80)
print("实验1: 检索成本影响 (真实LLM) - 迷你测试")
print("=" * 80)
print("参数: 5道Hard题, 2个c_r点, 2张GPU")
print("预计时间: ~5分钟")
print("=" * 80)
print()

from Exp_real_cost_impact import RealCostImpactExperiment

experiment = RealCostImpactExperiment(
    llm_model_path="/data/user/huangxiaolin/ARGO/RAG_Models/models/Qwen2.5-14B-Instruct",
    embedding_model_path="/data/user/huangxiaolin/ARGO/models/all-MiniLM-L6-v2",
    chroma_db_path="/data/user/huangxiaolin/ARGO2/ARGO/Environments/chroma_store",
    n_test_questions=5,  # 只测试5道题
    difficulty="hard",
    seed=42,
    gpu_ids=[0, 1]  # 使用2张GPU
)

print("\n开始运行实验...")
print("每道题大约需要1分钟\n")

# 运行实验  
results = experiment.run_experiment(
    c_r_min_multiplier=1.0,
    c_r_max_multiplier=3.0,
    n_steps=2  # 只测试2个点: c_r=0.02, 0.06
)

# 保存结果
experiment.save_results()

# 绘图
experiment.plot_results()

print("\n" + "=" * 80)
print("✅ 实验1迷你测试完成!")
print("=" * 80)
print("\n查看结果:")
print("  - JSON数据: draw_figs/data/")
print("  - 图表: figs/")
print("\n验证项目:")
print("  ✓ ChromaDB真实检索 (436,279个ORAN文档)")
print("  ✓ Qwen2.5-14B真实LLM推理")
print("  ✓ all-MiniLM-L6-v2嵌入模型")
print("  ✓ Hard难度问题")
print("  ✓ 多GPU并行")
print()
