#!/usr/bin/env python
"""
快速测试实验3 - 使用Medium难度和少量问题
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Exp_real_pareto_frontier import RealParetoFrontierExperiment


def main():
    print("=" * 80)
    print("快速测试实验3: 10道Medium题, 5个μ点")
    print("=" * 80)
    
    # 实验配置
    config = {
        'llm_model_path': '/data/user/huangxiaolin/ARGO/RAG_Models/models/Qwen2.5-3B-Instruct',
        'embedding_model_path': '/data/user/huangxiaolin/ARGO/models/all-MiniLM-L6-v2',
        'chroma_db_path': '/data/user/huangxiaolin/ARGO2/ARGO/Environments/chroma_store',
        'difficulty': 'medium',  # 使用Medium难度
        'n_test_questions': 10,  # 只用10道题测试
        'gpu_ids': [0, 1, 2, 3, 4, 5, 6, 7],
        'seed': 42
    }
    
    # 实验参数 (FIX v2: 聚焦在过渡区0-5)
    mu_min = 0.0
    mu_max = 5.0  # 聚焦在θ*变化的区间
    n_mu_steps = 5  # 只用5个μ点测试
    
    # 初始化实验
    print("\n初始化实验环境...")
    exp = RealParetoFrontierExperiment(
        llm_model_path=config['llm_model_path'],
        embedding_model_path=config['embedding_model_path'],
        chroma_db_path=config['chroma_db_path'],
        difficulty=config['difficulty'],
        n_test_questions=config['n_test_questions'],
        gpu_ids=config['gpu_ids'],
        seed=config['seed']
    )
    
    # 运行实验
    print("\n开始实验...")
    results = exp.run_experiment(
        mu_min=mu_min,
        mu_max=mu_max,
        n_mu_steps=n_mu_steps
    )
    
    # 保存结果
    save_path = exp.save_results()
    
    # 绘制图表
    print("\n生成可视化图表...")
    fig_path = exp.plot_pareto_frontier()
    threshold_fig_path = exp.plot_threshold_evolution()
    dashboard_fig_path = exp.plot_comprehensive_dashboard()
    latency_fig_path = exp.plot_latency_analysis()
    
    print("\n" + "=" * 80)
    print("✅ 快速测试完成!")
    print("=" * 80)
    print(f"结果已保存: {save_path}")
    print(f"\n生成的图表:")
    print(f"  1. Pareto边界图: {fig_path}")
    print(f"  2. 阈值演化图: {threshold_fig_path}")
    print(f"  3. 综合仪表板: {dashboard_fig_path}")
    print(f"  4. 延迟分析图: {latency_fig_path}")


if __name__ == "__main__":
    main()
