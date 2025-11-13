#!/usr/bin/env python
"""
完整实验2运行脚本: 检索成功率影响
========================================
验证ARGO如何根据期望成功率 p_s 自适应调整策略

实验设计:
- 问题池: 30道 Hard 难度问题
- p_s 扫描: 7个点 (0.50 → 0.95)
- 固定参数: c_r=0.05, c_p=0.02, delta_r=0.25, delta_p=0.08
- 对比策略: ARGO vs Always-Retrieve vs Always-Reason
- 模型: Qwen2.5-3B-Instruct (用户选择)
- GPU: 8张 RTX 3060

预期结果:
- ARGO应该随着 p_s 增加而增加检索次数
- Always-Retrieve应该保持恒定检索次数
- Always-Reason应该始终不检索

运行时间: ~40-50分钟
"""

import os
import sys

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Exp_real_success_impact import RealSuccessImpactExperiment


def main():
    """运行完整实验2"""
    
    print("=" * 80)
    print("完整实验2: 检索成功率影响 (真实LLM)")
    print("=" * 80)
    print("模型: Qwen2.5-3B-Instruct")
    print("参数: 30道Hard题, 7个p_s点, 8张GPU")
    print("策略: ARGO (动态) vs 3种基线 (固定)")
    print("预计时间: ~40-50分钟")
    print("=" * 80)
    print()
    
    # 实验配置
    config = {
        'llm_model_path': '/data/user/huangxiaolin/ARGO/RAG_Models/models/Qwen2.5-3B-Instruct',
        'embedding_model_path': '/data/user/huangxiaolin/ARGO/models/all-MiniLM-L6-v2',
        'chroma_db_path': '/data/user/huangxiaolin/ARGO2/ARGO/Environments/chroma_store',
        'difficulty': 'hard',  # 注意是小写
        'n_test_questions': 30,
        'gpu_ids': [0, 1, 2, 3, 4, 5, 6, 7],  # 8张GPU
        'seed': 42
    }
    
    # 实验参数: p_s 从低到高扫描
    p_s_min = 0.50
    p_s_max = 0.95
    n_steps = 7
    
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
    
    # 初始化实验
    print("初始化实验环境...")
    exp = RealSuccessImpactExperiment(
        llm_model_path=config['llm_model_path'],
        embedding_model_path=config['embedding_model_path'],
        chroma_db_path=config['chroma_db_path'],
        difficulty=config['difficulty'],
        n_test_questions=config['n_test_questions'],
        gpu_ids=config['gpu_ids'],
        seed=config['seed']
    )
    
    print("\n" + "=" * 80)
    print("开始实验 - 检索成功率影响")
    print("=" * 80)
    print(f"p_s范围: {p_s_min:.2f} ~ {p_s_max:.2f} (扫描 {n_steps} 个点)")
    print(f"问题数量: {config['n_test_questions']}")
    print(f"总评估次数: {n_steps} × 3策略 × {config['n_test_questions']}题 = {n_steps * 3 * config['n_test_questions']}")
    print("=" * 80)
    print()
    
    # 运行实验
    results = exp.run_experiment(
        p_s_min=p_s_min,
        p_s_max=p_s_max,
        n_steps=n_steps
    )
    
    print("\n" + "=" * 80)
    print("✅ 完整实验2完成!")
    print("=" * 80)
    print(f"结果已保存: {results['save_path']}")
    print(f"图表已保存: figs/exp2_real_*.png")
    print()
    print("查看图表以可视化:")
    print("  1. figs/exp2_real_success_vs_quality.png - 质量对比")
    print("  2. figs/exp2_real_success_vs_retrievals.png - 检索次数对比 ⭐核心")
    print("  3. figs/exp2_real_success_vs_accuracy.png - 准确率对比")
    

if __name__ == "__main__":
    main()
