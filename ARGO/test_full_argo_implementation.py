#!/usr/bin/env python
"""
测试完整ARGO实现
==================
验证Decomposer, Retriever/Reasoner, History维护, Synthesizer的正确性
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Exp_3B_quick_validation import RealCostImpactExperiment

def test_full_argo():
    """测试完整ARGO系统的执行"""
    
    print("="*80)
    print("🧪 测试完整ARGO系统实现")
    print("="*80)
    
    # 初始化实验
    print("\n1️⃣ 初始化实验环境...")
    exp = RealCostImpactExperiment(test_mode='small')
    
    # 加载一个测试问题
    print("\n2️⃣ 加载测试问题...")
    test_question = exp.test_questions[0]  # 使用实验已加载的测试问题
    print(f"   问题: {test_question['question'][:100]}...")
    print(f"   正确答案: {test_question['correct_answer']}")
    
    # 求解MDP获取阈值
    print("\n3️⃣ 求解MDP获取阈值...")
    c_r = 0.05
    theta_cont, theta_star = exp.solve_mdp(c_r)
    print(f"   Θ_cont = {theta_cont:.4f}")
    print(f"   Θ* = {theta_star:.4f}")
    
    # 执行完整ARGO策略
    print("\n4️⃣ 执行完整ARGO策略...")
    print("   (包括 Decomposer → Retriever/Reasoner → History → Synthesizer)")
    
    result = exp.simulate_argo_policy(test_question, theta_cont, theta_star, c_r)
    
    print("\n5️⃣ 执行结果:")
    print(f"   总步数: {result['steps']}")
    print(f"   历史长度: {result.get('history_length', 0)}")
    print(f"   检索次数: {result['retrieval_count']}")
    print(f"   推理次数: {result['reason_count']}")
    print(f"   最终质量: {result['quality']:.4f}")
    print(f"   总成本: {result['cost']:.4f}")
    print(f"   答案正确: {result['correct']}")
    
    # 测试纯检索和纯推理
    print("\n6️⃣ 对比测试...")
    
    # Always-Retrieve
    print("\n   测试 Always-Retrieve:")
    ar_result = exp.simulate_always_retrieve_policy(test_question, c_r, theta_star)
    print(f"   检索次数: {ar_result['retrieval_count']}, 正确: {ar_result['correct']}")
    
    # Always-Reason
    print("\n   测试 Always-Reason:")
    arn_result = exp.simulate_always_reason_policy(test_question, theta_star)
    print(f"   推理次数: {arn_result['reason_count']}, 正确: {arn_result['correct']}")
    
    print("\n" + "="*80)
    print("✅ 测试完成！")
    print("="*80)
    
    print("\n📊 关键验证点:")
    print(f"   ✓ 是否维护了历史? {'✅ 是' if result.get('history_length', 0) > 0 else '❌ 否'}")
    print(f"   ✓ 检索和推理都有执行? {'✅ 是' if result['retrieval_count'] > 0 and result['reason_count'] > 0 else '⚠️ 只有一种'}")
    print(f"   ✓ 步数等于历史长度? {'✅ 是' if result['steps'] == result.get('history_length', 0) else '❌ 否'}")
    
    return result

if __name__ == '__main__':
    test_full_argo()
