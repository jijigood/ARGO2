"""
Phase 1 验证脚本
快速测试History追踪和参数修正

验证内容:
1. History完整性（子查询、响应、中间答案）
2. 成本参数正确性（c_r=0.05, c_p=0.02）
3. 推理链可追踪性
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compare_mdp_vs_fixed_multigpu import run_comparison

def validate_phase1():
    """验证Phase 1改进"""
    
    print("=" * 80)
    print("Phase 1 验证测试")
    print("=" * 80)
    print()
    
    # 清理GPU缓存
    import torch
    torch.cuda.empty_cache()
    
    # 小规模测试: 10个简单问题 (使用3B模型避免OOM)
    print("运行小规模测试: 10个问题 (使用Qwen2.5-3B避免显存不足)...\n")
    
    results = run_comparison(
        model_name="Qwen/Qwen2.5-3B-Instruct",  # 改用3B模型
        n_questions=10,
        difficulty="easy",
        fixed_k=3,
        gpu_mode="single",
        gpu_ids=[0],
        seed=42
    )
    
    print("\n" + "=" * 80)
    print("验证结果")
    print("=" * 80)
    
    # 1. 检查History完整性
    print("\n1. 检查History完整性")
    print("-" * 40)
    
    mdp_sample = results['mdp_strategy']['results'][0]
    fixed_sample = results['fixed_strategy']['results'][0]
    
    print(f"\nMDP策略 - 第1个问题的history:")
    for i, step in enumerate(mdp_sample['history'][:3], 1):
        print(f"\n  步骤 {i}:")
        print(f"    - action: {step['action']}")
        print(f"    - subquery: {step['subquery'][:50] if step['subquery'] else 'None'}...")
        print(f"    - response: {step['response'][:50] if step['response'] else 'None'}...")
        print(f"    - intermediate_answer: {step['intermediate_answer']}")
        print(f"    - confidence: {step['confidence']}")
        print(f"    - uncertainty: {step['uncertainty']}")
        print(f"    - cost: {step['cost']:.3f}")
    
    # 检查必需字段
    required_fields = [
        'iteration', 'action', 'subquery', 'retrieved_docs',
        'retrieval_success', 'response', 'intermediate_answer',
        'confidence', 'uncertainty', 'cost', 'U_before', 'U_after'
    ]
    
    missing_fields = []
    for field in required_fields:
        if field not in mdp_sample['history'][0]:
            missing_fields.append(field)
    
    if missing_fields:
        print(f"\n  ❌ 缺少字段: {missing_fields}")
    else:
        print(f"\n  ✅ 所有必需字段都存在!")
    
    # 2. 检查成本参数
    print("\n\n2. 检查成本参数")
    print("-" * 40)
    
    # MDP策略
    mdp_retrieve_costs = []
    mdp_reason_costs = []
    
    for result in results['mdp_results']:
        prev_cost = 0.0
        for step in result['history']:
            cost_delta = step['cost'] - prev_cost
            if step['action'] == 'retrieve':
                mdp_retrieve_costs.append(cost_delta)
            elif step['action'] == 'reason':
                mdp_reason_costs.append(cost_delta)
            prev_cost = step['cost']
    
    avg_c_r_mdp = sum(mdp_retrieve_costs) / len(mdp_retrieve_costs) if mdp_retrieve_costs else 0
    avg_c_p_mdp = sum(mdp_reason_costs) / len(mdp_reason_costs) if mdp_reason_costs else 0
    
    print(f"\nMDP策略成本:")
    print(f"  - c_r (检索成本): {avg_c_r_mdp:.3f} (期望: 0.050)")
    print(f"  - c_p (推理成本): {avg_c_p_mdp:.3f} (期望: 0.020)")
    
    # Fixed策略
    fixed_retrieve_costs = []
    fixed_reason_costs = []
    
    for result in results['fixed_results']:
        prev_cost = 0.0
        for step in result['history']:
            cost_delta = step['cost'] - prev_cost
            if step['action'] == 'retrieve':
                fixed_retrieve_costs.append(cost_delta)
            elif step['action'] == 'reason':
                fixed_reason_costs.append(cost_delta)
            prev_cost = step['cost']
    
    avg_c_r_fixed = sum(fixed_retrieve_costs) / len(fixed_retrieve_costs) if fixed_retrieve_costs else 0
    avg_c_p_fixed = sum(fixed_reason_costs) / len(fixed_reason_costs) if fixed_reason_costs else 0
    
    print(f"\nFixed策略成本:")
    print(f"  - c_r (检索成本): {avg_c_r_fixed:.3f} (期望: 0.050)")
    print(f"  - c_p (推理成本): {avg_c_p_fixed:.3f} (期望: 0.020)")
    
    # 验证
    c_r_correct = abs(avg_c_r_mdp - 0.05) < 0.001 and abs(avg_c_r_fixed - 0.05) < 0.001
    c_p_correct = abs(avg_c_p_mdp - 0.02) < 0.001 and abs(avg_c_p_fixed - 0.02) < 0.001
    
    if c_r_correct and c_p_correct:
        print(f"\n  ✅ 成本参数正确!")
    else:
        print(f"\n  ❌ 成本参数不正确!")
        if not c_r_correct:
            print(f"     c_r 应为 0.05")
        if not c_p_correct:
            print(f"     c_p 应为 0.02")
    
    # 3. 推理链可追踪性
    print("\n\n3. 推理链可追踪性测试")
    print("-" * 40)
    
    # 提取一个问题的完整推理链
    sample_result = results['mdp_results'][0]
    
    print(f"\n问题 ID: {sample_result['question_id']}")
    print(f"正确答案: {sample_result['correct']}")
    print(f"预测答案: {sample_result['predicted']}")
    print(f"是否正确: {'✓' if sample_result['is_correct'] else '✗'}")
    print(f"\n推理链轨迹:")
    
    for step in sample_result['history']:
        action_symbol = {
            'retrieve': 'R',
            'reason': 'P',
            'terminate': 'T'
        }.get(step['action'], '?')
        
        unc = step['uncertainty'] if step['uncertainty'] is not None else 'N/A'
        print(f"  {step['iteration']:2d}. [{action_symbol}] U={1-unc if unc != 'N/A' else 'N/A'}, Cost={step['cost']:.3f}")
        
        if step['action'] == 'reason' and step['intermediate_answer']:
            print(f"      → Answer: {step['intermediate_answer']}")
    
    # 检查是否可以提取(q_t, r_t)对
    qa_pairs = []
    for step in sample_result['history']:
        if step['action'] == 'reason' and step['response']:
            qa_pairs.append({
                'iteration': step['iteration'],
                'subquery': step['subquery'],
                'response': step['response'],
                'answer': step['intermediate_answer']
            })
    
    print(f"\n提取的(q_t, r_t)对: {len(qa_pairs)} 个")
    for i, qa in enumerate(qa_pairs, 1):
        print(f"  {i}. Q: {qa['subquery'][:40]}...")
        print(f"     R: {qa['response'][:40]}...")
        print(f"     A: {qa['answer']}")
    
    if qa_pairs:
        print(f"\n  ✅ 推理链可完整追踪!")
    else:
        print(f"\n  ⚠️  没有reason步骤，无法提取QA对")
    
    # 总结
    print("\n\n" + "=" * 80)
    print("Phase 1 验证总结")
    print("=" * 80)
    
    checks = {
        "History字段完整性": len(missing_fields) == 0,
        "成本参数正确性": c_r_correct and c_p_correct,
        "推理链可追踪性": len(qa_pairs) > 0
    }
    
    print()
    for check_name, passed in checks.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {status} - {check_name}")
    
    all_passed = all(checks.values())
    
    if all_passed:
        print("\n🎉 Phase 1 所有验证通过! 可以进入Phase 2.")
    else:
        print("\n⚠️  部分验证失败，请检查代码.")
    
    return all_passed


if __name__ == '__main__':
    try:
        success = validate_phase1()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 验证过程出错: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
