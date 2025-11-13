"""
简化的Phase 1验证脚本
快速验证改进是否成功
"""

import json
import os

def main():
    print("="*80)
    print("Phase 1 验证测试 (简化版)")
    print("="*80)
    
    # 读取刚才生成的结果文件
    result_files = [f for f in os.listdir('results/multi_gpu_comparison') 
                    if 'Qwen2.5-3B' in f and 'easy' in f]
    
    if not result_files:
        print("❌ 未找到结果文件！请先运行实验。")
        return False
    
    result_file = f'results/multi_gpu_comparison/{result_files[0]}'
    print(f"\n读取结果文件: {result_file}\n")
    
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    # 1. 检查History完整性
    print("\n1. ✓ 检查History完整性")
    print("-" * 40)
    
    mdp_sample = results['mdp_strategy']['results'][0]
    
    required_fields = [
        'iteration', 'action', 'subquery', 'retrieved_docs',
        'retrieval_success', 'response', 'intermediate_answer',
        'confidence', 'uncertainty', 'cost', 'U_before', 'U_after'
    ]
    
    first_step = mdp_sample['history'][0]
    missing_fields = [f for f in required_fields if f not in first_step]
    
    if missing_fields:
        print(f"  ❌ 缺少字段: {missing_fields}")
        return False
    else:
        print(f"  ✅ 所有12个必需字段都存在")
        print(f"\n  示例 (第1步):")
        print(f"    - action: {first_step['action']}")
        print(f"    - subquery: {first_step['subquery'][:60]}...")
        print(f"    - response: {first_step['response'][:60] if first_step['response'] else 'None'}...")
        print(f"    - intermediate_answer: {first_step['intermediate_answer']}")
        print(f"    - confidence: {first_step['confidence']}")
        print(f"    - cost: {first_step['cost']:.3f}")
    
    # 2. 检查成本参数
    print("\n2. ✓ 检查成本参数正确性")
    print("-" * 40)
    
    expected_fixed_cost = 3 * 0.05 + 1 * 0.02  # 3次retrieve + 1次reason
    actual_fixed_cost = results['fixed_strategy']['avg_cost']
    
    print(f"  Fixed策略 (k=3):")
    print(f"    期望: 3×0.05 + 1×0.02 = {expected_fixed_cost:.3f}")
    print(f"    实际: {actual_fixed_cost:.3f}")
    print(f"    差异: {abs(actual_fixed_cost - expected_fixed_cost):.4f}")
    
    if abs(actual_fixed_cost - expected_fixed_cost) < 0.01:
        print(f"  ✅ 成本参数正确 (c_r=0.05, c_p=0.02)")
    else:
        print(f"  ❌ 成本参数不正确")
        return False
    
    # 3. 检查推理链可追踪性
    print("\n3. ✓ 检查推理链可追踪性")
    print("-" * 40)
    
    # 统计有中间答案的步骤
    reason_steps = [s for s in mdp_sample['history'] if s['action'] == 'reason']
    
    print(f"  问题ID: {mdp_sample['question_id']}")
    print(f"  总步骤数: {len(mdp_sample['history'])}")
    print(f"  Reason步骤: {len(reason_steps)}")
    print(f"  有中间答案的步骤: {sum(1 for s in reason_steps if s['intermediate_answer'])}")
    
    if len(reason_steps) > 0 and all(s['intermediate_answer'] for s in reason_steps):
        print(f"  ✅ 所有reason步骤都记录了中间答案")
    else:
        print(f"  ❌ 部分reason步骤缺少中间答案")
        return False
    
    # 4. 总结
    print("\n" + "="*80)
    print("Phase 1 验证总结")
    print("="*80)
    print()
    print("  ✅ 通过 - History字段完整性 (12个必需字段)")
    print("  ✅ 通过 - 成本参数正确性 (c_r=0.05, c_p=0.02)")
    print("  ✅ 通过 - 推理链可追踪性 (中间答案记录)")
    print()
    print("🎉 Phase 1 所有验证通过! 可以进入Phase 2.")
    print()
    
    # 5. 显示改进效果
    print("="*80)
    print("改进效果对比")
    print("="*80)
    print()
    print(f"MDP策略:")
    print(f"  准确率: {results['mdp_strategy']['accuracy']:.2%}")
    print(f"  平均成本: {results['mdp_strategy']['avg_cost']:.3f}")
    print(f"  平均迭代: {results['mdp_strategy']['avg_iterations']:.1f}")
    print()
    print(f"Fixed策略 (k=3):")
    print(f"  准确率: {results['fixed_strategy']['accuracy']:.2%}")
    print(f"  平均成本: {results['fixed_strategy']['avg_cost']:.3f}")
    print(f"  平均迭代: {results['fixed_strategy']['avg_iterations']:.1f}")
    print()
    print(f"改进:")
    print(f"  准确率: +{results['improvement']['accuracy']:.2%}")
    print(f"  成本差异: +{results['improvement']['cost']:.3f}")
    print()
    
    return True

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
