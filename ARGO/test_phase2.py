"""
Phase 2.1 验证脚本
验证检索成功率 p_s = 0.8 的实现
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_mdp_vs_fixed_multigpu import run_comparison

def validate_phase2():
    """验证Phase 2.1改进"""
    
    print("=" * 80)
    print("Phase 2.1 验证测试: 检索成功率 p_s = 0.8")
    print("=" * 80)
    print()
    
    # 清理GPU缓存
    import torch
    torch.cuda.empty_cache()
    
    # 小规模测试: 20个问题
    print("运行测试: 20个问题 (使用Qwen2.5-3B)...\n")
    
    results = run_comparison(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        n_questions=20,
        difficulty="easy",
        fixed_k=3,
        gpu_mode="single",
        gpu_ids=[0],
        seed=42  # 固定种子以便复现
    )
    
    print("\n" + "=" * 80)
    print("Phase 2.1 验证结果")
    print("=" * 80)
    
    # 1. 检查检索成功率
    print("\n1. 检查检索成功率")
    print("-" * 40)
    
    mdp_results = results['mdp_strategy']['results']
    
    total_retrievals = 0
    successful_retrievals = 0
    
    for result in mdp_results:
        for step in result['history']:
            if step['action'] == 'retrieve':
                total_retrievals += 1
                if step['retrieval_success']:
                    successful_retrievals += 1
    
    actual_success_rate = successful_retrievals / total_retrievals if total_retrievals > 0 else 0
    expected_success_rate = 0.8
    
    print(f"\n检索统计:")
    print(f"  总检索次数: {total_retrievals}")
    print(f"  成功次数: {successful_retrievals}")
    print(f"  失败次数: {total_retrievals - successful_retrievals}")
    print(f"  实际成功率: {actual_success_rate:.2%}")
    print(f"  期望成功率: {expected_success_rate:.2%}")
    print(f"  差异: {abs(actual_success_rate - expected_success_rate):.2%}")
    
    # 统计学检验: 20次实验，期望值16次成功，允许±3的波动
    success_rate_ok = abs(actual_success_rate - expected_success_rate) < 0.15
    
    if success_rate_ok:
        print(f"\n  ✅ 检索成功率符合预期 (p_s ≈ 0.8)")
    else:
        print(f"\n  ⚠️ 检索成功率偏差较大 (样本量可能不足)")
    
    # 2. 检查失败时U不变
    print("\n2. 检查失败检索时U不变")
    print("-" * 40)
    
    u_unchanged_on_failure = True
    failure_examples = []
    
    for result in mdp_results:
        for step in result['history']:
            if step['action'] == 'retrieve' and not step['retrieval_success']:
                if step['U_before'] is not None and step['U_after'] is not None:
                    if abs(step['U_before'] - step['U_after']) > 0.001:
                        u_unchanged_on_failure = False
                        failure_examples.append({
                            'iteration': step['iteration'],
                            'U_before': step['U_before'],
                            'U_after': step['U_after']
                        })
    
    if u_unchanged_on_failure:
        print(f"  ✅ 检索失败时U保持不变")
    else:
        print(f"  ❌ 检索失败时U发生变化:")
        for ex in failure_examples[:3]:
            print(f"    - Iter {ex['iteration']}: U {ex['U_before']:.2f} → {ex['U_after']:.2f}")
    
    # 3. 检查成本消耗
    print("\n3. 检查失败检索仍消耗成本")
    print("-" * 40)
    
    cost_consumed = True
    
    for result in mdp_results:
        prev_cost = 0.0
        for step in result['history']:
            if step['action'] == 'retrieve':
                if step['cost'] - prev_cost < 0.04:  # 应该至少增加c_r=0.05
                    cost_consumed = False
                prev_cost = step['cost']
    
    if cost_consumed:
        print(f"  ✅ 所有检索操作都消耗成本")
    else:
        print(f"  ❌ 部分检索操作未正确消耗成本")
    
    # 4. 显示示例
    print("\n4. 推理链示例 (含检索失败)")
    print("-" * 40)
    
    # 找一个包含检索失败的示例
    for result in mdp_results:
        has_failure = any(
            step['action'] == 'retrieve' and not step['retrieval_success']
            for step in result['history']
        )
        if has_failure:
            print(f"\n问题ID: {result['question_id']}")
            print(f"总步骤: {len(result['history'])}")
            print(f"\n前5步:")
            for i, step in enumerate(result['history'][:5], 1):
                if step['action'] == 'retrieve':
                    status = "✓ success" if step['retrieval_success'] else "✗ failed"
                    u_change = ""
                    if step['U_before'] is not None:
                        u_change = f", U: {step['U_before']:.2f}→{step['U_after']:.2f}"
                    print(f"  {i}. Retrieve {status}{u_change}, Cost: {step['cost']:.3f}")
                else:
                    print(f"  {i}. {step['action'].capitalize()}, Cost: {step['cost']:.3f}")
            break
    
    # 5. 总结
    print("\n" + "=" * 80)
    print("Phase 2.1 验证总结")
    print("=" * 80)
    print()
    
    all_passed = success_rate_ok and u_unchanged_on_failure and cost_consumed
    
    if all_passed:
        print("  ✅ 通过 - 检索成功率实现 (p_s ≈ 0.8)")
        print("  ✅ 通过 - 失败时U不变")
        print("  ✅ 通过 - 失败时仍消耗成本")
        print()
        print("🎉 Phase 2.1 所有验证通过! 可以进入Phase 2.2.")
    else:
        print(f"  {'✅' if success_rate_ok else '❌'} - 检索成功率")
        print(f"  {'✅' if u_unchanged_on_failure else '❌'} - 失败时U不变")
        print(f"  {'✅' if cost_consumed else '❌'} - 失败时消耗成本")
        print()
        print("⚠️ 部分验证未通过，请检查实现")
    
    print()
    
    return all_passed

if __name__ == '__main__':
    success = validate_phase2()
    exit(0 if success else 1)
