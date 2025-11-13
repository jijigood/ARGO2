"""
测试选择题功能
================

验证ARGO系统对O-RAN Benchmark选择题的支持。
"""

import json
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.argo_system import ARGO_System


def test_single_mcq():
    """测试单个选择题"""
    print("="*80)
    print("测试1: 单个选择题")
    print("="*80)
    
    # 初始化系统（使用小模型快速测试）
    argo = ARGO_System(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        retriever_mode="mock",  # 使用mock模式快速测试
        use_mdp=False,
        max_steps=2,
        verbose=True
    )
    
    # 准备测试数据
    question = "What is the role of the SM Fanout module in an O-DU when an E2 message is received?"
    options = [
        "It interacts with the E2 handler module to send the message to the appropriate internal module.",
        "It consults the SM Catalog module to identify the relevant SM specific modules and APIs.",
        "It maps E2 messages to their corresponding receiver modules and message contents.",
        "It sends the E2 message through the E2 Sender module."
    ]
    correct_answer = "2"
    
    print(f"\n问题: {question}")
    print(f"\n选项:")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    print(f"\n正确答案: {correct_answer}")
    print("\n" + "="*80)
    
    # 推理
    answer, choice, history, metadata = argo.answer_question(
        question=question,
        options=options,
        return_history=True
    )
    
    # 结果
    print("\n" + "="*80)
    print("结果:")
    print("="*80)
    print(f"详细答案: {answer[:300]}...")
    print(f"\n选择的选项: {choice}")
    print(f"正确答案: {correct_answer}")
    print(f"判定: {'✅ 正确' if choice == correct_answer else '❌ 错误'}")
    print(f"\n推理步数: {metadata['total_steps']}")
    print(f"耗时: {metadata['elapsed_time']:.2f}秒")
    
    return choice == correct_answer


def test_batch_mcq():
    """测试批量选择题（从数据集加载）"""
    print("\n\n" + "="*80)
    print("测试2: 批量选择题（从数据集）")
    print("="*80)
    
    # 加载数据集
    dataset_path = "ORAN-Bench-13K/Benchmark/fin_H_clean.json"
    if not os.path.exists(dataset_path):
        print(f"⚠️  数据集未找到: {dataset_path}")
        print("跳过批量测试")
        return True
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"加载数据集: {len(dataset)} 题")
    
    # 初始化系统
    argo = ARGO_System(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        retriever_mode="mock",
        use_mdp=False,
        max_steps=2,
        verbose=False  # 关闭详细输出
    )
    
    # 测试前5题
    num_samples = min(5, len(dataset))
    results = []
    
    print(f"\n处理前 {num_samples} 题...")
    
    for i, item in enumerate(dataset[:num_samples]):
        question_text = item[0]
        raw_options = item[1]
        correct_answer = item[2]
        
        # 清理选项（移除 "1. ", "2. " 等前缀）
        options = [opt.split('. ', 1)[1] if '. ' in opt else opt 
                  for opt in raw_options]
        
        print(f"\n题目 {i+1}/{num_samples}: {question_text[:60]}...")
        
        # 推理
        try:
            answer, choice, _, metadata = argo.answer_question(
                question=question_text,
                options=options,
                return_history=False
            )
            
            is_correct = (choice == correct_answer)
            result = {
                'question': question_text,
                'predicted': choice,
                'correct': correct_answer,
                'is_correct': is_correct,
                'steps': metadata['total_steps']
            }
            results.append(result)
            
            status = "✅" if is_correct else "❌"
            print(f"  预测: {choice}, 正确: {correct_answer} {status}")
            
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            results.append({
                'question': question_text,
                'predicted': None,
                'correct': correct_answer,
                'is_correct': False,
                'steps': 0
            })
    
    # 统计
    print("\n" + "="*80)
    print("批量测试结果:")
    print("="*80)
    
    total = len(results)
    correct_count = sum(1 for r in results if r['is_correct'])
    accuracy = correct_count / total if total > 0 else 0
    
    print(f"总题数: {total}")
    print(f"正确数: {correct_count}")
    print(f"准确率: {accuracy*100:.2f}%")
    
    # 详细结果
    print("\n详细结果:")
    for i, r in enumerate(results, 1):
        status = "✅" if r['is_correct'] else "❌"
        print(f"{i}. {status} 预测={r['predicted']}, 正确={r['correct']}")
    
    return accuracy > 0  # 只要有题目正确就算通过


def test_format_extraction():
    """测试格式提取"""
    print("\n\n" + "="*80)
    print("测试3: 格式提取")
    print("="*80)
    
    from src.synthesizer import AnswerSynthesizer
    
    # 创建一个简单的synthesizer实例（不需要真实model）
    class MockModel:
        pass
    
    class MockTokenizer:
        pass
    
    synthesizer = AnswerSynthesizer(
        model=MockModel(),
        tokenizer=MockTokenizer()
    )
    
    # 测试用例
    test_cases = [
        {
            'name': '完整格式',
            'raw': '<answer long>Detailed explanation...</answer long><answer short>Option 2 is correct</answer short><choice>2</choice>',
            'expected_choice': '2'
        },
        {
            'name': '仅有choice标签',
            'raw': 'Some reasoning text. <choice>3</choice>',
            'expected_choice': '3'
        },
        {
            'name': '回退提取 - Option',
            'raw': 'Based on the analysis, Option 4 is the correct answer.',
            'expected_choice': '4'
        },
        {
            'name': '回退提取 - 中文',
            'raw': '根据分析，选项1是正确答案。',
            'expected_choice': '1'
        }
    ]
    
    all_passed = True
    for test in test_cases:
        answer, choice = synthesizer._postprocess_answer(
            test['raw'], 
            has_options=True
        )
        
        passed = (choice == test['expected_choice'])
        status = "✅" if passed else "❌"
        
        print(f"\n{status} {test['name']}")
        print(f"   输入: {test['raw'][:60]}...")
        print(f"   提取: {choice}")
        print(f"   期望: {test['expected_choice']}")
        
        if not passed:
            all_passed = False
    
    return all_passed


def main():
    """运行所有测试"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "ARGO 选择题功能测试" + " "*20 + "║")
    print("╚" + "="*78 + "╝")
    
    tests = [
        ("单个选择题", test_single_mcq),
        ("批量选择题", test_batch_mcq),
        ("格式提取", test_format_extraction)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ 测试失败: {name}")
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # 总结
    print("\n\n" + "="*80)
    print("测试总结")
    print("="*80)
    
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{status}: {name}")
    
    total = len(results)
    passed_count = sum(1 for _, p in results if p)
    
    print(f"\n总计: {passed_count}/{total} 测试通过")
    
    if passed_count == total:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print(f"\n⚠️  {total - passed_count} 个测试失败")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
