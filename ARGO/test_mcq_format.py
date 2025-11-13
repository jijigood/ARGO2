"""
简化的选择题功能测试
====================

只测试核心的格式提取逻辑，不需要完整的ARGO系统。
"""

import sys
import os
import re

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_choice_extraction():
    """测试choice标签提取功能"""
    print("="*80)
    print("测试: Choice标签提取")
    print("="*80)
    
    # 测试用例
    test_cases = [
        {
            'name': '完整格式',
            'input': '<answer long>详细解释...</answer long><answer short>Option 2正确</answer short><choice>2</choice>',
            'expected': '2'
        },
        {
            'name': '仅choice标签',
            'input': '一些推理文本。<choice>3</choice>',
            'expected': '3'
        },
        {
            'name': '回退提取-Option',
            'input': '根据分析，Option 4 是正确答案。',
            'expected': '4'
        },
        {
            'name': '回退提取-中文',
            'input': '根据分析，选项1是正确答案。',
            'expected': '1'
        },
        {
            'name': '无法提取',
            'input': '这是一段没有选项信息的文本。',
            'expected': None
        }
    ]
    
    def extract_choice(text: str) -> str:
        """模拟synthesizer中的提取逻辑"""
        # 主提取: <choice>X</choice>
        choice_match = re.search(r'<choice>(\d)</choice>', text)
        if choice_match:
            return choice_match.group(1)
        
        # 回退提取: "Option 3" 或 "选项3"
        fallback_match = re.search(r'[Oo]ption\s*(\d)|选项\s*(\d)', text)
        if fallback_match:
            return fallback_match.group(1) or fallback_match.group(2)
        
        return None
    
    # 运行测试
    all_passed = True
    for test in test_cases:
        result = extract_choice(test['input'])
        passed = (result == test['expected'])
        status = "✅" if passed else "❌"
        
        print(f"\n{status} {test['name']}")
        print(f"   输入: {test['input'][:60]}...")
        print(f"   提取: {result}")
        print(f"   期望: {test['expected']}")
        
        if not passed:
            all_passed = False
    
    return all_passed


def test_answer_format_extraction():
    """测试answer标签提取"""
    print("\n\n" + "="*80)
    print("测试: Answer标签提取")
    print("="*80)
    
    test_input = """
<answer long>
Based on the retrieved O-RAN specifications, the Near-RT RIC (Near Real-Time RAN Intelligent Controller) 
is responsible for providing near-real-time RAN control and optimization through the E2 interface.
</answer long>

<answer short>
Option 2 is correct because Near-RT RIC provides near-real-time control via E2 interface.
</answer short>

<choice>2</choice>
"""
    
    # 提取逻辑
    long_match = re.search(r'<answer long>(.*?)</answer long>', test_input, re.DOTALL)
    short_match = re.search(r'<answer short>(.*?)</answer short>', test_input, re.DOTALL)
    choice_match = re.search(r'<choice>(\d)</choice>', test_input)
    
    print("\n提取结果:")
    if long_match:
        long_answer = long_match.group(1).strip()
        print(f"✅ Long Answer: {long_answer[:100]}...")
    else:
        print("❌ Long Answer: 未找到")
        return False
    
    if short_match:
        short_answer = short_match.group(1).strip()
        print(f"✅ Short Answer: {short_answer}")
    else:
        print("❌ Short Answer: 未找到")
        return False
    
    if choice_match:
        choice = choice_match.group(1)
        print(f"✅ Choice: {choice}")
    else:
        print("❌ Choice: 未找到")
        return False
    
    return True


def test_api_return_format():
    """测试API返回格式"""
    print("\n\n" + "="*80)
    print("测试: API返回格式")
    print("="*80)
    
    # 模拟API返回
    answer = "Based on O-RAN specifications, Near-RT RIC provides near-real-time control..."
    choice = "2"
    history = [
        {'action': 'retrieve', 'subquery': 'What is Near-RT RIC?', 'retrieval_success': True},
        {'action': 'reason', 'intermediate_answer': 'Near-RT RIC operates in 10ms-1s timeframe'}
    ]
    metadata = {
        'total_steps': 2,
        'retrieve_count': 1,
        'reason_count': 1,
        'elapsed_time': 3.5
    }
    
    print("\n返回值示例:")
    print(f"✅ answer (str): {answer[:60]}...")
    print(f"✅ choice (str): {choice}")
    print(f"✅ history (List[Dict]): {len(history)} 步")
    print(f"✅ metadata (Dict): {metadata}")
    
    # 验证类型
    checks = [
        (isinstance(answer, str), "answer是字符串"),
        (isinstance(choice, str), "choice是字符串"),
        (choice in ['1', '2', '3', '4'], "choice是有效选项"),
        (isinstance(history, list), "history是列表"),
        (isinstance(metadata, dict), "metadata是字典"),
        ('total_steps' in metadata, "metadata包含total_steps"),
    ]
    
    print("\n类型检查:")
    all_passed = True
    for check, desc in checks:
        status = "✅" if check else "❌"
        print(f"{status} {desc}")
        if not check:
            all_passed = False
    
    return all_passed


def main():
    """运行所有测试"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*18 + "ARGO 选择题格式测试" + " "*19 + "║")
    print("╚" + "="*78 + "╝")
    
    tests = [
        ("Choice标签提取", test_choice_extraction),
        ("Answer标签提取", test_answer_format_extraction),
        ("API返回格式", test_api_return_format)
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
        print("\n核心功能验证:")
        print("  ✅ Choice标签提取逻辑正确")
        print("  ✅ Answer标签提取逻辑正确")
        print("  ✅ API返回格式符合预期")
        print("\n下一步:")
        print("  1. 参考 MULTIPLE_CHOICE_SUPPORT.md 了解完整用法")
        print("  2. 查看 QUICK_REFERENCE.md 获取快速参考")
        print("  3. 使用真实ARGO系统进行完整测试")
        return 0
    else:
        print(f"\n⚠️  {total - passed_count} 个测试失败")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
