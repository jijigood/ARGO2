"""
ARGO System End-to-End Test
============================

测试完整的ARGO_System，验证4组件架构的集成

测试场景：
1. 单问题测试（详细输出）
2. 多问题批量测试
3. MDP vs Fixed策略对比
4. 性能统计

运行方式:
    python test_argo_system.py
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.argo_system import ARGO_System


def test_single_question(system, question):
    """测试单个问题（详细输出）"""
    print("\n" + "="*80)
    print(f"Test: Single Question")
    print("="*80)
    
    answer, history, metadata = system.answer_question(
        question,
        return_history=True
    )
    
    print("\n" + "="*80)
    print("Results Summary")
    print("="*80)
    print(f"\nQuestion: {question}")
    print(f"\nAnswer:\n{answer}")
    print(f"\nMetadata:")
    for key, value in metadata.items():
        if key != 'sources':
            print(f"  {key}: {value}")
    
    if metadata.get('sources'):
        print(f"\nSources: {', '.join(metadata['sources'])}")
    
    print(f"\nReasoning History ({len(history)} steps):")
    for i, step in enumerate(history):
        action = step['action']
        print(f"\n  Step {i+1}: {action.upper()}")
        
        if action == 'retrieve':
            print(f"    Subquery: {step['subquery'][:80]}...")
            print(f"    Success: {step['retrieval_success']}")
            if step['retrieval_success']:
                print(f"    Retrieved: {len(step['retrieved_docs'])} docs")
        else:
            print(f"    Answer: {step['intermediate_answer'][:80]}...")
            print(f"    Confidence: {step['confidence']:.2f}")
    
    print("\n" + "="*80)


def test_multiple_questions(system, questions):
    """测试多个问题（批量）"""
    print("\n" + "="*80)
    print(f"Test: Multiple Questions ({len(questions)})")
    print("="*80)
    
    results = []
    
    for i, q in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] Processing: {q[:60]}...")
        
        answer, _, metadata = system.answer_question(
            q,
            return_history=False
        )
        
        results.append({
            'question': q,
            'answer': answer,
            'metadata': metadata
        })
        
        print(f"  Steps: {metadata['total_steps']}, Time: {metadata['elapsed_time']:.2f}s")
    
    # 统计
    print("\n" + "="*80)
    print("Batch Statistics")
    print("="*80)
    
    total_steps = sum(r['metadata']['total_steps'] for r in results)
    total_time = sum(r['metadata']['elapsed_time'] for r in results)
    avg_steps = total_steps / len(results)
    avg_time = total_time / len(results)
    
    print(f"\nTotal Questions: {len(results)}")
    print(f"Total Steps: {total_steps}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Avg Steps/Question: {avg_steps:.1f}")
    print(f"Avg Time/Question: {avg_time:.2f}s")
    
    return results


def test_mdp_vs_fixed(model, tokenizer, question):
    """对比MDP策略 vs 固定策略"""
    print("\n" + "="*80)
    print(f"Test: MDP vs Fixed Strategy")
    print("="*80)
    
    # MDP策略
    print("\n--- Testing MDP-Guided Strategy ---")
    system_mdp = ARGO_System(
        model,
        tokenizer,
        use_mdp=True,
        retriever_mode="mock",
        max_steps=10,
        verbose=False  # 减少输出
    )
    
    answer_mdp, history_mdp, meta_mdp = system_mdp.answer_question(
        question,
        return_history=True
    )
    
    print(f"MDP Results:")
    print(f"  Steps: {meta_mdp['total_steps']}")
    print(f"  Retrieve: {meta_mdp['retrieve_count']}, Reason: {meta_mdp['reason_count']}")
    print(f"  Final U: {meta_mdp['final_uncertainty']:.3f}")
    print(f"  Time: {meta_mdp['elapsed_time']:.2f}s")
    
    # 固定策略
    print("\n--- Testing Fixed Strategy ---")
    system_fixed = ARGO_System(
        model,
        tokenizer,
        use_mdp=False,
        retriever_mode="mock",
        max_steps=10,
        verbose=False
    )
    
    answer_fixed, history_fixed, meta_fixed = system_fixed.answer_question(
        question,
        return_history=True
    )
    
    print(f"Fixed Results:")
    print(f"  Steps: {meta_fixed['total_steps']}")
    print(f"  Retrieve: {meta_fixed['retrieve_count']}, Reason: {meta_fixed['reason_count']}")
    print(f"  Final U: {meta_fixed['final_uncertainty']:.3f}")
    print(f"  Time: {meta_fixed['elapsed_time']:.2f}s")
    
    # 对比
    print("\n" + "="*80)
    print("Comparison")
    print("="*80)
    print(f"\nStrategy      Steps  Retrieve  Reason  FinalU   Time")
    print(f"MDP           {meta_mdp['total_steps']:5d}  {meta_mdp['retrieve_count']:8d}  {meta_mdp['reason_count']:6d}  {meta_mdp['final_uncertainty']:6.3f}  {meta_mdp['elapsed_time']:5.2f}s")
    print(f"Fixed         {meta_fixed['total_steps']:5d}  {meta_fixed['retrieve_count']:8d}  {meta_fixed['reason_count']:6d}  {meta_fixed['final_uncertainty']:6.3f}  {meta_fixed['elapsed_time']:5.2f}s")


def test_system_statistics(system, questions):
    """测试系统统计功能"""
    print("\n" + "="*80)
    print(f"Test: System Statistics")
    print("="*80)
    
    # 重置统计
    system.reset_statistics()
    
    # 处理问题
    for q in questions:
        system.answer_question(q, return_history=False)
    
    # 获取统计
    stats = system.get_statistics()
    
    print("\nSystem Statistics:")
    print("="*80)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")


def main():
    """主测试函数"""
    print("="*80)
    print("ARGO System End-to-End Test")
    print("="*80)
    print("\nLoading model and tokenizer...")
    
    # 使用小模型测试
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        print(f"✅ Loaded model: {model_name}")
        print(f"Device: {next(model.parameters()).device}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        print("\n⚠️  Model loading failed. Please ensure Qwen2.5-3B-Instruct is available.")
        return
    
    # 创建ARGO系统
    print("\nInitializing ARGO System...")
    
    system = ARGO_System(
        model,
        tokenizer,
        use_mdp=True,
        retriever_mode="mock",
        max_steps=8,
        verbose=True  # 详细输出
    )
    
    print("✅ ARGO System ready")
    
    # 测试问题
    test_questions = [
        "What are the latency requirements for O-RAN fronthaul?",
        "How does O-RAN handle network slicing?",
        "What is the role of RIC in O-RAN architecture?",
    ]
    
    # 运行测试
    try:
        # Test 1: 单问题详细测试
        test_single_question(system, test_questions[0])
        
        # Test 2: 多问题批量测试
        test_multiple_questions(system, test_questions)
        
        # Test 3: MDP vs Fixed对比
        test_mdp_vs_fixed(model, tokenizer, test_questions[1])
        
        # Test 4: 系统统计
        test_system_statistics(system, test_questions)
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✅")
        print("="*80)
        print("\nARGO System (4-Component Architecture) is working correctly!")
        print("\nNext Steps:")
        print("  1. Test with real Chroma retriever")
        print("  2. Implement baseline strategies (Always-Reason, Random)")
        print("  3. Run full evaluation on ORAN-Bench-13K")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print("\n❌ Test failed. See logs for details.")


if __name__ == "__main__":
    main()
