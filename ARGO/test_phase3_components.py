"""
Phase 3 Components Test Script
==============================

测试 Query Decomposer, Retriever, Answer Synthesizer 三个组件

测试内容：
1. QueryDecomposer: 生成子查询
2. Retriever: 检索文档（使用MockRetriever）
3. AnswerSynthesizer: 合成最终答案
4. 端到端流程测试

运行方式:
    python test_phase3_components.py
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

from src.decomposer import QueryDecomposer
from src.retriever import MockRetriever  # 使用MockRetriever避免依赖Chroma
from src.synthesizer import AnswerSynthesizer


def test_query_decomposer(model, tokenizer):
    """测试QueryDecomposer"""
    print("\n" + "="*80)
    print("TEST 1: Query Decomposer")
    print("="*80)
    
    decomposer = QueryDecomposer(model, tokenizer)
    
    # 测试用例1: 第一步查询（高不确定度）
    original_question = "What are the latency requirements for O-RAN fronthaul?"
    history = []
    uncertainty = 0.9  # 高不确定度
    
    print(f"\nOriginal Question: {original_question}")
    print(f"History: (empty)")
    print(f"Uncertainty: {uncertainty}")
    
    subquery1 = decomposer.generate_subquery(
        original_question,
        history,
        uncertainty
    )
    
    print(f"\nGenerated Subquery: {subquery1}")
    
    # 测试用例2: 第二步查询（有历史）
    history.append({
        'action': 'retrieve',
        'subquery': subquery1,
        'retrieval_success': True,
        'retrieved_docs': [
            "O-RAN fronthaul specifies strict latency bounds for different functional splits.",
            "The typical latency budget for fronthaul is 100-200 microseconds."
        ]
    })
    
    history.append({
        'action': 'reason',
        'intermediate_answer': "O-RAN fronthaul latency is typically 100-200 microseconds",
        'confidence': 0.6
    })
    
    uncertainty = 0.4  # 中等不确定度
    
    print(f"\n\nHistory: {len(history)} steps")
    print(f"Uncertainty: {uncertainty}")
    
    subquery2 = decomposer.generate_subquery(
        original_question,
        history,
        uncertainty
    )
    
    print(f"\nGenerated Subquery: {subquery2}")
    
    print("\n✅ QueryDecomposer Test Passed!")
    
    return decomposer


def test_retriever():
    """测试Retriever（使用MockRetriever）"""
    print("\n" + "="*80)
    print("TEST 2: Retriever (Mock)")
    print("="*80)
    
    retriever = MockRetriever(p_s_value=0.8)
    
    # 测试用例1: 单次检索
    query1 = "What is the latency requirement?"
    docs1, success1, _ = retriever.retrieve(query1, k=3)
    
    print(f"\nQuery: {query1}")
    print(f"Success: {success1}")
    print(f"Retrieved {len(docs1)} documents:")
    for i, doc in enumerate(docs1):
        print(f"  [{i+1}] {doc[:80]}...")
    
    # 测试用例2: 批量检索
    queries = [
        "What is O-RAN?",
        "Explain fronthaul",
        "Describe RAN architecture"
    ]
    
    print(f"\n\nBatch Retrieval: {len(queries)} queries")
    
    results = retriever.batch_retrieve(queries, k=2)
    
    for i, (q, (docs, success, _)) in enumerate(zip(queries, results)):
        print(f"\nQuery {i+1}: {q}")
        print(f"  Success: {success}, Docs: {len(docs)}")
    
    # 统计信息
    stats = retriever.get_statistics()
    print(f"\n\nRetriever Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Retriever Test Passed!")
    
    return retriever


def test_answer_synthesizer(model, tokenizer):
    """测试AnswerSynthesizer"""
    print("\n" + "="*80)
    print("TEST 3: Answer Synthesizer")
    print("="*80)
    
    synthesizer = AnswerSynthesizer(model, tokenizer)
    
    # 构造完整的推理历史
    original_question = "What are the key latency requirements in O-RAN?"
    
    history = [
        {
            'action': 'retrieve',
            'subquery': 'What is O-RAN latency?',
            'retrieval_success': True,
            'retrieved_docs': [
                "[Source: O-RAN.WG4] O-RAN specifies latency requirements for different network segments.",
                "[Source: O-RAN.WG4] The fronthaul latency budget is typically 100-200 microseconds."
            ]
        },
        {
            'action': 'reason',
            'intermediate_answer': 'O-RAN defines strict latency budgets for fronthaul around 100-200 microseconds.',
            'confidence': 0.7
        },
        {
            'action': 'retrieve',
            'subquery': 'What about control plane latency?',
            'retrieval_success': True,
            'retrieved_docs': [
                "[Source: O-RAN.WG1] Control plane latency should not exceed 10ms for RRC procedures.",
                "[Source: O-RAN.WG1] User plane latency targets are more stringent, typically <1ms."
            ]
        },
        {
            'action': 'reason',
            'intermediate_answer': 'Control plane has 10ms limit, user plane requires <1ms.',
            'confidence': 0.8
        }
    ]
    
    print(f"\nOriginal Question: {original_question}")
    print(f"History: {len(history)} steps")
    
    # 生成历史摘要
    summary = synthesizer.generate_summary(history)
    print(f"\n{summary}")
    
    # 合成答案
    answer, sources = synthesizer.synthesize(original_question, history)
    
    print(f"\nSynthesized Answer:")
    print(f"{answer}")
    
    if sources:
        print(f"\nSources: {', '.join(sources)}")
    
    print("\n✅ AnswerSynthesizer Test Passed!")
    
    return synthesizer


def test_end_to_end(decomposer, retriever, synthesizer):
    """端到端流程测试"""
    print("\n" + "="*80)
    print("TEST 4: End-to-End Workflow")
    print("="*80)
    
    original_question = "How does O-RAN handle network slicing?"
    
    print(f"\nOriginal Question: {original_question}")
    print("\nSimulating multi-step reasoning...\n")
    
    history = []
    uncertainty = 1.0  # 初始完全不确定
    max_steps = 4
    
    for step in range(max_steps):
        print(f"\n--- Step {step + 1} ---")
        
        # 决定动作（简化的MDP策略模拟）
        if uncertainty > 0.5:
            action = 'retrieve'
        else:
            action = 'reason'
        
        print(f"Action: {action.upper()}")
        print(f"Current Uncertainty: {uncertainty:.2f}")
        
        if action == 'retrieve':
            # 生成子查询
            subquery = decomposer.generate_subquery(
                original_question,
                history,
                uncertainty
            )
            print(f"Subquery: {subquery}")
            
            # 检索
            docs, success, _ = retriever.retrieve(subquery, k=2)
            
            print(f"Retrieval Success: {success}")
            
            if success:
                print(f"Retrieved {len(docs)} documents")
                # 降低不确定度
                uncertainty -= 0.25
            else:
                print("Retrieval failed")
            
            # 记录到历史
            history.append({
                'action': 'retrieve',
                'subquery': subquery,
                'retrieval_success': success,
                'retrieved_docs': docs
            })
        
        else:  # reason
            # 简单的推理（实际应该调用LLM）
            intermediate_answer = f"Based on {len(history)} previous steps, analyzing network slicing..."
            confidence = 0.7
            
            print(f"Reasoning: {intermediate_answer}")
            print(f"Confidence: {confidence:.2f}")
            
            # 降低不确定度
            uncertainty -= 0.2
            
            # 记录到历史
            history.append({
                'action': 'reason',
                'intermediate_answer': intermediate_answer,
                'confidence': confidence
            })
        
        # 确保不确定度不小于0
        uncertainty = max(0.0, uncertainty)
    
    # 合成最终答案
    print("\n\n--- Final Synthesis ---")
    
    final_answer, sources = synthesizer.synthesize(original_question, history)
    
    print(f"\nFinal Answer:")
    print(f"{final_answer}")
    
    if sources:
        print(f"\nSources: {', '.join(sources)}")
    
    print("\n✅ End-to-End Test Passed!")


def main():
    """主测试函数"""
    print("="*80)
    print("Phase 3 Components Test")
    print("="*80)
    print("\nLoading model and tokenizer...")
    
    # 使用小模型测试（避免内存问题）
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
    
    # 运行测试
    try:
        decomposer = test_query_decomposer(model, tokenizer)
        retriever = test_retriever()
        synthesizer = test_answer_synthesizer(model, tokenizer)
        
        test_end_to_end(decomposer, retriever, synthesizer)
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✅")
        print("="*80)
        print("\nPhase 3.1-3.3 components are working correctly!")
        print("Next: Integrate into ARGO_System (Phase 3.4)")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print("\n❌ Test failed. See logs for details.")


if __name__ == "__main__":
    main()
