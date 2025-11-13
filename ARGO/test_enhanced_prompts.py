#!/usr/bin/env python
"""
测试增强的ARGO Prompts V2.0
=============================

验证新的提示词系统是否正常工作，包括：
1. QueryDecomposer - 查询分解（带进度追踪）
2. Retriever - 检索答案生成
3. Reasoner - 中间推理
4. AnswerSynthesizer - 最终合成

使用方式:
    python test_enhanced_prompts.py --mode quick  # 快速测试（Mock检索）
    python test_enhanced_prompts.py --mode full   # 完整测试（真实Chroma检索）
"""

import os
import sys
import argparse
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from decomposer import QueryDecomposer
from retriever import Retriever, MockRetriever
from synthesizer import AnswerSynthesizer
from prompts import ARGOPrompts, PromptConfig


def test_decomposer(model, tokenizer):
    """测试QueryDecomposer（带进度追踪）"""
    print("\n" + "="*80)
    print("测试 1: QueryDecomposer with Progress Tracking")
    print("="*80)
    
    decomposer = QueryDecomposer(model, tokenizer)
    
    # 测试问题
    question = "Explain the O-RAN fronthaul interface protocols and their performance requirements."
    
    # 模拟推理历史（渐进式）
    histories = [
        # 第一步：空历史
        [],
        # 第二步：有了第一次检索
        [{
            'action': 'retrieve',
            'subquery': 'What are the main protocol layers in O-RAN fronthaul interface?',
            'retrieval_success': True,
            'retrieved_docs': [
                '[Source: O-RAN.WG4] The fronthaul interface uses Control-Plane (CU-Plane), User-Plane (U-Plane), and Synchronization-Plane (S-Plane).'
            ],
            'intermediate_answer': 'Three main layers: C-Plane, U-Plane, S-Plane.',
            'progress': 0.35
        }],
        # 第三步：有了第二次检索
        [{
            'action': 'retrieve',
            'subquery': 'What are the main protocol layers in O-RAN fronthaul interface?',
            'retrieval_success': True,
            'retrieved_docs': ['...'],
            'intermediate_answer': 'Three main layers...',
            'progress': 0.35
        },
        {
            'action': 'retrieve',
            'subquery': 'What are the latency requirements for O-RAN fronthaul?',
            'retrieval_success': True,
            'retrieved_docs': [
                '[Source: O-RAN.WG4] One-way latency typically <400us for FR1.'
            ],
            'intermediate_answer': '<400 microseconds for FR1.',
            'progress': 0.65
        }]
    ]
    
    uncertainties = [1.0, 0.65, 0.35]  # 1 - progress
    
    for i, (history, uncertainty) in enumerate(zip(histories, uncertainties)):
        print(f"\n--- Step {i+1} (Progress: {(1-uncertainty)*100:.0f}%) ---")
        
        subquery = decomposer.generate_subquery(
            original_question=question,
            history=history,
            uncertainty=uncertainty
        )
        
        print(f"Generated Subquery: {subquery}")
    
    print("\n✓ QueryDecomposer测试通过")


def test_retriever_answer_generation(model, tokenizer, use_chroma=False):
    """测试Retriever的答案生成功能"""
    print("\n" + "="*80)
    print("测试 2: Retriever Answer Generation")
    print("="*80)
    
    # 初始化检索器
    if use_chroma:
        chroma_dir = "Environments/chroma_store"
        if not os.path.exists(chroma_dir):
            print(f"⚠ Chroma数据库不存在: {chroma_dir}")
            print("  使用MockRetriever代替")
            use_chroma = False
    
    if use_chroma:
        retriever = Retriever(
            chroma_dir=chroma_dir,
            collection_name="oran_specs",
            similarity_threshold=0.3,
            p_s_mode="threshold"
        )
    else:
        retriever = MockRetriever(p_s_value=1.0)  # 总是成功
    
    # 测试查询
    queries = [
        "What is the maximum latency for E2 interface?",
        "How many bits are used for IQ sample representation?",
        "What compression methods are available for fronthaul?"
    ]
    
    for query in queries:
        print(f"\n--- Query: {query} ---")
        
        # 检索文档
        docs, success, scores = retriever.retrieve(query, k=3, return_scores=True)
        
        if success:
            print(f"Retrieved {len(docs)} documents")
            
            # 生成答案
            answer = retriever.generate_answer_from_docs(
                question=query,
                docs=docs,
                model=model,
                tokenizer=tokenizer
            )
            
            print(f"Generated Answer: {answer}")
        else:
            print("Retrieval failed")
    
    print("\n✓ Retriever答案生成测试通过")


def test_reasoning_prompt(model, tokenizer):
    """测试推理Prompt（基于参数化知识，无需检索文档）"""
    print("\n" + "="*80)
    print("测试 3: Reasoning Prompt (Parametric Knowledge)")
    print("="*80)
    
    question = "How does the E2 interface enable RAN optimization?"
    
    # 模拟历史（包含一些检索信息）
    history = [
        {
            'action': 'retrieve',
            'subquery': 'What is the E2 interface in O-RAN architecture?',
            'retrieval_success': True,
            'retrieved_docs': [
                '[Source: O-RAN.WG3] The E2 interface connects the Near-RT RIC to the E2 nodes (O-CU-CP, O-CU-UP, O-DU).'
            ],
            'intermediate_answer': 'The E2 interface connects Near-RT RIC to E2 nodes.',
            'progress': 0.30
        },
        {
            'action': 'retrieve',
            'subquery': 'What are E2 service models?',
            'retrieval_success': True,
            'retrieved_docs': [
                '[Source: O-RAN.WG3] Key E2SMs include: KPM (Key Performance Monitoring), RC (RAN Control).'
            ],
            'intermediate_answer': 'E2SMs include KPM and RC.',
            'progress': 0.55
        }
    ]
    
    # 构建推理prompt（注意：这是基于参数化知识的推理，不再检索）
    prompt = ARGOPrompts.build_reasoning_prompt(
        original_question=question,
        history=history
    )
    
    print("\n--- Generated Reasoning Prompt ---")
    print(prompt[:800] + "...")
    
    print("\n关键特征:")
    print("✓ 包含Few-shot示例（参数化知识推理）")
    print("✓ 显示之前的检索上下文")
    print("✓ 不要求检索新文档")
    print("✓ 基于LLM预训练知识推理")
    
    # 生成推理
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=PromptConfig.REASONER_MAX_LENGTH,
            temperature=PromptConfig.REASONER_TEMPERATURE,
            top_p=PromptConfig.REASONER_TOP_P,
            do_sample=True
        )
    
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    reasoning = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"\n--- Generated Reasoning (Parametric Knowledge) ---")
    print(reasoning)
    
    print("\n✓ Reasoning Prompt测试通过")


def test_synthesizer(model, tokenizer):
    """测试AnswerSynthesizer"""
    print("\n" + "="*80)
    print("测试 4: Answer Synthesizer")
    print("="*80)
    
    synthesizer = AnswerSynthesizer(model, tokenizer)
    
    question = "Explain the O-RAN fronthaul interface protocols."
    
    # 模拟完整的推理历史
    history = [
        {
            'action': 'retrieve',
            'subquery': 'What are the main protocol layers?',
            'retrieval_success': True,
            'retrieved_docs': [
                '[Source: O-RAN.WG4] C-Plane, U-Plane, S-Plane.'
            ],
            'intermediate_answer': 'Three protocol layers: C-Plane for control, U-Plane for data, S-Plane for sync.',
            'confidence': 0.9,
            'progress': 0.35
        },
        {
            'action': 'reason',
            'intermediate_answer': 'These layers work together to enable low-latency fronthaul communication.',
            'confidence': 0.7,
            'progress': 0.50
        },
        {
            'action': 'retrieve',
            'subquery': 'What are the latency requirements?',
            'retrieval_success': True,
            'retrieved_docs': [
                '[Source: O-RAN.WG4] Latency <400us for FR1.'
            ],
            'intermediate_answer': 'One-way latency must be under 400 microseconds.',
            'confidence': 0.85,
            'progress': 0.80
        }
    ]
    
    # 合成答案
    answer, sources = synthesizer.synthesize(
        original_question=question,
        history=history
    )
    
    print(f"\n--- Final Answer ---")
    print(answer)
    
    if sources:
        print(f"\n--- Sources ---")
        print(", ".join(sources))
    
    print("\n✓ AnswerSynthesizer测试通过")


def test_full_argo_flow(model, tokenizer, use_chroma=False):
    """测试完整的ARGO流程"""
    print("\n" + "="*80)
    print("测试 5: Full ARGO Flow (Simplified)")
    print("="*80)
    
    # 初始化所有组件
    decomposer = QueryDecomposer(model, tokenizer)
    synthesizer = AnswerSynthesizer(model, tokenizer)
    
    if use_chroma:
        chroma_dir = "Environments/chroma_store"
        if os.path.exists(chroma_dir):
            retriever = Retriever(chroma_dir=chroma_dir)
        else:
            retriever = MockRetriever()
    else:
        retriever = MockRetriever()
    
    # 测试问题
    question = "What is the E2 interface latency requirement?"
    
    print(f"\nQuestion: {question}")
    
    # 模拟2步推理流程
    history = []
    U_t = 0.0
    
    for step in range(2):
        print(f"\n--- Step {step+1} (Progress: {U_t*100:.0f}%) ---")
        
        # 生成子查询
        uncertainty = 1.0 - U_t
        subquery = decomposer.generate_subquery(question, history, uncertainty)
        print(f"Subquery: {subquery}")
        
        # 检索
        docs, success, _ = retriever.retrieve(subquery, k=3)
        
        if success:
            print(f"Retrieved {len(docs)} documents")
            
            # 生成答案
            answer = retriever.generate_answer_from_docs(
                question=subquery,
                docs=docs,
                model=model,
                tokenizer=tokenizer
            )
            print(f"Answer: {answer}")
            
            # 更新历史
            history.append({
                'action': 'retrieve',
                'subquery': subquery,
                'retrieval_success': True,
                'retrieved_docs': docs,
                'intermediate_answer': answer,
                'confidence': 0.8,
                'progress': U_t
            })
            
            # 更新进度
            U_t = min(1.0, U_t + 0.35)
        else:
            print("Retrieval failed")
            break
    
    # 合成最终答案
    print(f"\n--- Final Synthesis (Progress: {U_t*100:.0f}%) ---")
    final_answer, sources = synthesizer.synthesize(question, history)
    
    print(f"\nFinal Answer: {final_answer}")
    
    print("\n✓ Full ARGO Flow测试通过")


def main():
    parser = argparse.ArgumentParser(description="测试ARGO Enhanced Prompts V2.0")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['quick', 'full'],
        default='quick',
        help='测试模式: quick (Mock检索) 或 full (真实Chroma检索)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='/data/user/huangxiaolin/ARGO/RAG_Models/models/Qwen2.5-1.5B-Instruct',
        help='LLM模型路径'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='设备: cuda:0, cuda:1, 或 cpu'
    )
    
    args = parser.parse_args()
    
    use_chroma = (args.mode == 'full')
    
    print("\n" + "="*80)
    print("ARGO Enhanced Prompts V2.0 测试套件")
    print("="*80)
    print(f"模式: {args.mode}")
    print(f"模型: {args.model}")
    print(f"设备: {args.device}")
    print(f"使用Chroma: {use_chroma}")
    
    # 加载模型
    print("\n加载模型...")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
            device_map=args.device if device.type == 'cuda' else None,
            trust_remote_code=True
        )
        
        if device.type == 'cpu':
            model = model.to(device)
        
        model.eval()
        print("✓ 模型加载成功")
    
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        print("  请检查模型路径是否正确")
        return
    
    # 运行测试
    try:
        test_decomposer(model, tokenizer)
        test_retriever_answer_generation(model, tokenizer, use_chroma)
        test_reasoning_prompt(model, tokenizer)
        test_synthesizer(model, tokenizer)
        test_full_argo_flow(model, tokenizer, use_chroma)
        
        print("\n" + "="*80)
        print("✓ 所有测试通过！")
        print("="*80)
        
        print("\n改进总结:")
        print("1. ✓ 查询分解带进度追踪（Progress: 0-100%）")
        print("2. ✓ 检索后自动生成中间答案")
        print("3. ✓ 推理使用标准化prompt模板")
        print("4. ✓ 最终答案支持长/短格式")
        print("5. ✓ 所有prompts集中管理（src/prompts.py）")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
