"""
Quick Optimization Test - Single Query
======================================

快速测试单个query的优化效果
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import ARGO_System


def test_single_query(model_name, max_decomposer_tokens, max_synthesizer_tokens):
    """测试单个query"""
    
    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"Decomposer max_tokens: {max_decomposer_tokens}")
    print(f"Synthesizer max_tokens: {max_synthesizer_tokens}")
    print('='*70)
    
    # 加载模型
    print("Loading model...")
    start_load = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    load_time = time.time() - start_load
    print(f"✅ Model loaded in {load_time:.1f}s")
    
    # 创建系统
    system = ARGO_System(
        model,
        tokenizer,
        use_mdp=True,
        retriever_mode='mock',
        max_steps=3,  # 减少步数以加快测试
        verbose=False
    )
    
    # 优化参数
    system.decomposer.max_subquery_length = max_decomposer_tokens
    system.synthesizer.max_answer_length = max_synthesizer_tokens
    system.decomposer.temperature = 0.5
    system.synthesizer.temperature = 0.2
    
    # 测试单个问题
    question = "What is O-RAN?"
    
    print(f"\nQuestion: {question}")
    print("Running...")
    
    start = time.time()
    answer, history, metadata = system.answer_question(question, return_history=True)
    elapsed = time.time() - start
    
    print(f"\n✅ Completed in {elapsed:.1f}s")
    print(f"Steps: {metadata['total_steps']}")
    print(f"Retrieve: {metadata['retrieve_count']}, Reason: {metadata['reason_count']}")
    print(f"Answer: {answer[:100]}...")
    
    return elapsed


def main():
    print("="*70)
    print("Quick Optimization Test")
    print("="*70)
    
    # 测试3种配置
    configs = [
        ("Qwen/Qwen2.5-3B-Instruct", 128, 512, "Baseline"),
        ("Qwen/Qwen2.5-3B-Instruct", 50, 200, "Optimized Params"),
        ("Qwen/Qwen2.5-1.5B-Instruct", 50, 200, "Optimized Model + Params"),
    ]
    
    results = []
    
    for model_name, max_dec, max_syn, label in configs:
        try:
            elapsed = test_single_query(model_name, max_dec, max_syn)
            results.append((label, elapsed))
        except Exception as e:
            print(f"❌ Failed: {e}")
            if "1.5B" in model_name:
                print("⚠️  1.5B model may not be downloaded, skipping...")
                continue
            else:
                raise
    
    # 总结
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    baseline_time = results[0][1] if results else None
    
    for label, elapsed in results:
        if baseline_time and label != "Baseline":
            speedup = baseline_time / elapsed
            print(f"{label:30s}: {elapsed:6.1f}s (Speedup: {speedup:.2f}x)")
        else:
            print(f"{label:30s}: {elapsed:6.1f}s (Baseline)")
    
    print("="*70)


if __name__ == "__main__":
    main()
