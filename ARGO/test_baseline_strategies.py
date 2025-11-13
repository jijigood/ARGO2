"""
Quick Test for Baseline Strategies
===================================

快速测试基线策略是否正常工作
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import (
    ARGO_System,
    AlwaysReasonStrategy,
    RandomStrategy,
    FixedThresholdStrategy
)

print("Loading model...")
model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print(f"✅ Model loaded\n")

test_question = "What is O-RAN?"

strategies = {
    'MDP': ARGO_System(model, tokenizer, use_mdp=True, retriever_mode='mock', max_steps=3, verbose=False),
    'Fixed': FixedThresholdStrategy(model, tokenizer, retriever_mode='mock', max_steps=3, verbose=False),
    'Always-Reason': AlwaysReasonStrategy(model, tokenizer, retriever_mode='mock', max_steps=3, verbose=False),
    'Random': RandomStrategy(model, tokenizer, retriever_mode='mock', max_steps=3, verbose=False)
}

print("="*80)
print("Testing All Strategies")
print("="*80)
print(f"Question: {test_question}\n")

for name, strategy in strategies.items():
    print(f"\n--- {name} Strategy ---")
    try:
        answer, _, metadata = strategy.answer_question(test_question, return_history=False)
        
        print(f"✅ Success!")
        print(f"  Steps: {metadata['total_steps']}")
        print(f"  Retrieve: {metadata['retrieve_count']}, Reason: {metadata['reason_count']}")
        print(f"  Time: {metadata['elapsed_time']:.1f}s")
        print(f"  Answer: {answer[:100]}...")
        
    except Exception as e:
        print(f"❌ Failed: {e}")

print("\n" + "="*80)
print("Quick Test Complete!")
print("="*80)
