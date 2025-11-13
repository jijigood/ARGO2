"""
最小化测试：单个query诊断
"""
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import ARGO_System

print("="*80)
print("最小化测试：单个query")
print("="*80)

# 1. 加载模型
print("\n1. 加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
print("✅ 模型加载完成")

# 2. 创建ARGO系统
print("\n2. 创建ARGO系统...")
mdp_config = {
    'mdp': {
        'delta_r': 0.25,
        'delta_p': 0.08,
        'c_r': 0.05,
        'c_p': 0.02,
        'p_s': 0.8,
        'gamma': 0.98,
        'U_grid_size': 1000
    },
    'quality': {
        'mode': 'linear',
        'k': 1.0
    }
}

system = ARGO_System(
    model, tokenizer,
    retriever_mode='mock',
    mdp_config=mdp_config,
    max_steps=3,  # 降低到3步
    decomposer_max_tokens=30,  # 进一步降低
    synthesizer_max_tokens=100,  # 进一步降低
    use_mdp=True
)
print("✅ 系统创建完成")

# 3. 测试query
print("\n3. 运行测试query...")
print("   max_steps=3, decomposer_tokens=30, synthesizer_tokens=100")

question = "What is O-RAN?"

print(f"\n问题: {question}")
print(f"开始时间: {time.strftime('%H:%M:%S')}")

start_time = time.time()

try:
    answer, history, metadata = system.answer_question(
        question,
        return_history=True
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ 成功完成!")
    print(f"耗时: {elapsed:.2f}秒")
    print(f"\n答案: {answer[:200]}...")
    print(f"\n元数据:")
    print(f"  - 总步数: {metadata.get('total_steps', 'N/A')}")
    print(f"  - 检索次数: {metadata.get('retrieve_count', 'N/A')}")
    print(f"  - 推理次数: {metadata.get('reason_count', 'N/A')}")
    
except KeyboardInterrupt:
    print(f"\n❌ KeyboardInterrupt after {time.time() - start_time:.2f}秒")
    print("这确认了问题：即使是单个query也被中断")
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("测试完成")
print("="*80)
