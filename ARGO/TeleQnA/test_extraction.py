"""
快速测试脚本 - 测试ORAN提取功能
只处理前10个问题,用于验证功能和prompt
"""

import json
import os
from vllm import LLM, SamplingParams


# ========== 配置 ==========
MODEL_PATH = "/data/user/huangxiaolin/ARGO/RAG_Models/models/Qwen2.5-14B-Instruct"
INPUT_FILE = "/data/user/huangxiaolin/ARGO2/ARGO/TeleQnA/TeleQnA.txt"
TEST_SIZE = 10  # 只测试前10个问题
TENSOR_PARALLEL_SIZE = 8


# ========== Prompt模板 ==========
EXTRACTION_PROMPT = """You are an expert in O-RAN (Open Radio Access Network) technology. Your task is to determine whether a question and its answer are **exclusively** related to O-RAN knowledge.

**O-RAN (Open Radio Access Network)** refers to:
- The O-RAN Alliance specifications and architecture
- O-RAN components: O-CU (Central Unit), O-DU (Distributed Unit), O-RU (Radio Unit)
- O-RAN interfaces: E2, A1, O1, O2, F1, fronthaul, etc.
- RAN Intelligent Controllers: Near-RT RIC, Non-RT RIC
- xApps and rApps
- O-RAN specific protocols, procedures, and implementations
- O-RAN network slicing, QoS, and resource management
- O-RAN specific use cases and deployment scenarios

**NOT O-RAN** (should be excluded):
- General 3GPP specifications (unless specifically about O-RAN implementation)
- Generic telecommunications concepts (VPN, encryption, MIMO, etc. that are not O-RAN specific)
- IEEE standards (802.11, 802.15.4, etc.)
- Mathematical concepts, general wireless theory
- Non-O-RAN network architectures
- Cloud/edge computing concepts (unless specifically O-RAN related)
- General lexicon or acronyms

Below is a question with its answer from the TeleQnA dataset:

**Question:** {question}
**Options:**
{options}
**Answer:** {answer}
**Explanation:** {explanation}
**Category:** {category}

**Task:** Determine if this question is **exclusively** about O-RAN knowledge.

**Output Format:** 
Respond with ONLY "YES" or "NO", followed by a brief reason (one line).

Example:
- "YES - This question is about O-RAN architecture components and interfaces."
- "NO - This is a general 3GPP specification, not specific to O-RAN."

Your response:"""


def load_sample_questions(file_path: str, sample_size: int = 10) -> dict:
    """加载样本问题"""
    print(f"Loading sample questions from: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    data = json.loads(content)
    
    # 只取前N个问题
    sample_data = {}
    for i, (key, value) in enumerate(data.items()):
        if i >= sample_size:
            break
        sample_data[key] = value
    
    print(f"✓ Loaded {len(sample_data)} sample questions")
    return sample_data


def format_question_for_prompt(question_data: dict) -> str:
    """将问题数据格式化为prompt"""
    question = question_data.get('question', '')
    options_text = ""
    
    for i in range(1, 6):
        option_key = f'option {i}'
        if option_key in question_data:
            options_text += f"{option_key}: {question_data[option_key]}\n"
    
    answer = question_data.get('answer', '')
    explanation = question_data.get('explanation', '')
    category = question_data.get('category', '')
    
    prompt = EXTRACTION_PROMPT.format(
        question=question,
        options=options_text.strip(),
        answer=answer,
        explanation=explanation,
        category=category
    )
    
    return prompt


def parse_llm_response(response: str) -> tuple:
    """解析LLM响应"""
    response = response.strip()
    
    if response.upper().startswith('YES'):
        is_oran = True
        reason = response[3:].strip(' -:')
    elif response.upper().startswith('NO'):
        is_oran = False
        reason = response[2:].strip(' -:')
    else:
        if 'YES' in response.upper()[:20]:
            is_oran = True
        elif 'NO' in response.upper()[:20]:
            is_oran = False
        else:
            is_oran = False
        reason = response
    
    return is_oran, reason


def main():
    print(f"\n{'='*60}")
    print(f"Quick Test: ORAN QA Extraction")
    print(f"Testing with first {TEST_SIZE} questions")
    print(f"{'='*60}\n")
    
    # 1. 加载样本数据
    dataset = load_sample_questions(INPUT_FILE, TEST_SIZE)
    
    # 2. 初始化vLLM
    print(f"\nInitializing vLLM model...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        max_model_len=4096,
        trust_remote_code=True,
        dtype="float16",
        gpu_memory_utilization=0.9,
    )
    
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=200,
    )
    
    print("✓ vLLM model loaded\n")
    
    # 3. 处理问题
    print(f"Processing {len(dataset)} questions...\n")
    
    prompts = []
    question_ids = []
    
    for q_id, q_data in dataset.items():
        prompt = format_question_for_prompt(q_data)
        prompts.append(prompt)
        question_ids.append(q_id)
    
    # 批量推理
    outputs = llm.generate(prompts, sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    
    # 4. 显示结果
    oran_count = 0
    print(f"{'='*80}")
    
    for q_id, q_data, response in zip(question_ids, dataset.values(), responses):
        is_oran, reason = parse_llm_response(response)
        
        if is_oran:
            oran_count += 1
        
        print(f"\nQuestion ID: {q_id}")
        print(f"Question: {q_data['question'][:100]}...")
        print(f"Category: {q_data.get('category', 'N/A')}")
        print(f"Is ORAN: {'✓ YES' if is_oran else '✗ NO'}")
        print(f"Reason: {reason}")
        print(f"LLM Response: {response}")
        print(f"{'-'*80}")
    
    # 5. 统计
    print(f"\n{'='*80}")
    print(f"Test Summary:")
    print(f"  Total questions: {len(dataset)}")
    print(f"  ORAN questions: {oran_count} ({oran_count/len(dataset)*100:.1f}%)")
    print(f"  Non-ORAN questions: {len(dataset)-oran_count} ({(len(dataset)-oran_count)/len(dataset)*100:.1f}%)")
    print(f"{'='*80}\n")
    
    print("✓ Quick test completed!")
    print("\nIf results look good, run the full extraction:")
    print("  ./run_extraction.sh")


if __name__ == "__main__":
    main()
