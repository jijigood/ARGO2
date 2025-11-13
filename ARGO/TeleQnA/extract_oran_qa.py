"""
从TeleQnA数据集中提取仅涉及ORAN知识的QA问答
使用vLLM进行8卡GPU并行推理
"""

import json
import os
import re
from typing import Dict, List, Optional
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams


# ========== 配置 ==========
MODEL_PATH = "/data/user/huangxiaolin/ARGO/RAG_Models/models/Qwen2.5-14B-Instruct"
INPUT_FILE = "/data/user/huangxiaolin/ARGO2/ARGO/TeleQnA/TeleQnA.txt"
OUTPUT_FILE = "/data/user/huangxiaolin/ARGO2/ARGO/TeleQnA/TeleQnA_ORAN_only.json"
LOG_FILE = "/data/user/huangxiaolin/ARGO2/ARGO/TeleQnA/extraction_log.txt"

# vLLM配置
TENSOR_PARALLEL_SIZE = 8  # 使用8张GPU
MAX_MODEL_LEN = 4096
BATCH_SIZE = 32  # 批处理大小


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


# ========== 辅助函数 ==========
def load_teleqna_dataset(file_path: str) -> Dict:
    """加载TeleQnA数据集"""
    print(f"Loading TeleQnA dataset from: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 解析JSON格式
    try:
        data = json.loads(content)
        print(f"✓ Loaded {len(data)} questions")
        return data
    except json.JSONDecodeError as e:
        print(f"✗ JSON parsing error: {e}")
        return {}


def format_question_for_prompt(question_data: Dict) -> str:
    """将问题数据格式化为prompt"""
    question = question_data.get('question', '')
    options_text = ""
    
    for i in range(1, 6):  # 最多5个选项
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


def parse_llm_response(response: str) -> tuple[bool, str]:
    """解析LLM响应,提取YES/NO判断和理由"""
    response = response.strip()
    
    # 检查是否以YES或NO开头
    if response.upper().startswith('YES'):
        is_oran = True
        reason = response[3:].strip(' -:')
    elif response.upper().startswith('NO'):
        is_oran = False
        reason = response[2:].strip(' -:')
    else:
        # 尝试在文本中查找YES或NO
        if 'YES' in response.upper()[:20]:
            is_oran = True
        elif 'NO' in response.upper()[:20]:
            is_oran = False
        else:
            # 默认为NO(保守策略)
            is_oran = False
        reason = response
    
    return is_oran, reason


def batch_inference(llm: LLM, prompts: List[str], sampling_params: SamplingParams) -> List[str]:
    """批量推理"""
    outputs = llm.generate(prompts, sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    return responses


def extract_oran_questions(
    dataset: Dict,
    llm: LLM,
    sampling_params: SamplingParams,
    batch_size: int = 32
) -> tuple[Dict, List[Dict]]:
    """
    提取ORAN相关问题
    
    Returns:
        oran_questions: 仅包含ORAN知识的问题字典
        extraction_log: 提取日志列表
    """
    oran_questions = {}
    extraction_log = []
    
    # 准备所有问题
    all_questions = list(dataset.items())
    total = len(all_questions)
    
    print(f"\n{'='*60}")
    print(f"Starting ORAN extraction with vLLM")
    print(f"Total questions: {total}")
    print(f"Batch size: {batch_size}")
    print(f"GPU parallelism: {TENSOR_PARALLEL_SIZE}")
    print(f"{'='*60}\n")
    
    # 分批处理
    for i in tqdm(range(0, total, batch_size), desc="Processing batches"):
        batch = all_questions[i:i+batch_size]
        
        # 准备batch的prompts
        prompts = []
        for q_id, q_data in batch:
            prompt = format_question_for_prompt(q_data)
            prompts.append(prompt)
        
        # 批量推理
        responses = batch_inference(llm, prompts, sampling_params)
        
        # 处理结果
        for (q_id, q_data), response in zip(batch, responses):
            is_oran, reason = parse_llm_response(response)
            
            # 记录日志
            log_entry = {
                'question_id': q_id,
                'question': q_data.get('question', '')[:100],  # 截取前100字符
                'is_oran': is_oran,
                'reason': reason,
                'llm_response': response
            }
            extraction_log.append(log_entry)
            
            # 如果是ORAN问题,添加到结果中
            if is_oran:
                oran_questions[q_id] = q_data
    
    return oran_questions, extraction_log


def save_results(oran_questions: Dict, extraction_log: List[Dict]):
    """保存结果"""
    # 保存ORAN问题
    print(f"\nSaving {len(oran_questions)} ORAN questions to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(oran_questions, f, ensure_ascii=False, indent=2)
    
    # 保存日志
    print(f"Saving extraction log to: {LOG_FILE}")
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        for log_entry in extraction_log:
            f.write(f"{'='*80}\n")
            f.write(f"Question ID: {log_entry['question_id']}\n")
            f.write(f"Question: {log_entry['question']}\n")
            f.write(f"Is ORAN: {log_entry['is_oran']}\n")
            f.write(f"Reason: {log_entry['reason']}\n")
            f.write(f"LLM Response: {log_entry['llm_response']}\n")
    
    # 统计信息
    oran_count = len(oran_questions)
    total_count = len(extraction_log)
    non_oran_count = total_count - oran_count
    
    print(f"\n{'='*60}")
    print(f"Extraction Summary:")
    print(f"  Total questions: {total_count}")
    print(f"  ORAN questions: {oran_count} ({oran_count/total_count*100:.2f}%)")
    print(f"  Non-ORAN questions: {non_oran_count} ({non_oran_count/total_count*100:.2f}%)")
    print(f"{'='*60}")


def main():
    """主函数"""
    print(f"\n{'#'*60}")
    print(f"# ORAN QA Extraction from TeleQnA Dataset")
    print(f"# Using: {MODEL_PATH}")
    print(f"# GPUs: {TENSOR_PARALLEL_SIZE}")
    print(f"{'#'*60}\n")
    
    # 1. 加载数据集
    dataset = load_teleqna_dataset(INPUT_FILE)
    if not dataset:
        print("✗ Failed to load dataset. Exiting.")
        return
    
    # 2. 初始化vLLM模型
    print(f"\nInitializing vLLM model...")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Tensor Parallel Size: {TENSOR_PARALLEL_SIZE}")
    
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
        dtype="float16",
        gpu_memory_utilization=0.9,  # 使用90%的GPU内存
    )
    
    # 采样参数
    sampling_params = SamplingParams(
        temperature=0.0,  # 确定性输出
        top_p=1.0,
        max_tokens=200,  # 简短响应即可
    )
    
    print("✓ vLLM model loaded successfully\n")
    
    # 3. 提取ORAN问题
    oran_questions, extraction_log = extract_oran_questions(
        dataset=dataset,
        llm=llm,
        sampling_params=sampling_params,
        batch_size=BATCH_SIZE
    )
    
    # 4. 保存结果
    save_results(oran_questions, extraction_log)
    
    print("\n✓ Extraction completed successfully!")


if __name__ == "__main__":
    main()
