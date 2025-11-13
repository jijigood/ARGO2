"""
集成真实 RAG 系统的示例代码
此文件展示如何将 Qwen2.5-14B-Instruct 和检索器集成到评估框架中
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

# ============================
# 1. 模型加载（仅需执行一次）
# ============================

def load_qwen_model(model_path="/home/data2/huangxiaolin2/models/Qwen2.5-14B-Instruct"):
    """
    加载 Qwen2.5-14B-Instruct 模型
    
    Args:
        model_path: 模型路径
        
    Returns:
        model, tokenizer
    """
    print(f"Loading model from {model_path}...")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # 加载模型（自动使用多 GPU）
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # 使用 FP16 节省显存
        device_map="auto",          # 自动分配到多 GPU
        trust_remote_code=True
    )
    
    print(f"Model loaded on devices: {model.hf_device_map}")
    return model, tokenizer


# ============================
# 2. 检索器加载
# ============================

def load_retriever():
    """
    加载向量检索器
    
    Returns:
        retriever: 检索器对象
    """
    # 方法 1: 使用已有的 RAG_Models
    try:
        from RAG_Models.retrieval import build_vector_store
        retriever = build_vector_store()
        print("Retriever loaded from RAG_Models")
        return retriever
    except ImportError:
        print("Warning: RAG_Models not found, using fallback")
    
    # 方法 2: 使用 LangChain
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        retriever = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
        print("Retriever loaded with Chroma")
        return retriever
    except ImportError:
        print("Error: Cannot load retriever")
        return None


# ============================
# 3. RAG 推理函数
# ============================

def rag_inference(
    model,
    tokenizer,
    retriever,
    question_dict: dict,
    top_k: int = 5
) -> str:
    """
    使用 RAG 回答问题
    
    Args:
        model: LLM 模型
        tokenizer: Tokenizer
        retriever: 检索器
        question_dict: 问题字典 {'question': ..., 'options': [...]}
        top_k: 检索 Top-K 文档
        
    Returns:
        LLM 输出字符串
    """
    question_text = question_dict['question']
    options = question_dict['options']
    
    # 步骤 1: 检索相关文档
    if retriever is not None:
        docs = retriever.similarity_search(question_text, k=top_k)
        context = "\n\n".join([doc.page_content for doc in docs])
    else:
        context = "[No context available]"
    
    # 步骤 2: 构建提示
    prompt = f"""You are an expert on O-RAN (Open Radio Access Network) technology.
Based on the following context from O-RAN specifications, answer the multiple choice question.

**IMPORTANT**: Output ONLY the number of the correct answer (1, 2, 3, or 4). Do not include any explanation.

Context:
{context}

Question:
{question_text}

Options:
1. {options[0]}
2. {options[1]}
3. {options[2]}
4. {options[3]}

Answer (output only the number):"""
    
    # 步骤 3: LLM 推理
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,          # 只需要输出一个数字
            temperature=0.1,            # 低温度减少随机性
            do_sample=False,            # 使用贪婪解码
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 步骤 4: 解码输出
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取答案部分（去掉 prompt）
    answer_text = generated_text[len(prompt):].strip()
    
    return answer_text


def extract_answer_number(llm_output: str) -> int:
    """
    从 LLM 输出提取答案数字 (1-4)
    
    Args:
        llm_output: LLM 输出字符串
        
    Returns:
        答案数字 (1-4) 或 0（无法解析）
    """
    # 清理输出
    output = llm_output.strip()
    
    # 模式 1: 纯数字
    if output in ['1', '2', '3', '4']:
        return int(output)
    
    # 模式 2: "Answer: N" 或 "Answer is N"
    match = re.search(r'answer[:\s]+(\d)', output, re.IGNORECASE)
    if match:
        num = int(match.group(1))
        if 1 <= num <= 4:
            return num
    
    # 模式 3: 查找第一个 1-4 的数字
    match = re.search(r'[1-4]', output)
    if match:
        return int(match.group())
    
    # 无法解析
    print(f"Warning: Cannot extract answer from '{output}'")
    return 0


# ============================
# 4. 批量评估函数
# ============================

def evaluate_with_real_rag(
    model,
    tokenizer,
    retriever,
    questions: list,
    top_k: int = 5
) -> dict:
    """
    使用真实 RAG 系统评估问题
    
    Args:
        model: LLM 模型
        tokenizer: Tokenizer
        retriever: 检索器
        questions: 问题列表
        top_k: 检索 Top-K
        
    Returns:
        评估结果
    """
    results = {
        'correct': 0,
        'total': len(questions),
        'details': []
    }
    
    for i, q in enumerate(questions):
        print(f"\r[{i+1}/{len(questions)}] Processing...", end='')
        
        # RAG 推理
        llm_output = rag_inference(model, tokenizer, retriever, q, top_k=top_k)
        
        # 提取答案
        predicted = extract_answer_number(llm_output)
        correct = q['correct_answer']
        
        is_correct = (predicted == correct)
        results['correct'] += int(is_correct)
        
        results['details'].append({
            'question_id': q['id'],
            'question': q['question'],
            'predicted': predicted,
            'correct': correct,
            'is_correct': is_correct,
            'llm_output': llm_output
        })
    
    print()  # 换行
    results['accuracy'] = results['correct'] / results['total']
    return results


# ============================
# 5. 主函数（示例用法）
# ============================

def main():
    """
    主函数：展示如何使用真实 RAG 进行评估
    """
    print("=" * 80)
    print("真实 RAG 系统评估示例")
    print("=" * 80)
    
    # 加载基准
    from oran_benchmark_loader import ORANBenchmark
    benchmark = ORANBenchmark()
    
    # 采样问题（先用小样本测试）
    questions = benchmark.sample_questions(n=10, difficulty='easy', seed=42)
    print(f"\nSampled {len(questions)} questions for testing")
    
    # 加载模型和检索器
    print("\nLoading model and retriever...")
    model, tokenizer = load_qwen_model()
    retriever = load_retriever()
    
    # 运行评估
    print("\nRunning evaluation...")
    results = evaluate_with_real_rag(
        model, tokenizer, retriever,
        questions, top_k=5
    )
    
    # 显示结果
    print("\n" + "=" * 80)
    print(f"Accuracy: {results['accuracy']:.3f} ({results['correct']}/{results['total']})")
    print("=" * 80)
    
    # 显示错误案例
    print("\nError Analysis:")
    for detail in results['details']:
        if not detail['is_correct']:
            print(f"\n✗ Question: {detail['question'][:100]}...")
            print(f"  Predicted: {detail['predicted']}, Correct: {detail['correct']}")
            print(f"  LLM Output: {detail['llm_output']}")
    
    return results


# ============================
# 6. 集成到 Exp_RAG_benchmark.py
# ============================

"""
如何集成到现有评估框架:

1. 在 Exp_RAG_benchmark.py 文件顶部添加:
   from integrate_real_rag import load_qwen_model, load_retriever, rag_inference, extract_answer_number

2. 在 run_benchmark_experiment() 函数中:
   
   # 在函数开始处加载模型（只加载一次）
   if use_real_rag:
       model, tokenizer = load_qwen_model()
       retriever = load_retriever()
   
3. 在 evaluate_rag_on_benchmark() 函数中替换第 67-78 行:
   
   if use_real_rag and RAG_AVAILABLE:
       llm_output = rag_inference(
           model, tokenizer, retriever,
           q, top_k=retrieval_config['top_k']
       )
       predicted = extract_answer_number(llm_output)
   else:
       # 保留原有的模拟逻辑
       predicted = np.random.choice([1, 2, 3, 4])

4. 运行实验:
   python Exp_RAG_benchmark.py --use_real_rag
"""

if __name__ == "__main__":
    # 测试真实 RAG 系统
    results = main()
    
    # 保存结果
    import json
    with open('real_rag_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Results saved to real_rag_test_results.json")
