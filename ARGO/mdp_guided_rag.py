"""
ARGO MDP-Guided RAG 系统
结合 MDP 最优策略和真实 RAG 系统的完整实现

核心思想:
1. MDP 策略控制 Retrieve/Reason/Terminate 决策
2. 每次 Retrieve 增加成本，提高 uncertainty reduction
3. 根据当前 uncertainty 和成本，动态决定是否继续检索
4. 最终在 uncertainty 足够低或成本过高时 Terminate
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import json
from typing import Dict, List, Tuple
import datetime

# 导入 ARGO_MDP 项目的核心组件
import sys
sys.path.insert(0, '../ARGO_MDP/src')
try:
    from mdp_solver import MDPSolver
    from env_argo import ARGOEnv
    MDP_AVAILABLE = True
except ImportError:
    print("Warning: ARGO_MDP not available. Using fallback policies.")
    MDP_AVAILABLE = False

# 导入 RAG 组件
from oran_benchmark_loader import ORANBenchmark
try:
    from RAG_Models.retrieval import build_vector_store
    from RAG_Models.document_loader import load_documents
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# 导入 LLM（如果可用）
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


class MDPGuidedRAG:
    """
    MDP 引导的 RAG 系统
    
    核心流程:
    1. 初始 uncertainty U = U_max (最大不确定性)
    2. While True:
        a. 根据当前 U 和累积成本，查询 MDP 策略
        b. 如果 action = Terminate: 结束，输出当前答案
        c. 如果 action = Retrieve: 检索文档，更新 U (降低)
        d. 如果 action = Reason: LLM 推理，更新 U (降低)
    3. 返回最终答案
    """
    
    def __init__(
        self,
        mdp_config_path: str = "../ARGO_MDP/configs/base.yaml",
        model_path: str = None,
        retriever = None,
        use_real_llm: bool = False
    ):
        """
        初始化 MDP-Guided RAG 系统
        
        Args:
            mdp_config_path: MDP 配置文件路径
            model_path: LLM 模型路径
            retriever: 检索器对象
            use_real_llm: 是否使用真实 LLM（否则模拟）
        """
        self.use_real_llm = use_real_llm and LLM_AVAILABLE
        
        # 加载 MDP 策略
        if MDP_AVAILABLE:
            import yaml
            with open(mdp_config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.mdp_solver = MDPSolver(config)
            self.mdp_solver.solve()  # 计算最优策略
            
            # 获取最优阈值
            self.theta_star = self.mdp_solver.theta_star
            self.theta_cont = self.mdp_solver.theta_cont
            
            print(f"MDP Policy Loaded:")
            print(f"  θ* (termination) = {self.theta_star:.3f}")
            print(f"  θ_cont (continue) = {self.theta_cont:.3f}")
        else:
            # 使用默认阈值
            self.theta_star = 0.5
            self.theta_cont = 0.2
            print("Using default thresholds (MDP not available)")
        
        # 加载 LLM（如果使用真实模型）
        if self.use_real_llm:
            print(f"Loading LLM from {model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            print("LLM loaded successfully")
        else:
            self.model = None
            self.tokenizer = None
        
        # 设置检索器
        self.retriever = retriever
        
        # 统计信息
        self.stats = {
            'total_retrievals': 0,
            'total_reasons': 0,
            'total_terminates': 0
        }
    
    def get_action(self, uncertainty: float, cumulative_cost: float) -> str:
        """
        根据 MDP 策略获取下一步动作
        
        Args:
            uncertainty: 当前不确定性 U ∈ [0, 1]
            cumulative_cost: 累积成本
            
        Returns:
            action: 'retrieve', 'reason', 或 'terminate'
        """
        # MDP 决策逻辑（基于最优阈值）
        if uncertainty >= self.theta_star:
            # 高不确定性 → Retrieve（获取更多信息）
            return 'retrieve'
        elif uncertainty >= self.theta_cont:
            # 中等不确定性 → Reason（LLM 推理）
            return 'reason'
        else:
            # 低不确定性 → Terminate（输出答案）
            return 'terminate'
    
    def retrieve_documents(
        self,
        question: str,
        retrieved_docs: List[str],
        top_k: int = 3
    ) -> Tuple[List[str], float]:
        """
        检索相关文档
        
        Args:
            question: 问题文本
            retrieved_docs: 已检索的文档列表
            top_k: 检索数量
            
        Returns:
            new_docs: 新检索到的文档
            delta_u: 不确定性减少量
        """
        if self.retriever is None:
            # 模拟检索
            new_docs = [f"[Simulated doc {len(retrieved_docs)+i+1}]" for i in range(top_k)]
            delta_u = 0.15  # 模拟不确定性减少
        else:
            # 真实检索
            results = self.retriever.similarity_search(
                question, 
                k=top_k,
                filter={"doc_id": {"$nin": [d.metadata.get('doc_id') for d in retrieved_docs]}}
            )
            new_docs = results
            
            # 根据检索文档数量估计不确定性减少
            delta_u = min(len(new_docs) * 0.05, 0.2)
        
        self.stats['total_retrievals'] += 1
        return new_docs, delta_u
    
    def reason_with_llm(
        self,
        question: dict,
        context: List[str],
        current_answer: int = None
    ) -> Tuple[int, float, str]:
        """
        使用 LLM 进行推理
        
        Args:
            question: 问题字典
            context: 上下文文档列表
            current_answer: 当前答案（如果有）
            
        Returns:
            answer: 预测答案 (1-4)
            delta_u: 不确定性减少量
            llm_output: LLM 原始输出
        """
        if not self.use_real_llm:
            # 模拟推理
            answer = np.random.choice([1, 2, 3, 4])
            delta_u = 0.1
            llm_output = f"Simulated answer: {answer}"
        else:
            # 真实 LLM 推理
            context_text = "\n\n".join([
                doc.page_content if hasattr(doc, 'page_content') else str(doc)
                for doc in context
            ])
            
            prompt = f"""Based on the following context, answer the multiple choice question.
Output ONLY the number (1, 2, 3, or 4) of the correct answer.

Context:
{context_text}

Question: {question['question']}

Options:
1. {question['options'][0]}
2. {question['options'][1]}
3. {question['options'][2]}
4. {question['options'][3]}

Answer (1-4):"""
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            llm_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            llm_output = llm_output[len(prompt):].strip()
            
            # 提取答案
            answer = self._extract_answer(llm_output)
            
            # 根据是否改变答案估计不确定性减少
            if current_answer is None or answer != current_answer:
                delta_u = 0.12  # 答案改变，不确定性减少较多
            else:
                delta_u = 0.08  # 答案不变，不确定性减少较少
        
        self.stats['total_reasons'] += 1
        return answer, delta_u, llm_output
    
    def _extract_answer(self, llm_output: str) -> int:
        """从 LLM 输出提取答案"""
        import re
        
        # 尝试多种模式
        if llm_output.strip() in ['1', '2', '3', '4']:
            return int(llm_output.strip())
        
        match = re.search(r'[1-4]', llm_output)
        if match:
            return int(match.group())
        
        return np.random.choice([1, 2, 3, 4])  # 无法解析，随机猜测
    
    def answer_question(
        self,
        question: dict,
        max_iterations: int = 5,
        verbose: bool = False
    ) -> Dict:
        """
        使用 MDP-Guided RAG 回答问题
        
        Args:
            question: 问题字典
            max_iterations: 最大迭代次数
            verbose: 是否打印详细信息
            
        Returns:
            result: 包含答案、迭代历史等信息的字典
        """
        # 初始化状态
        uncertainty = 1.0  # 初始最大不确定性
        cumulative_cost = 0.0
        retrieved_docs = []
        current_answer = None
        iteration_history = []
        
        for iteration in range(max_iterations):
            # 查询 MDP 策略
            action = self.get_action(uncertainty, cumulative_cost)
            
            if verbose:
                print(f"\n  Iteration {iteration+1}: U={uncertainty:.3f}, Cost={cumulative_cost:.2f}, Action={action}")
            
            # 执行动作
            if action == 'terminate':
                self.stats['total_terminates'] += 1
                break
            
            elif action == 'retrieve':
                # 检索文档
                new_docs, delta_u = self.retrieve_documents(
                    question['question'],
                    retrieved_docs,
                    top_k=3
                )
                retrieved_docs.extend(new_docs)
                uncertainty = max(0, uncertainty - delta_u)
                cumulative_cost += 0.1  # 检索成本
                
                if verbose:
                    print(f"    Retrieved {len(new_docs)} docs, ΔU={-delta_u:.3f}")
            
            elif action == 'reason':
                # LLM 推理
                answer, delta_u, llm_output = self.reason_with_llm(
                    question,
                    retrieved_docs,
                    current_answer
                )
                current_answer = answer
                uncertainty = max(0, uncertainty - delta_u)
                cumulative_cost += 0.05  # 推理成本
                
                if verbose:
                    print(f"    LLM answer: {answer}, ΔU={-delta_u:.3f}")
            
            # 记录历史
            iteration_history.append({
                'iteration': iteration + 1,
                'action': action,
                'uncertainty': uncertainty,
                'cost': cumulative_cost,
                'num_docs': len(retrieved_docs),
                'current_answer': current_answer
            })
        
        # 如果没有推理过，强制推理一次
        if current_answer is None:
            current_answer, _, _ = self.reason_with_llm(question, retrieved_docs)
        
        return {
            'predicted_answer': current_answer,
            'correct_answer': question['correct_answer'],
            'is_correct': current_answer == question['correct_answer'],
            'iterations': len(iteration_history),
            'final_uncertainty': uncertainty,
            'total_cost': cumulative_cost,
            'num_retrievals': sum(1 for h in iteration_history if h['action'] == 'retrieve'),
            'num_reasons': sum(1 for h in iteration_history if h['action'] == 'reason'),
            'history': iteration_history
        }


def run_mdp_rag_experiment(
    n_questions: int = 50,
    difficulty: str = None,
    use_real_llm: bool = False,
    seed: int = 42
):
    """
    运行 MDP-Guided RAG 实验
    
    Args:
        n_questions: 测试问题数量
        difficulty: 难度级别
        use_real_llm: 是否使用真实 LLM
        seed: 随机种子
    """
    print("=" * 80)
    print("MDP-Guided RAG Experiment on ORAN-Bench-13K")
    print("=" * 80)
    print(f"Questions: {n_questions}")
    print(f"Difficulty: {difficulty or 'mixed'}")
    print(f"Use Real LLM: {use_real_llm}")
    print(f"Seed: {seed}")
    print("=" * 80)
    
    # 加载基准
    benchmark = ORANBenchmark()
    questions = benchmark.sample_questions(n=n_questions, difficulty=difficulty, seed=seed)
    
    # 初始化 MDP-RAG 系统
    mdp_rag = MDPGuidedRAG(
        mdp_config_path="../ARGO_MDP/configs/base.yaml",
        model_path="/home/data2/huangxiaolin2/models/Qwen2.5-14B-Instruct" if use_real_llm else None,
        retriever=None,  # TODO: 集成真实检索器
        use_real_llm=use_real_llm
    )
    
    # 评估
    results = []
    correct = 0
    
    for i, q in enumerate(questions):
        print(f"\n[{i+1}/{n_questions}] {q['question'][:80]}...")
        
        result = mdp_rag.answer_question(q, verbose=True)
        results.append(result)
        
        if result['is_correct']:
            correct += 1
            print(f"  ✓ Correct! ({correct}/{i+1})")
        else:
            print(f"  ✗ Wrong. Predicted: {result['predicted_answer']}, Correct: {result['correct_answer']}")
    
    # 统计
    accuracy = correct / n_questions
    avg_iterations = np.mean([r['iterations'] for r in results])
    avg_cost = np.mean([r['total_cost'] for r in results])
    avg_retrievals = np.mean([r['num_retrievals'] for r in results])
    avg_reasons = np.mean([r['num_reasons'] for r in results])
    
    print("\n" + "=" * 80)
    print("Experiment Results")
    print("=" * 80)
    print(f"Accuracy: {accuracy:.3f} ({correct}/{n_questions})")
    print(f"Avg Iterations: {avg_iterations:.2f}")
    print(f"Avg Cost: {avg_cost:.3f}")
    print(f"Avg Retrievals per Question: {avg_retrievals:.2f}")
    print(f"Avg Reasons per Question: {avg_reasons:.2f}")
    print("=" * 80)
    
    # 保存结果
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            if hasattr(obj, 'item'):
                return obj.item()
            return super().default(obj)
    
    output = {
        'timestamp': datetime.datetime.now().isoformat(),
        'config': {
            'n_questions': n_questions,
            'difficulty': difficulty,
            'use_real_llm': use_real_llm,
            'seed': seed
        },
        'metrics': {
            'accuracy': float(accuracy),
            'avg_iterations': float(avg_iterations),
            'avg_cost': float(avg_cost),
            'avg_retrievals': float(avg_retrievals),
            'avg_reasons': float(avg_reasons)
        },
        'results': results
    }
    
    os.makedirs('results/mdp_rag', exist_ok=True)
    output_file = f"results/mdp_rag/exp_{difficulty or 'mixed'}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    
    print(f"\nResults saved to {output_file}")
    
    return output


if __name__ == "__main__":
    # 测试实验（使用模拟 LLM）
    print("Running MDP-Guided RAG experiment (simulated LLM)...\n")
    
    results = run_mdp_rag_experiment(
        n_questions=10,
        difficulty='easy',
        use_real_llm=False,  # 先用模拟测试
        seed=42
    )
