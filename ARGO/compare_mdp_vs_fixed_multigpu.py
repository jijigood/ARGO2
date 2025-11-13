"""
Multi-GPU MDP vs Fixed Strategy Comparison
对比MDP自适应策略 vs Fixed-K策略

符合ARGO V3.0规范：
- 修正成本参数 c_r=0.05, c_p=0.02
- 完整推理链追踪
- Phase2: 检索成功率p_s=0.8
"""

import sys
import os
import yaml
import torch
import json
import random
from typing import List, Dict
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mdp_rag_multi_gpu import MultiGPU_MDP_RAG
from oran_benchmark_loader import ORANBenchmark

# 加载配置
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'configs', 'multi_gpu.yaml')
with open(CONFIG_PATH, 'r') as f:
    CONFIG = yaml.safe_load(f)

import os
import sys
import torch
import numpy as np
import json
import datetime
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mdp_rag_multi_gpu import MultiGPU_MDP_RAG
from oran_benchmark_loader import ORANBenchmark


class MultiGPU_FixedStrategyRAG(MultiGPU_MDP_RAG):
    """
    多GPU固定策略 RAG
    """
    
    def __init__(
        self,
        model_name: str,
        k: int = 3,
        gpu_mode: str = "auto",
        gpu_ids: List[int] = None,
        max_memory_per_gpu: str = "10GB"
    ):
        """
        Args:
            k: 固定检索次数
        """
        super().__init__(
            model_name=model_name,
            use_mdp=False,
            gpu_mode=gpu_mode,
            gpu_ids=gpu_ids,
            max_memory_per_gpu=max_memory_per_gpu
        )
        self.k = k
        
        # 从配置加载成本参数
        self.c_r = CONFIG['mdp']['c_r']  # 0.05
        self.c_p = CONFIG['mdp']['c_p']  # 0.02
        
        print(f"\n✓ Fixed Strategy: Always retrieve {k} times before reasoning")
        print(f"  c_r = {self.c_r:.3f} (修正后)")
        print(f"  c_p = {self.c_p:.3f} (修正后)\n")
    
    def answer_question(self, question: dict, verbose: bool = False) -> Dict:
        """固定策略：k次检索 + 1次推理"""
        C = 0.0
        num_retrievals = 0
        num_reasons = 0
        history = []
        
        if verbose:
            print(f"\n  Q: {question['question'][:60]}...")
        
        # 阶段1：固定k次检索
        for i in range(self.k):
            if verbose:
                print(f"    Iter {i+1}: Action=retrieve (fixed)")
            
            # Phase2: 实现检索成功率 p_s
            retrieval_success = random.random() < CONFIG['mdp']['p_s']
            
            C += self.c_r  # 使用配置参数0.05
            num_retrievals += 1
            
            # 完整history追踪 - Retrieve
            history.append({
                'iteration': i + 1,
                'action': 'retrieve',
                'subquery': question['question'],  # TODO Phase3: Decomposer
                'retrieved_docs': [],  # TODO Phase3: 真实检索器
                'retrieval_success': retrieval_success,  # Phase2: 基于p_s
                'response': None,
                'intermediate_answer': None,
                'confidence': None,
                'uncertainty': None,  # Fixed策略不追踪U
                'cost': float(C),
                'U_before': None,
                'U_after': None
            })
            
            if verbose:
                status = "✓" if retrieval_success else "✗"
                print(f"      → {status} Retrieval {'success' if retrieval_success else 'failed'}")
        
        # 阶段2：1次推理
        if verbose:
            print(f"    Iter {self.k+1}: Action=reason (final)")
        
        answer, confidence = self.reason_with_llm(question, num_retrievals)
        llm_response = f"Based on O-RAN knowledge, the answer is {answer}"
        
        C += self.c_p  # 使用配置参数0.02
        num_reasons += 1
        
        # 完整history追踪 - Reason
        history.append({
            'iteration': self.k + 1,
            'action': 'reason',
            'subquery': question['question'],  # TODO Phase3: Decomposer
            'retrieved_docs': [],
            'retrieval_success': None,
            'response': llm_response,  # LLM完整响应
            'intermediate_answer': answer,  # 最终答案
            'confidence': float(confidence),
            'uncertainty': None,
            'cost': float(C),
            'U_before': None,
            'U_after': None
        })
        
        if verbose:
            print(f"      → Answer: {answer}, Confidence: {confidence:.2f}")
        
        is_correct = (answer == question['correct_answer'])
        
        if verbose:
            icon = "✓" if is_correct else "✗"
            print(f"  {icon} Predicted: {answer}, Correct: {question['correct_answer']}")
        
        return {
            'question_id': question['id'],
            'predicted': answer,
            'correct': question['correct_answer'],
            'is_correct': is_correct,
            'iterations': len(history),
            'num_retrievals': num_retrievals,
            'num_reasons': num_reasons,
            'total_cost': float(C),
            'history': history
        }


def run_comparison(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    n_questions: int = 100,
    difficulty: str = "medium",
    fixed_k: int = 3,
    gpu_mode: str = "auto",
    gpu_ids: List[int] = None,
    seed: int = 42
):
    """
    运行对比实验: MDP vs Fixed
    
    Args:
        model_name: 模型名称
        n_questions: 问题数量
        difficulty: 难度级别
        fixed_k: 固定策略的检索次数
        gpu_mode: GPU模式
        gpu_ids: 使用的GPU列表
        seed: 随机种子
    """
    
    print(f"\n{'='*80}")
    print(f"Multi-GPU MDP vs Fixed Comparison")
    print(f"{'='*80}")
    print(f"Model: {model_name}")
    print(f"Questions: {n_questions} ({difficulty})")
    print(f"Fixed K: {fixed_k}")
    print(f"GPU Mode: {gpu_mode}")
    print(f"GPUs: {gpu_ids if gpu_ids else 'all'}")
    print(f"Seed: {seed}")
    print(f"{'='*80}\n")
    
    # 加载数据集
    benchmark = ORANBenchmark()
    questions = benchmark.sample_questions(n_questions, difficulty, seed)
    
    # ========== 策略1: MDP-Guided ==========
    print(f"\n{'='*80}")
    print(f"Strategy 1: MDP-Guided")
    print(f"{'='*80}\n")
    
    mdp_rag = MultiGPU_MDP_RAG(
        model_name=model_name,
        use_mdp=True,
        gpu_mode=gpu_mode,
        gpu_ids=gpu_ids
    )
    
    mdp_results = []
    mdp_correct = 0
    mdp_total_cost = 0.0
    mdp_total_iterations = 0
    
    print(f"Evaluating MDP strategy on {len(questions)} questions...\n")
    
    for i, question in enumerate(questions, 1):
        result = mdp_rag.answer_question(question, verbose=(i <= 2))
        mdp_results.append(result)
        
        if result['is_correct']:
            mdp_correct += 1
        mdp_total_cost += result['total_cost']
        mdp_total_iterations += result['iterations']
        
        if i % 20 == 0:
            acc = mdp_correct / i
            avg_cost = mdp_total_cost / i
            avg_iter = mdp_total_iterations / i
            print(f"  MDP [{i}/{len(questions)}] Acc={acc:.2%}, AvgCost={avg_cost:.3f}, AvgIter={avg_iter:.1f}")
    
    mdp_accuracy = mdp_correct / len(questions)
    mdp_avg_cost = mdp_total_cost / len(questions)
    mdp_avg_iterations = mdp_total_iterations / len(questions)
    
    print(f"\n✓ MDP Strategy Complete:")
    print(f"  Accuracy: {mdp_accuracy:.2%} ({mdp_correct}/{len(questions)})")
    print(f"  Avg Cost: {mdp_avg_cost:.3f}")
    print(f"  Avg Iterations: {mdp_avg_iterations:.1f}")
    
    # 清理GPU内存
    del mdp_rag
    torch.cuda.empty_cache()
    
    # ========== 策略2: Fixed-K ==========
    print(f"\n{'='*80}")
    print(f"Strategy 2: Fixed-K (k={fixed_k})")
    print(f"{'='*80}\n")
    
    fixed_rag = MultiGPU_FixedStrategyRAG(
        model_name=model_name,
        k=fixed_k,
        gpu_mode=gpu_mode,
        gpu_ids=gpu_ids
    )
    
    fixed_results = []
    fixed_correct = 0
    fixed_total_cost = 0.0
    fixed_total_iterations = 0
    
    print(f"Evaluating Fixed strategy on {len(questions)} questions...\n")
    
    for i, question in enumerate(questions, 1):
        result = fixed_rag.answer_question(question, verbose=(i <= 2))
        fixed_results.append(result)
        
        if result['is_correct']:
            fixed_correct += 1
        fixed_total_cost += result['total_cost']
        fixed_total_iterations += result['iterations']
        
        if i % 20 == 0:
            acc = fixed_correct / i
            avg_cost = fixed_total_cost / i
            avg_iter = fixed_total_iterations / i
            print(f"  Fixed [{i}/{len(questions)}] Acc={acc:.2%}, AvgCost={avg_cost:.3f}, AvgIter={avg_iter:.1f}")
    
    fixed_accuracy = fixed_correct / len(questions)
    fixed_avg_cost = fixed_total_cost / len(questions)
    fixed_avg_iterations = fixed_total_iterations / len(questions)
    
    print(f"\n✓ Fixed Strategy Complete:")
    print(f"  Accuracy: {fixed_accuracy:.2%} ({fixed_correct}/{len(questions)})")
    print(f"  Avg Cost: {fixed_avg_cost:.3f}")
    print(f"  Avg Iterations: {fixed_avg_iterations:.1f}")
    
    # ========== 对比结果 ==========
    print(f"\n{'='*80}")
    print(f"COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"\n{'Strategy':<20} {'Accuracy':<15} {'Avg Cost':<15} {'Avg Iterations':<15}")
    print(f"{'-'*80}")
    print(f"{'MDP-Guided':<20} {mdp_accuracy:>6.2%} ({mdp_correct:>3}/{len(questions):>3})  {mdp_avg_cost:>8.3f}        {mdp_avg_iterations:>8.1f}")
    print(f"{'Fixed-K (k='+str(fixed_k)+')':<20} {fixed_accuracy:>6.2%} ({fixed_correct:>3}/{len(questions):>3})  {fixed_avg_cost:>8.3f}        {fixed_avg_iterations:>8.1f}")
    print(f"{'-'*80}")
    
    # 计算提升
    acc_improvement = mdp_accuracy - fixed_accuracy
    cost_diff = mdp_avg_cost - fixed_avg_cost
    
    print(f"\n{'Improvement':<20} {acc_improvement:>+6.2%}           {cost_diff:>+8.3f}        {mdp_avg_iterations - fixed_avg_iterations:>+8.1f}")
    print(f"{'='*80}\n")
    
    # 保存结果
    model_short = model_name.split('/')[-1]
    output = {
        'timestamp': datetime.datetime.now().isoformat(),
        'model': model_name,
        'gpu_mode': gpu_mode,
        'gpu_ids': gpu_ids if gpu_ids else list(range(torch.cuda.device_count())),
        'config': {
            'n_questions': n_questions,
            'difficulty': difficulty,
            'fixed_k': fixed_k,
            'seed': seed
        },
        'mdp_strategy': {
            'accuracy': mdp_accuracy,
            'avg_cost': mdp_avg_cost,
            'avg_iterations': mdp_avg_iterations,
            'results': mdp_results
        },
        'fixed_strategy': {
            'accuracy': fixed_accuracy,
            'avg_cost': fixed_avg_cost,
            'avg_iterations': fixed_avg_iterations,
            'results': fixed_results
        },
        'improvement': {
            'accuracy': float(acc_improvement),
            'cost': float(cost_diff)
        }
    }
    
    os.makedirs('results/multi_gpu_comparison', exist_ok=True)
    output_file = f'results/multi_gpu_comparison/{model_short}_{difficulty}_{n_questions}q_mdp_vs_fixed_k{fixed_k}.json'
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to: {output_file}\n")
    
    return output


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-GPU MDP vs Fixed Comparison")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model name or path")
    parser.add_argument("-n", "--n_questions", type=int, default=100,
                        help="Number of questions")
    parser.add_argument("-d", "--difficulty", type=str, default="medium",
                        choices=["easy", "medium", "hard", "mixed"],
                        help="Question difficulty")
    parser.add_argument("-k", "--fixed_k", type=int, default=3,
                        help="Fixed strategy retrieval count")
    parser.add_argument("--gpu_mode", type=str, default="auto",
                        choices=["auto", "single", "data_parallel", "accelerate"],
                        help="GPU mode")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=None,
                        help="GPU IDs to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    run_comparison(
        model_name=args.model,
        n_questions=args.n_questions,
        difficulty=args.difficulty,
        fixed_k=args.fixed_k,
        gpu_mode=args.gpu_mode,
        gpu_ids=args.gpu_ids,
        seed=args.seed
    )
