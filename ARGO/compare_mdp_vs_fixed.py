"""
对比实验：MDP-Guided vs Fixed Strategy
使用小模型 (Qwen2.5-1.5B/3B)
"""

import os
import sys
import numpy as np
import json
import datetime
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mdp_rag_small_llm import SmallLLM_MDP_RAG
from oran_benchmark_loader import ORANBenchmark


class FixedStrategyRAG(SmallLLM_MDP_RAG):
    """
    固定策略 RAG (不使用 MDP)
    总是执行：Retrieve k次 → Reason 1次 → Terminate
    """
    
    def __init__(self, model_name: str, k: int = 3, device: str = "auto"):
        """
        Args:
            k: 固定检索次数
        """
        super().__init__(model_name=model_name, use_mdp=False, device=device)
        self.k = k
        print(f"\n✓ Fixed Strategy: Always retrieve {k} times before reasoning\n")
    
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
            
            C += 0.1
            num_retrievals += 1
            
            history.append({
                'iteration': i + 1,
                'action': 'retrieve',
                'cost': float(C)
            })
        
        # 阶段2：1次推理
        if verbose:
            print(f"    Iter {self.k+1}: Action=reason (final)")
        
        answer, confidence = self.reason_with_llm(question, num_retrievals)
        C += 0.05
        num_reasons += 1
        
        history.append({
            'iteration': self.k + 1,
            'action': 'reason',
            'cost': float(C)
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
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    n_questions: int = 20,
    difficulty: str = "easy",
    fixed_k: int = 3,
    seed: int = 42
):
    """
    运行对比实验
    """
    print("="*80)
    print(f"MDP-Guided vs Fixed Strategy Comparison")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Questions: {n_questions} ({difficulty})")
    print(f"Fixed k: {fixed_k}")
    print(f"Seed: {seed}")
    print("="*80)
    
    # 加载问题
    np.random.seed(seed)
    benchmark = ORANBenchmark()
    questions = benchmark.sample_questions(n=n_questions, difficulty=difficulty, seed=seed)
    
    # ============ 实验1: MDP-Guided ============
    print("\n" + "="*80)
    print("Experiment 1: MDP-Guided Strategy")
    print("="*80)
    
    mdp_rag = SmallLLM_MDP_RAG(model_name=model_name, use_mdp=True)
    mdp_results = []
    mdp_correct = 0
    
    for i, q in enumerate(questions):
        print(f"\n[MDP {i+1}/{n_questions}]", end=" ")
        result = mdp_rag.answer_question(q, verbose=True)
        mdp_results.append(result)
        
        if result['is_correct']:
            mdp_correct += 1
    
    mdp_accuracy = mdp_correct / n_questions
    mdp_avg_cost = np.mean([r['total_cost'] for r in mdp_results])
    mdp_avg_iters = np.mean([r['iterations'] for r in mdp_results])
    mdp_total_retrievals = sum([r['num_retrievals'] for r in mdp_results])
    mdp_total_reasons = sum([r['num_reasons'] for r in mdp_results])
    
    # ============ 实验2: Fixed Strategy ============
    print("\n\n" + "="*80)
    print("Experiment 2: Fixed Strategy")
    print("="*80)
    
    fixed_rag = FixedStrategyRAG(model_name=model_name, k=fixed_k)
    fixed_results = []
    fixed_correct = 0
    
    for i, q in enumerate(questions):
        print(f"\n[Fixed {i+1}/{n_questions}]", end=" ")
        result = fixed_rag.answer_question(q, verbose=True)
        fixed_results.append(result)
        
        if result['is_correct']:
            fixed_correct += 1
    
    fixed_accuracy = fixed_correct / n_questions
    fixed_avg_cost = np.mean([r['total_cost'] for r in fixed_results])
    fixed_avg_iters = np.mean([r['iterations'] for r in fixed_results])
    fixed_total_retrievals = sum([r['num_retrievals'] for r in fixed_results])
    fixed_total_reasons = sum([r['num_reasons'] for r in fixed_results])
    
    # ============ 结果对比 ============
    print("\n\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    print(f"\nMDP-Guided Strategy:")
    print(f"  Accuracy:        {mdp_accuracy:.1%} ({mdp_correct}/{n_questions})")
    print(f"  Avg Cost:        {mdp_avg_cost:.3f}")
    print(f"  Avg Iterations:  {mdp_avg_iters:.2f}")
    print(f"  Total Retrievals: {mdp_total_retrievals}")
    print(f"  Total Reasons:    {mdp_total_reasons}")
    
    print(f"\nFixed Strategy (k={fixed_k}):")
    print(f"  Accuracy:        {fixed_accuracy:.1%} ({fixed_correct}/{n_questions})")
    print(f"  Avg Cost:        {fixed_avg_cost:.3f}")
    print(f"  Avg Iterations:  {fixed_avg_iters:.2f}")
    print(f"  Total Retrievals: {fixed_total_retrievals}")
    print(f"  Total Reasons:    {fixed_total_reasons}")
    
    print(f"\n{'─'*80}")
    print("Improvement:")
    acc_improvement = (mdp_accuracy - fixed_accuracy) / fixed_accuracy * 100
    cost_change = (mdp_avg_cost - fixed_avg_cost) / fixed_avg_cost * 100
    
    print(f"  Accuracy: {acc_improvement:+.1f}% ({mdp_accuracy:.1%} vs {fixed_accuracy:.1%})")
    print(f"  Cost:     {cost_change:+.1f}% ({mdp_avg_cost:.3f} vs {fixed_avg_cost:.3f})")
    
    if mdp_accuracy > fixed_accuracy:
        print(f"\n✓ MDP strategy achieves {acc_improvement:.1f}% higher accuracy!")
    else:
        print(f"\n⚠ Fixed strategy is better (unexpected)")
    
    print("="*80)
    
    # ============ 保存结果 ============
    output = {
        'timestamp': datetime.datetime.now().isoformat(),
        'model': model_name,
        'config': {
            'n_questions': n_questions,
            'difficulty': difficulty,
            'fixed_k': fixed_k,
            'seed': seed
        },
        'mdp_strategy': {
            'accuracy': float(mdp_accuracy),
            'avg_cost': float(mdp_avg_cost),
            'avg_iterations': float(mdp_avg_iters),
            'total_retrievals': int(mdp_total_retrievals),
            'total_reasons': int(mdp_total_reasons),
            'results': mdp_results
        },
        'fixed_strategy': {
            'accuracy': float(fixed_accuracy),
            'avg_cost': float(fixed_avg_cost),
            'avg_iterations': float(fixed_avg_iters),
            'total_retrievals': int(fixed_total_retrievals),
            'total_reasons': int(fixed_total_reasons),
            'results': fixed_results
        },
        'comparison': {
            'accuracy_improvement_percent': float(acc_improvement),
            'cost_change_percent': float(cost_change),
            'mdp_better': bool(mdp_accuracy > fixed_accuracy)
        }
    }
    
    os.makedirs('results/comparison', exist_ok=True)
    model_short = model_name.split('/')[-1]
    output_file = f"results/comparison/{model_short}_{difficulty}_{n_questions}q_mdp_vs_fixed_k{fixed_k}.json"
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    return output


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare MDP-Guided vs Fixed Strategy')
    parser.add_argument('--model', type=str, 
                       default="Qwen/Qwen2.5-1.5B-Instruct",
                       help='Model name or path')
    parser.add_argument('-n', '--num_questions', type=int, default=20,
                       help='Number of questions')
    parser.add_argument('-d', '--difficulty', type=str, 
                       choices=['easy', 'medium', 'hard'], default='easy',
                       help='Question difficulty')
    parser.add_argument('-k', '--fixed_k', type=int, default=3,
                       help='Number of retrievals for fixed strategy')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    run_comparison(
        model_name=args.model,
        n_questions=args.num_questions,
        difficulty=args.difficulty,
        fixed_k=args.fixed_k,
        seed=args.seed
    )
