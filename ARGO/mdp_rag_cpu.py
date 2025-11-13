"""
MDP-Guided RAG 系统 - CPU 版本
适用于 GTX 1080 Ti (CUDA 6.1) 不兼容情况

解决方案:
1. 使用 CPU 进行小规模测试
2. LLM 使用量化版本或更小模型
3. 或者使用模拟模式验证逻辑
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 禁用 GPU

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import json
from typing import Dict, List
import datetime

# 导入基准加载器
from oran_benchmark_loader import ORANBenchmark

# 尝试导入 MDP solver
sys.path.insert(0, '../ARGO_MDP/src')
try:
    from mdp_solver import MDPSolver
    MDP_AVAILABLE = True
except ImportError:
    print("Warning: ARGO_MDP not available. Using default thresholds.")
    MDP_AVAILABLE = False


class SimplifiedMDPRAG:
    """
    简化的 MDP-Guided RAG（CPU 友好版）
    
    主要改进:
    1. 不加载大型 LLM（使用规则或小模型）
    2. 完整的 MDP 逻辑（迭代、阈值、成本优化）
    3. 可以在 CPU 上快速验证
    """
    
    def __init__(self, use_mdp: bool = True):
        """
        初始化简化版 MDP-RAG
        
        Args:
            use_mdp: 是否使用 MDP 策略（否则用固定策略）
        """
        self.use_mdp = use_mdp
        
        if use_mdp and MDP_AVAILABLE:
            # 加载 MDP 策略
            import yaml
            config_path = "../ARGO_MDP/configs/base.yaml"
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                self.mdp_solver = MDPSolver(config)
                self.mdp_solver.solve()
                
                self.theta_star = self.mdp_solver.theta_star
                self.theta_cont = self.mdp_solver.theta_cont
                
                print(f"✓ MDP Policy Loaded:")
                print(f"  θ* (termination) = {self.theta_star:.3f}")
                print(f"  θ_cont (continue) = {self.theta_cont:.3f}")
            else:
                print(f"Warning: Config file not found at {config_path}")
                self.theta_star = 0.5
                self.theta_cont = 0.2
        else:
            # 默认阈值
            self.theta_star = 0.5
            self.theta_cont = 0.2
            print(f"✓ Using default thresholds: θ*={self.theta_star}, θ_cont={self.theta_cont}")
        
        # 统计
        self.stats = {
            'total_questions': 0,
            'total_retrievals': 0,
            'total_reasons': 0,
            'total_cost': 0.0
        }
    
    def get_action(self, uncertainty: float) -> str:
        """
        根据 MDP 策略决定动作
        
        Args:
            uncertainty: 当前不确定性 [0, 1]
            
        Returns:
            'retrieve', 'reason', 或 'terminate'
        """
        if uncertainty >= self.theta_star:
            return 'retrieve'
        elif uncertainty >= self.theta_cont:
            return 'reason'
        else:
            return 'terminate'
    
    def simulate_retrieve(self, question: str, iteration: int) -> float:
        """
        模拟检索（不需要真实检索器）
        
        Returns:
            delta_u: 不确定性减少量
        """
        # 根据迭代次数模拟收益递减
        base_reduction = 0.15
        diminishing_factor = 0.85 ** iteration
        delta_u = base_reduction * diminishing_factor
        
        self.stats['total_retrievals'] += 1
        return delta_u
    
    def simulate_reason(self, question: dict, iteration: int, current_answer: int = None) -> tuple:
        """
        模拟 LLM 推理（基于规则）
        
        Args:
            question: 问题字典
            iteration: 当前迭代次数
            current_answer: 当前答案
            
        Returns:
            (answer, delta_u)
        """
        # 简单规则：基于问题难度和迭代次数
        difficulty_map = {'easy': 0.7, 'medium': 0.5, 'hard': 0.3}
        base_acc = difficulty_map.get(question.get('difficulty', 'medium'), 0.5)
        
        # 迭代越多，越可能答对
        iteration_boost = min(iteration * 0.05, 0.2)
        final_acc = min(base_acc + iteration_boost, 0.9)
        
        # 决定答案
        if np.random.random() < final_acc:
            answer = question['correct_answer']
        else:
            wrong_options = [i for i in [1, 2, 3, 4] if i != question['correct_answer']]
            answer = np.random.choice(wrong_options)
        
        # 不确定性减少
        if current_answer == answer:
            delta_u = 0.08  # 答案不变，减少较少
        else:
            delta_u = 0.12  # 答案改变，减少较多
        
        self.stats['total_reasons'] += 1
        return answer, delta_u
    
    def answer_question(self, question: dict, max_iterations: int = 10, verbose: bool = False) -> Dict:
        """
        使用 MDP 策略回答问题
        
        Args:
            question: 问题字典
            max_iterations: 最大迭代次数
            verbose: 是否打印详细信息
            
        Returns:
            结果字典
        """
        # 初始化
        U = 1.0  # 初始不确定性
        C = 0.0  # 累积成本
        current_answer = None
        history = []
        
        if verbose:
            print(f"\n  Q: {question['question'][:60]}...")
        
        for iteration in range(max_iterations):
            # 查询 MDP 策略
            action = self.get_action(U)
            
            if verbose:
                print(f"    Iter {iteration+1}: U={U:.3f}, C={C:.3f}, Action={action}")
            
            if action == 'terminate':
                break
            
            elif action == 'retrieve':
                delta_u = self.simulate_retrieve(question['question'], iteration)
                U = max(0, U - delta_u)
                C += 0.1  # 检索成本
                
                if verbose:
                    print(f"      → Retrieved docs, ΔU=-{delta_u:.3f}")
            
            elif action == 'reason':
                answer, delta_u = self.simulate_reason(question, iteration, current_answer)
                current_answer = answer
                U = max(0, U - delta_u)
                C += 0.05  # 推理成本
                
                if verbose:
                    print(f"      → Reasoned, answer={answer}, ΔU=-{delta_u:.3f}")
            
            # 记录历史
            history.append({
                'iteration': iteration + 1,
                'action': action,
                'uncertainty': float(U),
                'cost': float(C),
                'answer': current_answer
            })
        
        # 如果没有答案，强制推理一次
        if current_answer is None:
            current_answer, _ = self.simulate_reason(question, 0)
        
        is_correct = (current_answer == question['correct_answer'])
        
        if verbose:
            result_icon = "✓" if is_correct else "✗"
            print(f"  {result_icon} Final: {current_answer} (Correct: {question['correct_answer']})")
        
        self.stats['total_questions'] += 1
        self.stats['total_cost'] += C
        
        return {
            'question_id': question['id'],
            'predicted': current_answer,
            'correct': question['correct_answer'],
            'is_correct': is_correct,
            'iterations': len(history),
            'final_uncertainty': float(U),
            'total_cost': float(C),
            'history': history
        }


def run_comparison_experiment(n_questions: int = 50, difficulty: str = None, seed: int = 42):
    """
    运行对比实验：MDP vs. Fixed Strategy
    
    Args:
        n_questions: 问题数量
        difficulty: 难度级别
        seed: 随机种子
    """
    print("=" * 80)
    print("MDP-Guided RAG vs. Fixed Strategy Comparison")
    print("=" * 80)
    print(f"Questions: {n_questions}, Difficulty: {difficulty or 'mixed'}, Seed: {seed}")
    print("=" * 80)
    
    # 加载问题
    np.random.seed(seed)
    benchmark = ORANBenchmark()
    questions = benchmark.sample_questions(n=n_questions, difficulty=difficulty, seed=seed)
    
    # 策略 1: MDP-Guided
    print("\n[1/2] Running MDP-Guided Strategy...")
    mdp_rag = SimplifiedMDPRAG(use_mdp=True)
    mdp_results = []
    
    for i, q in enumerate(questions):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{n_questions}")
        result = mdp_rag.answer_question(q, verbose=False)
        mdp_results.append(result)
    
    # 策略 2: Fixed (always retrieve k=3, then reason once)
    print("\n[2/2] Running Fixed Strategy (k=3)...")
    fixed_results = []
    
    for i, q in enumerate(questions):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{n_questions}")
        
        # 固定策略：检索 3 次，然后推理
        U = 1.0 - (3 * 0.15)  # 3 次检索
        answer, _ = SimplifiedMDPRAG().simulate_reason(q, iteration=3)
        
        fixed_results.append({
            'question_id': q['id'],
            'predicted': answer,
            'correct': q['correct_answer'],
            'is_correct': answer == q['correct_answer'],
            'iterations': 4,  # 3 retrieve + 1 reason
            'total_cost': 3 * 0.1 + 0.05  # 3 次检索 + 1 次推理
        })
    
    # 统计对比
    print("\n" + "=" * 80)
    print("Comparison Results")
    print("=" * 80)
    
    # MDP 统计
    mdp_acc = sum(r['is_correct'] for r in mdp_results) / len(mdp_results)
    mdp_avg_cost = np.mean([r['total_cost'] for r in mdp_results])
    mdp_avg_iters = np.mean([r['iterations'] for r in mdp_results])
    
    print(f"\n✦ MDP-Guided Strategy:")
    print(f"  Accuracy: {mdp_acc:.3f} ({sum(r['is_correct'] for r in mdp_results)}/{len(mdp_results)})")
    print(f"  Avg Cost: {mdp_avg_cost:.3f}")
    print(f"  Avg Iterations: {mdp_avg_iters:.2f}")
    print(f"  Total Retrievals: {mdp_rag.stats['total_retrievals']}")
    print(f"  Total Reasons: {mdp_rag.stats['total_reasons']}")
    
    # Fixed 统计
    fixed_acc = sum(r['is_correct'] for r in fixed_results) / len(fixed_results)
    fixed_avg_cost = np.mean([r['total_cost'] for r in fixed_results])
    
    print(f"\n✦ Fixed Strategy (k=3):")
    print(f"  Accuracy: {fixed_acc:.3f} ({sum(r['is_correct'] for r in fixed_results)}/{len(fixed_results)})")
    print(f"  Avg Cost: {fixed_avg_cost:.3f}")
    print(f"  Avg Iterations: 4.00 (fixed)")
    
    # 对比分析
    print(f"\n✦ Comparison:")
    print(f"  Accuracy Improvement: {(mdp_acc - fixed_acc) * 100:+.1f}%")
    print(f"  Cost Reduction: {(1 - mdp_avg_cost/fixed_avg_cost) * 100:.1f}%")
    print(f"  Efficiency Gain: {(mdp_acc/mdp_avg_cost) / (fixed_acc/fixed_avg_cost):.2f}x")
    
    print("=" * 80)
    
    # 保存结果
    output = {
        'timestamp': datetime.datetime.now().isoformat(),
        'config': {
            'n_questions': n_questions,
            'difficulty': difficulty,
            'seed': seed
        },
        'mdp_strategy': {
            'accuracy': float(mdp_acc),
            'avg_cost': float(mdp_avg_cost),
            'avg_iterations': float(mdp_avg_iters),
            'results': mdp_results
        },
        'fixed_strategy': {
            'accuracy': float(fixed_acc),
            'avg_cost': float(fixed_avg_cost),
            'results': fixed_results
        }
    }
    
    os.makedirs('results/comparison', exist_ok=True)
    output_file = f"results/comparison/mdp_vs_fixed_{difficulty or 'mixed'}_{n_questions}q.json"
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    return output


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MDP-Guided RAG Comparison (CPU-friendly)')
    parser.add_argument('-n', '--num_questions', type=int, default=50, help='Number of questions')
    parser.add_argument('-d', '--difficulty', type=str, choices=['easy', 'medium', 'hard'], 
                       default=None, help='Question difficulty')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # 运行对比实验
    run_comparison_experiment(
        n_questions=args.num_questions,
        difficulty=args.difficulty,
        seed=args.seed
    )
