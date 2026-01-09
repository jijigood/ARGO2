#!/usr/bin/env python
"""
Experiment 3: Pareto Frontier Analysis - V3 (Using ARGO_System)
================================================================
使用完整的 ARGO_System 来评估问题，确保与 Exp_real_cost_impact_v2.py 一致的准确率。

Key Features:
- 使用 ARGO_System.answer_question() 进行评估
- 对不同 μ 值创建不同的 ARGO_System 实例
- 统一的答案生成和评估流程

Author: ARGO Team
Version: 3.0 (Refactored with ARGO_System)
"""

import os
import sys
import torch
import numpy as np
import yaml
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from copy import deepcopy
from tqdm import tqdm
import pandas as pd

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
argo_root = '/data/user/huangxiaolin/ARGO2/ARGO'
sys.path.insert(0, argo_root)

# Import ARGO_System
try:
    from ARGO.src.argo_system import ARGO_System
except ImportError:
    from src.argo_system import ARGO_System

# Import data loader
from oran_benchmark_loader import ORANBenchmark


class ParetoFrontierExperimentV3:
    """
    Pareto Frontier Experiment using ARGO_System
    
    对每个 μ 值：
    1. 创建配置了该 μ 的 ARGO_System
    2. 使用 answer_question() 评估问题
    3. 收集 accuracy, cost, information_quality
    """
    
    def __init__(
        self,
        config_path: str,
        llm_model_path: str = "/data/user/huangxiaolin/ARGO/RAG_Models/models/Qwen2.5-3B-Instruct",
        embedding_model_path: str = "/data/user/huangxiaolin/ARGO/models/all-MiniLM-L6-v2",
        chroma_db_path: str = "/data/user/huangxiaolin/ARGO2/ARGO/Environments/chroma_store",
        n_test_questions: int = 20,
        difficulty: str = "hard",
        seed: int = 42,
        gpu_ids: List[int] = None,
        verbose: bool = False
    ):
        self.config_path = config_path
        self.llm_model_path = llm_model_path
        self.embedding_model_path = embedding_model_path
        self.chroma_db_path = chroma_db_path
        self.n_test_questions = n_test_questions
        self.difficulty = difficulty
        self.seed = seed
        self.gpu_ids = gpu_ids or [0, 1, 2, 3]
        self.verbose = verbose
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Load model once (shared across all ARGO_System instances)
        self._load_model()
        
        # Load questions
        self._load_questions()
        
        # Results storage
        self.results = {
            'argo_results': [],
            'baseline_results': {},
            'fixed_threshold_results': []
        }
    
    def _load_model(self):
        """Load LLM model (shared)"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"\nLoading LLM: {self.llm_model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.llm_model_path, 
            trust_remote_code=True
        )
        
        if len(self.gpu_ids) > 1:
            print(f"  Using {len(self.gpu_ids)} GPUs...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True
            ).to(f"cuda:{self.gpu_ids[0]}")
        
        print("✓ LLM loaded successfully")
    
    def _load_questions(self):
        """Load test questions"""
        print(f"\nLoading ORAN-Bench-13K dataset...")
        benchmark = ORANBenchmark()
        self.questions = benchmark.sample_questions(
            n=self.n_test_questions,
            difficulty=self.difficulty,
            seed=self.seed
        )
        print(f"✓ Loaded {len(self.questions)} {self.difficulty.upper()} questions")
    
    def _create_argo_system(self, mu: float) -> ARGO_System:
        """Create ARGO_System with specific μ value"""
        # Deep copy config and set μ
        mdp_config = deepcopy(self.config)
        mdp_config.setdefault('mdp', {})['mu'] = mu
        
        argo_system = ARGO_System(
            model=self.model,
            tokenizer=self.tokenizer,
            use_mdp=True,
            mdp_config=mdp_config,
            retriever_mode='chroma',
            chroma_dir=self.chroma_db_path,
            max_steps=10,
            verbose=self.verbose
        )
        
        return argo_system
    
    def evaluate_with_mu(self, mu: float) -> Dict:
        """Evaluate all questions with specific μ value"""
        print(f"\n  Creating ARGO_System with μ={mu:.2f}...")
        argo_system = self._create_argo_system(mu)
        
        # Get thresholds
        theta_cont = getattr(argo_system.mdp_solver, 'theta_cont', None)
        theta_star = getattr(argo_system.mdp_solver, 'theta_star', None)
        print(f"  θ_cont={theta_cont:.3f}, θ*={theta_star:.3f}")
        
        results = []
        correct_count = 0
        total_cost = 0.0
        total_info_quality = 0.0
        
        for i, q in enumerate(self.questions):
            question = q['question']
            options = q.get('options', [])
            ground_truth = str(q.get('answer', '')).strip()
            
            # Call ARGO_System
            answer, choice, history, metadata = argo_system.answer_question(
                question,
                return_history=True,
                options=options
            )
            
            # Check correctness
            correct = (str(choice).strip() == ground_truth) if choice else False
            if correct:
                correct_count += 1
            
            # Calculate cost
            retrieve_count = metadata.get('retrieve_count', 0) if metadata else 0
            reason_count = metadata.get('reason_count', 0) if metadata else 0
            c_r = self.config['mdp'].get('c_r', 0.08)
            c_p = self.config['mdp'].get('c_p', 0.02)
            cost = retrieve_count * c_r + reason_count * c_p
            total_cost += cost
            
            # Information quality (final progress / U_max)
            final_progress = metadata.get('final_progress', 1.0) if metadata else 1.0
            question_umax = metadata.get('question_umax', 1.0) if metadata else 1.0
            info_quality = min(final_progress / question_umax, 1.0) if question_umax > 0 else 1.0
            total_info_quality += info_quality
            
            results.append({
                'question_id': i,
                'correct': correct,
                'choice': choice,
                'ground_truth': ground_truth,
                'cost': cost,
                'info_quality': info_quality,
                'retrieve_count': retrieve_count,
                'reason_count': reason_count
            })
            
            if self.verbose:
                print(f"    Q{i+1}: {'✓' if correct else '✗'} (choice={choice}, gt={ground_truth})")
        
        n = len(self.questions)
        accuracy = correct_count / n if n > 0 else 0.0
        avg_cost = total_cost / n if n > 0 else 0.0
        avg_info_quality = total_info_quality / n if n > 0 else 0.0
        
        # Cleanup
        del argo_system
        
        return {
            'mu': mu,
            'theta_cont': theta_cont,
            'theta_star': theta_star,
            'accuracy': accuracy,
            'avg_cost': avg_cost,
            'avg_info_quality': avg_info_quality,
            'correct_count': correct_count,
            'total_questions': n,
            'individual_results': results
        }
    
    def evaluate_baseline(self, baseline_type: str) -> Dict:
        """Evaluate baseline strategies"""
        print(f"\n  Evaluating {baseline_type}...")
        
        # Create ARGO_System with extreme μ values for baselines
        if baseline_type == 'always_retrieve':
            mu = 0.0  # Prefer retrieval
        elif baseline_type == 'always_reason':
            mu = 1.0  # Prefer reasoning
        else:  # random
            mu = 0.5
        
        argo_system = self._create_argo_system(mu)
        
        results = []
        correct_count = 0
        total_cost = 0.0
        total_info_quality = 0.0
        
        for i, q in enumerate(self.questions):
            question = q['question']
            options = q.get('options', [])
            ground_truth = str(q.get('answer', '')).strip()
            
            answer, choice, history, metadata = argo_system.answer_question(
                question,
                return_history=True,
                options=options
            )
            
            correct = (str(choice).strip() == ground_truth) if choice else False
            if correct:
                correct_count += 1
            
            retrieve_count = metadata.get('retrieve_count', 0) if metadata else 0
            reason_count = metadata.get('reason_count', 0) if metadata else 0
            c_r = self.config['mdp'].get('c_r', 0.08)
            c_p = self.config['mdp'].get('c_p', 0.02)
            cost = retrieve_count * c_r + reason_count * c_p
            total_cost += cost
            
            final_progress = metadata.get('final_progress', 1.0) if metadata else 1.0
            question_umax = metadata.get('question_umax', 1.0) if metadata else 1.0
            info_quality = min(final_progress / question_umax, 1.0) if question_umax > 0 else 1.0
            total_info_quality += info_quality
            
            results.append({
                'correct': correct,
                'cost': cost,
                'info_quality': info_quality
            })
        
        n = len(self.questions)
        del argo_system
        
        return {
            'baseline_type': baseline_type,
            'accuracy': correct_count / n if n > 0 else 0.0,
            'avg_cost': total_cost / n if n > 0 else 0.0,
            'avg_info_quality': total_info_quality / n if n > 0 else 0.0
        }
    
    def run_experiment(
        self,
        mu_values: List[float] = None,
        n_mu_steps: int = 12
    ) -> Dict:
        """Run full Pareto frontier experiment"""
        
        if mu_values is None:
            mu_values = np.linspace(0.0, 1.0, n_mu_steps).tolist()
        
        print("=" * 80)
        print("Pareto Frontier Experiment V3 (Using ARGO_System)")
        print("=" * 80)
        print(f"μ values: {[f'{m:.2f}' for m in mu_values]}")
        print(f"Questions: {self.n_test_questions} ({self.difficulty})")
        print("=" * 80)
        
        # 1. Evaluate ARGO at different μ values
        print("\n" + "=" * 80)
        print("Phase 1: ARGO at different μ values")
        print("=" * 80)
        
        argo_results = []
        for i, mu in enumerate(mu_values):
            print(f"\n[{i+1}/{len(mu_values)}] μ = {mu:.2f}")
            result = self.evaluate_with_mu(mu)
            argo_results.append(result)
            print(f"  → Accuracy: {result['accuracy']:.1%}, "
                  f"Info Quality: {result['avg_info_quality']:.3f}, "
                  f"Cost: {result['avg_cost']:.3f}")
        
        self.results['argo_results'] = argo_results
        
        # 2. Evaluate baselines
        print("\n" + "=" * 80)
        print("Phase 2: Baseline Evaluation")
        print("=" * 80)
        
        for baseline in ['always_retrieve', 'always_reason', 'random']:
            result = self.evaluate_baseline(baseline)
            self.results['baseline_results'][baseline] = result
            print(f"  {baseline}: Accuracy={result['accuracy']:.1%}, "
                  f"Cost={result['avg_cost']:.3f}")
        
        # 3. Summary
        self._print_summary()
        
        # 4. Save results
        self._save_results()
        
        return self.results
    
    def _print_summary(self):
        """Print experiment summary"""
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)
        
        print("\nARGO Results (by μ):")
        print("-" * 60)
        print(f"{'μ':>6} {'θ_cont':>8} {'θ*':>8} {'Accuracy':>10} {'InfoQual':>10} {'Cost':>8}")
        print("-" * 60)
        
        for r in self.results['argo_results']:
            print(f"{r['mu']:6.2f} {r['theta_cont']:8.3f} {r['theta_star']:8.3f} "
                  f"{r['accuracy']:10.1%} {r['avg_info_quality']:10.3f} {r['avg_cost']:8.3f}")
        
        print("\nBaselines:")
        print("-" * 60)
        for name, r in self.results['baseline_results'].items():
            print(f"{name:20s}: Accuracy={r['accuracy']:.1%}, "
                  f"InfoQual={r['avg_info_quality']:.3f}, Cost={r['avg_cost']:.3f}")
        
        # Check Pareto dominance
        print("\n" + "=" * 80)
        print("Pareto Dominance Check")
        print("=" * 80)
        
        # Find best ARGO point
        best_argo = max(self.results['argo_results'], 
                       key=lambda x: x['avg_info_quality'] - x['avg_cost'])
        
        for name, baseline in self.results['baseline_results'].items():
            # Check if any ARGO point dominates this baseline
            dominated = False
            for argo in self.results['argo_results']:
                if (argo['avg_info_quality'] >= baseline['avg_info_quality'] and 
                    argo['avg_cost'] <= baseline['avg_cost'] and
                    (argo['avg_info_quality'] > baseline['avg_info_quality'] or 
                     argo['avg_cost'] < baseline['avg_cost'])):
                    dominated = True
                    break
            
            status = "✓ Dominated" if dominated else "✗ Not dominated"
            print(f"  {name}: {status}")
    
    def _save_results(self):
        """Save results to JSON"""
        output_dir = Path(argo_root) / "experiments" / "runners" / "draw_figs" / "data"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"exp3_pareto_v3_{timestamp}.json"
        
        # Prepare serializable results
        save_data = {
            'experiment': 'pareto_frontier_v3',
            'timestamp': timestamp,
            'config': {
                'n_questions': self.n_test_questions,
                'difficulty': self.difficulty,
                'seed': self.seed,
                'config_path': self.config_path
            },
            'argo_results': [
                {k: v for k, v in r.items() if k != 'individual_results'}
                for r in self.results['argo_results']
            ],
            'baseline_results': self.results['baseline_results']
        }
        
        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n✓ Results saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Pareto Frontier Experiment V3')
    parser.add_argument('--config-path', type=str, 
                       default='/data/user/huangxiaolin/ARGO2/ARGO/configs/pareto_optimized.yaml')
    parser.add_argument('--n-questions', type=int, default=20)
    parser.add_argument('--difficulty', type=str, default='hard',
                       choices=['easy', 'medium', 'hard'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpus', type=str, default='0,1,2,3')
    parser.add_argument('--n-mu-steps', type=int, default=12)
    parser.add_argument('--mu-min', type=float, default=0.0)
    parser.add_argument('--mu-max', type=float, default=1.0)
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    gpu_ids = [int(g) for g in args.gpus.split(',')]
    mu_values = np.linspace(args.mu_min, args.mu_max, args.n_mu_steps).tolist()
    
    print("=" * 80)
    print("Pareto Frontier Experiment V3")
    print("=" * 80)
    print(f"Config: {args.config_path}")
    print(f"Questions: {args.n_questions} ({args.difficulty})")
    print(f"μ range: [{args.mu_min}, {args.mu_max}] ({args.n_mu_steps} steps)")
    print(f"GPUs: {gpu_ids}")
    print("=" * 80)
    
    exp = ParetoFrontierExperimentV3(
        config_path=args.config_path,
        n_test_questions=args.n_questions,
        difficulty=args.difficulty,
        seed=args.seed,
        gpu_ids=gpu_ids,
        verbose=args.verbose
    )
    
    results = exp.run_experiment(mu_values=mu_values)
    
    print("\n" + "=" * 80)
    print("✅ Experiment Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
