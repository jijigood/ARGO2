"""
Experiment 2: Impact of Retrieval Success Rate (p_s) on Performance
====================================================================

Objective: Prove that ARGO adapts to unreliable retrieval environments,
while static policies fail to account for retrieval uncertainty.

Fixed Parameters:
- Test set: ORAN-Bench-13K
- MDP parameters: delta_r, delta_p, c_r, c_p, mu, gamma

Independent Variable:
- Retrieval Success Rate (p_s): sweep from 0.3 to 1.0

Expected Results:
- Graph 1 (p_s vs Quality): ARGO maintains quality better than baselines
- Graph 2 (p_s vs Retrievals): ARGO increases retrievals when p_s is low
- Graph 3 (p_s vs Actions): ARGO shifts from Retrieve to Reason when p_s is low

Author: ARGO Team
Date: 2025-10-29
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../ARGO_MDP/src'))

import numpy as np
import json
from typing import Dict, List, Tuple
import datetime
import yaml
import matplotlib.pyplot as plt
from pathlib import Path

from oran_benchmark_loader import ORANBenchmark
from mdp_solver import MDPSolver


class RetrievalSuccessExperiment:
    """
    Experiment 2: Impact of Retrieval Success Rate on Performance
    """
    
    def __init__(
        self,
        config_path: str = "configs/multi_gpu.yaml",
        n_test_questions: int = 100,
        difficulty: str = "medium",
        seed: int = 42
    ):
        """
        Initialize experiment
        
        Args:
            config_path: Path to MDP config
            n_test_questions: Number of test questions
            difficulty: Question difficulty level
            seed: Random seed
        """
        self.seed = seed
        self.n_test_questions = n_test_questions
        self.difficulty = difficulty
        
        # Load base config
        with open(config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        # Load benchmark
        print(f"Loading ORAN-Bench-13K (seed={seed})...")
        self.benchmark = ORANBenchmark()
        self.test_questions = self.benchmark.sample_questions(
            n=n_test_questions,
            difficulty=difficulty,
            seed=seed
        )
        print(f"Loaded {len(self.test_questions)} test questions")
        
        # Fixed MDP parameters (from config)
        self.delta_r = self.base_config['mdp']['delta_r']
        self.delta_p = self.base_config['mdp']['delta_p']
        self.c_r = self.base_config['mdp']['c_r']
        self.c_p = self.base_config['mdp']['c_p']
        self.mu = self.base_config['mdp']['mu']
        self.gamma = self.base_config['mdp']['gamma']
        
        print(f"\nFixed MDP Parameters:")
        print(f"  delta_r = {self.delta_r}")
        print(f"  delta_p = {self.delta_p}")
        print(f"  c_r = {self.c_r}")
        print(f"  c_p = {self.c_p}")
        print(f"  mu = {self.mu}")
        print(f"  gamma = {self.gamma}")
        
        # Results storage
        self.results = {}
    
    def create_mdp_config(self, p_s: float) -> Dict:
        """
        Create MDP config with specific p_s
        
        Args:
            p_s: Retrieval success rate
            
        Returns:
            Complete MDP config
        """
        config = {
            'mdp': {
                'U_max': self.base_config['mdp']['U_max'],
                'delta_r': self.delta_r,
                'delta_p': self.delta_p,
                'p_s': p_s,  # <-- Variable
                'c_r': self.c_r,
                'c_p': self.c_p,
                'mu': self.mu,
                'gamma': self.gamma,
                'U_grid_size': self.base_config['mdp'].get('U_grid_size', 
                              self.base_config['mdp'].get('grid_size', 101))
            },
            'quality': {
                'mode': self.base_config['mdp'].get('quality_function', 'linear'),
                'k': self.base_config['mdp'].get('quality_k', 5.0)
            },
            'solver': {
                'max_iterations': 1000,
                'convergence_threshold': 1e-6,
                'verbose': False
            }
        }
        return config
    
    def solve_mdp(self, p_s: float) -> Tuple[float, float]:
        """
        Solve MDP for given p_s
        
        Args:
            p_s: Retrieval success rate
            
        Returns:
            (theta_cont, theta_star): Optimal thresholds
        """
        print(f"\n  [MDP Solver] p_s = {p_s:.3f}")
        
        config = self.create_mdp_config(p_s)
        solver = MDPSolver(config)
        results = solver.solve()
        
        theta_cont = results['theta_cont']
        theta_star = results['theta_star']
        
        print(f"    θ_cont = {theta_cont:.4f}, θ* = {theta_star:.4f}")
        
        return theta_cont, theta_star
    
    def simulate_quality_function(self, U: float) -> float:
        """Quality function σ(U)"""
        if 'quality' in self.base_config:
            mode = self.base_config['quality'].get('mode', 'linear')
            k = self.base_config['quality'].get('k', 5.0)
        else:
            mode = self.base_config['mdp'].get('quality_function', 'linear')
            k = self.base_config['mdp'].get('quality_k', 5.0)
        
        U_max = self.base_config['mdp']['U_max']
        x = U / U_max
        
        if mode == "sigmoid":
            return 1.0 / (1.0 + np.exp(-k * (x - 0.5)))
        elif mode == "sqrt":
            return np.sqrt(x)
        elif mode == "saturating":
            return 1.0 - np.exp(-k * x)
        else:
            return x
    
    def simulate_argo_policy(
        self,
        question: Dict,
        theta_cont: float,
        theta_star: float,
        p_s: float  # Now variable
    ) -> Dict:
        """
        Simulate ARGO policy on a question
        """
        U = 0.0
        step = 0
        max_steps = 30  # Increased for low p_s scenarios
        retrieval_count = 0
        reason_count = 0
        
        while U < theta_star and step < max_steps:
            step += 1
            
            if U < theta_cont:
                # Retrieve
                retrieval_count += 1
                if np.random.random() < p_s:  # Use variable p_s
                    U = min(U + self.delta_r, 1.0)
            else:
                # Reason
                reason_count += 1
                U = min(U + self.delta_p, 1.0)
        
        quality = self.simulate_quality_function(U)
        
        return {
            'quality': quality,
            'retrieval_count': retrieval_count,
            'reason_count': reason_count,
            'total_steps': step,
            'final_U': U
        }
    
    def simulate_always_retrieve_policy(self, question: Dict, p_s: float) -> Dict:
        """Simulate Always-Retrieve with variable p_s"""
        U = 0.0
        step = 0
        max_steps = 30
        retrieval_count = 0
        theta_star = 0.9
        
        while U < theta_star and step < max_steps:
            step += 1
            retrieval_count += 1
            
            if np.random.random() < p_s:  # Use variable p_s
                U = min(U + self.delta_r, 1.0)
        
        quality = self.simulate_quality_function(U)
        
        return {
            'quality': quality,
            'retrieval_count': retrieval_count,
            'reason_count': 0,
            'total_steps': step,
            'final_U': U
        }
    
    def simulate_always_reason_policy(self, question: Dict, p_s: float) -> Dict:
        """Simulate Always-Reason (p_s doesn't matter)"""
        U = 0.0
        step = 0
        max_steps = 30
        reason_count = 0
        theta_star = 0.9
        
        while U < theta_star and step < max_steps:
            step += 1
            reason_count += 1
            U = min(U + self.delta_p, 1.0)
        
        quality = self.simulate_quality_function(U)
        
        return {
            'quality': quality,
            'retrieval_count': 0,
            'reason_count': reason_count,
            'total_steps': step,
            'final_U': U
        }
    
    def simulate_random_policy(self, question: Dict, p_s: float) -> Dict:
        """Simulate Random policy with variable p_s"""
        U = 0.0
        step = 0
        max_steps = 30
        retrieval_count = 0
        reason_count = 0
        theta_star = 0.9
        
        while U < theta_star and step < max_steps:
            step += 1
            
            if np.random.random() < 0.5:
                retrieval_count += 1
                if np.random.random() < p_s:  # Use variable p_s
                    U = min(U + self.delta_r, 1.0)
            else:
                reason_count += 1
                U = min(U + self.delta_p, 1.0)
        
        quality = self.simulate_quality_function(U)
        
        return {
            'quality': quality,
            'retrieval_count': retrieval_count,
            'reason_count': reason_count,
            'total_steps': step,
            'final_U': U
        }
    
    def evaluate_all_policies(
        self,
        p_s: float,
        theta_cont: float,
        theta_star: float
    ) -> Dict:
        """
        Evaluate all 4 policies on test set
        """
        print(f"\n  [Evaluating Policies] p_s = {p_s:.3f}")
        
        np.random.seed(self.seed)
        
        policies = {
            'ARGO': lambda q: self.simulate_argo_policy(q, theta_cont, theta_star, p_s),
            'Always-Retrieve': lambda q: self.simulate_always_retrieve_policy(q, p_s),
            'Always-Reason': lambda q: self.simulate_always_reason_policy(q, p_s),
            'Random': lambda q: self.simulate_random_policy(q, p_s)
        }
        
        results = {}
        
        for policy_name, policy_fn in policies.items():
            print(f"    {policy_name}...", end='')
            
            policy_results = []
            for question in self.test_questions:
                result = policy_fn(question)
                policy_results.append(result)
            
            avg_quality = np.mean([r['quality'] for r in policy_results])
            avg_retrievals = np.mean([r['retrieval_count'] for r in policy_results])
            avg_reasons = np.mean([r['reason_count'] for r in policy_results])
            avg_steps = np.mean([r['total_steps'] for r in policy_results])
            
            results[policy_name] = {
                'avg_quality': avg_quality,
                'avg_retrievals': avg_retrievals,
                'avg_reasons': avg_reasons,
                'avg_steps': avg_steps,
                'raw_results': policy_results
            }
            
            print(f" Q={avg_quality:.3f}, R={avg_retrievals:.1f}, P={avg_reasons:.1f}")
        
        return results
    
    def run_experiment(
        self,
        p_s_min: float = 0.3,
        p_s_max: float = 1.0,
        n_steps: int = 8
    ):
        """
        Run complete experiment: sweep p_s from 0.3 to 1.0
        """
        print("\n" + "=" * 80)
        print("Experiment 2: Impact of Retrieval Success Rate (p_s) on Performance")
        print("=" * 80)
        print(f"\nSweeping p_s from {p_s_min} to {p_s_max}")
        print(f"Number of steps: {n_steps}")
        print(f"Test questions: {len(self.test_questions)}")
        print("=" * 80)
        
        # Generate p_s values to sweep
        p_s_values = np.linspace(p_s_min, p_s_max, n_steps)
        
        print(f"\np_s values: {p_s_values}")
        
        # Storage for results
        self.results['p_s_values'] = p_s_values.tolist()
        self.results['policies'] = {
            'ARGO': {'quality': [], 'retrievals': [], 'reasons': [], 'thresholds': []},
            'Always-Retrieve': {'quality': [], 'retrievals': [], 'reasons': []},
            'Always-Reason': {'quality': [], 'retrievals': [], 'reasons': []},
            'Random': {'quality': [], 'retrievals': [], 'reasons': []}
        }
        
        # Run experiment for each p_s
        for i, p_s in enumerate(p_s_values):
            print(f"\n{'='*60}")
            print(f"Step {i+1}/{n_steps}: p_s = {p_s:.3f}")
            print(f"{'='*60}")
            
            # Step 1: Solve MDP for this p_s
            theta_cont, theta_star = self.solve_mdp(p_s)
            
            # Step 2: Evaluate all policies
            policy_results = self.evaluate_all_policies(p_s, theta_cont, theta_star)
            
            # Step 3: Store results
            self.results['policies']['ARGO']['quality'].append(
                policy_results['ARGO']['avg_quality']
            )
            self.results['policies']['ARGO']['retrievals'].append(
                policy_results['ARGO']['avg_retrievals']
            )
            self.results['policies']['ARGO']['reasons'].append(
                policy_results['ARGO']['avg_reasons']
            )
            self.results['policies']['ARGO']['thresholds'].append({
                'theta_cont': theta_cont,
                'theta_star': theta_star
            })
            
            for policy in ['Always-Retrieve', 'Always-Reason', 'Random']:
                self.results['policies'][policy]['quality'].append(
                    policy_results[policy]['avg_quality']
                )
                self.results['policies'][policy]['retrievals'].append(
                    policy_results[policy]['avg_retrievals']
                )
                self.results['policies'][policy]['reasons'].append(
                    policy_results[policy]['avg_reasons']
                )
        
        print("\n" + "=" * 80)
        print("Experiment Complete!")
        print("=" * 80)
        
        return self.results
    
    def save_results(self, output_dir: str = "draw_figs/data"):
        """Save experiment results to JSON"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exp2_retrieval_success_impact_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        results_with_metadata = {
            'experiment': 'Impact of Retrieval Success Rate',
            'timestamp': timestamp,
            'config': {
                'n_test_questions': self.n_test_questions,
                'difficulty': self.difficulty,
                'seed': self.seed,
                'delta_r': self.delta_r,
                'delta_p': self.delta_p,
                'c_r': self.c_r,
                'c_p': self.c_p,
                'mu': self.mu,
                'gamma': self.gamma
            },
            'results': self.results
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
        
        return filepath
    
    def plot_results(self, output_dir: str = "figs"):
        """Generate plots"""
        os.makedirs(output_dir, exist_ok=True)
        
        p_s_values = np.array(self.results['p_s_values'])
        
        # Figure 1: p_s vs Quality
        plt.figure(figsize=(10, 6))
        
        for policy in ['ARGO', 'Always-Retrieve', 'Always-Reason', 'Random']:
            quality = self.results['policies'][policy]['quality']
            
            if policy == 'ARGO':
                plt.plot(p_s_values, quality, marker='o', linewidth=2.5, 
                        label=policy, color='#2E86AB')
            elif policy == 'Always-Retrieve':
                plt.plot(p_s_values, quality, marker='s', linewidth=2, 
                        label=policy, color='#A23B72', linestyle='--')
            elif policy == 'Always-Reason':
                plt.plot(p_s_values, quality, marker='^', linewidth=2, 
                        label=policy, color='#F18F01', linestyle='--')
            else:
                plt.plot(p_s_values, quality, marker='x', linewidth=2, 
                        label=policy, color='#C73E1D', linestyle=':')
        
        plt.xlabel('Retrieval Success Rate ($p_s$)', fontsize=12)
        plt.ylabel('Average Answer Quality $E[Q(O)]$', fontsize=12)
        plt.title('Figure 1: Impact of Retrieval Success Rate on Quality', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig1_path = os.path.join(output_dir, 'exp2_ps_vs_quality.png')
        plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure 1 saved to: {fig1_path}")
        plt.close()
        
        # Figure 2: p_s vs Retrievals
        plt.figure(figsize=(10, 6))
        
        for policy in ['ARGO', 'Always-Retrieve', 'Always-Reason', 'Random']:
            retrievals = self.results['policies'][policy]['retrievals']
            
            if policy == 'ARGO':
                plt.plot(p_s_values, retrievals, marker='o', linewidth=2.5, 
                        label=policy, color='#2E86AB')
            elif policy == 'Always-Retrieve':
                plt.plot(p_s_values, retrievals, marker='s', linewidth=2, 
                        label=policy, color='#A23B72', linestyle='--')
            elif policy == 'Always-Reason':
                plt.plot(p_s_values, retrievals, marker='^', linewidth=2, 
                        label=policy, color='#F18F01', linestyle='--')
            else:
                plt.plot(p_s_values, retrievals, marker='x', linewidth=2, 
                        label=policy, color='#C73E1D', linestyle=':')
        
        plt.xlabel('Retrieval Success Rate ($p_s$)', fontsize=12)
        plt.ylabel('Average Retrieval Calls $E[R_T]$', fontsize=12)
        plt.title('Figure 2: Impact of Retrieval Success Rate on Retrieval Usage', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig2_path = os.path.join(output_dir, 'exp2_ps_vs_retrievals.png')
        plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
        print(f"Figure 2 saved to: {fig2_path}")
        plt.close()
        
        # Figure 3: Action Distribution (Stacked Bar)
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(p_s_values))
        width = 0.2
        
        for i, policy in enumerate(['ARGO', 'Always-Retrieve', 'Always-Reason', 'Random']):
            retrievals = self.results['policies'][policy]['retrievals']
            reasons = self.results['policies'][policy]['reasons']
            
            offset = (i - 1.5) * width
            plt.bar(x + offset, retrievals, width, label=f'{policy} (Retrieve)', 
                   alpha=0.8)
            plt.bar(x + offset, reasons, width, bottom=retrievals, 
                   label=f'{policy} (Reason)', alpha=0.6, hatch='//')
        
        plt.xlabel('Retrieval Success Rate ($p_s$)', fontsize=12)
        plt.ylabel('Average Action Count', fontsize=12)
        plt.title('Figure 3: Action Distribution vs. Retrieval Success Rate', 
                 fontsize=14, fontweight='bold')
        plt.xticks(x, [f'{p:.2f}' for p in p_s_values])
        plt.legend(fontsize=9, loc='best', ncol=2)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        fig3_path = os.path.join(output_dir, 'exp2_action_distribution.png')
        plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
        print(f"Figure 3 saved to: {fig3_path}")
        plt.close()


def main():
    """Main function: Run Experiment 2"""
    print("\n" + "=" * 80)
    print("ARGO Experiment 2: Impact of Retrieval Success Rate (p_s)")
    print("=" * 80)
    print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Initialize experiment
    exp = RetrievalSuccessExperiment(
        config_path="configs/multi_gpu.yaml",
        n_test_questions=100,
        difficulty="medium",
        seed=42
    )
    
    # Run experiment
    results = exp.run_experiment(
        p_s_min=0.3,   # Low success rate
        p_s_max=1.0,   # Perfect success rate
        n_steps=8      # Test 8 different p_s values
    )
    
    # Save results
    results_path = exp.save_results()
    
    # Generate plots
    print("\n" + "=" * 80)
    print("Generating Plots...")
    print("=" * 80)
    exp.plot_results()
    
    print("\n" + "=" * 80)
    print("Experiment 2 Complete!")
    print("=" * 80)
    print("\nKey Findings:")
    print("1. Check Figure 1 (ps_vs_quality.png): Quality degradation with low p_s")
    print("2. Check Figure 2 (ps_vs_retrievals.png): ARGO adapts retrieval count")
    print("3. Check Figure 3 (action_distribution.png): Retrieve↔Reason balance")
    print("=" * 80)


if __name__ == "__main__":
    main()
