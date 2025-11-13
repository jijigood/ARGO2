"""
Experiment 1: Impact of Retrieval Cost (c_r) on Performance
============================================================

Objective: Prove that ARGO is the only policy that intelligently adapts 
to rising operational costs, while static policies are inefficient.

Fixed Parameters:
- Test set: ORAN-Bench-13K
- MDP environmental: delta_r, delta_p, p_s
- Other costs/objectives: c_p, mu, gamma

Independent Variable:
- Retrieval Cost (c_r): sweep from c_p to 10 * c_p

Expected Results:
- Graph 1 (Cost vs Quality): ARGO maintains quality; baselines flat
- Graph 2 (Cost vs Retrievals): ARGO decreases sharply; baselines flat

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


class CostImpactExperiment:
    """
    Experiment 1: Impact of Retrieval Cost on Performance
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
        self.p_s = self.base_config['mdp']['p_s']
        self.c_p = self.base_config['mdp']['c_p']
        self.mu = self.base_config['mdp']['mu']
        self.gamma = self.base_config['mdp']['gamma']
        
        print(f"\nFixed MDP Parameters:")
        print(f"  delta_r = {self.delta_r}")
        print(f"  delta_p = {self.delta_p}")
        print(f"  p_s = {self.p_s}")
        print(f"  c_p = {self.c_p}")
        print(f"  mu = {self.mu}")
        print(f"  gamma = {self.gamma}")
        
        # Results storage
        self.results = {}
    
    def create_mdp_config(self, c_r: float) -> Dict:
        """
        Create MDP config with specific c_r
        
        Args:
            c_r: Retrieval cost
            
        Returns:
            Complete MDP config
        """
        config = {
            'mdp': {
                'U_max': self.base_config['mdp']['U_max'],
                'delta_r': self.delta_r,
                'delta_p': self.delta_p,
                'p_s': self.p_s,
                'c_r': c_r,  # <-- Variable
                'c_p': self.c_p,
                'mu': self.mu,
                'gamma': self.gamma,
                'U_grid_size': self.base_config['mdp'].get('U_grid_size', self.base_config['mdp'].get('grid_size', 101))
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
    
    def solve_mdp(self, c_r: float) -> Tuple[float, float]:
        """
        Solve MDP for given c_r
        
        Args:
            c_r: Retrieval cost
            
        Returns:
            (theta_cont, theta_star): Optimal thresholds
        """
        print(f"\n  [MDP Solver] c_r = {c_r:.3f}")
        
        config = self.create_mdp_config(c_r)
        solver = MDPSolver(config)
        results = solver.solve()
        
        theta_cont = results['theta_cont']
        theta_star = results['theta_star']
        
        print(f"    θ_cont = {theta_cont:.4f}, θ* = {theta_star:.4f}")
        
        return theta_cont, theta_star
    
    def simulate_quality_function(self, U: float) -> float:
        """
        Quality function σ(U)
        
        Using sigmoid as default (from config)
        """
        # Get quality config from mdp section (backwards compatible)
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
        else:  # linear
            return x
    
    def simulate_argo_policy(
        self,
        question: Dict,
        theta_cont: float,
        theta_star: float
    ) -> Dict:
        """
        Simulate ARGO policy on a question
        
        Args:
            question: Benchmark question
            theta_cont: Continuation threshold
            theta_star: Termination threshold
            
        Returns:
            Result dict with quality, retrieval_count, etc.
        """
        U = 0.0
        step = 0
        max_steps = 20
        retrieval_count = 0
        reason_count = 0
        
        while U < theta_star and step < max_steps:
            step += 1
            
            # ARGO decision: Retrieve if U < θ_cont, else Reason
            if U < theta_cont:
                # Retrieve
                retrieval_count += 1
                
                # Simulate retrieval success
                if np.random.random() < self.p_s:
                    U = min(U + self.delta_r, 1.0)
                # else: U stays the same
            else:
                # Reason
                reason_count += 1
                U = min(U + self.delta_p, 1.0)
        
        # Final quality
        quality = self.simulate_quality_function(U)
        
        return {
            'quality': quality,
            'retrieval_count': retrieval_count,
            'reason_count': reason_count,
            'total_steps': step,
            'final_U': U
        }
    
    def simulate_always_retrieve_policy(self, question: Dict) -> Dict:
        """
        Simulate Always-Retrieve policy
        
        Always executes Retrieve action until termination
        """
        U = 0.0
        step = 0
        max_steps = 20
        retrieval_count = 0
        theta_star = 0.9  # Fixed termination threshold
        
        while U < theta_star and step < max_steps:
            step += 1
            retrieval_count += 1
            
            # Always retrieve
            if np.random.random() < self.p_s:
                U = min(U + self.delta_r, 1.0)
        
        quality = self.simulate_quality_function(U)
        
        return {
            'quality': quality,
            'retrieval_count': retrieval_count,
            'reason_count': 0,
            'total_steps': step,
            'final_U': U
        }
    
    def simulate_always_reason_policy(self, question: Dict) -> Dict:
        """
        Simulate Always-Reason policy
        
        Always executes Reason action (never retrieve)
        """
        U = 0.0
        step = 0
        max_steps = 20
        reason_count = 0
        theta_star = 0.9  # Fixed termination threshold
        
        while U < theta_star and step < max_steps:
            step += 1
            reason_count += 1
            
            # Always reason
            U = min(U + self.delta_p, 1.0)
        
        quality = self.simulate_quality_function(U)
        
        return {
            'quality': quality,
            'retrieval_count': 0,
            'reason_count': reason_count,
            'total_steps': step,
            'final_U': U
        }
    
    def simulate_random_policy(self, question: Dict) -> Dict:
        """
        Simulate Random policy
        
        Randomly chooses Retrieve or Reason with 50% probability
        """
        U = 0.0
        step = 0
        max_steps = 20
        retrieval_count = 0
        reason_count = 0
        theta_star = 0.9  # Fixed termination threshold
        
        while U < theta_star and step < max_steps:
            step += 1
            
            # Random choice
            if np.random.random() < 0.5:
                # Retrieve
                retrieval_count += 1
                if np.random.random() < self.p_s:
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
    
    def evaluate_all_policies(
        self,
        c_r: float,
        theta_cont: float,
        theta_star: float
    ) -> Dict:
        """
        Evaluate all 4 policies on test set
        
        Args:
            c_r: Current retrieval cost
            theta_cont: ARGO's continuation threshold
            theta_star: ARGO's termination threshold
            
        Returns:
            Results for all policies
        """
        print(f"\n  [Evaluating Policies] c_r = {c_r:.3f}")
        
        # Set random seed for reproducibility
        np.random.seed(self.seed)
        
        policies = {
            'ARGO': lambda q: self.simulate_argo_policy(q, theta_cont, theta_star),
            'Always-Retrieve': self.simulate_always_retrieve_policy,
            'Always-Reason': self.simulate_always_reason_policy,
            'Random': self.simulate_random_policy
        }
        
        results = {}
        
        for policy_name, policy_fn in policies.items():
            print(f"    {policy_name}...", end='')
            
            policy_results = []
            for question in self.test_questions:
                result = policy_fn(question)
                policy_results.append(result)
            
            # Aggregate statistics
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
            
            print(f" Q={avg_quality:.3f}, R={avg_retrievals:.1f}")
        
        return results
    
    def run_experiment(
        self,
        c_r_min_multiplier: float = 1.0,
        c_r_max_multiplier: float = 10.0,
        n_steps: int = 10
    ):
        """
        Run complete experiment: sweep c_r from c_p to 10*c_p
        
        Args:
            c_r_min_multiplier: Min c_r as multiple of c_p
            c_r_max_multiplier: Max c_r as multiple of c_p
            n_steps: Number of c_r values to test
        """
        print("\n" + "=" * 80)
        print("Experiment 1: Impact of Retrieval Cost (c_r) on Performance")
        print("=" * 80)
        print(f"\nSweeping c_r from {c_r_min_multiplier}*c_p to {c_r_max_multiplier}*c_p")
        print(f"c_p = {self.c_p:.3f}")
        print(f"Number of steps: {n_steps}")
        print(f"Test questions: {len(self.test_questions)}")
        print("=" * 80)
        
        # Generate c_r values to sweep
        c_r_values = np.linspace(
            c_r_min_multiplier * self.c_p,
            c_r_max_multiplier * self.c_p,
            n_steps
        )
        
        print(f"\nc_r values: {c_r_values}")
        
        # Storage for results
        self.results['c_r_values'] = c_r_values.tolist()
        self.results['policies'] = {
            'ARGO': {'quality': [], 'retrievals': [], 'thresholds': []},
            'Always-Retrieve': {'quality': [], 'retrievals': []},
            'Always-Reason': {'quality': [], 'retrievals': []},
            'Random': {'quality': [], 'retrievals': []}
        }
        
        # Run experiment for each c_r
        for i, c_r in enumerate(c_r_values):
            print(f"\n{'='*60}")
            print(f"Step {i+1}/{n_steps}: c_r = {c_r:.4f} ({c_r/self.c_p:.1f}*c_p)")
            print(f"{'='*60}")
            
            # Step 1: Solve MDP for this c_r
            theta_cont, theta_star = self.solve_mdp(c_r)
            
            # Step 2: Evaluate all policies
            policy_results = self.evaluate_all_policies(c_r, theta_cont, theta_star)
            
            # Step 3: Store results
            self.results['policies']['ARGO']['quality'].append(
                policy_results['ARGO']['avg_quality']
            )
            self.results['policies']['ARGO']['retrievals'].append(
                policy_results['ARGO']['avg_retrievals']
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
        
        print("\n" + "=" * 80)
        print("Experiment Complete!")
        print("=" * 80)
        
        return self.results
    
    def save_results(self, output_dir: str = "draw_figs/data"):
        """
        Save experiment results to JSON
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exp1_retrieval_cost_impact_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Add metadata
        results_with_metadata = {
            'experiment': 'Impact of Retrieval Cost',
            'timestamp': timestamp,
            'config': {
                'n_test_questions': self.n_test_questions,
                'difficulty': self.difficulty,
                'seed': self.seed,
                'delta_r': self.delta_r,
                'delta_p': self.delta_p,
                'p_s': self.p_s,
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
        """
        Generate Figure 1 and Figure 2
        
        Figure 1: Retrieval Cost vs. Answer Quality
        Figure 2: Retrieval Cost vs. Average Retrievals
        """
        os.makedirs(output_dir, exist_ok=True)
        
        c_r_values = np.array(self.results['c_r_values'])
        c_r_normalized = c_r_values / self.c_p  # Normalize by c_p
        
        # Figure 1: Cost vs Quality
        plt.figure(figsize=(10, 6))
        
        for policy in ['ARGO', 'Always-Retrieve', 'Always-Reason', 'Random']:
            quality = self.results['policies'][policy]['quality']
            
            if policy == 'ARGO':
                plt.plot(c_r_normalized, quality, marker='o', linewidth=2.5, 
                        label=policy, color='#2E86AB')
            elif policy == 'Always-Retrieve':
                plt.plot(c_r_normalized, quality, marker='s', linewidth=2, 
                        label=policy, color='#A23B72', linestyle='--')
            elif policy == 'Always-Reason':
                plt.plot(c_r_normalized, quality, marker='^', linewidth=2, 
                        label=policy, color='#F18F01', linestyle='--')
            else:  # Random
                plt.plot(c_r_normalized, quality, marker='x', linewidth=2, 
                        label=policy, color='#C73E1D', linestyle=':')
        
        plt.xlabel('Retrieval Cost ($c_r$ / $c_p$)', fontsize=12)
        plt.ylabel('Average Answer Quality $E[Q(O)]$', fontsize=12)
        plt.title('Figure 1: Impact of Retrieval Cost on Answer Quality', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig1_path = os.path.join(output_dir, 'exp1_cost_vs_quality.png')
        plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure 1 saved to: {fig1_path}")
        plt.close()
        
        # Figure 2: Cost vs Retrievals
        plt.figure(figsize=(10, 6))
        
        for policy in ['ARGO', 'Always-Retrieve', 'Always-Reason', 'Random']:
            retrievals = self.results['policies'][policy]['retrievals']
            
            if policy == 'ARGO':
                plt.plot(c_r_normalized, retrievals, marker='o', linewidth=2.5, 
                        label=policy, color='#2E86AB')
            elif policy == 'Always-Retrieve':
                plt.plot(c_r_normalized, retrievals, marker='s', linewidth=2, 
                        label=policy, color='#A23B72', linestyle='--')
            elif policy == 'Always-Reason':
                plt.plot(c_r_normalized, retrievals, marker='^', linewidth=2, 
                        label=policy, color='#F18F01', linestyle='--')
            else:  # Random
                plt.plot(c_r_normalized, retrievals, marker='x', linewidth=2, 
                        label=policy, color='#C73E1D', linestyle=':')
        
        plt.xlabel('Retrieval Cost ($c_r$ / $c_p$)', fontsize=12)
        plt.ylabel('Average Retrieval Calls $E[R_T]$', fontsize=12)
        plt.title('Figure 2: Impact of Retrieval Cost on Retrieval Usage', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig2_path = os.path.join(output_dir, 'exp1_cost_vs_retrievals.png')
        plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
        print(f"Figure 2 saved to: {fig2_path}")
        plt.close()
        
        # Additional: Threshold evolution
        plt.figure(figsize=(10, 6))
        
        thresholds = self.results['policies']['ARGO']['thresholds']
        theta_cont_values = [t['theta_cont'] for t in thresholds]
        theta_star_values = [t['theta_star'] for t in thresholds]
        
        plt.plot(c_r_normalized, theta_cont_values, marker='o', linewidth=2.5, 
                label='$\\Theta_{cont}$ (Retrieve→Reason)', color='#2E86AB')
        plt.plot(c_r_normalized, theta_star_values, marker='s', linewidth=2.5, 
                label='$\\Theta^*$ (Termination)', color='#A23B72')
        
        plt.xlabel('Retrieval Cost ($c_r$ / $c_p$)', fontsize=12)
        plt.ylabel('Threshold Value', fontsize=12)
        plt.title('Supplementary: ARGO Threshold Evolution', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig3_path = os.path.join(output_dir, 'exp1_threshold_evolution.png')
        plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
        print(f"Supplementary Figure saved to: {fig3_path}")
        plt.close()


def main():
    """
    Main function: Run Experiment 1
    """
    print("\n" + "=" * 80)
    print("ARGO Experiment 1: Impact of Retrieval Cost (c_r)")
    print("=" * 80)
    print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Initialize experiment
    exp = CostImpactExperiment(
        config_path="configs/multi_gpu.yaml",
        n_test_questions=100,  # Use 100 questions for faster testing
        difficulty="medium",   # Medium difficulty
        seed=42
    )
    
    # Run experiment
    results = exp.run_experiment(
        c_r_min_multiplier=1.0,   # Start at c_r = c_p
        c_r_max_multiplier=10.0,  # End at c_r = 10*c_p
        n_steps=10                # Test 10 different c_r values
    )
    
    # Save results
    results_path = exp.save_results()
    
    # Generate plots
    print("\n" + "=" * 80)
    print("Generating Plots...")
    print("=" * 80)
    exp.plot_results()
    
    print("\n" + "=" * 80)
    print("Experiment 1 Complete!")
    print("=" * 80)
    print("\nKey Findings:")
    print("1. Check Figure 1 (cost_vs_quality.png): ARGO should maintain quality")
    print("2. Check Figure 2 (cost_vs_retrievals.png): ARGO should decrease retrievals sharply")
    print("3. Baselines should show flat lines (no adaptation)")
    print("=" * 80)


if __name__ == "__main__":
    main()
