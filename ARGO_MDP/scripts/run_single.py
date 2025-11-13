"""
Main Experiment Runner for ARGO
Runs MDP solver, ARGO policy, and baseline comparisons
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from typing import Dict, List

from src.mdp_solver import MDPSolver
from src.env_argo import ARGOEnv, MultiEpisodeRunner
from src.policy import (
    ThresholdPolicy, 
    AlwaysRetrievePolicy, 
    AlwaysReasonPolicy,
    FixedKRetrieveThenReasonPolicy,
    RandomPolicy,
    SingleThresholdPolicy
)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_argo_solver(config: Dict) -> Dict:
    """
    Run ARGO MDP solver to compute optimal policy
    
    Returns:
        Solver results including V*, Q*, and thresholds
    """
    print("=" * 80)
    print("ARGO MDP SOLVER")
    print("=" * 80)
    
    solver = MDPSolver(config)
    results = solver.solve()
    
    # Save results
    if config['output']['save_results']:
        solver.save_results(results, config['output']['results_dir'])
    
    print(f"\nSolver completed successfully!")
    print(f"  θ_cont = {results['theta_cont']:.6f}")
    print(f"  θ_star = {results['theta_star']:.6f}")
    
    return results


def run_policy_evaluation(config: Dict, policy, policy_name: str) -> Dict:
    """
    Evaluate a policy over multiple episodes
    
    Args:
        config: Configuration dictionary
        policy: Policy object
        policy_name: Name for logging
        
    Returns:
        Evaluation results
    """
    print(f"\n[Evaluating {policy_name}]")
    
    # Create environment
    env = ARGOEnv(
        mdp_config=config['mdp'],
        policy=policy,
        seed=config['experiment']['seed']
    )
    
    # Run episodes
    runner = MultiEpisodeRunner(
        env=env,
        policy=policy,
        num_episodes=config['experiment']['num_episodes'],
        max_steps=config['experiment']['max_steps_per_episode']
    )
    
    results = runner.run()
    summary = runner.get_summary()
    
    print(f"  Avg Reward: {summary['avg_reward']:.4f} ± {summary['std_reward']:.4f}")
    print(f"  Avg Quality: {summary['avg_quality']:.4f} ± {summary['std_quality']:.4f}")
    print(f"  Avg Cost: {summary['avg_cost']:.4f} ± {summary['std_cost']:.4f}")
    print(f"  Avg Steps: {summary['avg_steps']:.2f}")
    print(f"  Avg Retrieves: {summary['avg_retrieves']:.2f}")
    print(f"  Avg Reasons: {summary['avg_reasons']:.2f}")
    
    return {
        'policy_name': policy_name,
        'summary': summary,
        'episodes': results
    }


def run_baseline_comparison(config: Dict, solver_results: Dict) -> List[Dict]:
    """
    Run all baseline policies and ARGO for comparison
    
    Args:
        config: Configuration dictionary
        solver_results: Results from MDP solver
        
    Returns:
        List of evaluation results for each policy
    """
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON")
    print("=" * 80)
    
    all_results = []
    
    # 1. ARGO Threshold Policy
    argo_policy = ThresholdPolicy(
        theta_cont=solver_results['theta_cont'],
        theta_star=solver_results['theta_star']
    )
    argo_results = run_policy_evaluation(config, argo_policy, "ARGO")
    all_results.append(argo_results)
    
    # 2. Always Retrieve
    always_retrieve = AlwaysRetrievePolicy(U_max=config['mdp']['U_max'])
    retrieve_results = run_policy_evaluation(config, always_retrieve, "AlwaysRetrieve")
    all_results.append(retrieve_results)
    
    # 3. Always Reason
    always_reason = AlwaysReasonPolicy(U_max=config['mdp']['U_max'])
    reason_results = run_policy_evaluation(config, always_reason, "AlwaysReason")
    all_results.append(reason_results)
    
    # 4. Fixed K Retrieve Then Reason
    if config['baselines']['enabled']:
        for K in config['baselines']['fixed_K_values']:
            fixed_k_policy = FixedKRetrieveThenReasonPolicy(K=K, U_max=config['mdp']['U_max'])
            # Note: Need to reset counter for each episode
            fixed_results = run_policy_evaluation(config, fixed_k_policy, f"FixedK{K}")
            all_results.append(fixed_results)
    
    # 5. Random Policy
    random_policy = RandomPolicy(
        U_max=config['mdp']['U_max'],
        p_terminate=0.1,
        seed=config['experiment']['seed']
    )
    random_results = run_policy_evaluation(config, random_policy, "Random")
    all_results.append(random_results)
    
    # 6. Single Threshold (Ablation)
    single_threshold = SingleThresholdPolicy(theta=solver_results['theta_star'])
    single_results = run_policy_evaluation(config, single_threshold, "SingleThreshold")
    all_results.append(single_results)
    
    return all_results


def save_comparison_results(all_results: List[Dict], output_dir: str):
    """
    Save comparison results to CSV
    
    Args:
        all_results: List of evaluation results
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create summary table
    summary_data = []
    for result in all_results:
        summary = result['summary']
        row = {
            'Policy': result['policy_name'],
            'Avg_Reward': summary['avg_reward'],
            'Std_Reward': summary['std_reward'],
            'Avg_Quality': summary['avg_quality'],
            'Std_Quality': summary['std_quality'],
            'Avg_Cost': summary['avg_cost'],
            'Std_Cost': summary['std_cost'],
            'Avg_Steps': summary['avg_steps'],
            'Avg_Retrieves': summary['avg_retrieves'],
            'Avg_Reasons': summary['avg_reasons'],
            'Termination_Rate': summary['termination_rate']
        }
        summary_data.append(row)
    
    df_summary = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'policy_comparison.csv')
    df_summary.to_csv(summary_path, index=False)
    
    print(f"\nComparison results saved to {summary_path}")
    print("\nSummary Table:")
    print(df_summary.to_string(index=False))
    
    # Save detailed episode data for each policy
    for result in all_results:
        policy_name = result['policy_name']
        episodes = result['episodes']
        
        episode_data = []
        for ep in episodes:
            ep_summary = {
                'episode': ep['episode'],
                'total_reward': ep['total_reward'],
                'final_U': ep['final_U'],
                'final_quality': ep['final_quality'],
                'total_cost': ep['total_cost'],
                'num_steps': ep['num_steps'],
                'retrieve_count': ep['retrieve_count'],
                'reason_count': ep['reason_count'],
                'terminated': ep['terminated']
            }
            episode_data.append(ep_summary)
        
        df_episodes = pd.DataFrame(episode_data)
        episode_path = os.path.join(output_dir, f'{policy_name}_episodes.csv')
        df_episodes.to_csv(episode_path, index=False)


def run_sensitivity_analysis(config: Dict, output_dir: str):
    """
    Run sensitivity analysis on key parameters
    
    Args:
        config: Base configuration
        output_dir: Output directory
    """
    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS")
    print("=" * 80)
    
    base_config = config.copy()
    sensitivity_results = []
    
    # 1. Vary mu (cost weight)
    print("\n[1] Varying μ (cost weight)...")
    mu_values = [0.2, 0.4, 0.6, 0.8, 1.0]
    for mu in mu_values:
        test_config = base_config.copy()
        test_config['mdp']['mu'] = mu
        
        # Solve MDP
        solver = MDPSolver(test_config)
        solver.verbose = False
        results = solver.solve()
        
        # Evaluate
        policy = ThresholdPolicy(results['theta_cont'], results['theta_star'])
        env = ARGOEnv(test_config['mdp'], policy, seed=42)
        runner = MultiEpisodeRunner(env, policy, num_episodes=50, max_steps=50)
        runner.run()
        summary = runner.get_summary()
        
        sensitivity_results.append({
            'parameter': 'mu',
            'value': mu,
            'theta_cont': results['theta_cont'],
            'theta_star': results['theta_star'],
            'avg_reward': summary['avg_reward'],
            'avg_quality': summary['avg_quality'],
            'avg_cost': summary['avg_cost']
        })
        print(f"  μ={mu:.2f}: θ_cont={results['theta_cont']:.4f}, θ*={results['theta_star']:.4f}, R={summary['avg_reward']:.4f}")
    
    # 2. Vary p_s (success probability)
    print("\n[2] Varying p_s (retrieve success probability)...")
    p_s_values = [0.5, 0.6, 0.7, 0.8, 0.9]
    for p_s in p_s_values:
        test_config = base_config.copy()
        test_config['mdp']['p_s'] = p_s
        
        solver = MDPSolver(test_config)
        solver.verbose = False
        results = solver.solve()
        
        policy = ThresholdPolicy(results['theta_cont'], results['theta_star'])
        env = ARGOEnv(test_config['mdp'], policy, seed=42)
        runner = MultiEpisodeRunner(env, policy, num_episodes=50, max_steps=50)
        runner.run()
        summary = runner.get_summary()
        
        sensitivity_results.append({
            'parameter': 'p_s',
            'value': p_s,
            'theta_cont': results['theta_cont'],
            'theta_star': results['theta_star'],
            'avg_reward': summary['avg_reward'],
            'avg_quality': summary['avg_quality'],
            'avg_cost': summary['avg_cost']
        })
        print(f"  p_s={p_s:.2f}: θ_cont={results['theta_cont']:.4f}, θ*={results['theta_star']:.4f}, R={summary['avg_reward']:.4f}")
    
    # 3. Vary delta ratio
    print("\n[3] Varying δ_r/δ_p ratio...")
    ratios = [1.5, 1.75, 2.0, 2.5, 3.0]
    for ratio in ratios:
        test_config = base_config.copy()
        test_config['mdp']['delta_r'] = 0.1 * ratio
        test_config['mdp']['delta_p'] = 0.1
        
        solver = MDPSolver(test_config)
        solver.verbose = False
        results = solver.solve()
        
        policy = ThresholdPolicy(results['theta_cont'], results['theta_star'])
        env = ARGOEnv(test_config['mdp'], policy, seed=42)
        runner = MultiEpisodeRunner(env, policy, num_episodes=50, max_steps=50)
        runner.run()
        summary = runner.get_summary()
        
        sensitivity_results.append({
            'parameter': 'delta_ratio',
            'value': ratio,
            'theta_cont': results['theta_cont'],
            'theta_star': results['theta_star'],
            'avg_reward': summary['avg_reward'],
            'avg_quality': summary['avg_quality'],
            'avg_cost': summary['avg_cost']
        })
        print(f"  δ_r/δ_p={ratio:.2f}: θ_cont={results['theta_cont']:.4f}, θ*={results['theta_star']:.4f}, R={summary['avg_reward']:.4f}")
    
    # Save sensitivity results
    df_sensitivity = pd.DataFrame(sensitivity_results)
    sensitivity_path = os.path.join(output_dir, 'sensitivity_analysis.csv')
    df_sensitivity.to_csv(sensitivity_path, index=False)
    print(f"\nSensitivity analysis saved to {sensitivity_path}")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Run ARGO MDP Experiments')
    parser.add_argument('--config', type=str, default='configs/base.yaml',
                        help='Path to configuration file')
    parser.add_argument('--no-solver', action='store_true',
                        help='Skip MDP solver (use existing results)')
    parser.add_argument('--no-baselines', action='store_true',
                        help='Skip baseline comparison')
    parser.add_argument('--sensitivity', action='store_true',
                        help='Run sensitivity analysis')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print("ARGO: Optimal Policy-Based Adaptive RAG")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: {args.config}\n")
    
    # Step 1: Solve MDP
    if not args.no_solver:
        solver_results = run_argo_solver(config)
    else:
        print("Skipping MDP solver (using dummy thresholds)")
        solver_results = {
            'theta_cont': 0.3,
            'theta_star': 0.7,
            'U_grid': np.linspace(0, config['mdp']['U_max'], config['mdp']['U_grid_size']),
            'V': np.zeros(config['mdp']['U_grid_size']),
            'Q': np.zeros((config['mdp']['U_grid_size'], 3))
        }
    
    # Step 2: Run baseline comparison
    if not args.no_baselines:
        all_results = run_baseline_comparison(config, solver_results)
        save_comparison_results(all_results, config['output']['results_dir'])
    
    # Step 3: Sensitivity analysis (optional)
    if args.sensitivity:
        run_sensitivity_analysis(config, config['output']['results_dir'])
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED")
    print("=" * 80)
    print(f"Results saved to: {config['output']['results_dir']}/")


if __name__ == "__main__":
    main()
