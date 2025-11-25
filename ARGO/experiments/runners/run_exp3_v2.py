#!/usr/bin/env python
"""
Run Script for Experiment 3 V2: Pareto Frontier Analysis (Enhanced)
===================================================================
Executes the enhanced Pareto analysis with:
- Fixed-Threshold sweep
- Statistical validation
- Efficiency gap visualization
- 3B Model

Usage:
    python run_exp3_v2.py
"""

import os
import sys
import argparse

# Add path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Exp_real_pareto_frontier_v2 import ParetoFrontierExperimentV2

def main():
    parser = argparse.ArgumentParser(description='Run Experiment 3 V2')
    parser.add_argument('--n-questions', type=int, default=100, help='Number of questions')
    parser.add_argument('--difficulty', type=str, default='medium', help='Difficulty level')
    parser.add_argument('--mu-min', type=float, default=0.0, help='Minimum mu value')
    parser.add_argument('--mu-max', type=float, default=2.0, help='Maximum mu value')
    parser.add_argument('--n-mu-steps', type=int, default=15, help='Number of mu steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7', help='GPU IDs (comma separated)')
    
    args = parser.parse_args()
    
    gpu_ids = [int(x) for x in args.gpus.split(',')]
    
    print("=" * 80)
    print("Experiment 3 V2: Enhanced Pareto Frontier Analysis")
    print("=" * 80)
    print(f"Model: Qwen2.5-3B-Instruct")
    print(f"Questions: {args.n_questions} ({args.difficulty})")
    print(f"μ Range: [{args.mu_min}, {args.mu_max}] ({args.n_mu_steps} steps)")
    print(f"GPUs: {gpu_ids}")
    print("=" * 80)
    
    # Initialize experiment
    exp = ParetoFrontierExperimentV2(
        config_path="configs/multi_gpu_data_calibrated.yaml",
        llm_model_path="/data/user/huangxiaolin/ARGO/RAG_Models/models/Qwen2.5-3B-Instruct",
        embedding_model_path="/data/user/huangxiaolin/ARGO/models/all-MiniLM-L6-v2",
        chroma_db_path="/data/user/huangxiaolin/ARGO2/ARGO/Environments/chroma_store",
        n_test_questions=args.n_questions,
        difficulty=args.difficulty,
        seed=args.seed,
        gpu_ids=gpu_ids
    )
    
    # Run experiment
    exp.run_experiment(
        mu_min=args.mu_min,
        mu_max=args.mu_max,
        n_mu_steps=args.n_mu_steps
    )
    
    # Run validations
    print("\nRunning Validations...")
    exp.validate_threshold_monotonicity()
    exp.validate_pareto_dominance()
    exp.validate_mu_range()
    exp.compute_quality_accuracy_correlation()
    
    # Save and Plot
    print("\nSaving Results and Plots...")
    exp.save_results()
    exp.plot_pareto_with_efficiency_gap()
    exp.plot_threshold_evolution()
    exp.plot_pareto_accuracy()
    
    print("\nExperiment 3 V2 Completed Successfully!")

if __name__ == "__main__":
    main()
