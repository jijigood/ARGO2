"""
RAG Experiment with Actual Queries
Compares retrieval strategies on real O-RAN queries
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import random
from typing import Dict, List
import json
import datetime

from Env_RAG import Env_RAG
from query_generator import QueryGenerator

# Try to import RAG models if available
try:
    from RAG_Models.retrieval import build_vector_store
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("Warning: RAG_Models not available, running in simulation mode")


def run_query_based_experiment(num_queries: int = 100, seed: int = 42):
    """
    Run experiment with actual queries
    
    Args:
        num_queries: Number of queries to test
        seed: Random seed
    """
    print("=" * 80)
    print(f"Query-Based RAG Experiment (Seed={seed})")
    print("=" * 80)
    
    # Initialize query generator
    query_gen = QueryGenerator(seed=seed)
    
    # Generate test queries
    print(f"\nGenerating {num_queries} test queries...")
    test_queries = query_gen.generate_batch(num_queries)
    
    # Count by complexity
    complexity_counts = {1: 0, 2: 0, 3: 0}
    for _, complexity in test_queries:
        complexity_counts[complexity] += 1
    
    print(f"  Simple queries: {complexity_counts[1]}")
    print(f"  Medium queries: {complexity_counts[2]}")
    print(f"  Complex queries: {complexity_counts[3]}")
    
    # Initialize environment
    env = Env_RAG(cost_weight=0.1, seed=seed)
    
    # Initialize RAG system if available
    if RAG_AVAILABLE:
        print("\nBuilding vector store...")
        try:
            vector_store, retriever = build_vector_store()
            print("✓ Vector store ready")
        except Exception as e:
            print(f"✗ Vector store failed: {e}")
            print("  Falling back to simulation mode")
            RAG_AVAILABLE = False
    
    # Run experiments with different policies
    policies = {
        'optimal': lambda state: env.opt_policy(state),
        'fixed_k3': lambda state: env.fixed_policy(3),
        'fixed_k5': lambda state: env.fixed_policy(5),
        'adaptive': lambda state: env.adaptive_policy(state)
    }
    
    results = {}
    
    for policy_name, policy_fn in policies.items():
        print(f"\n[Testing {policy_name} policy]")
        
        policy_rewards = []
        policy_accuracies = []
        policy_costs = []
        query_results = []
        
        # Reset environment
        state = env.reset()
        
        for i, (query_text, complexity) in enumerate(test_queries):
            # Inject actual query complexity into state
            state[0] = complexity  # Set query complexity
            
            # Get action from policy
            action = policy_fn(state)
            
            if RAG_AVAILABLE:
                # TODO: Actual RAG retrieval with action parameters
                # top_k, use_rerank, use_filter = action
                # retrieved_docs = retriever.retrieve(query_text, top_k=top_k)
                # accuracy = evaluate_retrieval(retrieved_docs, ground_truth)
                pass
            
            # Step environment (simulated or real)
            next_state, reward, done, info = env.step(action)
            
            policy_rewards.append(reward)
            policy_accuracies.append(info["accuracy"])
            policy_costs.append(info["cost"])
            
            query_results.append({
                'query_id': i,
                'query_text': query_text,
                'complexity': complexity,
                'action': action,
                'reward': reward,
                'accuracy': info["accuracy"],
                'cost': info["cost"]
            })
            
            state = next_state
            if done:
                state = env.reset()
        
        # Calculate statistics
        results[policy_name] = {
            'avg_reward': np.mean(policy_rewards),
            'std_reward': np.std(policy_rewards),
            'avg_accuracy': np.mean(policy_accuracies),
            'std_accuracy': np.std(policy_accuracies),
            'avg_cost': np.mean(policy_costs),
            'std_cost': np.std(policy_costs),
            'queries': query_results
        }
        
        print(f"  Avg Reward: {results[policy_name]['avg_reward']:.3f} ± {results[policy_name]['std_reward']:.3f}")
        print(f"  Avg Accuracy: {results[policy_name]['avg_accuracy']:.3f} ± {results[policy_name]['std_accuracy']:.3f}")
        print(f"  Avg Cost: {results[policy_name]['avg_cost']:.3f} ± {results[policy_name]['std_cost']:.3f}")
    
    return results, test_queries


def analyze_by_complexity(results: Dict, test_queries: List):
    """
    Analyze performance by query complexity
    """
    print("\n" + "=" * 80)
    print("Performance by Query Complexity")
    print("=" * 80)
    
    for policy_name, policy_data in results.items():
        print(f"\n{policy_name.upper()} Policy:")
        
        # Group by complexity
        complexity_groups = {1: [], 2: [], 3: []}
        
        for query_result in policy_data['queries']:
            complexity = query_result['complexity']
            complexity_groups[complexity].append({
                'accuracy': query_result['accuracy'],
                'cost': query_result['cost'],
                'reward': query_result['reward']
            })
        
        # Calculate stats for each complexity
        complexity_names = {1: "Simple", 2: "Medium", 3: "Complex"}
        for complexity, name in complexity_names.items():
            if complexity_groups[complexity]:
                data = complexity_groups[complexity]
                accuracies = [d['accuracy'] for d in data]
                costs = [d['cost'] for d in data]
                rewards = [d['reward'] for d in data]
                
                print(f"  {name} (n={len(data)}):")
                print(f"    Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
                print(f"    Cost:     {np.mean(costs):.3f} ± {np.std(costs):.3f}")
                print(f"    Reward:   {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")


def save_results(results: Dict, queries: List, filename: str):
    """Save results with actual queries"""
    output_dir = "draw_figs/data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare output
    output = {
        'timestamp': datetime.datetime.now().isoformat(),
        'num_queries': len(queries),
        'queries': [{'text': q, 'complexity': c} for q, c in queries],
        'results': results
    }
    
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {filepath}")


if __name__ == "__main__":
    print("=" * 80)
    print("RAG Query-Based Experiment")
    print("=" * 80)
    print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"RAG Models Available: {RAG_AVAILABLE}")
    print("=" * 80)
    
    # Run experiment with actual queries
    results, test_queries = run_query_based_experiment(num_queries=100, seed=42)
    
    # Analyze by complexity
    analyze_by_complexity(results, test_queries)
    
    # Save results
    save_results(results, test_queries, "query_based_experiment.json")
    
    print("\n" + "=" * 80)
    print("Experiment completed!")
    print("=" * 80)
