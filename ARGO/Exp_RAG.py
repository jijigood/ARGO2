"""
RAG Experiment Script
运行RAG检索策略的完整实验
"""
import sys
import os
import numpy as np
import random
from typing import Dict, List
import json
import datetime

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Env_RAG import Env_RAG, test_opt_policy, test_fixed_policy, test_adaptive_policy

# 实验参数
SEEDS = [42, 123, 456, 789, 1024]
NUM_STEPS = 500
COST_WEIGHTS = [0.05, 0.1, 0.2, 0.5]  # 不同的成本权重


def run_comparison_experiment(num_steps: int = 500, seed: int = 42):
    """
    运行策略对比实验
    """
    print("=" * 80)
    print(f"Running Comparison Experiment (Seed={seed}, Steps={num_steps})")
    print("=" * 80)
    
    results = {}
    
    # 1. 测试最优策略
    print("\n[1/4] Testing Optimal Policy...")
    env = Env_RAG(cost_weight=0.1, seed=seed)
    state = env.reset()
    opt_rewards = []
    opt_accuracies = []
    opt_costs = []
    
    for step in range(num_steps):
        action = env.opt_policy(state)
        next_state, reward, done, info = env.step(action)
        opt_rewards.append(reward)
        opt_accuracies.append(info["accuracy"])
        opt_costs.append(info["cost"])
        state = next_state
        if done:
            state = env.reset()
    
    results["optimal"] = {
        "avg_reward": np.mean(opt_rewards),
        "avg_accuracy": np.mean(opt_accuracies),
        "avg_cost": np.mean(opt_costs),
        "success_rate": env.successful_queries / env.total_queries
    }
    print(f"  Avg Reward: {results['optimal']['avg_reward']:.2f}")
    print(f"  Avg Accuracy: {results['optimal']['avg_accuracy']:.3f}")
    print(f"  Success Rate: {results['optimal']['success_rate']:.3f}")
    
    # 2. 测试固定策略 (top_k=3, 5, 7)
    print("\n[2/4] Testing Fixed Policies...")
    for top_k in [3, 5, 7]:
        env = Env_RAG(cost_weight=0.1, seed=seed)
        state = env.reset()
        fixed_rewards = []
        fixed_accuracies = []
        fixed_costs = []
        
        for step in range(num_steps):
            action = env.fixed_policy(top_k)
            next_state, reward, done, info = env.step(action)
            fixed_rewards.append(reward)
            fixed_accuracies.append(info["accuracy"])
            fixed_costs.append(info["cost"])
            state = next_state
            if done:
                state = env.reset()
        
        results[f"fixed_k{top_k}"] = {
            "avg_reward": np.mean(fixed_rewards),
            "avg_accuracy": np.mean(fixed_accuracies),
            "avg_cost": np.mean(fixed_costs),
            "success_rate": env.successful_queries / env.total_queries
        }
        print(f"  top_k={top_k}: Reward={results[f'fixed_k{top_k}']['avg_reward']:.2f}, " +
              f"Accuracy={results[f'fixed_k{top_k}']['avg_accuracy']:.3f}")
    
    # 3. 测试自适应策略
    print("\n[3/4] Testing Adaptive Policy...")
    env = Env_RAG(cost_weight=0.1, seed=seed)
    state = env.reset()
    adap_rewards = []
    adap_accuracies = []
    adap_costs = []
    
    for step in range(num_steps):
        action = env.adaptive_policy(state)
        next_state, reward, done, info = env.step(action)
        adap_rewards.append(reward)
        adap_accuracies.append(info["accuracy"])
        adap_costs.append(info["cost"])
        state = next_state
        if done:
            state = env.reset()
    
    results["adaptive"] = {
        "avg_reward": np.mean(adap_rewards),
        "avg_accuracy": np.mean(adap_accuracies),
        "avg_cost": np.mean(adap_costs),
        "success_rate": env.successful_queries / env.total_queries
    }
    print(f"  Avg Reward: {results['adaptive']['avg_reward']:.2f}")
    print(f"  Avg Accuracy: {results['adaptive']['avg_accuracy']:.3f}")
    
    # 4. 测试随机策略 (baseline)
    print("\n[4/4] Testing Random Policy...")
    env = Env_RAG(cost_weight=0.1, seed=seed)
    state = env.reset()
    rand_rewards = []
    rand_accuracies = []
    rand_costs = []
    
    for step in range(num_steps):
        top_k = random.choice([1, 3, 5, 7, 10])
        use_rerank = random.choice([0, 1])
        use_filter = random.choice([0, 1])
        action = (top_k, use_rerank, use_filter)
        next_state, reward, done, info = env.step(action)
        rand_rewards.append(reward)
        rand_accuracies.append(info["accuracy"])
        rand_costs.append(info["cost"])
        state = next_state
        if done:
            state = env.reset()
    
    results["random"] = {
        "avg_reward": np.mean(rand_rewards),
        "avg_accuracy": np.mean(rand_accuracies),
        "avg_cost": np.mean(rand_costs),
        "success_rate": env.successful_queries / env.total_queries
    }
    print(f"  Avg Reward: {results['random']['avg_reward']:.2f}")
    print(f"  Avg Accuracy: {results['random']['avg_accuracy']:.3f}")
    
    return results


def run_cost_weight_experiment(num_steps: int = 500, seed: int = 42):
    """
    实验不同成本权重的影响
    """
    print("\n" + "=" * 80)
    print("Cost Weight Sensitivity Analysis")
    print("=" * 80)
    
    results = {}
    
    for cost_weight in COST_WEIGHTS:
        print(f"\nTesting cost_weight={cost_weight}...")
        env = Env_RAG(cost_weight=cost_weight, seed=seed)
        state = env.reset()
        
        rewards = []
        accuracies = []
        costs = []
        
        for step in range(num_steps):
            action = env.opt_policy(state)
            next_state, reward, done, info = env.step(action)
            rewards.append(reward)
            accuracies.append(info["accuracy"])
            costs.append(info["cost"])
            state = next_state
            if done:
                state = env.reset()
        
        results[f"weight_{cost_weight}"] = {
            "avg_reward": np.mean(rewards),
            "avg_accuracy": np.mean(accuracies),
            "avg_cost": np.mean(costs),
            "success_rate": env.successful_queries / env.total_queries
        }
        
        print(f"  Reward: {results[f'weight_{cost_weight}']['avg_reward']:.2f}, " +
              f"Accuracy: {results[f'weight_{cost_weight}']['avg_accuracy']:.3f}, " +
              f"Cost: {results[f'weight_{cost_weight}']['avg_cost']:.2f}")
    
    return results


def save_results(results: Dict, filename: str):
    """保存实验结果"""
    output_dir = "draw_figs/data"
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {filepath}")


def run_multi_seed_experiment():
    """运行多随机种子实验"""
    print("\n" + "=" * 80)
    print("Multi-Seed Experiment")
    print("=" * 80)
    
    all_results = {}
    
    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"Seed: {seed}")
        print(f"{'='*60}")
        
        results = run_comparison_experiment(num_steps=NUM_STEPS, seed=seed)
        all_results[f"seed_{seed}"] = results
        
        # 保存单个seed的结果
        save_results(
            results,
            f"comparison_seed{seed}_steps{NUM_STEPS}.json"
        )
    
    # 计算平均结果
    print("\n" + "=" * 80)
    print("Average Results Across All Seeds")
    print("=" * 80)
    
    policy_names = list(all_results[f"seed_{SEEDS[0]}"].keys())
    avg_results = {}
    
    for policy in policy_names:
        metrics = {}
        for metric in ["avg_reward", "avg_accuracy", "avg_cost", "success_rate"]:
            values = [all_results[f"seed_{seed}"][policy][metric] for seed in SEEDS]
            metrics[metric] = {
                "mean": np.mean(values),
                "std": np.std(values)
            }
        avg_results[policy] = metrics
        
        print(f"\n{policy}:")
        print(f"  Reward: {metrics['avg_reward']['mean']:.2f} ± {metrics['avg_reward']['std']:.2f}")
        print(f"  Accuracy: {metrics['avg_accuracy']['mean']:.3f} ± {metrics['avg_accuracy']['std']:.3f}")
        print(f"  Cost: {metrics['avg_cost']['mean']:.2f} ± {metrics['avg_cost']['std']:.2f}")
        print(f"  Success Rate: {metrics['success_rate']['mean']:.3f} ± {metrics['success_rate']['std']:.3f}")
    
    # 保存平均结果
    save_results(avg_results, f"comparison_average_steps{NUM_STEPS}.json")
    
    return avg_results


if __name__ == "__main__":
    print("=" * 80)
    print("RAG Retrieval Strategy MDP Experiment")
    print("=" * 80)
    print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Number of steps: {NUM_STEPS}")
    print(f"Seeds: {SEEDS}")
    print("=" * 80)
    
    # 1. 运行单次对比实验
    print("\n### PART 1: Single Run Comparison ###")
    results_single = run_comparison_experiment(num_steps=NUM_STEPS, seed=42)
    save_results(results_single, "comparison_single_run.json")
    
    # 2. 运行成本权重实验
    print("\n### PART 2: Cost Weight Analysis ###")
    results_cost = run_cost_weight_experiment(num_steps=NUM_STEPS, seed=42)
    save_results(results_cost, "cost_weight_analysis.json")
    
    # 3. 运行多种子实验
    print("\n### PART 3: Multi-Seed Experiment ###")
    results_multi = run_multi_seed_experiment()
    
    print("\n" + "=" * 80)
    print("All experiments completed!")
    print("=" * 80)
