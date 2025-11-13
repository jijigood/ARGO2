"""
RAG Retrieval Strategy MDP Environment
将RAG检索策略建模为马尔可夫决策过程
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import random
from typing import Tuple, List, Dict
import gym

# MDP参数
ALPHA = 0.1         # learning rate
GAMMA = 0.9         # discount factor
Episode_num = 1
Step_num = 100
Seed = 42

# RAG参数
TOP_K_RANGE = [1, 3, 5, 7, 10]  # 可选的top-k值
SIMILARITY_THRESHOLD_RANGE = [0.5, 0.6, 0.7, 0.8]  # 相似度阈值


class QueryComplexity:
    """查询复杂度分类"""
    SIMPLE = 1      # 简单查询(单个概念)
    MEDIUM = 2      # 中等查询(多个概念)
    COMPLEX = 3     # 复杂查询(需要推理)


class Env_RAG(gym.Env):
    """
    RAG检索策略的MDP环境
    
    状态空间 S = (query_complexity, retrieval_quality, cost_budget)
    - query_complexity: 查询复杂度 {1: 简单, 2: 中等, 3: 复杂}
    - retrieval_quality: 历史检索质量 [0, 100]
    - cost_budget: 剩余成本预算 [0, 100]
    
    动作空间 A = (top_k, use_rerank, use_filter)
    - top_k: 检索文档数量 {1, 3, 5, 7, 10}
    - use_rerank: 是否使用重排序 {0, 1}
    - use_filter: 是否使用类别过滤 {0, 1}
    
    奖励函数:
    - 正奖励: 检索准确率 * 100
    - 负奖励: 计算成本 (top_k * cost_per_doc + rerank_cost + filter_cost)
    - 目标: 最大化 (准确率 - 成本系数 * 成本)
    """
    
    def __init__(self, 
                 cost_weight: float = 0.1,
                 seed: int = Seed):
        super(Env_RAG, self).__init__()
        
        # 参数
        self.cost_weight = cost_weight
        self.seed = seed
        self.set_seed(seed)
        
        # 成本模型
        self.cost_per_doc = 1.0         # 每个文档的检索成本
        self.rerank_cost = 5.0          # 重排序成本
        self.filter_cost = 2.0          # 过滤成本
        
        # 状态空间
        self.complexity_levels = [1, 2, 3]
        self.quality_max = 100
        self.budget_max = 100
        
        # 动作空间
        self.top_k_options = TOP_K_RANGE
        self.use_rerank_options = [0, 1]
        self.use_filter_options = [0, 1]
        
        # 状态
        self.state = [QueryComplexity.SIMPLE, 50.0, 100.0]  # [complexity, quality, budget]
        
        # 统计
        self.total_queries = 0
        self.successful_queries = 0
        self.total_cost = 0.0
        
    def reset(self):
        """重置环境"""
        self.state = [QueryComplexity.SIMPLE, 50.0, 100.0]
        self.total_queries = 0
        self.successful_queries = 0
        self.total_cost = 0.0
        return self.state
    
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
    
    def generate_query(self) -> Tuple[int, str]:
        """
        生成新查询
        Returns:
            complexity: 查询复杂度
            query_type: 查询类型描述
        """
        complexity_prob = [0.4, 0.4, 0.2]  # Simple, Medium, Complex
        complexity = random.choices(self.complexity_levels, weights=complexity_prob, k=1)[0]
        
        query_types = {
            1: "simple_fact",      # 简单事实查询
            2: "multi_concept",    # 多概念查询
            3: "reasoning"         # 需要推理的复杂查询
        }
        
        return complexity, query_types[complexity]
    
    def calculate_retrieval_accuracy(self, 
                                     query_complexity: int,
                                     top_k: int,
                                     use_rerank: int,
                                     use_filter: int) -> float:
        """
        计算检索准确率
        准确率取决于查询复杂度和动作选择
        """
        # 基础准确率
        base_accuracy = {
            1: 0.8,   # 简单查询基础准确率80%
            2: 0.6,   # 中等查询基础准确率60%
            3: 0.4    # 复杂查询基础准确率40%
        }
        
        accuracy = base_accuracy[query_complexity]
        
        # top_k影响 - 更多文档可能提高召回率
        if query_complexity == 1:
            # 简单查询：top_k=1或3就够了，太多反而干扰
            if top_k <= 3:
                accuracy += 0.1
            elif top_k >= 7:
                accuracy -= 0.05
        elif query_complexity == 2:
            # 中等查询：top_k=5-7较好
            if 3 <= top_k <= 7:
                accuracy += 0.15
        else:  # complex
            # 复杂查询：需要更多文档
            if top_k >= 7:
                accuracy += 0.2
            elif top_k <= 3:
                accuracy -= 0.1
        
        # 重排序影响 - 对复杂查询帮助更大
        if use_rerank:
            accuracy += 0.05 * query_complexity
        
        # 过滤影响 - 可能提高精确率
        if use_filter:
            accuracy += 0.08
        
        # 添加随机噪声
        noise = np.random.normal(0, 0.05)
        accuracy = np.clip(accuracy + noise, 0.0, 1.0)
        
        return accuracy
    
    def calculate_cost(self, top_k: int, use_rerank: int, use_filter: int) -> float:
        """计算动作成本"""
        cost = top_k * self.cost_per_doc
        if use_rerank:
            cost += self.rerank_cost
        if use_filter:
            cost += self.filter_cost
        
        return cost
    
    def step(self, action: Tuple[int, int, int]) -> Tuple[List, float, bool, Dict]:
        """
        执行动作
        Args:
            action: (top_k, use_rerank, use_filter)
        Returns:
            next_state: 下一个状态
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        top_k, use_rerank, use_filter = action
        query_complexity = self.state[0]
        
        # 计算检索准确率
        accuracy = self.calculate_retrieval_accuracy(
            query_complexity, top_k, use_rerank, use_filter
        )
        
        # 计算成本
        cost = self.calculate_cost(top_k, use_rerank, use_filter)
        
        # 计算奖励: 准确率 - 成本权重 * 成本
        reward = accuracy * 100 - self.cost_weight * cost
        
        # 更新统计
        self.total_queries += 1
        if accuracy >= 0.7:  # 准确率>=70%视为成功
            self.successful_queries += 1
        self.total_cost += cost
        
        # 更新状态
        # 质量：基于准确率的移动平均
        old_quality = self.state[1]
        new_quality = 0.7 * old_quality + 0.3 * (accuracy * 100)
        
        # 预算：减少成本
        old_budget = self.state[2]
        new_budget = max(0, old_budget - cost)
        
        # 生成新查询
        new_complexity, query_type = self.generate_query()
        
        next_state = [new_complexity, new_quality, new_budget]
        self.state = next_state
        
        # 判断是否结束
        done = new_budget <= 0
        
        info = {
            "accuracy": accuracy,
            "cost": cost,
            "query_type": query_type,
            "success_rate": self.successful_queries / self.total_queries if self.total_queries > 0 else 0
        }
        
        return next_state, reward, done, info
    
    def opt_policy(self, state: List) -> Tuple[int, int, int]:
        """
        最优策略：根据查询复杂度和预算选择动作
        """
        complexity, quality, budget = state
        
        # 根据复杂度选择top_k
        if complexity == QueryComplexity.SIMPLE:
            top_k = 3
        elif complexity == QueryComplexity.MEDIUM:
            top_k = 5
        else:  # COMPLEX
            top_k = 7
        
        # 根据预算和复杂度决定是否使用rerank
        if budget >= 30 and complexity >= 2:
            use_rerank = 1
        else:
            use_rerank = 0
        
        # 根据预算决定是否使用filter
        if budget >= 20:
            use_filter = 1
        else:
            use_filter = 0
        
        return (top_k, use_rerank, use_filter)
    
    def fixed_policy(self, top_k: int = 5) -> Tuple[int, int, int]:
        """固定top-k策略"""
        return (top_k, 0, 0)
    
    def adaptive_policy(self, state: List) -> Tuple[int, int, int]:
        """自适应策略：根据历史质量调整"""
        complexity, quality, budget = state
        
        # 如果质量低，增加检索数量
        if quality < 50:
            top_k = 7
            use_rerank = 1
        elif quality < 70:
            top_k = 5
            use_rerank = 0
        else:
            top_k = 3
            use_rerank = 0
        
        use_filter = 1 if budget >= 20 else 0
        
        return (top_k, use_rerank, use_filter)


def test_opt_policy(num_steps: int = 100, seed: int = Seed):
    """测试最优策略"""
    env = Env_RAG(cost_weight=0.1, seed=seed)
    state = env.reset()
    
    total_reward = 0
    episode_rewards = []
    episode_info = []
    
    for step in range(num_steps):
        action = env.opt_policy(state)
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        episode_rewards.append(reward)
        episode_info.append(info)
        
        state = next_state
        
        if done:
            state = env.reset()
    
    avg_reward = total_reward / num_steps
    avg_accuracy = np.mean([info["accuracy"] for info in episode_info])
    avg_cost = env.total_cost / num_steps
    success_rate = env.successful_queries / env.total_queries
    
    print(f"Optimal Policy Results:")
    print(f"  Avg Reward: {avg_reward:.2f}")
    print(f"  Avg Accuracy: {avg_accuracy:.3f}")
    print(f"  Avg Cost: {avg_cost:.2f}")
    print(f"  Success Rate: {success_rate:.3f}")
    print(f"  Total Queries: {env.total_queries}")
    
    return {
        "avg_reward": avg_reward,
        "avg_accuracy": avg_accuracy,
        "avg_cost": avg_cost,
        "success_rate": success_rate
    }


def test_fixed_policy(top_k: int = 5, num_steps: int = 100, seed: int = Seed):
    """测试固定top-k策略"""
    env = Env_RAG(cost_weight=0.1, seed=seed)
    state = env.reset()
    
    total_reward = 0
    episode_info = []
    
    for step in range(num_steps):
        action = env.fixed_policy(top_k)
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        episode_info.append(info)
        
        state = next_state
        
        if done:
            state = env.reset()
    
    avg_reward = total_reward / num_steps
    avg_accuracy = np.mean([info["accuracy"] for info in episode_info])
    avg_cost = env.total_cost / num_steps
    success_rate = env.successful_queries / env.total_queries
    
    print(f"Fixed Policy (top_k={top_k}) Results:")
    print(f"  Avg Reward: {avg_reward:.2f}")
    print(f"  Avg Accuracy: {avg_accuracy:.3f}")
    print(f"  Avg Cost: {avg_cost:.2f}")
    print(f"  Success Rate: {success_rate:.3f}")
    
    return {
        "avg_reward": avg_reward,
        "avg_accuracy": avg_accuracy,
        "avg_cost": avg_cost,
        "success_rate": success_rate
    }


def test_adaptive_policy(num_steps: int = 100, seed: int = Seed):
    """测试自适应策略"""
    env = Env_RAG(cost_weight=0.1, seed=seed)
    state = env.reset()
    
    total_reward = 0
    episode_info = []
    
    for step in range(num_steps):
        action = env.adaptive_policy(state)
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        episode_info.append(info)
        
        state = next_state
        
        if done:
            state = env.reset()
    
    avg_reward = total_reward / num_steps
    avg_accuracy = np.mean([info["accuracy"] for info in episode_info])
    avg_cost = env.total_cost / num_steps
    success_rate = env.successful_queries / env.total_queries
    
    print(f"Adaptive Policy Results:")
    print(f"  Avg Reward: {avg_reward:.2f}")
    print(f"  Avg Accuracy: {avg_accuracy:.3f}")
    print(f"  Avg Cost: {avg_cost:.2f}")
    print(f"  Success Rate: {success_rate:.3f}")
    
    return {
        "avg_reward": avg_reward,
        "avg_accuracy": avg_accuracy,
        "avg_cost": avg_cost,
        "success_rate": success_rate
    }


if __name__ == "__main__":
    print("=" * 60)
    print("RAG Retrieval Strategy MDP Environment Test")
    print("=" * 60)
    
    num_steps = 200
    
    print("\n[1/3] Testing Optimal Policy...")
    test_opt_policy(num_steps)
    
    print("\n[2/3] Testing Fixed Policy (top_k=5)...")
    test_fixed_policy(top_k=5, num_steps=num_steps)
    
    print("\n[3/3] Testing Adaptive Policy...")
    test_adaptive_policy(num_steps)
