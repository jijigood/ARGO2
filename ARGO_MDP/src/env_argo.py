"""
ARGO Environment
Implements the MDP environment for adaptive RAG with threshold policy execution
"""
import numpy as np
import random
from typing import Tuple, Dict, List, Optional
import gym


class ARGOEnv(gym.Env):
    """
    ARGO Environment for Adaptive RAG
    
    State: U ∈ [0, U_max] (continuous information progress)
    Actions: {0: Retrieve, 1: Reason, 2: Terminate}
    
    Transition:
    - Retrieve: U' = min(U + δ_r, U_max) with prob p_s, else U' = U
    - Reason: U' = min(U + δ_p, U_max) deterministically
    - Terminate: Absorbing state
    
    Reward:
    - Step cost: -c_r for Retrieve, -c_p for Reason
    - Terminal reward: Q(O) - μ * C_T
    """
    
    def __init__(self, mdp_config: Dict, policy=None, seed: int = 42):
        """
        Initialize ARGO environment
        
        Args:
            mdp_config: MDP configuration dictionary
            policy: Policy object (optional)
            seed: Random seed
        """
        super(ARGOEnv, self).__init__()
        
        # MDP parameters
        self.U_max = mdp_config['U_max']
        self.delta_r = mdp_config['delta_r']
        self.delta_p = mdp_config['delta_p']
        self.p_s = mdp_config['p_s']
        self.c_r = mdp_config['c_r']
        self.c_p = mdp_config['c_p']
        self.mu = mdp_config['mu']
        self.gamma = mdp_config['gamma']
        
        # Policy
        self.policy = policy
        
        # State
        self.U = 0.0
        self.done = False
        self.accumulated_cost = 0.0
        
        # Action space
        self.action_space = [0, 1, 2]  # Retrieve, Reason, Terminate
        
        # Statistics
        self.episode_history = []
        self.current_trajectory = []
        
        # Set seed
        self.set_seed(seed)
    
    def set_seed(self, seed: int):
        """Set random seed"""
        random.seed(seed)
        np.random.seed(seed)
    
    def quality_function(self, U: float, mode: str = "sigmoid", k: float = 5.0) -> float:
        """
        Quality function Q(O)
        
        Args:
            U: Information progress
            mode: "sigmoid" or "linear"
            k: Sigmoid steepness
            
        Returns:
            Quality score
        """
        x = U / self.U_max
        
        if mode == "sigmoid":
            return 1.0 / (1.0 + np.exp(-k * (x - 0.5)))
        else:
            return x
    
    def reset(self) -> float:
        """
        Reset environment to initial state
        
        Returns:
            Initial state U = 0
        """
        self.U = 0.0
        self.done = False
        self.accumulated_cost = 0.0
        self.current_trajectory = []
        return self.U
    
    def step(self, action: int) -> Tuple[float, float, bool, Dict]:
        """
        Execute one step in environment
        
        Args:
            action: 0=Retrieve, 1=Reason, 2=Terminate
            
        Returns:
            next_state: Next U value
            reward: Immediate reward
            done: Episode termination flag
            info: Additional information
        """
        if self.done:
            raise ValueError("Episode already terminated. Call reset().")
        
        # Initialize info
        info = {
            'action': action,
            'U_before': self.U,
            'cost': 0.0,
            'success': None
        }
        
        # Execute action
        if action == 0:  # Retrieve
            cost = self.c_r
            self.accumulated_cost += cost
            
            # Stochastic transition
            success = random.random() < self.p_s
            if success:
                self.U = min(self.U + self.delta_r, self.U_max)
                info['success'] = True
            else:
                # U remains the same
                info['success'] = False
            
            reward = -cost
            
        elif action == 1:  # Reason
            cost = self.c_p
            self.accumulated_cost += cost
            
            # Deterministic transition
            self.U = min(self.U + self.delta_p, self.U_max)
            reward = -cost
            info['success'] = True
            
        elif action == 2:  # Terminate
            cost = 0.0
            self.accumulated_cost += cost
            
            # Terminal reward
            quality = self.quality_function(self.U)
            reward = quality - self.mu * self.accumulated_cost
            
            self.done = True
            info['success'] = True
            info['quality'] = quality
            info['total_cost'] = self.accumulated_cost
            
        else:
            raise ValueError(f"Invalid action: {action}")
        
        info['cost'] = cost
        info['U_after'] = self.U
        
        # Record trajectory
        self.current_trajectory.append({
            'U': self.U,
            'action': action,
            'reward': reward,
            'cost': cost,
            'done': self.done
        })
        
        return self.U, reward, self.done, info
    
    def run_episode(self, max_steps: int = 50, policy=None) -> Dict:
        """
        Run a complete episode with given policy
        
        Args:
            max_steps: Maximum steps per episode
            policy: Policy object (uses self.policy if None)
            
        Returns:
            Episode statistics
        """
        if policy is None:
            policy = self.policy
            
        if policy is None:
            raise ValueError("No policy provided")
        
        # Reset
        U = self.reset()
        trajectory = []
        total_reward = 0.0
        
        for t in range(max_steps):
            # Select action
            action = policy.act(U)
            
            # Execute step
            next_U, reward, done, info = self.step(action)
            
            total_reward += reward
            trajectory.append({
                't': t,
                'U': info['U_before'],
                'action': action,
                'reward': reward,
                'cost': info['cost'],
                'success': info['success'],
                'done': done
            })
            
            U = next_U
            
            if done:
                break
        
        # Compute episode statistics
        retrieve_count = sum(1 for step in trajectory if step['action'] == 0)
        reason_count = sum(1 for step in trajectory if step['action'] == 1)
        
        stats = {
            'trajectory': trajectory,
            'total_reward': total_reward,
            'final_U': U,
            'final_quality': self.quality_function(U) if done else 0.0,
            'total_cost': self.accumulated_cost,
            'num_steps': len(trajectory),
            'retrieve_count': retrieve_count,
            'reason_count': reason_count,
            'terminated': done
        }
        
        return stats


class MultiEpisodeRunner:
    """
    Run multiple episodes and collect statistics
    """
    
    def __init__(self, env: ARGOEnv, policy, num_episodes: int = 100, max_steps: int = 50):
        """
        Initialize runner
        
        Args:
            env: ARGO environment
            policy: Policy object
            num_episodes: Number of episodes to run
            max_steps: Maximum steps per episode
        """
        self.env = env
        self.policy = policy
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        
        self.results = []
    
    def run(self) -> List[Dict]:
        """
        Run all episodes
        
        Returns:
            List of episode statistics
        """
        for episode in range(self.num_episodes):
            stats = self.env.run_episode(max_steps=self.max_steps, policy=self.policy)
            stats['episode'] = episode
            self.results.append(stats)
        
        return self.results
    
    def get_summary(self) -> Dict:
        """
        Get summary statistics across all episodes
        
        Returns:
            Summary dictionary
        """
        rewards = [r['total_reward'] for r in self.results]
        qualities = [r['final_quality'] for r in self.results]
        costs = [r['total_cost'] for r in self.results]
        steps = [r['num_steps'] for r in self.results]
        retrieves = [r['retrieve_count'] for r in self.results]
        reasons = [r['reason_count'] for r in self.results]
        
        summary = {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_quality': np.mean(qualities),
            'std_quality': np.std(qualities),
            'avg_cost': np.mean(costs),
            'std_cost': np.std(costs),
            'avg_steps': np.mean(steps),
            'avg_retrieves': np.mean(retrieves),
            'avg_reasons': np.mean(reasons),
            'termination_rate': np.mean([r['terminated'] for r in self.results])
        }
        
        return summary
