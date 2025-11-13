"""
Policy Module
Implements ThresholdPolicy and baseline policies
"""
import numpy as np
import random
from typing import Optional


class ThresholdPolicy:
    """
    Optimal threshold-based policy for ARGO
    
    Policy structure:
    - If U >= θ_star: Terminate
    - If U < θ_cont: Retrieve
    - Otherwise: Reason
    """
    
    def __init__(self, theta_cont: float, theta_star: float):
        """
        Initialize threshold policy
        
        Args:
            theta_cont: Continuation threshold (Retrieve vs Reason)
            theta_star: Termination threshold
        """
        self.theta_cont = theta_cont
        self.theta_star = theta_star
        
        # Validate thresholds
        if theta_cont > theta_star:
            raise ValueError(f"theta_cont ({theta_cont}) must be <= theta_star ({theta_star})")
    
    def act(self, U: float) -> int:
        """
        Select action based on current state
        
        Args:
            U: Current information progress
            
        Returns:
            action: 0=Retrieve, 1=Reason, 2=Terminate
        """
        if U >= self.theta_star:
            return 2  # Terminate
        elif U < self.theta_cont:
            return 0  # Retrieve
        else:
            return 1  # Reason
    
    def __repr__(self):
        return f"ThresholdPolicy(θ_cont={self.theta_cont:.4f}, θ_star={self.theta_star:.4f})"


class AlwaysRetrievePolicy:
    """
    Baseline: Always retrieve until U_max, then terminate
    """
    
    def __init__(self, U_max: float = 1.0):
        self.U_max = U_max
    
    def act(self, U: float) -> int:
        if U >= self.U_max - 1e-6:
            return 2  # Terminate
        return 0  # Retrieve
    
    def __repr__(self):
        return "AlwaysRetrievePolicy"


class AlwaysReasonPolicy:
    """
    Baseline: Always reason until U_max, then terminate
    """
    
    def __init__(self, U_max: float = 1.0):
        self.U_max = U_max
    
    def act(self, U: float) -> int:
        if U >= self.U_max - 1e-6:
            return 2  # Terminate
        return 1  # Reason
    
    def __repr__(self):
        return "AlwaysReasonPolicy"


class FixedKRetrieveThenReasonPolicy:
    """
    Baseline: Retrieve K times, then always reason, terminate at U_max
    """
    
    def __init__(self, K: int, U_max: float = 1.0):
        self.K = K
        self.U_max = U_max
        self.retrieve_count = 0
    
    def act(self, U: float) -> int:
        if U >= self.U_max - 1e-6:
            return 2  # Terminate
        
        if self.retrieve_count < self.K:
            self.retrieve_count += 1
            return 0  # Retrieve
        else:
            return 1  # Reason
    
    def reset(self):
        """Reset counter for new episode"""
        self.retrieve_count = 0
    
    def __repr__(self):
        return f"FixedK{self.K}RetrieveThenReasonPolicy"


class RandomPolicy:
    """
    Baseline: Random action selection
    Terminate with probability p_terminate when U > threshold
    """
    
    def __init__(self, U_max: float = 1.0, p_terminate: float = 0.1, seed: Optional[int] = None):
        self.U_max = U_max
        self.p_terminate = p_terminate
        if seed is not None:
            random.seed(seed)
    
    def act(self, U: float) -> int:
        # Force terminate at U_max
        if U >= self.U_max - 1e-6:
            return 2
        
        # Random termination with probability
        if U > 0.5 * self.U_max and random.random() < self.p_terminate:
            return 2
        
        # Random choice between Retrieve and Reason
        return random.choice([0, 1])
    
    def __repr__(self):
        return f"RandomPolicy(p_term={self.p_terminate})"


class GreedyPolicy:
    """
    Greedy policy based on immediate Q-values
    Uses the Q-function from MDP solver
    """
    
    def __init__(self, Q: np.ndarray, U_grid: np.ndarray):
        """
        Initialize greedy policy
        
        Args:
            Q: Q-function array [grid_size, 3]
            U_grid: State grid
        """
        self.Q = Q
        self.U_grid = U_grid
    
    def get_state_index(self, U: float) -> int:
        """Get closest grid index"""
        idx = np.argmin(np.abs(self.U_grid - U))
        return idx
    
    def act(self, U: float) -> int:
        """Select action with highest Q-value"""
        idx = self.get_state_index(U)
        action = np.argmax(self.Q[idx, :])
        return action
    
    def __repr__(self):
        return "GreedyPolicy"


class SingleThresholdPolicy:
    """
    Ablation: Single threshold policy (no distinction between Retrieve and Reason)
    - If U >= θ: Terminate
    - Otherwise: Retrieve
    """
    
    def __init__(self, theta: float):
        self.theta = theta
    
    def act(self, U: float) -> int:
        if U >= self.theta:
            return 2  # Terminate
        return 0  # Always Retrieve
    
    def __repr__(self):
        return f"SingleThresholdPolicy(θ={self.theta:.4f})"


class NoRewardShapingPolicy:
    """
    Ablation: Policy trained without reward shaping
    Uses uniform costs instead of differentiated c_r and c_p
    """
    
    def __init__(self, theta_star: float, U_max: float = 1.0):
        self.theta_star = theta_star
        self.U_max = U_max
    
    def act(self, U: float) -> int:
        if U >= self.theta_star:
            return 2
        # Random between Retrieve and Reason (no preference)
        return random.choice([0, 1])
    
    def __repr__(self):
        return f"NoRewardShapingPolicy(θ*={self.theta_star:.4f})"
