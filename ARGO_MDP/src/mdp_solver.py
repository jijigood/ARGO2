"""
MDP Solver for ARGO
Implements value iteration to solve the Bellman equation and compute optimal thresholds
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import os


class MDPSolver:
    """
    Solves the MDP using value iteration and computes optimal thresholds
    
    State: U ∈ [0, U_max] (information progress)
    Actions: {0: Retrieve, 1: Reason, 2: Terminate}
    """
    
    def __init__(self, config: Dict):
        """
        Initialize MDP solver with configuration
        
        Args:
            config: Dictionary containing MDP parameters
        """
        # MDP parameters
        self.U_max = config['mdp']['U_max']
        self.delta_r = config['mdp']['delta_r']
        self.delta_p = config['mdp']['delta_p']
        self.p_s = config['mdp']['p_s']
        self.c_r = config['mdp']['c_r']
        self.c_p = config['mdp']['c_p']
        self.mu = config['mdp']['mu']
        self.gamma = config['mdp']['gamma']
        self.grid_size = config['mdp']['U_grid_size']
        
        # Reward Shaping parameters (Phase 2.2)
        self.use_reward_shaping = config.get('reward_shaping', {}).get('enabled', False)
        self.shaping_k = config.get('reward_shaping', {}).get('k', 1.0)
        
        # Quality function
        self.quality_mode = config['quality']['mode']
        self.quality_k = config['quality']['k']
        
        # Solver parameters
        self.max_iterations = config['solver']['max_iterations']
        self.convergence_threshold = config['solver']['convergence_threshold']
        self.verbose = config['solver']['verbose']
        
        # Discretize state space
        self.U_grid = np.linspace(0, self.U_max, self.grid_size)
        
        # Initialize value function and Q-function
        self.V = np.zeros(self.grid_size)
        self.Q = np.zeros((self.grid_size, 3))  # 3 actions
        
        # Thresholds
        self.theta_cont = None
        self.theta_star = None
        
    def quality_function(self, U: float) -> float:
        """
        Quality function σ(U) based on information progress U
        Phase 2.3: Extended with sqrt and saturating modes
        
        Args:
            U: Information progress
            
        Returns:
            Quality score
        """
        x = U / self.U_max
        
        if self.quality_mode == "sigmoid":
            # Sigmoid: σ(x) = 1 / (1 + e^{-k(x - 0.5)})
            return 1.0 / (1.0 + np.exp(-self.quality_k * (x - 0.5)))
        
        elif self.quality_mode == "sqrt":
            # Phase 2.3: Square root: σ(x) = √x
            return np.sqrt(x)
        
        elif self.quality_mode == "saturating":
            # Phase 2.3: Saturating: σ(x) = 1 - e^{-αx}
            # α (alpha) is stored in quality_k
            alpha = self.quality_k
            return 1.0 - np.exp(-alpha * x)
        
        else:  # linear (default)
            # Linear: σ(x) = x
            return x
    
    def potential_function(self, U: float) -> float:
        """
        Potential function Φ(U) for reward shaping
        Phase 2.2: Φ(U) = k * U
        
        Args:
            U: Information progress
            
        Returns:
            Potential value
        """
        return self.shaping_k * U
    
    def shaping_reward(self, U: float, U_next: float) -> float:
        """
        Compute potential-based shaping reward
        Phase 2.2: F(U, U') = γ * Φ(U') - Φ(U)
        
        Args:
            U: Current state
            U_next: Next state
            
        Returns:
            Shaping reward
        """
        if not self.use_reward_shaping:
            return 0.0
        
        return self.gamma * self.potential_function(U_next) - self.potential_function(U)
    
    def get_state_index(self, U: float) -> int:
        """Get grid index closest to U"""
        idx = np.argmin(np.abs(self.U_grid - U))
        return idx
    
    def transition(self, U: float, action: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute next state distribution for given state and action
        
        Args:
            U: Current information progress
            action: 0=Retrieve, 1=Reason, 2=Terminate
            
        Returns:
            next_states: Array of next state values
            probs: Corresponding probabilities
        """
        if action == 0:  # Retrieve
            # Success: U' = min(U + delta_r, U_max)
            U_success = min(U + self.delta_r, self.U_max)
            # Failure: U' = U
            U_fail = U
            
            next_states = np.array([U_success, U_fail])
            probs = np.array([self.p_s, 1 - self.p_s])
            
        elif action == 1:  # Reason
            # Deterministic: U' = min(U + delta_p, U_max)
            U_next = min(U + self.delta_p, self.U_max)
            
            next_states = np.array([U_next])
            probs = np.array([1.0])
            
        else:  # Terminate
            # Absorbing state
            next_states = np.array([U])
            probs = np.array([1.0])
            
        return next_states, probs
    
    def reward(self, U: float, action: int, is_terminal: bool = False) -> float:
        """
        Compute immediate reward with cost weight μ
        
        Args:
            U: Current information progress
            action: 0=Retrieve, 1=Reason, 2=Terminate
            is_terminal: Whether this is terminal reward
            
        Returns:
            Reward value
        """
        if is_terminal and action == 2:
            # Terminal reward: Q(O)
            # Cost is already penalized in step rewards with μ weight
            quality = self.quality_function(U)
            return quality
        else:
            # Step cost weighted by μ
            # Higher μ means more penalty for cost
            if action == 0:
                return -self.mu * self.c_r
            elif action == 1:
                return -self.mu * self.c_p
            else:
                return 0.0
    
    def value_iteration(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform value iteration to solve Bellman equation
        Phase 2.2: Includes optional reward shaping
        
        Returns:
            V: Optimal value function
            Q: Optimal Q-function
        """
        if self.verbose:
            print("Starting Value Iteration...")
            print(f"Grid size: {self.grid_size}, Gamma: {self.gamma}")
            if self.use_reward_shaping:
                print(f"Reward Shaping: ENABLED (k={self.shaping_k})")
            else:
                print(f"Reward Shaping: DISABLED")
        
        for iteration in range(self.max_iterations):
            V_old = self.V.copy()
            
            # Update each state
            for i, U in enumerate(self.U_grid):
                # Compute Q-values for each action
                for action in range(3):
                    if action == 2:  # Terminate
                        # Terminal reward without cost accumulation in this formulation
                        self.Q[i, action] = self.quality_function(U)
                    else:
                        # Expected value: R(U,a) + F(U,U') + γ * E[V(U')]
                        # Phase 2.2: Add shaping reward F(U,U')
                        next_states, probs = self.transition(U, action)
                        immediate_reward = self.reward(U, action)
                        
                        expected_value = 0.0
                        expected_shaping = 0.0
                        
                        for U_next, prob in zip(next_states, probs):
                            idx_next = self.get_state_index(U_next)
                            expected_value += prob * V_old[idx_next]
                            
                            # Phase 2.2: Add shaping reward
                            if self.use_reward_shaping:
                                expected_shaping += prob * self.shaping_reward(U, U_next)
                        
                        self.Q[i, action] = immediate_reward + expected_shaping + self.gamma * expected_value
                
                # Value is max over actions
                self.V[i] = np.max(self.Q[i, :])
            
            # Check convergence
            max_diff = np.max(np.abs(self.V - V_old))
            if self.verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: Max diff = {max_diff:.6e}")
            
            if max_diff < self.convergence_threshold:
                if self.verbose:
                    print(f"Converged after {iteration} iterations")
                break
        
        return self.V, self.Q
    
    def compute_thresholds(self) -> Tuple[float, float]:
        """
        Compute optimal thresholds from Q-function
        
        Returns:
            theta_cont: Continuation threshold (Retrieve vs Reason)
            theta_star: Termination threshold
        """
        # Find termination threshold: θ* where Q(U,2) becomes optimal
        theta_star = None
        for i in range(self.grid_size - 1, -1, -1):
            if self.Q[i, 2] >= max(self.Q[i, 0], self.Q[i, 1]):
                theta_star = self.U_grid[i]
                break
        
        if theta_star is None:
            theta_star = self.U_max
        
        # Find continuation threshold: θ_cont where Q(U,0) = Q(U,1) for U < θ*
        theta_cont = 0.0
        theta_star_idx = self.get_state_index(theta_star)
        
        for i in range(theta_star_idx):
            if self.Q[i, 0] < self.Q[i, 1]:
                theta_cont = self.U_grid[i]
                break
        
        self.theta_cont = theta_cont
        self.theta_star = theta_star
        
        if self.verbose:
            print(f"\nOptimal Thresholds:")
            print(f"  θ_cont = {theta_cont:.4f}")
            print(f"  θ_star = {theta_star:.4f}")
        
        return theta_cont, theta_star
    
    def solve(self) -> Dict:
        """
        Complete solve: value iteration + threshold computation
        
        Returns:
            Dictionary with results
        """
        V, Q = self.value_iteration()
        theta_cont, theta_star = self.compute_thresholds()
        
        results = {
            'V': V,
            'Q': Q,
            'theta_cont': theta_cont,
            'theta_star': theta_star,
            'U_grid': self.U_grid
        }
        
        return results
    
    def save_results(self, results: Dict, output_dir: str = "results"):
        """
        Save solver results to CSV
        
        Args:
            results: Dictionary from solve()
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save value function and Q-function
        df = pd.DataFrame({
            'U': results['U_grid'],
            'V': results['V'],
            'Q_retrieve': results['Q'][:, 0],
            'Q_reason': results['Q'][:, 1],
            'Q_terminate': results['Q'][:, 2]
        })
        df.to_csv(os.path.join(output_dir, 'value_function.csv'), index=False)
        
        # Save thresholds
        with open(os.path.join(output_dir, 'thresholds.txt'), 'w') as f:
            f.write(f"theta_cont: {results['theta_cont']:.6f}\n")
            f.write(f"theta_star: {results['theta_star']:.6f}\n")
        
        if self.verbose:
            print(f"\nResults saved to {output_dir}/")
