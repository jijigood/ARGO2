"""
Quick Test Script - Verify ARGO Installation and Basic Functionality
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
import numpy as np

def test_mdp_solver():
    """Test MDP solver"""
    print("=" * 60)
    print("Testing MDP Solver...")
    print("=" * 60)
    
    from src.mdp_solver import MDPSolver
    
    # Simple config
    config = {
        'mdp': {
            'U_max': 1.0,
            'delta_r': 0.15,
            'delta_p': 0.08,
            'p_s': 0.7,
            'c_r': 0.2,
            'c_p': 0.1,
            'mu': 0.6,
            'gamma': 1.0,
            'U_grid_size': 50
        },
        'quality': {
            'mode': 'sigmoid',
            'k': 5.0
        },
        'solver': {
            'max_iterations': 1000,
            'convergence_threshold': 1e-6,
            'verbose': True
        }
    }
    
    solver = MDPSolver(config)
    results = solver.solve()
    
    print(f"\n✓ MDP Solver Test Passed")
    print(f"  θ_cont = {results['theta_cont']:.4f}")
    print(f"  θ_star = {results['theta_star']:.4f}")
    print(f"  V*(0) = {results['V'][0]:.4f}")
    print(f"  V*(U_max) = {results['V'][-1]:.4f}")
    
    return results


def test_environment():
    """Test ARGO environment"""
    print("\n" + "=" * 60)
    print("Testing ARGO Environment...")
    print("=" * 60)
    
    from src.env_argo import ARGOEnv
    from src.policy import ThresholdPolicy
    
    mdp_config = {
        'U_max': 1.0,
        'delta_r': 0.15,
        'delta_p': 0.08,
        'p_s': 0.7,
        'c_r': 0.2,
        'c_p': 0.1,
        'mu': 0.6,
        'gamma': 1.0
    }
    
    # Create environment
    env = ARGOEnv(mdp_config=mdp_config, seed=42)
    
    # Create simple policy
    policy = ThresholdPolicy(theta_cont=0.3, theta_star=0.7)
    
    # Run one episode
    U = env.reset()
    total_reward = 0
    
    print("\nRunning sample episode:")
    for step in range(20):
        action = policy.act(U)
        action_names = ['Retrieve', 'Reason', 'Terminate']
        
        U_next, reward, done, info = env.step(action)
        total_reward += reward
        
        print(f"  Step {step}: U={U:.3f} → {action_names[action]} → U'={U_next:.3f}, R={reward:.3f}")
        
        U = U_next
        if done:
            print(f"  Episode terminated at step {step}")
            break
    
    print(f"\n✓ Environment Test Passed")
    print(f"  Total Reward: {total_reward:.4f}")
    print(f"  Final U: {U:.4f}")
    

def test_policies():
    """Test different policies"""
    print("\n" + "=" * 60)
    print("Testing Policies...")
    print("=" * 60)
    
    from src.policy import (
        ThresholdPolicy, AlwaysRetrievePolicy, AlwaysReasonPolicy,
        FixedKRetrieveThenReasonPolicy, RandomPolicy
    )
    
    U_test = [0.0, 0.3, 0.5, 0.7, 0.9]
    
    # Threshold Policy
    policy1 = ThresholdPolicy(theta_cont=0.4, theta_star=0.8)
    print(f"\n{policy1}")
    for U in U_test:
        action = policy1.act(U)
        print(f"  U={U:.1f} → {['Retrieve', 'Reason', 'Terminate'][action]}")
    
    # Always Retrieve
    policy2 = AlwaysRetrievePolicy(U_max=1.0)
    print(f"\n{policy2}")
    for U in [0.0, 0.5, 0.99, 1.0]:
        action = policy2.act(U)
        print(f"  U={U:.2f} → {['Retrieve', 'Reason', 'Terminate'][action]}")
    
    # Fixed K
    policy3 = FixedKRetrieveThenReasonPolicy(K=3, U_max=1.0)
    print(f"\n{policy3}")
    for i in range(5):
        action = policy3.act(0.5)
        print(f"  Call {i+1} → {['Retrieve', 'Reason', 'Terminate'][action]}")
    
    print(f"\n✓ Policy Test Passed")


def test_full_experiment():
    """Test full experiment flow"""
    print("\n" + "=" * 60)
    print("Testing Full Experiment Flow...")
    print("=" * 60)
    
    from src.mdp_solver import MDPSolver
    from src.env_argo import ARGOEnv, MultiEpisodeRunner
    from src.policy import ThresholdPolicy
    
    # Config
    config = {
        'mdp': {
            'U_max': 1.0,
            'delta_r': 0.15,
            'delta_p': 0.08,
            'p_s': 0.7,
            'c_r': 0.2,
            'c_p': 0.1,
            'mu': 0.6,
            'gamma': 1.0,
            'U_grid_size': 50
        },
        'quality': {'mode': 'sigmoid', 'k': 5.0},
        'solver': {
            'max_iterations': 1000,
            'convergence_threshold': 1e-6,
            'verbose': False
        }
    }
    
    # Solve MDP
    print("\n1. Solving MDP...")
    solver = MDPSolver(config)
    results = solver.solve()
    print(f"   Thresholds: θ_cont={results['theta_cont']:.4f}, θ_star={results['theta_star']:.4f}")
    
    # Create policy
    policy = ThresholdPolicy(results['theta_cont'], results['theta_star'])
    
    # Run episodes
    print("\n2. Running episodes...")
    env = ARGOEnv(mdp_config=config['mdp'], seed=42)
    runner = MultiEpisodeRunner(env, policy, num_episodes=10, max_steps=30)
    runner.run()
    summary = runner.get_summary()
    
    print(f"   Avg Reward: {summary['avg_reward']:.4f}")
    print(f"   Avg Quality: {summary['avg_quality']:.4f}")
    print(f"   Avg Cost: {summary['avg_cost']:.4f}")
    print(f"   Avg Steps: {summary['avg_steps']:.2f}")
    
    print(f"\n✓ Full Experiment Test Passed")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("ARGO MDP - Quick Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: MDP Solver
        test_mdp_solver()
        
        # Test 2: Environment
        test_environment()
        
        # Test 3: Policies
        test_policies()
        
        # Test 4: Full experiment
        test_full_experiment()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print("\nYou can now run the full experiment:")
        print("  python scripts/run_single.py --config configs/base.yaml\n")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
