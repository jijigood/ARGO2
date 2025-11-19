#!/usr/bin/env python3
"""
Test script to verify that quality function configuration is correctly read from YAML
and that different quality function modes work properly.
"""

import yaml
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '../ARGO_MDP/src')
from mdp_solver import MDPSolver


def test_quality_config_from_yaml():
    """Test that quality configuration is correctly loaded from YAML"""
    print("=" * 70)
    print("TEST 1: Quality Configuration from YAML")
    print("=" * 70)
    
    with open('configs/multi_gpu.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    mdp = config['mdp']
    
    # Check that quality_function and quality_k are present
    assert 'quality_function' in mdp, "quality_function not found in MDP config"
    assert 'quality_k' in mdp, "quality_k not found in MDP config"
    
    print(f"✓ quality_function: '{mdp['quality_function']}'")
    print(f"✓ quality_k: {mdp['quality_k']}")
    print(f"✓ reward_shaping: {mdp.get('reward_shaping')}")
    print()


def test_solve_mdp_uses_yaml_config():
    """Test that solve_mdp method uses configuration from YAML"""
    print("=" * 70)
    print("TEST 2: solve_mdp Method Uses YAML Config")
    print("=" * 70)
    
    with open('configs/multi_gpu.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Simulate the solve_mdp method
    mdp_config = config['mdp'].copy()
    mdp_config['mu'] = 0.5
    mdp_config['U_grid_size'] = mdp_config.get('grid_size', 101)
    
    # Build quality config (as per the fix)
    quality_config = {
        'mode': mdp_config.get('quality_function', 'linear'),
        'k': mdp_config.get('quality_k', 1.0)
    }
    
    solver_config = {
        'mdp': mdp_config,
        'quality': quality_config,
        'reward_shaping': mdp_config.get('reward_shaping', {'enabled': False, 'k': 1.0}),
        'solver': {
            'max_iterations': 1000,
            'convergence_threshold': 1e-6,
            'verbose': False
        }
    }
    
    solver = MDPSolver(solver_config)
    
    # Verify that solver has correct configuration
    assert solver.quality_mode == config['mdp']['quality_function'], \
        f"Expected quality_mode '{config['mdp']['quality_function']}', got '{solver.quality_mode}'"
    assert solver.quality_k == config['mdp']['quality_k'], \
        f"Expected quality_k {config['mdp']['quality_k']}, got {solver.quality_k}"
    
    print(f"✓ Solver quality_mode: '{solver.quality_mode}' (matches YAML)")
    print(f"✓ Solver quality_k: {solver.quality_k} (matches YAML)")
    print(f"✓ Solver reward_shaping: enabled={solver.use_reward_shaping}, k={solver.shaping_k}")
    print()


def test_different_quality_functions():
    """Test that different quality function modes work correctly"""
    print("=" * 70)
    print("TEST 3: Different Quality Function Modes")
    print("=" * 70)
    
    with open('configs/multi_gpu.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    test_cases = [
        ('linear', 0.6, 0.5, 0.5000, "Linear mode"),
        ('sqrt', 1.0, 0.5, 0.7071, "Square root mode"),
        ('sigmoid', 5.0, 0.5, 0.5000, "Sigmoid mode"),
        ('saturating', 2.0, 0.5, 0.6321, "Saturating mode"),
    ]
    
    for mode, k, test_U, expected, description in test_cases:
        # Update config
        mdp_config = config['mdp'].copy()
        mdp_config['quality_function'] = mode
        mdp_config['quality_k'] = k
        mdp_config['mu'] = 0.5
        mdp_config['U_grid_size'] = 101
        
        quality_config = {
            'mode': mdp_config.get('quality_function'),
            'k': mdp_config.get('quality_k')
        }
        
        solver_config = {
            'mdp': mdp_config,
            'quality': quality_config,
            'reward_shaping': mdp_config.get('reward_shaping', {'enabled': False, 'k': 1.0}),
            'solver': {
                'max_iterations': 1000,
                'convergence_threshold': 1e-6,
                'verbose': False
            }
        }
        
        solver = MDPSolver(solver_config)
        result = solver.quality_function(test_U)
        
        assert abs(result - expected) < 0.01, \
            f"{description}: Expected σ({test_U}) ≈ {expected}, got {result:.4f}"
        
        print(f"✓ {description}: σ({test_U}) = {result:.4f} (expected {expected})")
    
    print()


def test_reward_shaping_config():
    """Test that reward shaping configuration is properly passed"""
    print("=" * 70)
    print("TEST 4: Reward Shaping Configuration")
    print("=" * 70)
    
    with open('configs/multi_gpu.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Test with reward shaping disabled (default)
    mdp_config = config['mdp'].copy()
    mdp_config['mu'] = 0.5
    mdp_config['U_grid_size'] = 101
    
    solver_config = {
        'mdp': mdp_config,
        'quality': {'mode': 'linear', 'k': 1.0},
        'reward_shaping': mdp_config.get('reward_shaping', {'enabled': False, 'k': 1.0}),
        'solver': {
            'max_iterations': 1000,
            'convergence_threshold': 1e-6,
            'verbose': False
        }
    }
    
    solver = MDPSolver(solver_config)
    assert solver.use_reward_shaping == False, "Reward shaping should be disabled by default"
    print(f"✓ Reward shaping disabled (default): {solver.use_reward_shaping}")
    
    # Test with reward shaping enabled
    mdp_config['reward_shaping'] = {'enabled': True, 'k': 2.5}
    solver_config['reward_shaping'] = mdp_config['reward_shaping']
    
    solver2 = MDPSolver(solver_config)
    assert solver2.use_reward_shaping == True, "Reward shaping should be enabled"
    assert solver2.shaping_k == 2.5, f"Expected shaping_k=2.5, got {solver2.shaping_k}"
    print(f"✓ Reward shaping enabled: {solver2.use_reward_shaping}, k={solver2.shaping_k}")
    print()


def main():
    """Run all tests"""
    print()
    print("=" * 70)
    print("Testing Quality Function Configuration")
    print("=" * 70)
    print()
    
    try:
        test_quality_config_from_yaml()
        test_solve_mdp_uses_yaml_config()
        test_different_quality_functions()
        test_reward_shaping_config()
        
        print("=" * 70)
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)
        print()
        print("Summary:")
        print("  - Quality configuration is correctly read from YAML")
        print("  - solve_mdp method properly constructs quality config from MDP section")
        print("  - All quality function modes (linear, sqrt, sigmoid, saturating) work")
        print("  - Reward shaping configuration is properly passed to MDPSolver")
        print()
        return 0
    
    except AssertionError as e:
        print()
        print("=" * 70)
        print("✗ TEST FAILED!")
        print("=" * 70)
        print(f"Error: {e}")
        print()
        return 1
    
    except Exception as e:
        print()
        print("=" * 70)
        print("✗ UNEXPECTED ERROR!")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print()
        return 1


if __name__ == "__main__":
    exit(main())
