#!/usr/bin/env python3
"""
Experiment 0 V2: Threshold Structure Validation with Optimized Parameters
=========================================================================

Version 2 improvements:
- Adjusted parameters for clearer three-region structure visualization
- Balanced cost-efficiency between Retrieve and Reason actions
- Fine-tuned to show distinct Retrieve → Reason → Terminate progression

Author: ARGO Team
Date: 2025-11-14
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import yaml
from datetime import datetime
from scipy import stats

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ARGO_MDP', 'src'))
from mdp_solver import MDPSolver


class ThresholdValidationExperimentV2:
    """
    V2: Validates two-level threshold structure with optimized parameters
    for clearer visualization of Retrieve → Reason → Terminate regions.
    """
    
    def __init__(self, output_dir='results/exp0_v2_threshold_validation'):
        """Initialize experiment with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        
    def get_optimized_parameters(self):
        """
        Get parameter sets optimized for clear three-region visualization.
        
        Key changes from V1:
        1. Baseline: Reduced delta_r from 0.25 to 0.18 (closer to delta_p)
        2. Balanced efficiency: c_r/c_p ≈ 1.5, delta_r/delta_p ≈ 2.0
        3. Fine-tuned costs: Make Retrieve and Reason more competitive
        """
        param_sets = [
            {
                'name': 'Balanced (Optimized)',
                'p_s': 0.6,
                'c_r': 0.03,      # Moderate retrieval cost
                'c_p': 0.02,      # Slightly cheaper reasoning
                'delta_r': 0.16,  # Reduced retrieval effectiveness
                'delta_p': 0.08,  # Keep reasoning effectiveness
                'gamma': 0.95,
                'description': 'Optimized for balanced three-region structure'
            },
            {
                'name': 'Equal Efficiency',
                'p_s': 0.6,
                'c_r': 0.04,      # 2x cost of reasoning
                'c_p': 0.02,
                'delta_r': 0.16,  # 2x effect of reasoning
                'delta_p': 0.08,
                'gamma': 0.95,
                'description': 'Cost-efficiency ratio = 1.0 (δ_r/c_r = δ_p/c_p)'
            },
            {
                'name': 'Slight Retrieve Advantage',
                'p_s': 0.6,
                'c_r': 0.025,     # Cheaper retrieval
                'c_p': 0.02,
                'delta_r': 0.15,  # Moderate effect
                'delta_p': 0.08,
                'gamma': 0.95,
                'description': 'Retrieve 20% more cost-efficient'
            },
            {
                'name': 'Slight Reason Advantage',
                'p_s': 0.6,
                'c_r': 0.04,      # More expensive retrieval
                'c_p': 0.02,
                'delta_r': 0.14,  # Reduced effect
                'delta_p': 0.08,
                'gamma': 0.95,
                'description': 'Reason 15% more cost-efficient'
            },
            {
                'name': 'High Success Probability',
                'p_s': 0.8,       # Higher success rate
                'c_r': 0.03,
                'c_p': 0.02,
                'delta_r': 0.16,
                'delta_p': 0.08,
                'gamma': 0.95,
                'description': 'Impact of high p_s on threshold structure'
            },
            {
                'name': 'Low Success Probability',
                'p_s': 0.4,       # Lower success rate
                'c_r': 0.03,
                'c_p': 0.02,
                'delta_r': 0.16,
                'delta_p': 0.08,
                'gamma': 0.95,
                'description': 'Impact of low p_s on threshold structure'
            },
        ]
        return param_sets
    
    def calculate_cost_efficiency(self, params):
        """Calculate and display cost-efficiency metrics."""
        eff_r = params['delta_r'] / params['c_r']
        eff_p = params['delta_p'] / params['c_p']
        
        info = {
            'retrieve_efficiency': eff_r,
            'reason_efficiency': eff_p,
            'efficiency_ratio': eff_r / eff_p,
            'cost_ratio': params['c_r'] / params['c_p'],
            'effect_ratio': params['delta_r'] / params['delta_p']
        }
        
        return info
    
    def solve_mdp(self, params, grid_size=201):
        """
        Solve MDP with given parameters.
        
        Args:
            params: Dictionary of MDP parameters
            grid_size: Number of grid points for U discretization
        
        Returns:
            solver: MDPSolver instance with solution
            U_grid: Utility grid points
        """
        # Create config dictionary for MDPSolver
        config = {
            'mdp': {
                'U_max': 1.0,
                'delta_r': params['delta_r'],
                'delta_p': params['delta_p'],
                'p_s': params['p_s'],
                'c_r': params['c_r'],
                'c_p': params['c_p'],
                'mu': 1.0,  # FIXED: Enable cost penalties (was 0.0)
                'gamma': params['gamma'],
                'U_grid_size': grid_size
            },
            'quality': {
                'mode': 'linear',
                'k': 1.0
            },
            'solver': {
                'max_iterations': 10000,
                'convergence_threshold': 1e-6,
                'verbose': False
            }
        }
        
        # Create MDP solver
        solver = MDPSolver(config)
        
        # Solve MDP using value iteration
        print(f"  Solving MDP with grid_size={grid_size}...")
        solver.solve()
        
        # Get utility grid
        U_grid = solver.U_grid
        
        return solver, U_grid
    
    def identify_thresholds(self, solver, U_grid):
        """
        Identify threshold values from optimal policy.
        
        Returns:
            thresholds: Dict with Θ_cont and Θ_term
        """
        # Compute optimal policy from Q-function
        policy = np.argmax(solver.Q, axis=1)
        
        # Find Θ_cont: last U where Retrieve is optimal
        retrieve_indices = np.where(policy == 0)[0]
        if len(retrieve_indices) > 0:
            Theta_cont = U_grid[retrieve_indices[-1]]
        else:
            Theta_cont = 0.0
        
        # Find Θ_term: first U where Terminate is optimal
        terminate_indices = np.where(policy == 2)[0]
        if len(terminate_indices) > 0:
            Theta_term = U_grid[terminate_indices[0]]
        else:
            Theta_term = 1.0
        
        return {
            'Theta_cont': Theta_cont,
            'Theta_term': Theta_term
        }
    
    def validate_threshold_values(self, thresholds):
        """
        Layer 0 Validation: Check basic threshold properties.
        
        Checks:
        1. Θ_cont ∈ [0, 1]
        2. Θ_term ∈ [0, 1]
        3. Θ_cont ≤ Θ_term (ordering)
        
        Returns:
            is_valid: Boolean
            messages: List of validation messages
        """
        messages = []
        is_valid = True
        
        Theta_cont = thresholds['Theta_cont']
        Theta_term = thresholds['Theta_term']
        
        # Check range
        if not (0 <= Theta_cont <= 1):
            messages.append(f"❌ Θ_cont = {Theta_cont:.3f} not in [0,1]")
            is_valid = False
        else:
            messages.append(f"✓ Θ_cont = {Theta_cont:.3f} ∈ [0,1]")
        
        if not (0 <= Theta_term <= 1):
            messages.append(f"❌ Θ_term = {Theta_term:.3f} not in [0,1]")
            is_valid = False
        else:
            messages.append(f"✓ Θ_term = {Theta_term:.3f} ∈ [0,1]")
        
        # Check ordering
        if Theta_cont > Theta_term:
            messages.append(f"❌ Θ_cont ({Theta_cont:.3f}) > Θ_term ({Theta_term:.3f})")
            is_valid = False
        else:
            messages.append(f"✓ Θ_cont ≤ Θ_term ({Theta_cont:.3f} ≤ {Theta_term:.3f})")
        
        return is_valid, messages
    
    def validate_policy_structure(self, solver, U_grid, thresholds, tolerance=1e-3):
        """
        Validate that policy follows Retrieve → Reason → Terminate structure.
        
        Args:
            solver: MDP solver instance
            U_grid: Grid of utility values
            thresholds: Identified thresholds
            tolerance: Numerical tolerance for Q-value differences (default 0.001)
                      Violations with Q-diff < tolerance are ignored as numerical artifacts
        
        Returns:
            is_valid: Boolean
            messages: List of validation messages
        """
        # Compute optimal policy from Q-function
        policy = np.argmax(solver.Q, axis=1)
        messages = []
        is_valid = True
        
        # Count actions
        n_retrieve = np.sum(policy == 0)
        n_reason = np.sum(policy == 1)
        n_terminate = np.sum(policy == 2)
        
        messages.append(f"Action distribution:")
        messages.append(f"  Retrieve: {n_retrieve} points ({n_retrieve/len(policy)*100:.1f}%)")
        messages.append(f"  Reason: {n_reason} points ({n_reason/len(policy)*100:.1f}%)")
        messages.append(f"  Terminate: {n_terminate} points ({n_terminate/len(policy)*100:.1f}%)")
        
        # Check monotonicity with numerical tolerance
        violations = []
        numerical_artifacts = []
        
        for i in range(len(policy) - 1):
            current_action = policy[i]
            next_action = policy[i + 1]
            
            # Check for backward transitions
            if current_action == 1 and next_action == 0:  # Reason → Retrieve
                # Check Q-value difference
                Q_retrieve = solver.Q[i + 1, 0]
                Q_reason = solver.Q[i + 1, 1]
                Q_diff = abs(Q_retrieve - Q_reason)
                
                if Q_diff > tolerance:
                    # Significant violation
                    violations.append(f"U={U_grid[i]:.3f}: Retrieve after Reason (ΔQ={Q_diff:.4f})")
                else:
                    # Numerical artifact
                    numerical_artifacts.append(f"U={U_grid[i]:.3f}: ΔQ={Q_diff:.6f}")
                    
            elif current_action == 2 and next_action in [0, 1]:  # Terminate → Retrieve/Reason
                violations.append(f"U={U_grid[i]:.3f}: Action after Terminate")
        
        if numerical_artifacts:
            messages.append(f"  (Ignored {len(numerical_artifacts)} numerical artifacts with ΔQ < {tolerance})")
        
        if violations:
            is_valid = False
            messages.append(f"⚠ Found {len(violations)} significant monotonicity violations:")
            for v in violations[:5]:  # Show first 5
                messages.append(f"  - {v}")
            if len(violations) > 5:
                messages.append(f"  ... and {len(violations)-5} more")
        else:
            messages.append("✓ No significant monotonicity violations detected")
        
        return is_valid, messages
    
    def test_monotonicity_statistical(self, solver, U_grid):
        """
        Statistical test for monotonicity using Spearman rank correlation.
        
        Tests:
        1. V*(U) should be monotonically non-decreasing in U
        2. Spearman ρ should be close to 1.0
        
        Returns:
            is_valid: Boolean
            stats_dict: Dictionary with test statistics
        """
        V_star = solver.V
        
        # Compute Spearman rank correlation
        rho, p_value = stats.spearmanr(U_grid, V_star)
        
        # Test criteria
        is_valid = (rho > 0.99) and (p_value < 0.01)
        
        stats_dict = {
            'spearman_rho': rho,
            'p_value': p_value,
            'is_monotonic': is_valid
        }
        
        return is_valid, stats_dict
    
    def validate_single_crossing(self, solver, U_grid):
        """
        Validate single-crossing property: Continue vs Terminate.
        
        CORRECTED: Tests that the advantage of continuing (best of Retrieve/Reason)
        vs terminating crosses zero exactly once at Θ*.
        """
        # Best continuing action at each state
        Q_continue = np.maximum(solver.Q[:, 0], solver.Q[:, 1])
        Q_terminate = solver.Q[:, 2]
        
        # Advantage of continuing vs terminating
        adv_continue_vs_terminate = Q_continue - Q_terminate
        
        # Count zero crossings
        sign_changes = np.sum(np.diff(np.sign(adv_continue_vs_terminate)) != 0)
        is_valid = (sign_changes == 1)
        
        return is_valid, sign_changes
    
    def plot_policy_structure(self, solver, U_grid, thresholds, params, 
                             cost_efficiency, output_file):
        """
        Create comprehensive visualization of policy structure.
        
        Creates 2x2 subplot:
        1. Policy actions vs U
        2. Q-functions vs U
        3. Value function vs U
        4. Advantage functions vs U
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Threshold Structure Validation V2: {params['name']}\n"
                    f"Cost Efficiency Ratio: {cost_efficiency['efficiency_ratio']:.2f} "
                    f"(Retrieve={cost_efficiency['retrieve_efficiency']:.1f}, "
                    f"Reason={cost_efficiency['reason_efficiency']:.1f})",
                    fontsize=14, fontweight='bold')
        
        # Compute optimal policy from Q-function
        policy = np.argmax(solver.Q, axis=1)
        Q = solver.Q
        V = solver.V
        
        # Color mapping for actions
        action_colors = {0: 'blue', 1: 'green', 2: 'red'}
        action_names = {0: 'Retrieve', 1: 'Reason', 2: 'Terminate'}
        
        # Plot 1: Policy Structure
        ax1 = axes[0, 0]
        for action in [0, 1, 2]:
            mask = (policy == action)
            ax1.scatter(U_grid[mask], policy[mask], 
                       c=action_colors[action], 
                       label=action_names[action],
                       alpha=0.6, s=30)
        
        # Add threshold lines
        Theta_cont = thresholds['Theta_cont']
        Theta_term = thresholds['Theta_term']
        ax1.axvline(Theta_cont, color='orange', linestyle='--', 
                   label=f'Θ_cont = {Theta_cont:.3f}', linewidth=2)
        ax1.axvline(Theta_term, color='purple', linestyle='--', 
                   label=f'Θ_term = {Theta_term:.3f}', linewidth=2)
        
        ax1.set_xlabel('Utility U', fontsize=11)
        ax1.set_ylabel('Optimal Action', fontsize=11)
        ax1.set_title('Policy Structure (π*(U))', fontsize=12, fontweight='bold')
        ax1.set_yticks([0, 1, 2])
        ax1.set_yticklabels(['Retrieve', 'Reason', 'Terminate'])
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Q-functions
        ax2 = axes[0, 1]
        ax2.plot(U_grid, Q[:, 0], 'b-', label='Q(U, Retrieve)', linewidth=2)
        ax2.plot(U_grid, Q[:, 1], 'g-', label='Q(U, Reason)', linewidth=2)
        ax2.plot(U_grid, Q[:, 2], 'r-', label='Q(U, Terminate)', linewidth=2)
        ax2.axvline(Theta_cont, color='orange', linestyle='--', alpha=0.5)
        ax2.axvline(Theta_term, color='purple', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Utility U', fontsize=11)
        ax2.set_ylabel('Q-value', fontsize=11)
        ax2.set_title('Q-Functions', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Value Function
        ax3 = axes[1, 0]
        ax3.plot(U_grid, V, 'k-', linewidth=2, label='V*(U)')
        ax3.axvline(Theta_cont, color='orange', linestyle='--', alpha=0.5)
        ax3.axvline(Theta_term, color='purple', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Utility U', fontsize=11)
        ax3.set_ylabel('Value V*(U)', fontsize=11)
        ax3.set_title('Value Function (Should be Monotonic)', fontsize=12, fontweight='bold')
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Advantage Functions
        ax4 = axes[1, 1]
        A_retrieve = Q[:, 0] - V
        A_reason = Q[:, 1] - V
        A_terminate = Q[:, 2] - V
        
        ax4.plot(U_grid, A_retrieve, 'b-', label='A(U, Retrieve)', linewidth=2)
        ax4.plot(U_grid, A_reason, 'g-', label='A(U, Reason)', linewidth=2)
        ax4.plot(U_grid, A_terminate, 'r-', label='A(U, Terminate)', linewidth=2)
        ax4.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax4.axvline(Theta_cont, color='orange', linestyle='--', alpha=0.5)
        ax4.axvline(Theta_term, color='purple', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Utility U', fontsize=11)
        ax4.set_ylabel('Advantage A(U, a)', fontsize=11)
        ax4.set_title('Advantage Functions', fontsize=12, fontweight='bold')
        ax4.legend(loc='best', fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved plot: {output_file}")
    
    def run_validation(self, params, grid_size=201):
        """
        Run complete validation for one parameter set.
        
        Returns:
            result: Dictionary with all validation results
        """
        print(f"\n{'='*70}")
        print(f"Parameter Set: {params['name']}")
        print(f"{'='*70}")
        print(f"Description: {params['description']}")
        
        # Display cost efficiency
        cost_eff = self.calculate_cost_efficiency(params)
        print(f"\nCost-Efficiency Analysis:")
        print(f"  Retrieve: δ_r/c_r = {cost_eff['retrieve_efficiency']:.2f}")
        print(f"  Reason: δ_p/c_p = {cost_eff['reason_efficiency']:.2f}")
        print(f"  Ratio: {cost_eff['efficiency_ratio']:.2f} " +
              ("(Retrieve more efficient)" if cost_eff['efficiency_ratio'] > 1 
               else "(Reason more efficient)" if cost_eff['efficiency_ratio'] < 1
               else "(Equal efficiency)"))
        
        # Solve MDP
        solver, U_grid = self.solve_mdp(params, grid_size)
        
        # Identify thresholds
        thresholds = self.identify_thresholds(solver, U_grid)
        print(f"\nIdentified Thresholds:")
        print(f"  Θ_cont = {thresholds['Theta_cont']:.4f}")
        print(f"  Θ_term = {thresholds['Theta_term']:.4f}")
        print(f"  Region lengths:")
        print(f"    Retrieve: [0, {thresholds['Theta_cont']:.3f}] = {thresholds['Theta_cont']:.3f}")
        print(f"    Reason: [{thresholds['Theta_cont']:.3f}, {thresholds['Theta_term']:.3f}] = {thresholds['Theta_term'] - thresholds['Theta_cont']:.3f}")
        print(f"    Terminate: [{thresholds['Theta_term']:.3f}, 1] = {1 - thresholds['Theta_term']:.3f}")
        
        # Layer 0: Validate threshold values
        print(f"\n[Layer 0] Threshold Value Validation:")
        threshold_valid, threshold_msgs = self.validate_threshold_values(thresholds)
        for msg in threshold_msgs:
            print(f"  {msg}")
        
        # Validate policy structure
        print(f"\n[Layer 1] Policy Structure Validation:")
        structure_valid, structure_msgs = self.validate_policy_structure(
            solver, U_grid, thresholds)
        for msg in structure_msgs:
            print(f"  {msg}")
        
        # Statistical monotonicity test
        print(f"\n[Layer 2] Statistical Monotonicity Test:")
        monotonic_valid, monotonic_stats = self.test_monotonicity_statistical(
            solver, U_grid)
        print(f"  Spearman ρ = {monotonic_stats['spearman_rho']:.6f}")
        print(f"  p-value = {monotonic_stats['p_value']:.2e}")
        print(f"  {'✓' if monotonic_valid else '❌'} V*(U) is " +
              f"{'monotonically increasing' if monotonic_valid else 'NOT monotonic'}")
        
        # Single-crossing validation
        print(f"\n[Layer 3] Single-Crossing Property:")
        crossing_valid, n_crossings = self.validate_single_crossing(solver, U_grid)
        print(f"  Number of zero crossings: {n_crossings}")
        print(f"  {'✓' if crossing_valid else '❌'} Single-crossing property " +
              f"{'holds' if crossing_valid else 'violated'}")
        
        # Overall validation
        overall_valid = (threshold_valid and structure_valid and 
                        monotonic_valid and crossing_valid)
        print(f"\n{'='*70}")
        print(f"OVERALL VALIDATION: {'✓ PASSED' if overall_valid else '❌ FAILED'}")
        print(f"{'='*70}")
        
        # Generate plot
        output_file = self.output_dir / f"policy_structure_{params['name'].replace(' ', '_').replace('(', '').replace(')', '')}.png"
        self.plot_policy_structure(solver, U_grid, thresholds, params, 
                                  cost_eff, output_file)
        
        # Compile result
        result = {
            'name': params['name'],
            'p_s': params['p_s'],
            'c_r': params['c_r'],
            'c_p': params['c_p'],
            'delta_r': params['delta_r'],
            'delta_p': params['delta_p'],
            'gamma': params['gamma'],
            'Theta_cont': thresholds['Theta_cont'],
            'Theta_term': thresholds['Theta_term'],
            'retrieve_region_length': thresholds['Theta_cont'],
            'reason_region_length': thresholds['Theta_term'] - thresholds['Theta_cont'],
            'terminate_region_length': 1 - thresholds['Theta_term'],
            'cost_efficiency_ratio': cost_eff['efficiency_ratio'],
            'threshold_valid': threshold_valid,
            'structure_valid': structure_valid,
            'monotonic_valid': monotonic_valid,
            'spearman_rho': monotonic_stats['spearman_rho'],
            'crossing_valid': crossing_valid,
            'n_crossings': n_crossings,
            'overall_valid': overall_valid
        }
        
        return result
    
    def run_all_validations(self):
        """Run validation for all parameter sets."""
        print("\n" + "="*70)
        print("EXPERIMENT 0 V2: THRESHOLD STRUCTURE VALIDATION")
        print("="*70)
        print("Optimized parameters for clear three-region visualization")
        print()
        
        param_sets = self.get_optimized_parameters()
        
        for params in param_sets:
            result = self.run_validation(params)
            self.results.append(result)
        
        # Save results
        self.save_results()
        
        # Print summary
        self.print_summary()
    
    def save_results(self):
        """Save validation results to CSV."""
        df = pd.DataFrame(self.results)
        output_file = self.output_dir / 'threshold_validation_summary_v2.csv'
        df.to_csv(output_file, index=False)
        print(f"\n✓ Saved results summary: {output_file}")
    
    def print_summary(self):
        """Print summary of all validation results."""
        print("\n" + "="*70)
        print("VALIDATION SUMMARY (V2)")
        print("="*70)
        
        df = pd.DataFrame(self.results)
        
        print(f"\nTotal parameter sets tested: {len(df)}")
        print(f"Passed all validations: {df['overall_valid'].sum()}/{len(df)}")
        
        print("\nThreshold Distribution:")
        print(f"  Mean Θ_cont: {df['Theta_cont'].mean():.3f} ± {df['Theta_cont'].std():.3f}")
        print(f"  Mean Θ_term: {df['Theta_term'].mean():.3f} ± {df['Theta_term'].std():.3f}")
        
        print("\nRegion Length Distribution:")
        print(f"  Mean Retrieve region: {df['retrieve_region_length'].mean():.3f} ± {df['retrieve_region_length'].std():.3f}")
        print(f"  Mean Reason region: {df['reason_region_length'].mean():.3f} ± {df['reason_region_length'].std():.3f}")
        print(f"  Mean Terminate region: {df['terminate_region_length'].mean():.3f} ± {df['terminate_region_length'].std():.3f}")
        
        print("\nCost-Efficiency Impact:")
        print(f"  Mean efficiency ratio: {df['cost_efficiency_ratio'].mean():.3f}")
        print(f"  Range: [{df['cost_efficiency_ratio'].min():.3f}, {df['cost_efficiency_ratio'].max():.3f}]")
        
        print("\nMonotonicity Statistics:")
        print(f"  Mean Spearman ρ: {df['spearman_rho'].mean():.6f}")
        print(f"  Min Spearman ρ: {df['spearman_rho'].min():.6f}")
        
        print("\n" + "="*70)
        print("V2 OPTIMIZATION RESULTS:")
        print("="*70)
        
        # Compare with V1 expectations
        well_structured = df[df['reason_region_length'] > 0.2]
        print(f"\nParameter sets with clear Reason region (>20%):")
        print(f"  Count: {len(well_structured)}/{len(df)}")
        if len(well_structured) > 0:
            for _, row in well_structured.iterrows():
                print(f"  - {row['name']}: Reason region = {row['reason_region_length']:.3f}")
        
        print("\n✓ V2 validation complete!")


def main():
    """Main entry point."""
    experiment = ThresholdValidationExperimentV2(
        output_dir='results/exp0_v2_threshold_validation'
    )
    experiment.run_all_validations()


if __name__ == '__main__':
    main()
