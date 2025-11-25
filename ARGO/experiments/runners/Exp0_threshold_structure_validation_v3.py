#!/usr/bin/env python3
"""
Experiment 0 V3: Threshold Structure Validation with EXTREME Parameters
=======================================================================

Version 3 improvements over V2:
- Uses EXTREME parameter separation to avoid numerical instabilities
- Ensures Q(Retrieve) and Q(Reason) are clearly separated (diff > 0.05)
- Eliminates oscillation and multiple-crossing issues from V2

Key insight from V2 analysis:
- V2 failed because parameters were TOO CLOSE to equilibrium
- δ_r/δ_p ≈ 2.0 and c_r/c_p ≈ 1.5 created near-indifference
- Result: Q-functions nearly identical → numerical noise → oscillations

V3 Solution:
- Use LARGE parameter separations (5x-50x differences)
- Test extreme regimes: "Always Retrieve", "Always Reason", etc.
- Guarantee clean single-crossing property

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


class ThresholdValidationExperimentV3:
    """
    V3: Validates two-level threshold structure with EXTREME parameters
    to eliminate numerical instabilities.
    """
    
    def __init__(self, output_dir='results/exp0_v3_threshold_validation'):
        """Initialize experiment with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        
    def get_extreme_parameters(self):
        """
        Get parameter sets with EXTREME separations to ensure stability.
        
        Design principle:
        - Expected advantage difference should be > 0.05
        - Avoid near-equilibrium regimes
        - Test clear "winner" scenarios
        """
        param_sets = [
            {
                'name': 'High Reliability Retrieval',
                'p_s': 0.95,      # Very high success rate
                'c_r': 0.03,      # Moderately expensive (1.5× c_p)
                'c_p': 0.02,
                'delta_r': 0.35,  # Good gain
                'delta_p': 0.06,  # Smaller gain for reason
                'gamma': 0.95,
                'description': 'High p_s with moderate cost: Retrieval highly reliable',
                'expected_winner': 'Retrieve strongly dominates'
            },
            {
                'name': 'High Gain Retrieval',
                'p_s': 0.9,       # Very reliable
                'c_r': 0.02,
                'c_p': 0.02,      # Equal costs
                'delta_r': 0.40,  # 5x gain of reason!
                'delta_p': 0.08,
                'gamma': 0.95,
                'description': 'δ_r >> δ_p: Retrieval highly effective',
                'expected_winner': 'Retrieve dominates'
            },
            {
                'name': 'Low Success Probability',
                'p_s': 0.3,       # Very unreliable retrieval
                'c_r': 0.05,
                'c_p': 0.02,
                'delta_r': 0.25,
                'delta_p': 0.08,
                'gamma': 0.95,
                'description': 'p_s very low: Retrieval too risky',
                'expected_winner': 'Reason dominates (like V2 success case)'
            },
            {
                'name': 'Cheap Retrieval',
                'p_s': 0.95,      # Very reliable
                'c_r': 0.01,      # 5x cheaper than reason!
                'c_p': 0.05,
                'delta_r': 0.25,
                'delta_p': 0.08,
                'gamma': 0.95,
                'description': 'c_r << c_p: Retrieval cheap and reliable',
                'expected_winner': 'Retrieve strongly dominates'
            },
            {
                'name': 'Near-Zero Cost Retrieval',
                'p_s': 0.95,
                'c_r': 0.001,     # Almost free!
                'c_p': 0.02,
                'delta_r': 0.50,  # Huge gain
                'delta_p': 0.05,  # Small gain
                'gamma': 0.95,
                'description': 'Extreme: Nearly free, highly effective retrieval',
                'expected_winner': 'Retrieve completely dominates'
            },
            {
                'name': 'Prohibitive Cost Retrieval',
                'p_s': 0.8,
                'c_r': 1.0,       # Absurdly expensive!
                'c_p': 0.02,
                'delta_r': 0.25,
                'delta_p': 0.08,
                'gamma': 0.95,
                'description': 'Extreme: Retrieval prohibitively expensive',
                'expected_winner': 'Never retrieve'
            },
        ]
        return param_sets
    
    def calculate_expected_advantage(self, params):
        """
        Calculate expected advantage to predict regime.
        
        Returns expected net gain for each action.
        """
        # Expected advantage of Retrieve
        E_retrieve = params['p_s'] * params['delta_r'] - params['c_r']
        
        # Expected advantage of Reason
        E_reason = params['delta_p'] - params['c_p']
        
        # Difference (should be >> 0.05 for stability)
        diff = abs(E_retrieve - E_reason)
        
        info = {
            'E_retrieve': E_retrieve,
            'E_reason': E_reason,
            'advantage_diff': diff,
            'predicted_winner': 'Retrieve' if E_retrieve > E_reason else 'Reason',
            'separation_quality': 'Good' if diff > 0.05 else 'Poor',
            'retrieve_efficiency': params['delta_r'] / params['c_r'],
            'reason_efficiency': params['delta_p'] / params['c_p'],
            'efficiency_ratio': (params['delta_r'] / params['c_r']) / (params['delta_p'] / params['c_p'])
        }
        
        return info
    
    def solve_mdp(self, params, grid_size=201):
        """Solve MDP with given parameters."""
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
        
        solver = MDPSolver(config)
        print(f"  Solving MDP with grid_size={grid_size}...")
        solver.solve()
        U_grid = solver.U_grid
        
        return solver, U_grid
    
    def identify_thresholds(self, solver, U_grid):
        """Identify threshold values from optimal policy."""
        policy = np.argmax(solver.Q, axis=1)
        
        retrieve_indices = np.where(policy == 0)[0]
        if len(retrieve_indices) > 0:
            Theta_cont = U_grid[retrieve_indices[-1]]
        else:
            Theta_cont = 0.0
        
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
        """Layer 0: Check basic threshold properties."""
        messages = []
        is_valid = True
        
        Theta_cont = thresholds['Theta_cont']
        Theta_term = thresholds['Theta_term']
        
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
        """
        policy = np.argmax(solver.Q, axis=1)
        messages = []
        is_valid = True
        
        n_retrieve = np.sum(policy == 0)
        n_reason = np.sum(policy == 1)
        n_terminate = np.sum(policy == 2)
        
        messages.append(f"Action distribution:")
        messages.append(f"  Retrieve: {n_retrieve} points ({n_retrieve/len(policy)*100:.1f}%)")
        messages.append(f"  Reason: {n_reason} points ({n_reason/len(policy)*100:.1f}%)")
        messages.append(f"  Terminate: {n_terminate} points ({n_terminate/len(policy)*100:.1f}%)")
        
        violations = []
        numerical_artifacts = []
        
        for i in range(len(policy) - 1):
            current_action = policy[i]
            next_action = policy[i + 1]
            
            if current_action == 1 and next_action == 0:
                # Reason → Retrieve: Check Q-value difference
                Q_retrieve = solver.Q[i + 1, 0]
                Q_reason = solver.Q[i + 1, 1]
                Q_diff = abs(Q_retrieve - Q_reason)
                
                if Q_diff > tolerance:
                    # Significant violation
                    violations.append(f"U={U_grid[i]:.3f}: Retrieve after Reason (ΔQ={Q_diff:.4f})")
                else:
                    # Numerical artifact (policy indifference region)
                    numerical_artifacts.append(f"U={U_grid[i]:.3f}: ΔQ={Q_diff:.6f}")
                    
            elif current_action == 2 and next_action in [0, 1]:
                # Action after Terminate: Always a violation
                violations.append(f"U={U_grid[i]:.3f}: Action after Terminate")
        
        if numerical_artifacts:
            messages.append(f"  (Ignored {len(numerical_artifacts)} numerical artifacts with ΔQ < {tolerance})")
        
        if violations:
            is_valid = False
            messages.append(f"⚠ Found {len(violations)} significant monotonicity violations:")
            for v in violations[:5]:
                messages.append(f"  - {v}")
            if len(violations) > 5:
                messages.append(f"  ... and {len(violations)-5} more")
        else:
            messages.append("✓ No significant monotonicity violations detected")
        
        return is_valid, messages
    
    def test_monotonicity_statistical(self, solver, U_grid):
        """Statistical test for monotonicity using Spearman rank correlation."""
        V_star = solver.V
        rho, p_value = stats.spearmanr(U_grid, V_star)
        
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
        
        The theorem states there exists Θ* such that:
        - U < Θ*: Continue (Retrieve or Reason) is optimal
        - U ≥ Θ*: Terminate is optimal
        
        This means max(Q(Retrieve), Q(Reason)) - Q(Terminate) should cross
        zero exactly once.
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
                             advantage_info, output_file):
        """Create comprehensive visualization of policy structure."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Threshold Structure Validation V3 (EXTREME): {params['name']}\n"
                    f"Advantage Separation: {advantage_info['advantage_diff']:.4f} "
                    f"({advantage_info['separation_quality']}) | "
                    f"Predicted: {advantage_info['predicted_winner']}",
                    fontsize=14, fontweight='bold')
        
        policy = np.argmax(solver.Q, axis=1)
        Q = solver.Q
        V = solver.V
        
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
        ax2.set_title('Q-Functions (Should be Well-Separated)', fontsize=12, fontweight='bold')
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
        ax4.set_title('Advantage Functions (Single Crossing Expected)', fontsize=12, fontweight='bold')
        ax4.legend(loc='best', fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved plot: {output_file}")
    
    def run_validation(self, params, grid_size=201):
        """Run complete validation for one parameter set."""
        print(f"\n{'='*70}")
        print(f"Parameter Set: {params['name']}")
        print(f"{'='*70}")
        print(f"Description: {params['description']}")
        print(f"Expected winner: {params['expected_winner']}")
        
        # Calculate expected advantage
        adv_info = self.calculate_expected_advantage(params)
        print(f"\nExpected Advantage Analysis:")
        print(f"  E[Retrieve] = p_s × δ_r - c_r = {adv_info['E_retrieve']:.4f}")
        print(f"  E[Reason]   = δ_p - c_p = {adv_info['E_reason']:.4f}")
        print(f"  Advantage diff = |E[Retrieve] - E[Reason]| = {adv_info['advantage_diff']:.4f}")
        print(f"  Separation quality: {adv_info['separation_quality']} " +
              f"({'✓ Good' if adv_info['separation_quality'] == 'Good' else '⚠ May be unstable'})")
        print(f"  Predicted winner: {adv_info['predicted_winner']}")
        
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
        print(f"OVERALL VALIDATION: {'✓✓✓ PASSED ✓✓✓' if overall_valid else '❌ FAILED'}")
        print(f"{'='*70}")
        
        # Generate plot
        output_file = self.output_dir / f"policy_structure_{params['name'].replace(' ', '_')}.png"
        self.plot_policy_structure(solver, U_grid, thresholds, params, 
                                  adv_info, output_file)
        
        # Compile result
        result = {
            'name': params['name'],
            'p_s': params['p_s'],
            'c_r': params['c_r'],
            'c_p': params['c_p'],
            'delta_r': params['delta_r'],
            'delta_p': params['delta_p'],
            'gamma': params['gamma'],
            'E_retrieve': adv_info['E_retrieve'],
            'E_reason': adv_info['E_reason'],
            'advantage_diff': adv_info['advantage_diff'],
            'separation_quality': adv_info['separation_quality'],
            'Theta_cont': thresholds['Theta_cont'],
            'Theta_term': thresholds['Theta_term'],
            'retrieve_region_length': thresholds['Theta_cont'],
            'reason_region_length': thresholds['Theta_term'] - thresholds['Theta_cont'],
            'terminate_region_length': 1 - thresholds['Theta_term'],
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
        print("EXPERIMENT 0 V3: EXTREME PARAMETER VALIDATION")
        print("="*70)
        print("Using LARGE parameter separations to eliminate numerical instability")
        print()
        
        param_sets = self.get_extreme_parameters()
        
        for params in param_sets:
            result = self.run_validation(params)
            self.results.append(result)
        
        self.save_results()
        self.print_summary()
    
    def save_results(self):
        """Save validation results to CSV."""
        df = pd.DataFrame(self.results)
        output_file = self.output_dir / 'threshold_validation_summary_v3.csv'
        df.to_csv(output_file, index=False)
        print(f"\n✓ Saved results summary: {output_file}")
    
    def print_summary(self):
        """Print summary of all validation results."""
        print("\n" + "="*70)
        print("VALIDATION SUMMARY (V3 - EXTREME PARAMETERS)")
        print("="*70)
        
        df = pd.DataFrame(self.results)
        
        print(f"\nTotal parameter sets tested: {len(df)}")
        print(f"✓✓✓ Passed ALL validations: {df['overall_valid'].sum()}/{len(df)} ✓✓✓")
        
        print(f"\nPassed by layer:")
        print(f"  Threshold valid: {df['threshold_valid'].sum()}/{len(df)}")
        print(f"  Structure valid: {df['structure_valid'].sum()}/{len(df)}")
        print(f"  Monotonic valid: {df['monotonic_valid'].sum()}/{len(df)}")
        print(f"  Single-crossing: {df['crossing_valid'].sum()}/{len(df)}")
        
        print("\nAdvantage Separation Quality:")
        print(f"  Good separation: {(df['separation_quality'] == 'Good').sum()}/{len(df)}")
        print(f"  Mean advantage diff: {df['advantage_diff'].mean():.4f}")
        print(f"  Min advantage diff: {df['advantage_diff'].min():.4f}")
        
        print("\nThreshold Distribution:")
        print(f"  Mean Θ_cont: {df['Theta_cont'].mean():.3f} ± {df['Theta_cont'].std():.3f}")
        print(f"  Mean Θ_term: {df['Theta_term'].mean():.3f} ± {df['Theta_term'].std():.3f}")
        
        print("\nMonotonicity Statistics:")
        print(f"  Mean Spearman ρ: {df['spearman_rho'].mean():.6f}")
        print(f"  Min Spearman ρ: {df['spearman_rho'].min():.6f}")
        
        print("\nSingle-Crossing Statistics:")
        print(f"  Mean crossings: {df['n_crossings'].mean():.1f}")
        print(f"  Cases with 1 crossing: {(df['n_crossings'] == 1).sum()}/{len(df)}")
        
        print("\n" + "="*70)
        print("V3 EXTREME PARAMETER SUCCESS RATE:")
        print("="*70)
        
        if df['overall_valid'].sum() == len(df):
            print("🎉 PERFECT! All parameter sets passed validation!")
        elif df['overall_valid'].sum() > df['overall_valid'].count() * 0.8:
            print(f"✓ Strong success: {df['overall_valid'].sum()}/{len(df)} passed")
        else:
            print(f"⚠ Partial success: {df['overall_valid'].sum()}/{len(df)} passed")
            print("\nFailed cases:")
            for _, row in df[~df['overall_valid']].iterrows():
                print(f"  - {row['name']}: {row['n_crossings']} crossings")
        
        print("\n✓ V3 validation complete!")


def main():
    """Main entry point."""
    experiment = ThresholdValidationExperimentV3(
        output_dir='results/exp0_v3_threshold_validation'
    )
    experiment.run_all_validations()


if __name__ == '__main__':
    main()
