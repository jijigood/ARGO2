"""
Visualization Script: Plot Policy Comparison Results
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


def plot_policy_comparison(results_dir: str = 'results', output_dir: str = 'figs'):
    """
    Plot comparison of different policies
    """
    # Load comparison data
    df = pd.read_csv(os.path.join(results_dir, 'policy_comparison.csv'))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    policies = df['Policy']
    x = np.arange(len(policies))
    width = 0.6
    
    # Plot 1: Average Reward
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x, df['Avg_Reward'], width, yerr=df['Std_Reward'], 
                    capsize=5, alpha=0.8, color='steelblue')
    ax1.set_ylabel('Average Reward', fontsize=12)
    ax1.set_title('Policy Comparison: Reward', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(policies, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Highlight ARGO
    if 'ARGO' in policies.values:
        argo_idx = policies[policies == 'ARGO'].index[0]
        bars1[argo_idx].set_color('darkgreen')
        bars1[argo_idx].set_alpha(1.0)
    
    # Plot 2: Average Quality
    ax2 = axes[0, 1]
    bars2 = ax2.bar(x, df['Avg_Quality'], width, yerr=df['Std_Quality'],
                    capsize=5, alpha=0.8, color='coral')
    ax2.set_ylabel('Average Quality', fontsize=12)
    ax2.set_title('Policy Comparison: Quality', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(policies, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    if 'ARGO' in policies.values:
        argo_idx = policies[policies == 'ARGO'].index[0]
        bars2[argo_idx].set_color('darkgreen')
        bars2[argo_idx].set_alpha(1.0)
    
    # Plot 3: Average Cost
    ax3 = axes[1, 0]
    bars3 = ax3.bar(x, df['Avg_Cost'], width, yerr=df['Std_Cost'],
                    capsize=5, alpha=0.8, color='lightcoral')
    ax3.set_ylabel('Average Cost', fontsize=12)
    ax3.set_title('Policy Comparison: Cost', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(policies, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    if 'ARGO' in policies.values:
        argo_idx = policies[policies == 'ARGO'].index[0]
        bars3[argo_idx].set_color('darkgreen')
        bars3[argo_idx].set_alpha(1.0)
    
    # Plot 4: Action Distribution
    ax4 = axes[1, 1]
    width2 = 0.35
    x2 = np.arange(len(policies))
    bars4a = ax4.bar(x2 - width2/2, df['Avg_Retrieves'], width2, 
                     label='Retrieves', alpha=0.8, color='blue')
    bars4b = ax4.bar(x2 + width2/2, df['Avg_Reasons'], width2,
                     label='Reasons', alpha=0.8, color='green')
    ax4.set_ylabel('Average Count', fontsize=12)
    ax4.set_title('Policy Comparison: Action Distribution', fontsize=14, fontweight='bold')
    ax4.set_xticks(x2)
    ax4.set_xticklabels(policies, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'policy_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_trajectory_example(results_dir: str = 'results', output_dir: str = 'figs', 
                           policy_name: str = 'ARGO', episode_idx: int = 0):
    """
    Plot trajectory for a specific policy and episode
    """
    # Load episode data
    filename = os.path.join(results_dir, f'{policy_name}_episodes.csv')
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found, skipping trajectory plot")
        return
    
    # For detailed trajectory, we need the full trajectory data
    # Here we'll create a conceptual plot
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Simulated trajectory data (in actual implementation, load from detailed results)
    # This would come from episode['trajectory'] in the detailed results
    
    print(f"Note: Trajectory plotting requires detailed episode data storage")
    print(f"This is a placeholder. Implement detailed trajectory saving in run_single.py")
    
    plt.close()


def plot_cost_quality_tradeoff(results_dir: str = 'results', output_dir: str = 'figs'):
    """
    Plot cost-quality tradeoff scatter plot
    """
    # Load comparison data
    df = pd.read_csv(os.path.join(results_dir, 'policy_comparison.csv'))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Scatter plot
    colors = ['darkgreen' if p == 'ARGO' else 'steelblue' for p in df['Policy']]
    sizes = [200 if p == 'ARGO' else 100 for p in df['Policy']]
    
    for i, row in df.iterrows():
        ax.scatter(row['Avg_Cost'], row['Avg_Quality'], 
                  c=colors[i], s=sizes[i], alpha=0.7, edgecolors='black', linewidth=1.5)
        ax.annotate(row['Policy'], (row['Avg_Cost'], row['Avg_Quality']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('Average Cost', fontsize=12)
    ax.set_ylabel('Average Quality', fontsize=12)
    ax.set_title('Cost-Quality Tradeoff', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add Pareto frontier hint (upper-left is better)
    ax.annotate('Better\n(Low Cost, High Quality)', xy=(0.05, 0.95), xycoords='axes fraction',
               fontsize=10, ha='left', va='top', 
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, 'cost_quality_tradeoff.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_sensitivity_analysis(results_dir: str = 'results', output_dir: str = 'figs'):
    """
    Plot sensitivity analysis results
    """
    sensitivity_file = os.path.join(results_dir, 'sensitivity_analysis.csv')
    if not os.path.exists(sensitivity_file):
        print("Sensitivity analysis results not found, skipping...")
        return
    
    df = pd.read_csv(sensitivity_file)
    
    # Get unique parameters
    parameters = df['parameter'].unique()
    
    # Create subplots
    n_params = len(parameters)
    fig, axes = plt.subplots(n_params, 2, figsize=(14, 5 * n_params))
    
    if n_params == 1:
        axes = axes.reshape(1, -1)
    
    for i, param in enumerate(parameters):
        param_data = df[df['parameter'] == param]
        
        # Plot thresholds
        ax1 = axes[i, 0]
        ax1.plot(param_data['value'], param_data['theta_cont'], 'o-', label='θ_cont', linewidth=2)
        ax1.plot(param_data['value'], param_data['theta_star'], 's-', label='θ_star', linewidth=2)
        ax1.set_xlabel(f'{param}', fontsize=12)
        ax1.set_ylabel('Threshold Value', fontsize=12)
        ax1.set_title(f'Thresholds vs {param}', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot performance
        ax2 = axes[i, 1]
        ax2.plot(param_data['value'], param_data['avg_reward'], 'o-', label='Reward', linewidth=2)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(param_data['value'], param_data['avg_quality'], 's-', 
                     color='orange', label='Quality', linewidth=2)
        ax2.set_xlabel(f'{param}', fontsize=12)
        ax2.set_ylabel('Average Reward', fontsize=12, color='blue')
        ax2_twin.set_ylabel('Average Quality', fontsize=12, color='orange')
        ax2.set_title(f'Performance vs {param}', fontsize=13, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2_twin.tick_params(axis='y', labelcolor='orange')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, 'sensitivity_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot Policy Comparison Results')
    parser.add_argument('--results', type=str, default='results', help='Results directory')
    parser.add_argument('--output', type=str, default='figs', help='Output directory')
    
    args = parser.parse_args()
    
    print("Generating policy comparison visualizations...")
    plot_policy_comparison(args.results, args.output)
    plot_cost_quality_tradeoff(args.results, args.output)
    plot_sensitivity_analysis(args.results, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
