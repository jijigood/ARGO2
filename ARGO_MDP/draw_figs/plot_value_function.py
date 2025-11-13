"""
Visualization Script: Plot Value Function and Thresholds
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


def plot_value_function(results_dir: str = 'results', output_dir: str = 'figs'):
    """
    Plot V*(U) and Q*(U, a) functions
    """
    # Load data
    df = pd.read_csv(os.path.join(results_dir, 'value_function.csv'))
    
    # Read thresholds
    with open(os.path.join(results_dir, 'thresholds.txt'), 'r') as f:
        lines = f.readlines()
        theta_cont = float(lines[0].split(':')[1].strip())
        theta_star = float(lines[1].split(':')[1].strip())
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Value Function V*(U)
    ax1.plot(df['U'], df['V'], 'b-', linewidth=2, label='V*(U)')
    ax1.axvline(theta_cont, color='g', linestyle='--', linewidth=1.5, label=f'θ_cont = {theta_cont:.3f}')
    ax1.axvline(theta_star, color='r', linestyle='--', linewidth=1.5, label=f'θ_star = {theta_star:.3f}')
    ax1.set_xlabel('Information Progress U', fontsize=12)
    ax1.set_ylabel('Value V*(U)', fontsize=12)
    ax1.set_title('Optimal Value Function', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot 2: Q-Function
    ax2.plot(df['U'], df['Q_retrieve'], 'b-', linewidth=2, label='Q*(U, Retrieve)')
    ax2.plot(df['U'], df['Q_reason'], 'g-', linewidth=2, label='Q*(U, Reason)')
    ax2.plot(df['U'], df['Q_terminate'], 'r-', linewidth=2, label='Q*(U, Terminate)')
    ax2.axvline(theta_cont, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax2.axvline(theta_star, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax2.set_xlabel('Information Progress U', fontsize=12)
    ax2.set_ylabel('Q*(U, a)', fontsize=12)
    ax2.set_title('Q-Function for Each Action', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'value_function.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_action_selection(results_dir: str = 'results', output_dir: str = 'figs'):
    """
    Plot optimal action selection regions
    """
    # Load data
    df = pd.read_csv(os.path.join(results_dir, 'value_function.csv'))
    
    # Determine optimal action for each state
    actions = np.argmax(df[['Q_retrieve', 'Q_reason', 'Q_terminate']].values, axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color code actions
    colors = ['blue', 'green', 'red']
    labels = ['Retrieve', 'Reason', 'Terminate']
    
    for action in [0, 1, 2]:
        mask = actions == action
        if np.any(mask):
            ax.scatter(df['U'][mask], np.ones(np.sum(mask)) * action, 
                      c=colors[action], s=20, label=labels[action], alpha=0.6)
    
    ax.set_xlabel('Information Progress U', fontsize=12)
    ax.set_ylabel('Optimal Action', fontsize=12)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Retrieve', 'Reason', 'Terminate'])
    ax.set_title('Optimal Action Selection by State', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, 'action_selection.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_threshold_diagram(results_dir: str = 'results', output_dir: str = 'figs'):
    """
    Create threshold diagram showing policy regions
    """
    # Read thresholds
    with open(os.path.join(results_dir, 'thresholds.txt'), 'r') as f:
        lines = f.readlines()
        theta_cont = float(lines[0].split(':')[1].strip())
        theta_star = float(lines[1].split(':')[1].strip())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 3))
    
    # Draw regions
    U_max = 1.0
    
    # Retrieve region [0, theta_cont)
    ax.barh(0, theta_cont, left=0, height=0.5, color='blue', alpha=0.3, label='Retrieve')
    ax.text(theta_cont/2, 0, 'Retrieve', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Reason region [theta_cont, theta_star)
    if theta_star > theta_cont:
        ax.barh(0, theta_star - theta_cont, left=theta_cont, height=0.5, color='green', alpha=0.3, label='Reason')
        ax.text((theta_cont + theta_star)/2, 0, 'Reason', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Terminate region [theta_star, U_max]
    ax.barh(0, U_max - theta_star, left=theta_star, height=0.5, color='red', alpha=0.3, label='Terminate')
    ax.text((theta_star + U_max)/2, 0, 'Terminate', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Add threshold lines
    ax.axvline(theta_cont, color='darkgreen', linestyle='--', linewidth=2)
    ax.axvline(theta_star, color='darkred', linestyle='--', linewidth=2)
    
    # Annotations
    ax.text(theta_cont, 0.6, f'θ_cont = {theta_cont:.3f}', ha='center', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='darkgreen'))
    ax.text(theta_star, 0.6, f'θ_star = {theta_star:.3f}', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='darkred'))
    
    ax.set_xlim(0, U_max)
    ax.set_ylim(-0.5, 1)
    ax.set_xlabel('Information Progress U', fontsize=12)
    ax.set_yticks([])
    ax.set_title('ARGO Threshold Policy Regions', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, 'threshold_diagram.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot Value Function and Thresholds')
    parser.add_argument('--results', type=str, default='results', help='Results directory')
    parser.add_argument('--output', type=str, default='figs', help='Output directory')
    
    args = parser.parse_args()
    
    print("Generating value function visualizations...")
    plot_value_function(args.results, args.output)
    plot_action_selection(args.results, args.output)
    plot_threshold_diagram(args.results, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
