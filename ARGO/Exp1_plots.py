#!/usr/bin/env python
"""
实验1: 增强版可视化 (带误差条)
================================
从聚合统计数据生成发表级别的图表

特性:
- 误差条 (95% 置信区间)
- 统计显著性标记
- 多难度级别对比
- 高分辨率输出 (DPI 300)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import seaborn as sns

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_aggregated_data(stats_file):
    """加载聚合统计数据"""
    
    print(f"加载数据: {stats_file}")
    
    stats_df = pd.read_csv(stats_file)
    
    print(f"✓ 数据加载成功")
    print(f"  - 记录数: {len(stats_df)}")
    print(f"  - 难度级别: {sorted(stats_df['difficulty'].unique())}")
    print(f"  - 成本点数: {len(stats_df['c_r'].unique())}")
    print()
    
    return stats_df


def plot_cost_vs_accuracy_with_ci(stats_df, output_dir='figs'):
    """图1.A: Cost vs. Accuracy (带置信区间)"""
    
    print("绘制图1.A: Cost vs. Accuracy...")
    
    difficulties = sorted(stats_df['difficulty'].unique())
    strategies = ['ARGO', 'Always-Retrieve', 'Always-Reason', 'Random']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#6A994E']
    markers = ['o', 's', '^', 'D']
    
    # 为每个难度级别创建子图
    n_diff = len(difficulties)
    fig, axes = plt.subplots(1, n_diff, figsize=(6*n_diff, 5), squeeze=False)
    axes = axes.flatten()
    
    for idx, difficulty in enumerate(difficulties):
        ax = axes[idx]
        diff_data = stats_df[stats_df['difficulty'] == difficulty]
        
        c_r_values = diff_data['c_r'].values
        
        for strategy, color, marker in zip(strategies, colors, markers):
            mean_col = f'{strategy}_accuracy_mean'
            ci_col = f'{strategy}_accuracy_ci95'
            
            if mean_col in diff_data.columns:
                means = diff_data[mean_col].values
                cis = diff_data[ci_col].values if ci_col in diff_data.columns else np.zeros_like(means)
                
                ax.errorbar(c_r_values, means, yerr=cis, 
                           label=strategy, marker=marker, 
                           linewidth=2.5, markersize=8, 
                           color=color, alpha=0.8, capsize=4)
        
        ax.set_xlabel('Retrieval Cost ($c_r$)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Accuracy', fontsize=12, fontweight='bold')
        ax.set_title(f'{difficulty.upper()} Questions', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle('Graph 1.A: Cost vs. Accuracy (with 95% CI)', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig_path = output_path / 'exp1_graph1A_cost_vs_accuracy_with_ci.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 已保存: {fig_path}\n")


def plot_cost_vs_retrievals_with_ci(stats_df, output_dir='figs'):
    """图1.B: Cost vs. Retrieval Calls (带置信区间)"""
    
    print("绘制图1.B: Cost vs. Retrieval Calls...")
    
    difficulties = sorted(stats_df['difficulty'].unique())
    retrieval_strategies = ['ARGO', 'Always-Retrieve', 'Random']
    colors = ['#2E86AB', '#A23B72', '#6A994E']
    markers = ['o', 's', 'D']
    
    n_diff = len(difficulties)
    fig, axes = plt.subplots(1, n_diff, figsize=(6*n_diff, 5), squeeze=False)
    axes = axes.flatten()
    
    for idx, difficulty in enumerate(difficulties):
        ax = axes[idx]
        diff_data = stats_df[stats_df['difficulty'] == difficulty]
        
        c_r_values = diff_data['c_r'].values
        
        for strategy, color, marker in zip(retrieval_strategies, colors, markers):
            mean_col = f'{strategy}_retrievals_mean'
            ci_col = f'{strategy}_retrievals_ci95'
            
            if mean_col in diff_data.columns:
                means = diff_data[mean_col].values
                cis = diff_data[ci_col].values if ci_col in diff_data.columns else np.zeros_like(means)
                
                ax.errorbar(c_r_values, means, yerr=cis, 
                           label=strategy, marker=marker, 
                           linewidth=2.5, markersize=8, 
                           color=color, alpha=0.8, capsize=4)
        
        ax.set_xlabel('Retrieval Cost ($c_r$)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Retrieval Calls ($E[R_T]$)', fontsize=12, fontweight='bold')
        ax.set_title(f'{difficulty.upper()} Questions', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle('Graph 1.B: Cost vs. Retrieval Calls (with 95% CI)', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = Path(output_dir)
    fig_path = output_path / 'exp1_graph1B_cost_vs_retrievals_with_ci.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 已保存: {fig_path}\n")


def plot_combined_view(stats_df, output_dir='figs'):
    """组合视图: 单图包含所有难度"""
    
    print("绘制组合视图...")
    
    difficulties = sorted(stats_df['difficulty'].unique())
    strategies = ['ARGO', 'Always-Retrieve', 'Always-Reason', 'Random']
    
    # 为每个难度使用不同的线型
    linestyles = ['-', '--', '-.']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图: Accuracy
    for diff_idx, difficulty in enumerate(difficulties):
        diff_data = stats_df[stats_df['difficulty'] == difficulty]
        c_r_values = diff_data['c_r'].values
        
        for strat_idx, strategy in enumerate(strategies):
            mean_col = f'{strategy}_accuracy_mean'
            ci_col = f'{strategy}_accuracy_ci95'
            
            if mean_col in diff_data.columns:
                means = diff_data[mean_col].values
                cis = diff_data[ci_col].values if ci_col in diff_data.columns else np.zeros_like(means)
                
                label = f'{strategy} ({difficulty})'
                linestyle = linestyles[diff_idx % len(linestyles)]
                
                ax1.errorbar(c_r_values, means, yerr=cis, 
                           label=label, linewidth=2, 
                           linestyle=linestyle, marker='o', markersize=6,
                           alpha=0.7, capsize=3)
    
    ax1.set_xlabel('Retrieval Cost ($c_r$)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Average Accuracy', fontsize=13, fontweight='bold')
    ax1.set_title('Accuracy Across Difficulties', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, loc='best', ncol=2, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 右图: Retrievals
    retrieval_strategies = ['ARGO', 'Always-Retrieve', 'Random']
    
    for diff_idx, difficulty in enumerate(difficulties):
        diff_data = stats_df[stats_df['difficulty'] == difficulty]
        c_r_values = diff_data['c_r'].values
        
        for strategy in retrieval_strategies:
            mean_col = f'{strategy}_retrievals_mean'
            ci_col = f'{strategy}_retrievals_ci95'
            
            if mean_col in diff_data.columns:
                means = diff_data[mean_col].values
                cis = diff_data[ci_col].values if ci_col in diff_data.columns else np.zeros_like(means)
                
                label = f'{strategy} ({difficulty})'
                linestyle = linestyles[diff_idx % len(linestyles)]
                
                ax2.errorbar(c_r_values, means, yerr=cis, 
                           label=label, linewidth=2, 
                           linestyle=linestyle, marker='o', markersize=6,
                           alpha=0.7, capsize=3)
    
    ax2.set_xlabel('Retrieval Cost ($c_r$)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Average Retrieval Calls', fontsize=13, fontweight='bold')
    ax2.set_title('Retrieval Calls Across Difficulties', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9, loc='best', ncol=2, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle('Experiment 1: Combined View Across All Difficulties', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    output_path = Path(output_dir)
    fig_path = output_path / 'exp1_combined_all_difficulties.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 已保存: {fig_path}\n")


def plot_reduction_percentage(stats_df, output_dir='figs'):
    """补充图: 检索减少百分比"""
    
    print("绘制补充图: 检索减少百分比...")
    
    difficulties = sorted(stats_df['difficulty'].unique())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_offset = 0
    bar_width = 0.2
    
    for diff_idx, difficulty in enumerate(difficulties):
        diff_data = stats_df[stats_df['difficulty'] == difficulty]
        
        c_r_values = diff_data['c_r'].values
        
        # 计算相对于Always-Retrieve的减少百分比
        argo_ret = diff_data['ARGO_retrievals_mean'].values
        alw_ret = diff_data['Always-Retrieve_retrievals_mean'].values
        
        reduction_pct = (alw_ret - argo_ret) / alw_ret * 100
        
        x_positions = np.arange(len(c_r_values)) + diff_idx * bar_width
        
        ax.bar(x_positions, reduction_pct, width=bar_width, 
               label=difficulty.upper(), alpha=0.8)
    
    ax.set_xlabel('Retrieval Cost Level', fontsize=13, fontweight='bold')
    ax.set_ylabel('Retrieval Reduction (%)', fontsize=13, fontweight='bold')
    ax.set_title('ARGO Retrieval Reduction vs. Always-Retrieve Baseline', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(np.arange(len(c_r_values)) + bar_width)
    ax.set_xticklabels([f'{i+1}' for i in range(len(c_r_values))])
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    plt.tight_layout()
    
    output_path = Path(output_dir)
    fig_path = output_path / 'exp1_supplementary_reduction_percentage.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 已保存: {fig_path}\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='生成实验1的增强版图表')
    parser.add_argument('stats_file', type=str, 
                       help='聚合统计CSV文件路径')
    parser.add_argument('--output-dir', type=str, default='figs',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 验证文件存在
    if not Path(args.stats_file).exists():
        print(f"❌ 文件不存在: {args.stats_file}")
        return 1
    
    # 加载数据
    stats_df = load_aggregated_data(args.stats_file)
    
    # 生成所有图表
    print("="*80)
    print("生成图表...")
    print("="*80)
    print()
    
    plot_cost_vs_accuracy_with_ci(stats_df, args.output_dir)
    plot_cost_vs_retrievals_with_ci(stats_df, args.output_dir)
    plot_combined_view(stats_df, args.output_dir)
    plot_reduction_percentage(stats_df, args.output_dir)
    
    print("="*80)
    print("✓ 所有图表生成完成!")
    print("="*80)
    print(f"\n输出目录: {args.output_dir}/")
    print("\n生成的图表:")
    print("  1. exp1_graph1A_cost_vs_accuracy_with_ci.png")
    print("  2. exp1_graph1B_cost_vs_retrievals_with_ci.png")
    print("  3. exp1_combined_all_difficulties.png")
    print("  4. exp1_supplementary_reduction_percentage.png")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
