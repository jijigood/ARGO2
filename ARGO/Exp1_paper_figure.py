#!/usr/bin/env python
"""
论文图表生成脚本 - IEEE ICC格式
================================
生成两面板图表：
- Panel (a): Retrieval Calls vs Cost (合并所有难度)
- Panel (b): Accuracy Comparison (ARGO vs Baselines)
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import seaborn as sns
pd.set_option('mode.chained_assignment', None)  # 避免警告

# 设置全局字体（对齐AOI脚本样式）
matplotlib.rcParams.update({
    "mathtext.fontset": 'stix'
})

# 字体样式字典（对齐AOI脚本）
legend_front = {
    'size': 12,
}

xy_descri = {
    'weight': 'bold',
    'size': 12,
}

# 设置绘图风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300
sns.set_style("whitegrid")


def set_border(ax_):
    """设置边框粗细（对齐AOI脚本样式）"""
    bwith = 0.5
    ax_.spines['bottom'].set_linewidth(bwith)
    ax_.spines['left'].set_linewidth(bwith)
    ax_.spines['top'].set_linewidth(bwith)
    ax_.spines['right'].set_linewidth(bwith)


def load_aggregated_data(stats_file):
    """加载聚合统计数据"""
    print(f"加载数据: {stats_file}")
    stats_df = pd.read_csv(stats_file)
    print(f"✓ 数据加载成功")
    print(f"  - 记录数: {len(stats_df)}")
    print(f"  - 难度级别: {sorted(stats_df['difficulty'].unique())}")
    print(f"  - 成本点数: {len(stats_df['c_r'].unique())}")
    return stats_df


def plot_paper_figure_two_panel(stats_df, output_dir='figs', c_p=0.02):
    """
    生成论文两面板图表
    
    Panel (a): Retrieval Calls vs Cost (合并所有难度)
    Panel (b): Accuracy Comparison (ARGO vs Baselines)
    """
    
    print("\n" + "="*80)
    print("生成论文两面板图表")
    print("="*80)
    
    # 创建两面板图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # ========================================================================
    # Panel (a): Retrieval Calls vs Cost (合并所有难度)
    # ========================================================================
    print("\n绘制 Panel (a): Retrieval Calls vs Cost...")
    
    difficulties = sorted(stats_df['difficulty'].unique())
    colors = {'easy': '#2ecc71', 'medium': '#3498db', 'hard': '#e74c3c'}
    markers = {'easy': 'o', 'medium': 's', 'hard': '^'}
    labels = {'easy': 'Easy', 'medium': 'Medium', 'hard': 'Hard'}
    
    # 获取所有成本点
    c_r_values = sorted(stats_df['c_r'].unique())
    
    # 绘制每个难度的检索次数
    for difficulty in difficulties:
        diff_data = stats_df[stats_df['difficulty'] == difficulty].sort_values('c_r')
        
        retrievals = diff_data['ARGO_retrievals_mean'].values
        cis = diff_data['ARGO_retrievals_ci95'].values if 'ARGO_retrievals_ci95' in diff_data.columns else np.zeros_like(retrievals)
        c_r_diff = diff_data['c_r'].values
        
        ax1.errorbar(c_r_diff, retrievals, yerr=cis,
                    label=labels[difficulty], 
                    marker=markers[difficulty],
                    color=colors[difficulty],
                    linewidth=2.0,
                    markersize=6,
                    capsize=3,
                    alpha=0.8)
    
    # 添加垂直线标记phase transition (c_r = c_p)
    ax1.axvline(x=c_p, color='gray', linestyle='--', linewidth=1.5, 
                alpha=0.7, label=f'$c_r = c_p$')
    
    # 添加水平线显示Always-Retrieve baseline
    always_retrieve_value = 10.0  # 固定值
    ax1.axhline(y=always_retrieve_value, color='purple', linestyle=':', 
                linewidth=1.5, alpha=0.7, label='Always-Retrieve')
    
    ax1.set_xlabel('Retrieval Cost ($c_r$)', fontdict=xy_descri)
    ax1.set_ylabel('Average Retrieval Calls', fontdict=xy_descri)
    ax1.set_title('(a) Retrieval Calls vs Cost', fontweight='bold', pad=10)
    ax1.legend(loc='upper right', prop=legend_front, framealpha=1.0)
    ax1.set_ylim(0, 11)
    ax1.grid(True, linewidth=0.5, alpha=0.3, linestyle='--')
    set_border(ax1)
    ax1.tick_params(labelsize=10)
    
    # 添加文本标注phase transition
    ax1.text(c_p, ax1.get_ylim()[1] * 0.95, f'$c_r=c_p$', 
             ha='center', va='top', fontsize=8, color='gray',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # ========================================================================
    # Panel (b): Accuracy vs Cost (ARGO vs Baselines)
    # ========================================================================
    print("绘制 Panel (b): Accuracy vs Cost (ARGO vs Baselines)...")
    
    # 检查是否有baseline数据
    has_baseline = 'Always-Retrieve_accuracy_mean' in stats_df.columns and \
                   stats_df['Always-Retrieve_accuracy_mean'].notna().any()
    
    # 绘制ARGO在不同难度下的accuracy随成本变化
    for difficulty in difficulties:
        diff_data = stats_df[stats_df['difficulty'] == difficulty].sort_values('c_r')
        
        accuracies = diff_data['ARGO_accuracy_mean'].values
        cis = diff_data['ARGO_accuracy_ci95'].values if 'ARGO_accuracy_ci95' in diff_data.columns else np.zeros_like(accuracies)
        c_r_diff = diff_data['c_r'].values
        
        ax2.errorbar(c_r_diff, accuracies, yerr=cis,
                    label=f'ARGO ({labels[difficulty]})', 
                    marker=markers[difficulty],
                    color=colors[difficulty],
                    linewidth=2.0,
                    markersize=6,
                    capsize=3,
                    alpha=0.8)
    
    # 添加Always-Retrieve baseline（如果可用）
    if has_baseline:
        for difficulty in difficulties:
            diff_data = stats_df[stats_df['difficulty'] == difficulty]
            if len(diff_data) > 0:
                ar_acc = diff_data['Always-Retrieve_accuracy_mean'].iloc[0]
                ar_ci = diff_data['Always-Retrieve_accuracy_ci95'].iloc[0] if 'Always-Retrieve_accuracy_ci95' in diff_data.columns else 0.0
                
                if not pd.isna(ar_acc):
                    # 绘制水平线表示baseline
                    ax2.axhline(y=ar_acc, color=colors[difficulty], linestyle=':', 
                               linewidth=1.5, alpha=0.5, 
                               label=f'Always-Retrieve ({labels[difficulty]})')
        
        # 添加图例说明
        print("  ✓ 已添加Always-Retrieve baseline对比")
    
    # 添加垂直线标记phase transition (c_r = c_p)
    ax2.axvline(x=c_p, color='gray', linestyle='--', linewidth=1.5, 
                alpha=0.7)
    
    ax2.set_xlabel('Retrieval Cost ($c_r$)', fontdict=xy_descri)
    ax2.set_ylabel('Accuracy', fontdict=xy_descri)
    ax2.set_title('(b) Accuracy vs Cost', fontweight='bold', pad=10)
    ax2.legend(loc='best', prop=legend_front, framealpha=1.0, ncol=2)
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, linewidth=0.5, alpha=0.3, linestyle='--')
    set_border(ax2)
    ax2.tick_params(labelsize=10)
    
    # 添加文本标注phase transition
    ax2.text(c_p, ax2.get_ylim()[1] * 0.95, f'$c_r=c_p$', 
             ha='center', va='top', fontsize=8, color='gray',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # ========================================================================
    # 整体标题和布局
    # ========================================================================
    fig.suptitle('Cost-Adaptive Behavior of ARGO', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存PNG格式
    fig_path = output_path / 'exp1_paper_figure_two_panel.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ 已保存PNG: {fig_path}")
    
    # 保存PDF格式（对齐AOI脚本）
    fig_path_pdf = output_path / 'exp1_paper_figure_two_panel.pdf'
    plt.savefig(fig_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"✓ 已保存PDF: {fig_path_pdf}")
    
    plt.close()
    
    return fig_path, fig_path_pdf


def generate_paper_table(stats_df, output_dir='figs', c_p=0.02):
    """
    生成论文表格：Performance Comparison Across Retrieval Cost Levels
    """
    
    print("\n" + "="*80)
    print("生成论文表格")
    print("="*80)
    
    # 选择代表性的成本点
    representative_costs = [0.005, 0.02, 0.05]
    difficulties = sorted(stats_df['difficulty'].unique())
    
    # 准备表格数据
    table_rows = []
    
    for difficulty in difficulties:
        for cost_val in representative_costs:
            diff_data = stats_df[
                (stats_df['difficulty'] == difficulty) & 
                (stats_df['c_r'] == cost_val)
            ]
            
            if len(diff_data) > 0:
                row = diff_data.iloc[0]
                
                # 提取数据
                argo_acc = row['ARGO_accuracy_mean'] * 100
                argo_ret = row['ARGO_retrievals_mean']
                ar_acc = row['Always-Retrieve_accuracy_mean'] * 100 if 'Always-Retrieve_accuracy_mean' in row and not pd.isna(row['Always-Retrieve_accuracy_mean']) else None
                ar_ret = 10.0  # 固定值
                
                # 计算减少百分比
                reduction = (1 - argo_ret / ar_ret) * 100 if ar_ret > 0 else 0
                
                # 判断是否为最佳值（在成本点内）
                cost_data = stats_df[
                    (stats_df['difficulty'] == difficulty) & 
                    (stats_df['c_r'].isin(representative_costs))
                ]
                is_best_acc = argo_acc == cost_data['ARGO_accuracy_mean'].max() * 100
                is_best_ret = argo_ret == cost_data['ARGO_retrievals_mean'].min()
                
                table_rows.append({
                    'difficulty': difficulty.capitalize(),
                    'c_r': cost_val,
                    'argo_acc': argo_acc,
                    'argo_ret': argo_ret,
                    'ar_acc': ar_acc,
                    'ar_ret': ar_ret,
                    'reduction': reduction,
                    'is_best_acc': is_best_acc,
                    'is_best_ret': is_best_ret
                })
    
    # 生成LaTeX表格
    latex_table = []
    latex_table.append("\\begin{table}[h]")
    latex_table.append("\\centering")
    latex_table.append("\\caption{Performance Comparison Across Retrieval Cost Levels}")
    latex_table.append("\\label{tab:performance_comparison}")
    latex_table.append("\\begin{tabular}{lccccc}")
    latex_table.append("\\toprule")
    latex_table.append("Difficulty & $c_r$ & ARGO Acc. & ARGO Ret. & AR Acc.$^\\dagger$ & Reduction \\\\")
    latex_table.append("\\midrule")
    
    for row in table_rows:
        diff = row['difficulty']
        c_r = row['c_r']
        argo_acc = f"{row['argo_acc']:.1f}\\%"
        argo_ret = f"{row['argo_ret']:.2f}"
        ar_acc = f"{row['ar_acc']:.1f}\\%" if row['ar_acc'] is not None else "---"
        reduction = f"{row['reduction']:.1f}\\%"
        
        # 加粗最佳值
        if row['is_best_acc']:
            argo_acc = f"\\textbf{{{argo_acc}}}"
        if row['is_best_ret']:
            argo_ret = f"\\textbf{{{argo_ret}}}"
        
        latex_table.append(f"{diff} & {c_r:.3f} & {argo_acc} & {argo_ret} & {ar_acc} & {reduction} \\\\")
    
    latex_table.append("\\bottomrule")
    latex_table.append("\\end{tabular}")
    latex_table.append("\\begin{flushleft}\\footnotesize")
    latex_table.append("$^\\dagger$AR = Always-Retrieve baseline. Bold indicates best in category.")
    latex_table.append("\\end{flushleft}")
    latex_table.append("\\end{table}")
    
    # 保存LaTeX表格
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    latex_path = output_path / 'exp1_paper_table.tex'
    with open(latex_path, 'w') as f:
        f.write('\n'.join(latex_table))
    
    print(f"\n✓ LaTeX表格已保存: {latex_path}")
    
    # 同时生成Markdown格式（便于查看）
    md_table = []
    md_table.append("## Table I: Performance Comparison Across Retrieval Cost Levels\n")
    md_table.append("| Difficulty | $c_r$ | ARGO Acc. | ARGO Ret. | AR Acc.† | Reduction |")
    md_table.append("|:-----------|:-----:|:---------:|:---------:|:--------:|:---------:|")
    
    for row in table_rows:
        diff = row['difficulty']
        c_r = row['c_r']
        argo_acc = f"{row['argo_acc']:.1f}%"
        argo_ret = f"{row['argo_ret']:.2f}"
        ar_acc = f"{row['ar_acc']:.1f}%" if row['ar_acc'] is not None else "---"
        reduction = f"{row['reduction']:.1f}%"
        
        # 加粗最佳值
        if row['is_best_acc']:
            argo_acc = f"**{argo_acc}**"
        if row['is_best_ret']:
            argo_ret = f"**{argo_ret}**"
        
        md_table.append(f"| {diff} | {c_r:.3f} | {argo_acc} | {argo_ret} | {ar_acc} | {reduction} |")
    
    md_table.append("\n*†AR = Always-Retrieve baseline. Bold indicates best in category.*")
    
    md_path = output_path / 'exp1_paper_table.md'
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_table))
    
    print(f"✓ Markdown表格已保存: {md_path}")
    
    return latex_path, md_path


def main():
    parser = argparse.ArgumentParser(description='生成论文图表和表格')
    parser.add_argument('--stats-file', type=str, 
                       default='draw_figs/data/exp1_aggregated_from_csv_20260112_021552.csv',
                       help='聚合统计数据文件路径')
    parser.add_argument('--output-dir', type=str, default='figs',
                       help='输出目录')
    parser.add_argument('--c-p', type=float, default=0.02,
                       help='Phase transition阈值 (c_p)')
    
    args = parser.parse_args()
    
    # 加载数据
    stats_df = load_aggregated_data(args.stats_file)
    
    # 生成两面板图表
    fig_path, fig_path_pdf = plot_paper_figure_two_panel(stats_df, args.output_dir, args.c_p)
    
    # 生成表格
    latex_path, md_path = generate_paper_table(stats_df, args.output_dir, args.c_p)
    
    print("\n" + "="*80)
    print("✓ 所有文件生成完成!")
    print("="*80)
    print(f"\n生成的文件:")
    print(f"  - 图表PNG: {fig_path}")
    print(f"  - 图表PDF: {fig_path_pdf}")
    print(f"  - LaTeX表格: {latex_path}")
    print(f"  - Markdown表格: {md_path}")


if __name__ == '__main__':
    main()
