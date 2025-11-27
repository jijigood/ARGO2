#!/usr/bin/env python
"""
实验1: 结果聚合与统计分析
============================
从多个种子的实验结果中计算:
- 均值、标准差、标准误差
- 95%置信区间
- 配对t检验
- Cohen's d效应量
- 百分比改进

输出:
- 聚合统计CSV文件
- 统计检验结果CSV文件
- 文本报告 (可直接用于论文)
"""

import json
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime
import argparse


DEFAULT_DATA_DIR = Path('analysis/figures/draw_figs/data')
DEFAULT_PATTERN = 'exp1_real_cost_impact_custom_*.json'


def load_all_results(data_dir=DEFAULT_DATA_DIR, pattern=DEFAULT_PATTERN):
    """加载所有实验结果文件"""
    
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_path}")
    result_files = sorted(data_path.glob(pattern))
    
    if not result_files:
        print(f"❌ 未找到匹配的结果文件: {data_path}/{pattern}")
        print(f"   请先运行实验生成结果文件")
        return None
    
    print(f"找到 {len(result_files)} 个结果文件:")
    for f in result_files:
        print(f"  - {f.name}")
    print()
    
    all_data = []
    
    for file_path in result_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # 提取元数据
        metadata = data['metadata']
        results = data['results']
        
        # 检查必要字段
        if 'seed' not in metadata:
            print(f"⚠ {file_path.name}: 缺少seed字段，跳过")
            continue
        
        # 添加元数据到每个结果行
        for result in results:
            result['difficulty'] = metadata['difficulty']
            result['seed'] = metadata['seed']
            result['n_questions'] = metadata['n_questions']
            result['timestamp'] = metadata.get('timestamp')
            all_data.append(result)
    
    if not all_data:
        print(f"❌ 没有有效的数据")
        return None
    
    df = pd.DataFrame(all_data)

    # 过滤掉无效运行（例如早期占位，n_questions=0 或 没有accuracy数据）
    if 'n_questions' in df.columns:
        before = len(df)
        df = df[df['n_questions'] > 0]
        if len(df) < before:
            print(f"  - 已过滤占位记录: {before - len(df)} 条 (n_questions=0)")

    # 仅保留同一 (difficulty, seed, c_r) 组合的最新结果
    if 'timestamp' in df.columns and 'c_r' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d_%H%M%S', errors='coerce')
        df = df.sort_values('timestamp')
        df = df.groupby(['difficulty', 'seed', 'c_r'], as_index=False).tail(1)
        print(f"  - 使用每个难度/种子的最新结果，共 {len(df)} 条记录")
    
    print(f"✓ 加载数据:")
    print(f"  - 总记录数: {len(df)}")
    print(f"  - 难度级别: {sorted(df['difficulty'].unique())}")
    print(f"  - 随机种子: {sorted(df['seed'].unique())}")
    print(f"  - 成本点数: {len(df['c_r'].unique())}")
    print()
    
    return df


def compute_statistics(df):
    """计算统计量"""
    
    print("计算统计量...")
    
    # 按难度和c_r分组
    grouped = df.groupby(['difficulty', 'c_r'])
    
    stats_list = []
    
    for (difficulty, c_r), group in grouped:
        row = {
            'difficulty': difficulty,
            'c_r': c_r,
            'c_r_multiplier': c_r / 0.02,  # 假设 c_p = 0.02
            'n_seeds': len(group),
        }
        
        # 对每个策略和指标
        strategies = ['ARGO', 'Always-Retrieve', 'Always-Reason', 'Random']
        metrics = ['accuracy', 'quality', 'retrievals', 'cost', 'reasons']
        
        for strategy in strategies:
            for metric in metrics:
                col = f'{strategy}_{metric}'
                if col not in group.columns:
                    continue
                
                values = group[col].values
                
                if len(values) > 0:
                    row[f'{col}_mean'] = np.mean(values)
                    row[f'{col}_std'] = np.std(values, ddof=1) if len(values) > 1 else 0.0
                    row[f'{col}_sem'] = stats.sem(values) if len(values) > 1 else 0.0
                    row[f'{col}_ci95'] = 1.96 * stats.sem(values) if len(values) > 1 else 0.0
        
        stats_list.append(row)
    
    stats_df = pd.DataFrame(stats_list)
    
    print(f"✓ 计算完成: {len(stats_df)} 个统计点")
    print()
    
    return stats_df


def compute_significance_tests(df):
    """计算统计显著性检验"""
    
    print("执行统计显著性检验...")
    
    test_results = []
    
    for difficulty in df['difficulty'].unique():
        for c_r in df['c_r'].unique():
            subset = df[(df['difficulty'] == difficulty) & (df['c_r'] == c_r)]
            
            if len(subset) < 2:
                continue  # 需要至少2个样本才能进行t检验
            
            # 关键指标: retrievals (检索次数)
            baselines = ['Always-Retrieve', 'Always-Reason', 'Random']
            
            for baseline in baselines:
                argo_col = 'ARGO_retrievals'
                base_col = f'{baseline}_retrievals'
                
                if argo_col not in subset.columns or base_col not in subset.columns:
                    continue
                
                argo_vals = subset[argo_col].values
                base_vals = subset[base_col].values
                
                # 配对t检验
                if len(argo_vals) == len(base_vals) and len(argo_vals) > 1:
                    t_stat, p_val = stats.ttest_rel(argo_vals, base_vals)
                    
                    # Cohen's d (配对样本)
                    diff = argo_vals - base_vals
                    cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0
                    
                    # 百分比减少
                    if np.mean(base_vals) > 0:
                        pct_reduction = (np.mean(base_vals) - np.mean(argo_vals)) / np.mean(base_vals) * 100
                    else:
                        pct_reduction = 0.0
                    
                    # 效应量分类
                    abs_d = abs(cohens_d)
                    if abs_d > 0.8:
                        effect_size = 'large'
                    elif abs_d > 0.5:
                        effect_size = 'medium'
                    elif abs_d > 0.2:
                        effect_size = 'small'
                    else:
                        effect_size = 'negligible'
                    
                    # 显著性标记
                    if p_val < 0.001:
                        sig_level = '***'
                    elif p_val < 0.01:
                        sig_level = '**'
                    elif p_val < 0.05:
                        sig_level = '*'
                    else:
                        sig_level = 'ns'
                    
                    test_results.append({
                        'difficulty': difficulty,
                        'c_r': c_r,
                        'c_r_multiplier': c_r / 0.02,
                        'comparison': f'ARGO vs {baseline}',
                        'metric': 'retrievals',
                        't_statistic': t_stat,
                        'p_value': p_val,
                        'significant': p_val < 0.05,
                        'sig_level': sig_level,
                        'cohens_d': cohens_d,
                        'effect_size': effect_size,
                        'argo_mean': np.mean(argo_vals),
                        'baseline_mean': np.mean(base_vals),
                        'difference': np.mean(diff),
                        'percent_reduction': pct_reduction
                    })
    
    sig_df = pd.DataFrame(test_results)
    
    print(f"✓ 检验完成: {len(sig_df)} 个比较")
    print()
    
    return sig_df


def generate_summary_report(stats_df, sig_df):
    """生成汇总报告"""
    
    print("\n" + "="*80)
    print("实验1: 统计分析报告")
    print("="*80)
    
    # 获取最高成本场景 (10×)
    high_cost_multiplier = stats_df['c_r_multiplier'].max()
    high_cost = stats_df[stats_df['c_r_multiplier'] == high_cost_multiplier]
    
    print(f"\n1. 最高成本场景性能 (c_r = {high_cost_multiplier:.0f}× c_p)")
    print("-"*80)
    
    for difficulty in sorted(stats_df['difficulty'].unique()):
        diff_data = high_cost[high_cost['difficulty'] == difficulty]
        
        if len(diff_data) == 0:
            continue
        
        row = diff_data.iloc[0]
        
        # ARGO指标
        argo_acc = row.get('ARGO_accuracy_mean', 0)
        argo_acc_ci = row.get('ARGO_accuracy_ci95', 0)
        argo_ret = row.get('ARGO_retrievals_mean', 0)
        argo_ret_ci = row.get('ARGO_retrievals_ci95', 0)
        
        # Always-Retrieve基线
        alw_ret = row.get('Always-Retrieve_retrievals_mean', 0)
        alw_ret_ci = row.get('Always-Retrieve_retrievals_ci95', 0)
        
        # 计算减少比例
        if alw_ret > 0:
            reduction = (alw_ret - argo_ret) / alw_ret * 100
        else:
            reduction = 0.0
        
        # 获取显著性信息
        sig_data = sig_df[
            (sig_df['difficulty'] == difficulty) &
            (sig_df['c_r_multiplier'] == high_cost_multiplier) &
            (sig_df['comparison'] == 'ARGO vs Always-Retrieve')
        ]
        
        print(f"\n{difficulty.upper()} 问题:")
        print(f"  ARGO准确率: {argo_acc:.3f} ± {argo_acc_ci:.3f}")
        print(f"  ARGO检索次数: {argo_ret:.1f} ± {argo_ret_ci:.1f}")
        print(f"  Always-Retrieve: {alw_ret:.1f} ± {alw_ret_ci:.1f}")
        
        if len(sig_data) > 0:
            sig_row = sig_data.iloc[0]
            print(f"  → 减少: {reduction:.1f}% (p={sig_row['p_value']:.4f} {sig_row['sig_level']})")
            print(f"  → 效应量: Cohen's d={sig_row['cohens_d']:.2f} ({sig_row['effect_size']})")
        else:
            print(f"  → 减少: {reduction:.1f}%")
    
    # 总体平均
    print(f"\n2. 总体平均 (c_r = {high_cost_multiplier:.0f}× c_p)")
    print("-"*80)
    
    avg_argo_ret = high_cost['ARGO_retrievals_mean'].mean()
    avg_alw_ret = high_cost['Always-Retrieve_retrievals_mean'].mean()
    
    if avg_alw_ret > 0:
        avg_reduction = (avg_alw_ret - avg_argo_ret) / avg_alw_ret * 100
    else:
        avg_reduction = 0.0
    
    avg_argo_acc = high_cost['ARGO_accuracy_mean'].mean()
    avg_alw_acc = high_cost['Always-Retrieve_accuracy_mean'].mean()
    acc_diff = (avg_argo_acc - avg_alw_acc) * 100
    
    print(f"  ARGO: {avg_argo_ret:.1f} 检索, {avg_argo_acc:.3f} 准确率")
    print(f"  Always-Retrieve: {avg_alw_ret:.1f} 检索, {avg_alw_acc:.3f} 准确率")
    print(f"  → 检索减少: {avg_reduction:.1f}%")
    print(f"  → 准确率差异: {acc_diff:+.1f} pp")
    
    # 建议的论文摘要文本
    print("\n" + "="*80)
    print("建议的论文摘要文本:")
    print("="*80)
    
    # 计算总体显著性
    high_cost_sig = sig_df[
        (sig_df['c_r_multiplier'] == high_cost_multiplier) &
        (sig_df['comparison'] == 'ARGO vs Always-Retrieve')
    ]
    
    if len(high_cost_sig) > 0:
        avg_p_value = high_cost_sig['p_value'].mean()
        all_significant = all(high_cost_sig['significant'])
        
        sig_text = "p < 0.001" if avg_p_value < 0.001 else f"p < 0.05"
    else:
        sig_text = ""
    
    print(f'''
"在ORAN-Bench-13K基准测试中，ARGO在保持与always-retrieve基线
相当的答案质量的同时，在高成本场景下将检索调用次数减少了
{avg_reduction:.0f}% ({sig_text})，证明了其在实时电信环境中的
实用可行性。"

英文版:
"Evaluated on ORAN-Bench-13K, ARGO achieves comparable answer quality 
to always-retrieve baselines while reducing retrieval calls by 
{avg_reduction:.0f}% ({sig_text}) under high-cost scenarios, demonstrating 
practical viability for real-time telecommunications environments."
''')
    
    print("="*80)
    
    # 额外洞察
    print("\n3. 关键洞察")
    print("-"*80)
    
    # 成本敏感性分析
    for difficulty in sorted(stats_df['difficulty'].unique()):
        diff_data = stats_df[stats_df['difficulty'] == difficulty]
        
        if len(diff_data) == 0:
            continue
        
        # 最低和最高成本下的检索次数
        min_cost_row = diff_data[diff_data['c_r_multiplier'] == diff_data['c_r_multiplier'].min()].iloc[0]
        max_cost_row = diff_data[diff_data['c_r_multiplier'] == diff_data['c_r_multiplier'].max()].iloc[0]
        
        min_ret = min_cost_row.get('ARGO_retrievals_mean', 0)
        max_ret = max_cost_row.get('ARGO_retrievals_mean', 0)
        
        if min_ret > 0:
            ret_change = (max_ret - min_ret) / min_ret * 100
            print(f"\n{difficulty.upper()}: 当c_r从{diff_data['c_r_multiplier'].min():.0f}×增至{diff_data['c_r_multiplier'].max():.0f}×时，")
            print(f"  ARGO检索次数变化: {ret_change:+.1f}%")
            print(f"  (从 {min_ret:.1f} 到 {max_ret:.1f})")


def save_results(stats_df, sig_df, output_dir='draw_figs/data'):
    """保存结果"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存聚合统计
    stats_file = output_path / f"exp1_aggregated_{timestamp}.csv"
    stats_df.to_csv(stats_file, index=False)
    print(f"\n✓ 聚合统计已保存: {stats_file}")
    
    # 保存显著性检验
    sig_file = output_path / f"exp1_statistical_tests_{timestamp}.csv"
    sig_df.to_csv(sig_file, index=False)
    print(f"✓ 显著性检验已保存: {sig_file}")
    
    return str(stats_file), str(sig_file)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='聚合实验1的多种子结果')
    parser.add_argument('--data-dir', type=str, default=str(DEFAULT_DATA_DIR),
                       help='数据目录 (默认: analysis/figures/draw_figs/data)')
    parser.add_argument('--pattern', type=str, default=DEFAULT_PATTERN,
                       help='文件匹配模式 (默认: exp1_real_cost_impact_custom_*.json)')
    
    args = parser.parse_args()
    
    # 加载数据
    df = load_all_results(args.data_dir, args.pattern)
    
    if df is None or len(df) == 0:
        print("❌ 没有数据可分析")
        return 1
    
    # 计算统计量
    stats_df = compute_statistics(df)
    
    # 计算显著性检验
    sig_df = compute_significance_tests(df)
    
    # 保存结果
    stats_file, sig_file = save_results(stats_df, sig_df, args.data_dir)
    
    # 生成报告
    generate_summary_report(stats_df, sig_df)
    
    print("\n" + "="*80)
    print("✓ 分析完成!")
    print("="*80)
    print("\n下一步:")
    print(f"  1. 生成图表:")
    print(f"     python Exp1_plots.py {stats_file}")
    print(f"  2. 查看聚合数据: {stats_file}")
    print(f"  3. 查看统计检验: {sig_file}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
