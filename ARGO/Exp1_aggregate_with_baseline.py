#!/usr/bin/env python
"""
聚合ARGO和所有Baseline数据并生成更新的图表
==========================================
1. 加载ARGO聚合数据
2. 聚合所有baseline数据 (Always-Retrieve, Always-Reason, Random)
3. 合并数据
4. 生成更新的图表和表格
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
from scipy import stats
import glob


def aggregate_single_baseline(data_dir: Path, baseline_name: str, file_pattern: str):
    """聚合单个基线策略的数据"""
    
    all_baseline_files = sorted(data_dir.glob(file_pattern))
    
    if not all_baseline_files:
        print(f"  ⚠️  未找到 {baseline_name} 数据文件")
        return None
    
    # 按配置分组，只使用每个配置的最新文件
    config_files = {}
    for file_path in all_baseline_files:
        filename = file_path.stem
        parts = filename.split('_')
        
        difficulty = None
        seed = None
        
        for part in parts:
            if part in ['easy', 'medium', 'hard']:
                difficulty = part
            elif part.startswith('seed'):
                try:
                    seed = int(part[4:])
                except ValueError:
                    pass
        
        if difficulty and seed:
            key = f"{difficulty}_seed{seed}"
            if key not in config_files or file_path.stat().st_mtime > config_files[key].stat().st_mtime:
                config_files[key] = file_path
    
    baseline_files = list(config_files.values())
    print(f"  找到 {len(baseline_files)} 个 {baseline_name} 文件")
    
    all_baseline_data = []
    
    for file_path in baseline_files:
        try:
            filename = file_path.stem
            parts = filename.split('_')
            
            difficulty = None
            seed = None
            
            for part in parts:
                if part in ['easy', 'medium', 'hard']:
                    difficulty = part
                elif part.startswith('seed'):
                    try:
                        seed = int(part[4:])
                    except ValueError:
                        pass
            
            if difficulty is None or seed is None:
                continue
            
            df = pd.read_csv(file_path)
            
            accuracy = df['correct'].mean() if 'correct' in df.columns else 0.0
            n_questions = len(df)
            n_correct = df['correct'].sum() if 'correct' in df.columns else 0
            retrievals = df['retrieve_count'].mean() if 'retrieve_count' in df.columns else 0.0
            
            all_baseline_data.append({
                'difficulty': difficulty,
                'seed': seed,
                'accuracy': accuracy,
                'retrievals': retrievals,
                'n_questions': n_questions,
                'n_correct': n_correct
            })
            
        except Exception as e:
            print(f"  ❌ 处理文件 {file_path.name} 时出错: {e}")
            continue
    
    if not all_baseline_data:
        return None
    
    baseline_df = pd.DataFrame(all_baseline_data)
    
    # 按难度聚合
    baseline_stats_list = []
    
    for difficulty in baseline_df['difficulty'].unique():
        diff_data = baseline_df[baseline_df['difficulty'] == difficulty]
        
        all_correct = diff_data['n_correct'].sum()
        all_total = diff_data['n_questions'].sum()
        overall_accuracy = all_correct / all_total if all_total > 0 else 0.0
        avg_retrievals = diff_data['retrievals'].mean()
        
        # 计算置信区间 (Wilson score interval)
        if all_total > 0:
            p = overall_accuracy
            n = all_total
            z = 1.96
            denominator = 1 + z**2 / n
            centre_adjusted_probability = (p + z**2 / (2 * n)) / denominator
            adjusted_standard_deviation = np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
            lower_bound = centre_adjusted_probability - z * adjusted_standard_deviation
            upper_bound = centre_adjusted_probability + z * adjusted_standard_deviation
            ci95 = (upper_bound - lower_bound) / 2
        else:
            ci95 = 0.0
        
        baseline_stats_list.append({
            'difficulty': difficulty,
            'accuracy_mean': overall_accuracy,
            'accuracy_ci95': ci95,
            'retrievals_mean': avg_retrievals,
            'retrievals_ci95': 0.0,  # 简化处理
            'total_questions': all_total,
            'total_correct': all_correct
        })
    
    return pd.DataFrame(baseline_stats_list)


def aggregate_baseline_data(data_dir='draw_figs/data'):
    """聚合所有baseline数据"""
    
    print("="*80)
    print("聚合所有 Baseline 数据")
    print("="*80)
    
    data_path = Path(data_dir)
    
    # 定义所有基线策略及其文件模式
    baselines = {
        'Always-Retrieve': 'exp1_baseline_always_retrieve_*.csv',
        'Always-Reason': 'exp1_baseline_always_reason_*.csv',
        'Random': 'exp1_baseline_random_*.csv',
        'Direct-LLM': 'exp1_baseline_direct_llm_*.csv'
    }
    
    all_stats = {}
    
    for baseline_name, pattern in baselines.items():
        print(f"\n处理 {baseline_name}...")
        stats = aggregate_single_baseline(data_path, baseline_name, pattern)
        if stats is not None and len(stats) > 0:
            all_stats[baseline_name] = stats
            print(f"  ✓ {baseline_name}: {sorted(stats['difficulty'].unique())}")
            for _, row in stats.iterrows():
                print(f"    - {row['difficulty']}: {row['accuracy_mean']*100:.1f}% ({row['total_correct']}/{row['total_questions']})")
    
    if not all_stats:
        print("\n❌ 没有找到任何有效的baseline数据")
        return None
    
    print(f"\n✓ Baseline数据聚合完成")
    print(f"  找到的基线: {list(all_stats.keys())}")
    
    return all_stats


def merge_with_argo_data(argo_file, baseline_stats_dict, output_dir='draw_figs/data'):
    """将所有baseline数据合并到ARGO聚合数据中"""
    
    print("\n" + "="*80)
    print("合并ARGO和所有Baseline数据")
    print("="*80)
    
    # 加载ARGO数据
    argo_df = pd.read_csv(argo_file)
    print(f"✓ 加载ARGO数据: {len(argo_df)} 行")
    
    # 基线名称映射
    baseline_column_map = {
        'Always-Retrieve': 'Always-Retrieve',
        'Always-Reason': 'Always-Reason',
        'Random': 'Random',
        'Direct-LLM': 'Direct-LLM'
    }
    
    # 为每个基线添加列 (初始化为 NaN)
    for baseline_name in baseline_column_map.values():
        argo_df[f'{baseline_name}_accuracy_mean'] = np.nan
        argo_df[f'{baseline_name}_accuracy_ci95'] = np.nan
        argo_df[f'{baseline_name}_retrievals_mean'] = np.nan
        argo_df[f'{baseline_name}_retrievals_ci95'] = np.nan
    
    # 为每个难度添加所有baseline数据
    for idx, row in argo_df.iterrows():
        difficulty = row['difficulty']
        
        for baseline_name, column_prefix in baseline_column_map.items():
            if baseline_name in baseline_stats_dict:
                baseline_stats = baseline_stats_dict[baseline_name]
                baseline_row = baseline_stats[baseline_stats['difficulty'] == difficulty]
                
                if len(baseline_row) > 0:
                    argo_df.at[idx, f'{column_prefix}_accuracy_mean'] = baseline_row.iloc[0]['accuracy_mean']
                    argo_df.at[idx, f'{column_prefix}_accuracy_ci95'] = baseline_row.iloc[0]['accuracy_ci95']
                    argo_df.at[idx, f'{column_prefix}_retrievals_mean'] = baseline_row.iloc[0]['retrievals_mean']
                    argo_df.at[idx, f'{column_prefix}_retrievals_ci95'] = baseline_row.iloc[0]['retrievals_ci95']
    
    # 打印合并结果摘要
    print("\n合并结果摘要:")
    for baseline_name in baseline_column_map.values():
        col = f'{baseline_name}_accuracy_mean'
        if col in argo_df.columns:
            non_null = argo_df[col].notna().sum()
            if non_null > 0:
                avg_acc = argo_df[col].dropna().mean()
                print(f"  {baseline_name}: {non_null} 行数据, 平均准确率 {avg_acc*100:.1f}%")
            else:
                print(f"  {baseline_name}: 无数据")
    
    # 保存合并后的数据
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    merged_file = output_path / f"exp1_aggregated_with_baseline_{timestamp}.csv"
    argo_df.to_csv(merged_file, index=False)
    
    print(f"\n✓ 合并数据已保存: {merged_file}")
    
    return argo_df, merged_file


def main():
    parser = argparse.ArgumentParser(description='聚合ARGO和所有Baseline数据')
    parser.add_argument('--argo-file', type=str,
                       default=None,
                       help='ARGO聚合数据文件路径 (默认自动查找最新)')
    parser.add_argument('--data-dir', type=str, default='draw_figs/data',
                       help='数据目录')
    parser.add_argument('--output-dir', type=str, default='draw_figs/data',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 如果未指定ARGO文件，自动查找最新的
    if args.argo_file is None:
        data_path = Path(args.data_dir)
        argo_files = sorted(data_path.glob('exp1_aggregated_from_csv_*.csv'), 
                           key=lambda x: x.stat().st_mtime, reverse=True)
        if argo_files:
            args.argo_file = str(argo_files[0])
            print(f"✓ 自动选择最新ARGO文件: {args.argo_file}")
        else:
            print("❌ 未找到ARGO聚合数据文件")
            print("   请先运行: python experiments/runners/aggregate_and_plot_from_csv.py")
            return
    
    # 聚合所有baseline数据
    baseline_stats_dict = aggregate_baseline_data(args.data_dir)
    
    if baseline_stats_dict is None:
        print("\n⚠️  未找到baseline数据，但仍将使用ARGO数据生成文件")
        baseline_stats_dict = {}
    
    # 合并数据
    merged_df, merged_file = merge_with_argo_data(args.argo_file, baseline_stats_dict, args.output_dir)
    
    print("\n" + "="*80)
    print("✓ 数据聚合和合并完成!")
    print("="*80)
    print(f"\n生成的文件:")
    print(f"  - 合并数据: {merged_file}")
    
    # 打印最终统计
    print("\n最终数据摘要:")
    print(f"  - 总行数: {len(merged_df)}")
    print(f"  - 难度级别: {sorted(merged_df['difficulty'].unique()) if 'difficulty' in merged_df.columns else 'N/A'}")
    print(f"  - 成本点数: {merged_df['c_r'].nunique() if 'c_r' in merged_df.columns else 'N/A'}")
    
    baselines_with_data = []
    for baseline in ['Always-Retrieve', 'Always-Reason', 'Random', 'Direct-LLM']:
        col = f'{baseline}_accuracy_mean'
        if col in merged_df.columns and merged_df[col].notna().any():
            baselines_with_data.append(baseline)
    
    if baselines_with_data:
        print(f"  - 包含基线: {baselines_with_data}")
    else:
        print("  - ⚠️  无基线数据 (需要运行基线实验)")
    
    print(f"\n下一步:")
    print(f"  1. 生成论文图表: python Exp1_paper_figure.py --stats-file {merged_file}")
    print(f"  2. 或使用默认绘图: python Exp1_plots.py {merged_file}")


if __name__ == '__main__':
    main()
