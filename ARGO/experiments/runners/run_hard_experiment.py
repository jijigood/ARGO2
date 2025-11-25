"""
Phase 4.3 - Hard Difficulty Experiment (20 queries)
===================================================

专注于Hard难度问题，更能体现ARGO策略的差异

配置:
- 20 questions from fin_H.json (Hard difficulty)
- 2 strategies: MDP-Guided vs Always-Reason
- Optimized parameters for speed
"""

import os
import sys
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import ARGO_System, AlwaysReasonStrategy
from run_small_scale_experiment import ORANBenchLoader, MCQAEvaluator


def run_hard_experiment(
    strategy_name: str,
    system: ARGO_System,
    questions: List[Dict],
    output_dir: str
) -> pd.DataFrame:
    """运行Hard难度实验"""
    
    print(f"\n{'='*80}")
    print(f"Running: {strategy_name} (Hard Difficulty)")
    print('='*80)
    
    results = []
    evaluator = MCQAEvaluator()
    
    start_time = time.time()
    
    for i, item in enumerate(tqdm(questions, desc=strategy_name)):
        # 格式化问题
        formatted_question = evaluator.format_mcqa_question(item)
        
        # 运行系统
        query_start = time.time()
        
        try:
            answer, history, metadata = system.answer_question(
                formatted_question,
                return_history=True
            )
            
            query_time = time.time() - query_start
            
            # 提取预测答案
            predicted = evaluator.extract_answer(answer)
            
            # 评估
            is_correct = evaluator.evaluate(predicted, item['answer'])
            
            # 记录结果
            result = {
                'strategy': strategy_name,
                'question_id': i,
                'difficulty': 'H',
                'question': item['question'][:150],
                'ground_truth': item['answer'],
                'predicted': predicted,
                'is_correct': is_correct,
                'total_steps': metadata['total_steps'],
                'retrieve_count': metadata['retrieve_count'],
                'reason_count': metadata['reason_count'],
                'final_uncertainty': metadata['final_uncertainty'],
                'query_time_seconds': query_time,
                'answer_preview': answer[:200]
            }
            
            results.append(result)
            
            # 实时显示进度
            if (i + 1) % 5 == 0 or (i + 1) == len(questions):
                correct_so_far = sum(r['is_correct'] for r in results)
                acc_so_far = correct_so_far / len(results) * 100
                avg_time = np.mean([r['query_time_seconds'] for r in results])
                
                print(f"  Progress: {i+1}/{len(questions)} | "
                      f"Accuracy: {acc_so_far:.1f}% | "
                      f"Avg Time: {avg_time:.1f}s/query")
            
            # 定期保存
            if (i + 1) % 5 == 0:
                df_temp = pd.DataFrame(results)
                df_temp.to_csv(
                    os.path.join(output_dir, f'{strategy_name}_temp.csv'),
                    index=False
                )
        
        except Exception as e:
            logger.error(f"Failed on question {i}: {e}")
            
            query_time = time.time() - query_start
            
            # 记录失败
            result = {
                'strategy': strategy_name,
                'question_id': i,
                'difficulty': 'H',
                'question': item['question'][:150],
                'ground_truth': item['answer'],
                'predicted': 'ERROR',
                'is_correct': False,
                'total_steps': 0,
                'retrieve_count': 0,
                'reason_count': 0,
                'final_uncertainty': 1.0,
                'query_time_seconds': query_time,
                'answer_preview': f'ERROR: {str(e)[:100]}'
            }
            results.append(result)
    
    total_time = time.time() - start_time
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 统计
    accuracy = df['is_correct'].mean()
    avg_steps = df['total_steps'].mean()
    avg_time = df['query_time_seconds'].mean()
    
    print(f"\n{strategy_name} Final Results:")
    print(f"  ✓ Accuracy: {accuracy*100:.1f}% ({df['is_correct'].sum()}/{len(df)})")
    print(f"  ⏱  Avg Time: {avg_time:.1f}s/query")
    print(f"  📊 Avg Steps: {avg_steps:.2f}")
    print(f"  🔄 Avg Retrieves: {df['retrieve_count'].mean():.2f}")
    print(f"  💭 Avg Reasons: {df['reason_count'].mean():.2f}")
    print(f"  ⏰ Total Time: {total_time/60:.1f} minutes")
    
    # 保存最终结果
    df.to_csv(os.path.join(output_dir, f'{strategy_name}_results.csv'), index=False)
    
    return df


def analyze_hard_results(all_results: Dict[str, pd.DataFrame], output_dir: str):
    """分析Hard难度结果"""
    
    print("\n" + "="*80)
    print("Hard Difficulty Experiment Analysis")
    print("="*80)
    
    # 1. 汇总统计
    summary = []
    
    for strategy_name, df in all_results.items():
        stats = {
            'Strategy': strategy_name,
            'Accuracy (%)': df['is_correct'].mean() * 100,
            'Correct': df['is_correct'].sum(),
            'Total': len(df),
            'Avg Time (s)': df['query_time_seconds'].mean(),
            'Avg Steps': df['total_steps'].mean(),
            'Avg Retrieves': df['retrieve_count'].mean(),
            'Avg Reasons': df['reason_count'].mean(),
            'Total Time (min)': df['query_time_seconds'].sum() / 60
        }
        summary.append(stats)
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(output_dir, 'summary_hard.csv'), index=False)
    
    print("\n📊 Summary Statistics (Hard Difficulty):")
    print(summary_df.to_string(index=False))
    
    # 2. 对比分析
    if len(all_results) == 2:
        strategies = list(all_results.keys())
        df1 = all_results[strategies[0]]
        df2 = all_results[strategies[1]]
        
        print(f"\n🔍 Comparison: {strategies[0]} vs {strategies[1]}")
        print("-" * 80)
        
        acc_diff = df1['is_correct'].mean() - df2['is_correct'].mean()
        time_diff = df1['query_time_seconds'].mean() - df2['query_time_seconds'].mean()
        
        print(f"Accuracy Difference: {acc_diff*100:+.1f}% "
              f"({strategies[0]}: {df1['is_correct'].mean()*100:.1f}% vs "
              f"{strategies[1]}: {df2['is_correct'].mean()*100:.1f}%)")
        
        print(f"Time Difference: {time_diff:+.1f}s "
              f"({strategies[0]}: {df1['query_time_seconds'].mean():.1f}s vs "
              f"{strategies[1]}: {df2['query_time_seconds'].mean():.1f}s)")
        
        # 显著性分析
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(df1['is_correct'], df2['is_correct'])
        
        print(f"\nStatistical Significance (t-test):")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"  ✅ Difference is statistically significant (p < 0.05)")
        else:
            print(f"  ⚠️  Difference is NOT statistically significant (p >= 0.05)")
    
    # 3. 可视化
    visualize_hard_results(all_results, summary_df, output_dir)
    
    # 4. 生成LaTeX表格
    generate_hard_latex(summary_df, output_dir)
    
    print(f"\n✅ All results saved to {output_dir}/")


def visualize_hard_results(all_results: Dict[str, pd.DataFrame], 
                           summary_df: pd.DataFrame,
                           output_dir: str):
    """可视化Hard难度结果"""
    
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    strategies = list(all_results.keys())
    colors = ['#3498db', '#e74c3c']
    
    # 1. 准确率对比
    ax1 = fig.add_subplot(gs[0, 0])
    accuracies = summary_df['Accuracy (%)'].values
    bars = ax1.bar(range(len(strategies)), accuracies, color=colors, 
                   edgecolor='black', alpha=0.8, width=0.6)
    ax1.set_xticks(range(len(strategies)))
    ax1.set_xticklabels(strategies, fontsize=11)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy on Hard Questions', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 2. 平均时间对比
    ax2 = fig.add_subplot(gs[0, 1])
    avg_times = summary_df['Avg Time (s)'].values
    bars = ax2.bar(range(len(strategies)), avg_times, color=colors, 
                   edgecolor='black', alpha=0.8, width=0.6)
    ax2.set_xticks(range(len(strategies)))
    ax2.set_xticklabels(strategies, fontsize=11)
    ax2.set_ylabel('Avg Time (seconds)', fontsize=12)
    ax2.set_title('Efficiency Comparison', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for i, (bar, t) in enumerate(zip(bars, avg_times)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{t:.1f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 3. Retrieve vs Reason
    ax3 = fig.add_subplot(gs[0, 2])
    x = np.arange(len(strategies))
    width = 0.35
    
    retrieves = summary_df['Avg Retrieves'].values
    reasons = summary_df['Avg Reasons'].values
    
    ax3.bar(x - width/2, retrieves, width, label='Retrieve', 
            color='#3498db', alpha=0.8, edgecolor='black')
    ax3.bar(x + width/2, reasons, width, label='Reason', 
            color='#e74c3c', alpha=0.8, edgecolor='black')
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(strategies, fontsize=11)
    ax3.set_ylabel('Average Count', fontsize=12)
    ax3.set_title('Action Distribution', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4-5. 每个策略的逐题准确率
    for idx, strategy in enumerate(strategies):
        ax = fig.add_subplot(gs[1, idx])
        df = all_results[strategy]
        
        x = range(len(df))
        y = df['is_correct'].astype(int)
        
        ax.scatter(x, y, alpha=0.6, s=80, color=colors[idx], edgecolor='black')
        ax.plot(x, pd.Series(y).rolling(5, min_periods=1).mean(), 
                color=colors[idx], linewidth=2, label='Rolling Avg (5)')
        
        ax.set_xlabel('Question Index', fontsize=11)
        ax.set_ylabel('Correct (1) / Wrong (0)', fontsize=11)
        ax.set_title(f'{strategy} - Per-Question Results', fontsize=12, fontweight='bold')
        ax.set_ylim(-0.1, 1.1)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    
    # 6. 准确率vs时间散点图
    ax6 = fig.add_subplot(gs[1, 2])
    
    for i, strategy in enumerate(strategies):
        df = all_results[strategy]
        # 每个query的时间和是否正确
        correct = df[df['is_correct'] == True]
        incorrect = df[df['is_correct'] == False]
        
        ax6.scatter(correct['query_time_seconds'], 
                   np.ones(len(correct)) + i*0.1, 
                   s=60, alpha=0.6, color=colors[i], marker='o', label=f'{strategy} (Correct)')
        ax6.scatter(incorrect['query_time_seconds'], 
                   np.zeros(len(incorrect)) + i*0.1, 
                   s=60, alpha=0.6, color=colors[i], marker='x', label=f'{strategy} (Wrong)')
    
    ax6.set_xlabel('Query Time (seconds)', fontsize=11)
    ax6.set_ylabel('Correctness', fontsize=11)
    ax6.set_yticks([0, 1])
    ax6.set_yticklabels(['Wrong', 'Correct'])
    ax6.set_title('Time vs Correctness', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=8, loc='upper right')
    ax6.grid(alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'hard_experiment_results.png'), 
                dpi=300, bbox_inches='tight')
    print(f"✅ Saved visualization to {output_dir}/hard_experiment_results.png")
    plt.close()


def generate_hard_latex(summary_df: pd.DataFrame, output_dir: str):
    """生成LaTeX表格"""
    
    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{ARGO Strategy Comparison on Hard ORAN-Bench Questions (N=20)}",
        "\\label{tab:argo_hard_comparison}",
        "\\begin{tabular}{lcccccc}",
        "\\hline",
        "\\textbf{Strategy} & \\textbf{Accuracy} & \\textbf{Correct/Total} & \\textbf{Avg Time (s)} & \\textbf{Avg Steps} & \\textbf{Retrieves} & \\textbf{Reasons} \\\\",
        "\\hline",
    ]
    
    for _, row in summary_df.iterrows():
        strategy = row['Strategy'].replace('_', '\\_')
        acc = row['Accuracy (%)']
        correct = int(row['Correct'])
        total = int(row['Total'])
        time_s = row['Avg Time (s)']
        steps = row['Avg Steps']
        retrieves = row['Avg Retrieves']
        reasons = row['Avg Reasons']
        
        line = f"{strategy} & {acc:.1f}\\% & {correct}/{total} & {time_s:.1f} & {steps:.2f} & {retrieves:.2f} & {reasons:.2f} \\\\"
        latex_lines.append(line)
    
    latex_lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    latex_content = "\n".join(latex_lines)
    
    with open(os.path.join(output_dir, 'hard_results_table.tex'), 'w') as f:
        f.write(latex_content)
    
    print(f"✅ Saved LaTeX table to {output_dir}/hard_results_table.tex")


def main():
    """主函数"""
    
    print("="*80)
    print("ARGO Phase 4.3 - Hard Difficulty Experiment (20 queries)")
    print("="*80)
    
    # 配置
    N_QUESTIONS = 20
    DIFFICULTY = 'H'  # Hard only
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    OUTPUT_DIR = "results/phase4.3_hard"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 加载Hard难度数据
    print(f"\n1. Loading ORAN-Bench-13K (Hard difficulty only)...")
    loader = ORANBenchLoader()
    
    # 只从Hard中采样
    questions = loader.sample(n=N_QUESTIONS, difficulties=['H'], random_seed=42)
    
    print(f"✅ Sampled {len(questions)} Hard questions")
    
    # 2. 加载模型
    print(f"\n2. Loading model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print("✅ Model loaded")
    
    # 3. 创建2种策略（对比最明显）
    print("\n3. Initializing strategies...")
    
    strategies = {}
    
    # MDP-Guided (ARGO)
    print("  - Creating MDP-Guided strategy...")
    argo_system = ARGO_System(
        model, tokenizer,
        use_mdp=True,
        retriever_mode='mock',
        max_steps=4,  # 减少步数加快速度
        verbose=False
    )
    # 优化参数
    argo_system.decomposer.max_subquery_length = 50
    argo_system.synthesizer.max_answer_length = 150
    argo_system.decomposer.temperature = 0.3
    argo_system.synthesizer.temperature = 0.1
    
    strategies['MDP_Guided'] = argo_system
    
    # Always Reason (Pure LLM baseline)
    print("  - Creating Always-Reason baseline...")
    always_system = AlwaysReasonStrategy(
        model, tokenizer,
        retriever_mode='mock',
        max_steps=4,
        verbose=False
    )
    # 相同优化参数
    always_system.decomposer.max_subquery_length = 50
    always_system.synthesizer.max_answer_length = 150
    always_system.decomposer.temperature = 0.3
    always_system.synthesizer.temperature = 0.1
    
    strategies['Always_Reason'] = always_system
    
    print(f"✅ Created {len(strategies)} strategies")
    
    # 4. 运行实验
    print("\n4. Running Hard difficulty experiments...")
    print("   (Estimated time: ~6-8 minutes for 20 queries × 2 strategies)")
    
    all_results = {}
    
    for strategy_name, system in strategies.items():
        try:
            df = run_hard_experiment(strategy_name, system, questions, OUTPUT_DIR)
            all_results[strategy_name] = df
        except Exception as e:
            logger.error(f"Failed to run {strategy_name}: {e}", exc_info=True)
            print(f"❌ {strategy_name} failed: {e}")
    
    # 5. 分析结果
    if len(all_results) > 0:
        print("\n5. Analyzing results...")
        analyze_hard_results(all_results, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("HARD DIFFICULTY EXPERIMENT COMPLETED! ✅")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("  - MDP_Guided_results.csv: MDP strategy详细结果")
    print("  - Always_Reason_results.csv: Always-Reason baseline详细结果")
    print("  - summary_hard.csv: 汇总统计表")
    print("  - hard_experiment_results.png: 可视化图表")
    print("  - hard_results_table.tex: LaTeX表格")
    
    # 显示关键结论
    if len(all_results) == 2:
        strategies_list = list(all_results.keys())
        df1 = all_results[strategies_list[0]]
        df2 = all_results[strategies_list[1]]
        
        acc1 = df1['is_correct'].mean() * 100
        acc2 = df2['is_correct'].mean() * 100
        
        print(f"\n🎯 Key Findings:")
        print(f"  - {strategies_list[0]}: {acc1:.1f}% accuracy")
        print(f"  - {strategies_list[1]}: {acc2:.1f}% accuracy")
        print(f"  - Difference: {abs(acc1-acc2):.1f}%")
        
        if acc1 > acc2:
            print(f"  ✅ {strategies_list[0]} outperforms {strategies_list[1]} on Hard questions!")
        elif acc2 > acc1:
            print(f"  ✅ {strategies_list[1]} outperforms {strategies_list[0]} on Hard questions!")
        else:
            print(f"  → Similar performance")


if __name__ == "__main__":
    main()
