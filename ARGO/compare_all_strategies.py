"""
Compare All Strategies - Phase 4.1
===================================

对比4种策略在相同问题上的表现：
1. MDP-Guided: ARGO_System (use_mdp=True)
2. Fixed-Threshold: FixedThresholdStrategy
3. Always-Reason: AlwaysReasonStrategy  
4. Random: RandomStrategy

评估指标：
- 准确性: Answer quality (需要人工评估或参考答案)
- 效率: Total steps, Time per question
- 成本: Retrieve actions, Reason actions
- 鲁棒性: 对检索失败的容忍度

运行方式:
    python compare_all_strategies.py
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import time

# 设置日志
logging.basicConfig(
    level=logging.WARNING,  # 减少输出
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import (
    ARGO_System,
    AlwaysReasonStrategy,
    RandomStrategy,
    FixedThresholdStrategy
)


# 测试问题集
TEST_QUESTIONS = [
    "What are the latency requirements for O-RAN fronthaul?",
    "How does O-RAN handle network slicing?",
    "What is the role of RIC in O-RAN architecture?",
    "Explain the O-RAN near-RT RIC and non-RT RIC differences.",
    "What are the key components of O-RAN architecture?",
]


def create_strategy(
    strategy_name: str,
    model,
    tokenizer,
    max_steps: int = 8
):
    """
    创建策略实例
    
    Args:
        strategy_name: 策略名称
        model: LLM模型
        tokenizer: Tokenizer
        max_steps: 最大步数
    
    Returns:
        策略实例
    """
    common_kwargs = {
        'retriever_mode': 'mock',
        'max_steps': max_steps,
        'verbose': False  # 关闭详细输出
    }
    
    if strategy_name == 'MDP':
        return ARGO_System(
            model, tokenizer,
            use_mdp=True,
            **common_kwargs
        )
    
    elif strategy_name == 'Fixed':
        return FixedThresholdStrategy(
            model, tokenizer,
            theta_cont=0.5,
            theta_star=0.9,
            **common_kwargs
        )
    
    elif strategy_name == 'Always-Reason':
        return AlwaysReasonStrategy(
            model, tokenizer,
            **common_kwargs
        )
    
    elif strategy_name == 'Random':
        return RandomStrategy(
            model, tokenizer,
            retrieve_probability=0.5,
            **common_kwargs
        )
    
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def run_comparison(
    model,
    tokenizer,
    questions: List[str],
    strategies: List[str] = None
):
    """
    运行策略对比实验
    
    Args:
        model: LLM模型
        tokenizer: Tokenizer
        questions: 测试问题列表
        strategies: 策略名称列表
    
    Returns:
        结果DataFrame
    """
    if strategies is None:
        strategies = ['MDP', 'Fixed', 'Always-Reason', 'Random']
    
    results = []
    
    print("="*80)
    print("Running Strategy Comparison")
    print("="*80)
    print(f"Questions: {len(questions)}")
    print(f"Strategies: {', '.join(strategies)}")
    print("="*80 + "\n")
    
    for strategy_name in strategies:
        print(f"\n{'='*80}")
        print(f"Testing Strategy: {strategy_name}")
        print(f"{'='*80}")
        
        # 创建策略
        strategy = create_strategy(strategy_name, model, tokenizer)
        
        strategy_results = []
        
        for i, question in enumerate(questions):
            print(f"\n[{i+1}/{len(questions)}] {question[:60]}...")
            
            try:
                # 运行策略
                start_time = time.time()
                answer, history, metadata = strategy.answer_question(
                    question,
                    return_history=True
                )
                end_time = time.time()
                
                # 记录结果
                result = {
                    'strategy': strategy_name,
                    'question_id': i,
                    'question': question,
                    'answer': answer,
                    'total_steps': metadata['total_steps'],
                    'retrieve_count': metadata['retrieve_count'],
                    'reason_count': metadata['reason_count'],
                    'successful_retrievals': metadata['successful_retrievals'],
                    'elapsed_time': metadata['elapsed_time'],
                    'final_U': metadata['final_uncertainty']
                }
                
                strategy_results.append(result)
                results.append(result)
                
                print(f"  Steps: {result['total_steps']}, "
                      f"Retrieve: {result['retrieve_count']}, "
                      f"Reason: {result['reason_count']}, "
                      f"Time: {result['elapsed_time']:.1f}s")
                
            except Exception as e:
                logger.error(f"Error processing question {i}: {e}", exc_info=True)
                print(f"  ❌ Failed: {e}")
        
        # 策略统计
        if strategy_results:
            avg_steps = sum(r['total_steps'] for r in strategy_results) / len(strategy_results)
            avg_time = sum(r['elapsed_time'] for r in strategy_results) / len(strategy_results)
            avg_retrieve = sum(r['retrieve_count'] for r in strategy_results) / len(strategy_results)
            avg_reason = sum(r['reason_count'] for r in strategy_results) / len(strategy_results)
            
            print(f"\nStrategy Summary:")
            print(f"  Avg Steps: {avg_steps:.1f}")
            print(f"  Avg Retrieve: {avg_retrieve:.1f}")
            print(f"  Avg Reason: {avg_reason:.1f}")
            print(f"  Avg Time: {avg_time:.1f}s")
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    return df


def analyze_results(df: pd.DataFrame, output_dir: str = "results"):
    """
    分析和可视化结果
    
    Args:
        df: 结果DataFrame
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("Analysis Results")
    print("="*80)
    
    # 1. 按策略汇总统计
    print("\n1. Summary by Strategy:")
    print("-" * 80)
    
    summary = df.groupby('strategy').agg({
        'total_steps': ['mean', 'std'],
        'retrieve_count': ['mean', 'std'],
        'reason_count': ['mean', 'std'],
        'elapsed_time': ['mean', 'std'],
        'final_U': ['mean', 'std']
    }).round(2)
    
    print(summary)
    
    # 保存到CSV
    summary.to_csv(os.path.join(output_dir, 'strategy_summary.csv'))
    print(f"\n✅ Saved to {output_dir}/strategy_summary.csv")
    
    # 2. 详细结果
    df.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)
    print(f"✅ Saved to {output_dir}/detailed_results.csv")
    
    # 3. 可视化
    create_visualizations(df, output_dir)
    
    # 4. 对比表格
    create_comparison_table(df, output_dir)


def create_visualizations(df: pd.DataFrame, output_dir: str):
    """创建可视化图表"""
    
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    strategies = df['strategy'].unique()
    
    # 1. 总步数对比
    ax = axes[0, 0]
    df.groupby('strategy')['total_steps'].mean().plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title('Average Total Steps per Strategy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Steps')
    ax.set_xlabel('Strategy')
    ax.tick_params(axis='x', rotation=45)
    
    # 2. 时间对比
    ax = axes[0, 1]
    df.groupby('strategy')['elapsed_time'].mean().plot(kind='bar', ax=ax, color='lightcoral')
    ax.set_title('Average Elapsed Time per Strategy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (seconds)')
    ax.set_xlabel('Strategy')
    ax.tick_params(axis='x', rotation=45)
    
    # 3. Retrieve vs Reason 比例
    ax = axes[1, 0]
    retrieve_counts = df.groupby('strategy')['retrieve_count'].mean()
    reason_counts = df.groupby('strategy')['reason_count'].mean()
    
    x = range(len(strategies))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], retrieve_counts, width, label='Retrieve', color='#3498db')
    ax.bar([i + width/2 for i in x], reason_counts, width, label='Reason', color='#e74c3c')
    
    ax.set_title('Retrieve vs Reason Actions', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count')
    ax.set_xlabel('Strategy')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45)
    ax.legend()
    
    # 4. Final U_t 分布
    ax = axes[1, 1]
    df.boxplot(column='final_U', by='strategy', ax=ax)
    ax.set_title('Final Uncertainty Distribution', fontsize=12, fontweight='bold')
    ax.set_ylabel('Final U_t')
    ax.set_xlabel('Strategy')
    plt.sca(ax)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strategy_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved visualization to {output_dir}/strategy_comparison.png")
    plt.close()


def create_comparison_table(df: pd.DataFrame, output_dir: str):
    """创建论文级别的对比表格"""
    
    print("\n" + "="*80)
    print("Strategy Comparison Table")
    print("="*80)
    
    # 计算统计
    table_data = []
    
    for strategy in df['strategy'].unique():
        strategy_df = df[df['strategy'] == strategy]
        
        row = {
            'Strategy': strategy,
            'Avg Steps': f"{strategy_df['total_steps'].mean():.1f} ± {strategy_df['total_steps'].std():.1f}",
            'Avg Retrieve': f"{strategy_df['retrieve_count'].mean():.1f}",
            'Avg Reason': f"{strategy_df['reason_count'].mean():.1f}",
            'Avg Time (s)': f"{strategy_df['elapsed_time'].mean():.1f} ± {strategy_df['elapsed_time'].std():.1f}",
            'Final U_t': f"{strategy_df['final_U'].mean():.3f}",
        }
        
        table_data.append(row)
    
    table_df = pd.DataFrame(table_data)
    
    print(table_df.to_string(index=False))
    
    # 保存LaTeX格式
    latex_table = table_df.to_latex(index=False)
    with open(os.path.join(output_dir, 'comparison_table.tex'), 'w') as f:
        f.write(latex_table)
    
    print(f"\n✅ Saved LaTeX table to {output_dir}/comparison_table.tex")


def main():
    """主函数"""
    print("="*80)
    print("ARGO Strategy Comparison Experiment")
    print("="*80)
    
    # 加载模型
    print("\nLoading model and tokenizer...")
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        print(f"✅ Loaded model: {model_name}")
        print(f"Device: {next(model.parameters()).device}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        print(f"\n⚠️  Model loading failed: {e}")
        return
    
    # 运行对比实验
    try:
        df = run_comparison(
            model,
            tokenizer,
            questions=TEST_QUESTIONS,
            strategies=['MDP', 'Fixed', 'Always-Reason', 'Random']
        )
        
        # 分析结果
        analyze_results(df, output_dir="results/strategy_comparison")
        
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETED! ✅")
        print("="*80)
        print("\nResults saved to: results/strategy_comparison/")
        print("  - strategy_summary.csv: 汇总统计")
        print("  - detailed_results.csv: 详细结果")
        print("  - strategy_comparison.png: 可视化图表")
        print("  - comparison_table.tex: LaTeX表格")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        print(f"\n❌ Experiment failed: {e}")


if __name__ == "__main__":
    main()
