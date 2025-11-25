"""
Phase 4.3 - Small-Scale Experiment Framework
============================================

在ORAN-Bench-13K上进行小规模实验 (100 queries)
对比4种策略：MDP-Guided, Fixed-Threshold, Always-Reason, Random

目标:
1. 验证各策略的准确率 (Accuracy)
2. 测量延迟和效率 (Latency, Steps, Retrieve/Reason ratio)
3. 分析成本效益 (Cost-effectiveness)
4. 生成论文级别结果表格和图表
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
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import ARGO_System, AlwaysReasonStrategy, RandomStrategy, FixedThresholdStrategy


class ORANBenchLoader:
    """加载ORAN-Bench-13K数据集"""
    
    def __init__(self, benchmark_dir: str = "data/benchmark/ORAN-Bench-13K/Benchmark"):
        self.benchmark_dir = benchmark_dir
        self.data = {
            'E': [],  # Easy
            'M': [],  # Medium
            'H': []   # Hard
        }
        
        self._load_data()
    
    def _load_data(self):
        """加载所有难度级别的数据"""
        for difficulty in ['E', 'M', 'H']:
            filepath = os.path.join(self.benchmark_dir, f'fin_{difficulty}.json')
            
            if not os.path.exists(filepath):
                logger.warning(f"File not found: {filepath}")
                continue
            
            with open(filepath, 'r', encoding='utf-8') as f:
                # ORAN-Bench格式: 每行一个JSON数组 [question, options, answer]
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        item = json.loads(line)
                        if len(item) == 3:
                            question, options, answer = item
                            
                            # 转换为统一格式
                            self.data[difficulty].append({
                                'question': question,
                                'options': options,
                                'answer': answer,  # e.g., "3" (1-indexed)
                                'difficulty': difficulty
                            })
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line: {line[:100]}...")
                        continue
        
        # 打印统计
        total = sum(len(self.data[d]) for d in ['E', 'M', 'H'])
        print(f"\n✅ Loaded ORAN-Bench-13K:")
        print(f"  Easy: {len(self.data['E'])} questions")
        print(f"  Medium: {len(self.data['M'])} questions")
        print(f"  Hard: {len(self.data['H'])} questions")
        print(f"  Total: {total} questions")
    
    def sample(self, n: int = 100, difficulties: List[str] = ['E', 'M', 'H'], 
               random_seed: int = 42) -> List[Dict]:
        """
        采样n个问题
        
        Args:
            n: 采样数量
            difficulties: 包含的难度级别
            random_seed: 随机种子
        
        Returns:
            采样的问题列表
        """
        random.seed(random_seed)
        
        # 合并所有难度的问题
        all_questions = []
        for d in difficulties:
            all_questions.extend(self.data[d])
        
        # 随机采样
        if n >= len(all_questions):
            sampled = all_questions
        else:
            sampled = random.sample(all_questions, n)
        
        print(f"\n✅ Sampled {len(sampled)} questions from {difficulties}")
        
        return sampled
    
    def get_all(self, difficulties: List[str] = ['E', 'M', 'H']) -> List[Dict]:
        """获取所有问题"""
        all_questions = []
        for d in difficulties:
            all_questions.extend(self.data[d])
        return all_questions


class MCQAEvaluator:
    """多选题答案评估器"""
    
    @staticmethod
    def format_mcqa_question(item: Dict) -> str:
        """
        格式化为LLM输入
        
        Args:
            item: 包含question, options的字典
        
        Returns:
            格式化的问题字符串
        """
        question = item['question']
        options = item['options']
        
        # 格式化选项
        formatted_options = "\n".join(options)
        
        prompt = f"{question}\n\n{formatted_options}\n\nPlease select the correct answer (1, 2, 3, or 4)."
        
        return prompt
    
    @staticmethod
    def extract_answer(response: str) -> str:
        """
        从LLM回复中提取答案
        
        Args:
            response: LLM生成的回答
        
        Returns:
            提取的答案 (1, 2, 3, 4) 或 "N/A"
        """
        # 尝试多种模式提取答案
        import re
        
        # 模式1: "答案是X" / "The answer is X"
        patterns = [
            r'answer is\s*["\']?(\d)["\']?',
            r'correct answer is\s*["\']?(\d)["\']?',
            r'选择\s*["\']?(\d)["\']?',
            r'答案是\s*["\']?(\d)["\']?',
            r'^(\d)[\.\)]',  # 开头的数字
            r'\((\d)\)',      # 括号中的数字
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer = match.group(1)
                if answer in ['1', '2', '3', '4']:
                    return answer
        
        # 模式2: 直接找第一个1-4的数字
        for char in response:
            if char in ['1', '2', '3', '4']:
                return char
        
        return "N/A"
    
    @staticmethod
    def evaluate(predicted: str, ground_truth: str) -> bool:
        """
        评估答案是否正确
        
        Args:
            predicted: 预测答案
            ground_truth: 正确答案
        
        Returns:
            是否正确
        """
        return predicted == ground_truth


def run_experiment(
    strategy_name: str,
    system: ARGO_System,
    questions: List[Dict],
    output_dir: str
) -> pd.DataFrame:
    """
    运行单个策略的实验
    
    Args:
        strategy_name: 策略名称
        system: ARGO系统实例
        questions: 问题列表
        output_dir: 输出目录
    
    Returns:
        结果DataFrame
    """
    print(f"\n{'='*80}")
    print(f"Running: {strategy_name}")
    print('='*80)
    
    results = []
    evaluator = MCQAEvaluator()
    
    start_time = time.time()
    
    for i, item in enumerate(tqdm(questions, desc=strategy_name)):
        # 格式化问题
        formatted_question = evaluator.format_mcqa_question(item)
        
        # 运行系统
        try:
            answer, history, metadata = system.answer_question(
                formatted_question,
                return_history=True
            )
            
            # 提取预测答案
            predicted = evaluator.extract_answer(answer)
            
            # 评估
            is_correct = evaluator.evaluate(predicted, item['answer'])
            
            # 记录结果
            result = {
                'strategy': strategy_name,
                'question_id': i,
                'difficulty': item['difficulty'],
                'question': item['question'][:100],  # 截断以节省空间
                'ground_truth': item['answer'],
                'predicted': predicted,
                'is_correct': is_correct,
                'total_steps': metadata['total_steps'],
                'retrieve_count': metadata['retrieve_count'],
                'reason_count': metadata['reason_count'],
                'final_uncertainty': metadata['final_uncertainty'],
                'answer_length': len(answer)
            }
            
            results.append(result)
            
            # 定期保存
            if (i + 1) % 10 == 0:
                df_temp = pd.DataFrame(results)
                df_temp.to_csv(
                    os.path.join(output_dir, f'{strategy_name}_temp.csv'),
                    index=False
                )
        
        except Exception as e:
            logger.error(f"Failed on question {i}: {e}")
            
            # 记录失败
            result = {
                'strategy': strategy_name,
                'question_id': i,
                'difficulty': item['difficulty'],
                'question': item['question'][:100],
                'ground_truth': item['answer'],
                'predicted': 'ERROR',
                'is_correct': False,
                'total_steps': 0,
                'retrieve_count': 0,
                'reason_count': 0,
                'final_uncertainty': 1.0,
                'answer_length': 0
            }
            results.append(result)
    
    total_time = time.time() - start_time
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 添加统计信息
    accuracy = df['is_correct'].mean()
    avg_steps = df['total_steps'].mean()
    avg_retrieves = df['retrieve_count'].mean()
    avg_reasons = df['reason_count'].mean()
    
    print(f"\n{strategy_name} Results:")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  Avg Steps: {avg_steps:.2f}")
    print(f"  Avg Retrieves: {avg_retrieves:.2f}")
    print(f"  Avg Reasons: {avg_reasons:.2f}")
    print(f"  Total Time: {total_time:.1f}s ({total_time/len(questions):.1f}s/query)")
    
    # 保存最终结果
    df.to_csv(os.path.join(output_dir, f'{strategy_name}_results.csv'), index=False)
    
    return df


def analyze_results(all_results: Dict[str, pd.DataFrame], output_dir: str):
    """
    分析并可视化所有策略的结果
    
    Args:
        all_results: {strategy_name: DataFrame}
        output_dir: 输出目录
    """
    print("\n" + "="*80)
    print("Analysis & Visualization")
    print("="*80)
    
    # 1. 汇总统计
    summary = []
    
    for strategy_name, df in all_results.items():
        stats = {
            'Strategy': strategy_name,
            'Accuracy (%)': df['is_correct'].mean() * 100,
            'Avg Steps': df['total_steps'].mean(),
            'Avg Retrieves': df['retrieve_count'].mean(),
            'Avg Reasons': df['reason_count'].mean(),
            'Avg Answer Length': df['answer_length'].mean(),
            'Questions': len(df)
        }
        
        # 按难度分解
        for difficulty in ['E', 'M', 'H']:
            df_diff = df[df['difficulty'] == difficulty]
            if len(df_diff) > 0:
                stats[f'Accuracy_{difficulty} (%)'] = df_diff['is_correct'].mean() * 100
        
        summary.append(stats)
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(output_dir, 'summary_statistics.csv'), index=False)
    
    print("\nSummary Statistics:")
    print(summary_df.to_string(index=False))
    
    # 2. 生成可视化
    visualize_results(all_results, summary_df, output_dir)
    
    # 3. 生成LaTeX表格
    generate_latex_table(summary_df, output_dir)
    
    print(f"\n✅ All results saved to {output_dir}/")


def visualize_results(all_results: Dict[str, pd.DataFrame], 
                     summary_df: pd.DataFrame,
                     output_dir: str):
    """生成可视化图表"""
    
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    strategies = list(all_results.keys())
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    # 1. 准确率对比（总体）
    ax1 = fig.add_subplot(gs[0, 0])
    accuracies = summary_df['Accuracy (%)'].values
    ax1.bar(range(len(strategies)), accuracies, color=colors, edgecolor='black', alpha=0.8)
    ax1.set_xticks(range(len(strategies)))
    ax1.set_xticklabels(strategies, rotation=15, ha='right')
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_title('Overall Accuracy', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    
    for i, (strat, acc) in enumerate(zip(strategies, accuracies)):
        ax1.text(i, acc + 2, f'{acc:.1f}%', ha='center', fontsize=9, fontweight='bold')
    
    # 2. 按难度分解准确率
    ax2 = fig.add_subplot(gs[0, 1])
    
    difficulties = ['E', 'M', 'H']
    x = np.arange(len(difficulties))
    width = 0.2
    
    for i, strategy in enumerate(strategies):
        accs = [summary_df.loc[summary_df['Strategy'] == strategy, f'Accuracy_{d} (%)'].values[0] 
                if f'Accuracy_{d} (%)' in summary_df.columns else 0
                for d in difficulties]
        ax2.bar(x + i*width, accs, width, label=strategy, color=colors[i], alpha=0.8)
    
    ax2.set_xlabel('Difficulty', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('Accuracy by Difficulty', fontsize=12, fontweight='bold')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(difficulties)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. 平均步数对比
    ax3 = fig.add_subplot(gs[0, 2])
    avg_steps = summary_df['Avg Steps'].values
    ax3.bar(range(len(strategies)), avg_steps, color=colors, edgecolor='black', alpha=0.8)
    ax3.set_xticks(range(len(strategies)))
    ax3.set_xticklabels(strategies, rotation=15, ha='right')
    ax3.set_ylabel('Average Steps', fontsize=11)
    ax3.set_title('Efficiency: Steps per Query', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Retrieve vs Reason 对比
    ax4 = fig.add_subplot(gs[1, 0])
    
    x = np.arange(len(strategies))
    retrieves = summary_df['Avg Retrieves'].values
    reasons = summary_df['Avg Reasons'].values
    
    ax4.bar(x - 0.2, retrieves, 0.4, label='Retrieve', color='#3498db', alpha=0.8)
    ax4.bar(x + 0.2, reasons, 0.4, label='Reason', color='#e74c3c', alpha=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(strategies, rotation=15, ha='right')
    ax4.set_ylabel('Average Count', fontsize=11)
    ax4.set_title('Action Distribution', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. 准确率 vs 效率散点图
    ax5 = fig.add_subplot(gs[1, 1])
    
    for i, strategy in enumerate(strategies):
        ax5.scatter(summary_df.loc[summary_df['Strategy'] == strategy, 'Avg Steps'],
                   summary_df.loc[summary_df['Strategy'] == strategy, 'Accuracy (%)'],
                   s=200, color=colors[i], label=strategy, alpha=0.7, edgecolor='black')
    
    ax5.set_xlabel('Average Steps', fontsize=11)
    ax5.set_ylabel('Accuracy (%)', fontsize=11)
    ax5.set_title('Accuracy vs Efficiency', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # 6. 答案长度对比
    ax6 = fig.add_subplot(gs[1, 2])
    avg_lengths = summary_df['Avg Answer Length'].values
    ax6.bar(range(len(strategies)), avg_lengths, color=colors, edgecolor='black', alpha=0.8)
    ax6.set_xticks(range(len(strategies)))
    ax6.set_xticklabels(strategies, rotation=15, ha='right')
    ax6.set_ylabel('Average Answer Length (chars)', fontsize=11)
    ax6.set_title('Answer Verbosity', fontsize=12, fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)
    
    # 7-9. 各策略的步数分布
    for idx, strategy in enumerate(strategies[:3]):  # 只显示前3个
        ax = fig.add_subplot(gs[2, idx])
        df = all_results[strategy]
        
        ax.hist(df['total_steps'], bins=10, color=colors[idx], edgecolor='black', alpha=0.7)
        ax.set_xlabel('Total Steps', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{strategy} Step Distribution', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'experiment_results.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved visualization to {output_dir}/experiment_results.png")
    plt.close()


def generate_latex_table(summary_df: pd.DataFrame, output_dir: str):
    """生成LaTeX格式的表格"""
    
    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Comparison of ARGO Strategies on ORAN-Bench-13K (100 queries)}",
        "\\label{tab:argo_comparison}",
        "\\begin{tabular}{lcccccc}",
        "\\hline",
        "\\textbf{Strategy} & \\textbf{Accuracy} & \\textbf{Acc (E)} & \\textbf{Acc (M)} & \\textbf{Acc (H)} & \\textbf{Avg Steps} & \\textbf{Avg Retrieves} \\\\",
        "\\hline",
    ]
    
    for _, row in summary_df.iterrows():
        strategy = row['Strategy'].replace('_', '\\_')
        acc = row['Accuracy (%)']
        acc_e = row.get('Accuracy_E (%)', 0)
        acc_m = row.get('Accuracy_M (%)', 0)
        acc_h = row.get('Accuracy_H (%)', 0)
        steps = row['Avg Steps']
        retrieves = row['Avg Retrieves']
        
        line = f"{strategy} & {acc:.1f}\\% & {acc_e:.1f}\\% & {acc_m:.1f}\\% & {acc_h:.1f}\\% & {steps:.2f} & {retrieves:.2f} \\\\"
        latex_lines.append(line)
    
    latex_lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    latex_content = "\n".join(latex_lines)
    
    with open(os.path.join(output_dir, 'results_table.tex'), 'w') as f:
        f.write(latex_content)
    
    print(f"✅ Saved LaTeX table to {output_dir}/results_table.tex")


def main():
    """主函数"""
    
    print("="*80)
    print("ARGO Phase 4.3 - Small-Scale Experiment")
    print("="*80)
    
    # 配置
    N_QUESTIONS = 100  # 小规模实验
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # 使用优化后的模型
    OUTPUT_DIR = "results/phase4.3_small"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 加载数据
    print("\n1. Loading ORAN-Bench-13K...")
    loader = ORANBenchLoader()
    questions = loader.sample(n=N_QUESTIONS, difficulties=['E', 'M', 'H'], random_seed=42)
    
    # 2. 加载模型
    print(f"\n2. Loading model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print("✅ Model loaded")
    
    # 3. 创建4种策略
    print("\n3. Initializing strategies...")
    
    strategies = {}
    
    # MDP-Guided (ARGO)
    argo_system = ARGO_System(
        model, tokenizer,
        use_mdp=True,
        retriever_mode='mock',
        max_steps=6,
        verbose=False
    )
    # 应用优化参数
    argo_system.decomposer.max_subquery_length = 50
    argo_system.synthesizer.max_answer_length = 200
    argo_system.decomposer.temperature = 0.5
    argo_system.synthesizer.temperature = 0.2
    
    strategies['MDP_Guided'] = argo_system
    
    # Fixed Threshold
    fixed_system = FixedThresholdStrategy(
        model, tokenizer,
        theta_cont=0.5,  # 固定阈值
        retriever_mode='mock',
        max_steps=6,
        verbose=False
    )
    fixed_system.decomposer.max_subquery_length = 50
    fixed_system.synthesizer.max_answer_length = 200
    fixed_system.decomposer.temperature = 0.5
    fixed_system.synthesizer.temperature = 0.2
    
    strategies['Fixed_Threshold'] = fixed_system
    
    # Always Reason
    always_system = AlwaysReasonStrategy(
        model, tokenizer,
        retriever_mode='mock',
        max_steps=6,
        verbose=False
    )
    always_system.decomposer.max_subquery_length = 50
    always_system.synthesizer.max_answer_length = 200
    always_system.decomposer.temperature = 0.5
    always_system.synthesizer.temperature = 0.2
    
    strategies['Always_Reason'] = always_system
    
    # Random
    random_system = RandomStrategy(
        model, tokenizer,
        retrieve_probability=0.5,
        retriever_mode='mock',
        max_steps=6,
        verbose=False
    )
    random_system.decomposer.max_subquery_length = 50
    random_system.synthesizer.max_answer_length = 200
    random_system.decomposer.temperature = 0.5
    random_system.synthesizer.temperature = 0.2
    
    strategies['Random'] = random_system
    
    print(f"✅ Created {len(strategies)} strategies")
    
    # 4. 运行实验
    print("\n4. Running experiments...")
    all_results = {}
    
    for strategy_name, system in strategies.items():
        try:
            df = run_experiment(strategy_name, system, questions, OUTPUT_DIR)
            all_results[strategy_name] = df
        except Exception as e:
            logger.error(f"Failed to run {strategy_name}: {e}", exc_info=True)
            print(f"❌ {strategy_name} failed: {e}")
    
    # 5. 分析结果
    if len(all_results) > 0:
        print("\n5. Analyzing results...")
        analyze_results(all_results, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED! ✅")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("  - *_results.csv: Detailed results for each strategy")
    print("  - summary_statistics.csv: Summary table")
    print("  - experiment_results.png: Visualizations")
    print("  - results_table.tex: LaTeX table for paper")


if __name__ == "__main__":
    main()
