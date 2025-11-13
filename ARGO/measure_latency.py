"""
Latency Measurement - Phase 4.2
================================

测量ARGO系统的延迟性能，分析瓶颈，验证是否满足O-RAN实时性要求

目标:
- 测量每个query的总延迟
- 验证 latency ≤ 1000ms (O-RAN要求)
- 分解延迟：Decomposer, Retriever, Reasoner, Synthesizer
- 识别性能瓶颈
- 提供优化建议

运行方式:
    python measure_latency.py
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import logging

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import ARGO_System


class LatencyProfiler:
    """
    延迟性能分析器
    
    功能:
    1. 细粒度计时：每个组件的耗时
    2. 统计分析：平均、中位数、P95、P99
    3. 瓶颈识别：找出最耗时的组件
    4. 可视化：延迟分布图、瀑布图
    """
    
    def __init__(self, system: ARGO_System):
        """
        Args:
            system: ARGO_System实例
        """
        self.system = system
        self.measurements = []
        
        # 劫持系统方法以插入计时
        self._wrap_methods()
    
    def _wrap_methods(self):
        """包装系统方法以插入计时器"""
        
        # 保存原始方法
        self.original_generate_subquery = self.system.decomposer.generate_subquery
        self.original_retrieve = self.system.retriever.retrieve
        self.original_synthesize = self.system.synthesizer.synthesize
        
        # 包装Decomposer
        def timed_generate_subquery(*args, **kwargs):
            start = time.time()
            result = self.original_generate_subquery(*args, **kwargs)
            elapsed = time.time() - start
            
            if not hasattr(self, 'current_step_timings'):
                self.current_step_timings = {}
            self.current_step_timings['decomposer'] = elapsed
            
            return result
        
        # 包装Retriever
        def timed_retrieve(*args, **kwargs):
            start = time.time()
            result = self.original_retrieve(*args, **kwargs)
            elapsed = time.time() - start
            
            if not hasattr(self, 'current_step_timings'):
                self.current_step_timings = {}
            self.current_step_timings['retriever'] = elapsed
            
            return result
        
        # 包装Synthesizer
        def timed_synthesize(*args, **kwargs):
            start = time.time()
            result = self.original_synthesize(*args, **kwargs)
            elapsed = time.time() - start
            
            if not hasattr(self, 'current_step_timings'):
                self.current_step_timings = {}
            self.current_step_timings['synthesizer'] = elapsed
            
            return result
        
        # 替换方法
        self.system.decomposer.generate_subquery = timed_generate_subquery
        self.system.retriever.retrieve = timed_retrieve
        self.system.synthesizer.synthesize = timed_synthesize
    
    def measure_question(
        self,
        question: str
    ) -> Dict:
        """
        测量单个问题的延迟
        
        Args:
            question: 输入问题
        
        Returns:
            延迟分解字典
        """
        self.current_step_timings = {}
        
        # 总计时
        start_total = time.time()
        
        # 运行系统
        answer, history, metadata = self.system.answer_question(
            question,
            return_history=True
        )
        
        total_time = time.time() - start_total
        
        # 收集各组件耗时
        decomposer_total = 0
        retriever_total = 0
        reasoner_total = 0
        
        for step in history:
            if step['action'] == 'retrieve':
                # Decomposer + Retriever
                decomposer_total += self.current_step_timings.get('decomposer', 0)
                retriever_total += self.current_step_timings.get('retriever', 0)
            else:
                # Reasoner (推理在_execute_reason中，我们需要估算)
                # 简化：用总时间减去其他组件
                pass
        
        synthesizer_time = self.current_step_timings.get('synthesizer', 0)
        
        # 估算其他时间（MDP计算、系统开销）
        accounted_time = decomposer_total + retriever_total + synthesizer_time
        overhead_time = total_time - accounted_time
        
        measurement = {
            'question': question,
            'total_latency_ms': total_time * 1000,
            'decomposer_ms': decomposer_total * 1000,
            'retriever_ms': retriever_total * 1000,
            'synthesizer_ms': synthesizer_time * 1000,
            'overhead_ms': overhead_time * 1000,
            'total_steps': metadata['total_steps'],
            'retrieve_count': metadata['retrieve_count'],
            'reason_count': metadata['reason_count'],
            'meets_requirement': total_time * 1000 <= 1000  # ≤1000ms
        }
        
        self.measurements.append(measurement)
        
        return measurement
    
    def measure_batch(
        self,
        questions: List[str],
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        批量测量延迟
        
        Args:
            questions: 问题列表
            verbose: 是否打印进度
        
        Returns:
            结果DataFrame
        """
        if verbose:
            print("="*80)
            print("Latency Measurement")
            print("="*80)
            print(f"Questions: {len(questions)}")
            print("="*80 + "\n")
        
        for i, question in enumerate(questions):
            if verbose:
                print(f"\n[{i+1}/{len(questions)}] {question[:60]}...")
            
            measurement = self.measure_question(question)
            
            if verbose:
                print(f"  Total: {measurement['total_latency_ms']:.0f}ms")
                print(f"  Decomposer: {measurement['decomposer_ms']:.0f}ms")
                print(f"  Retriever: {measurement['retriever_ms']:.0f}ms")
                print(f"  Synthesizer: {measurement['synthesizer_ms']:.0f}ms")
                print(f"  Overhead: {measurement['overhead_ms']:.0f}ms")
                print(f"  Meets 1000ms? {'✅ YES' if measurement['meets_requirement'] else '❌ NO'}")
        
        df = pd.DataFrame(self.measurements)
        
        return df
    
    def analyze(self, df: pd.DataFrame, output_dir: str = "results/latency"):
        """
        分析延迟数据
        
        Args:
            df: 测量结果DataFrame
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*80)
        print("Latency Analysis")
        print("="*80)
        
        # 1. 总体统计
        print("\n1. Overall Statistics:")
        print("-" * 80)
        
        stats = df['total_latency_ms'].describe()
        print(f"Total Latency (ms):")
        print(f"  Mean:   {stats['mean']:.1f}")
        print(f"  Median: {stats['50%']:.1f}")
        print(f"  Std:    {stats['std']:.1f}")
        print(f"  Min:    {stats['min']:.1f}")
        print(f"  Max:    {stats['max']:.1f}")
        print(f"  P95:    {df['total_latency_ms'].quantile(0.95):.1f}")
        print(f"  P99:    {df['total_latency_ms'].quantile(0.99):.1f}")
        
        # 检查O-RAN要求
        meets_count = df['meets_requirement'].sum()
        total_count = len(df)
        pass_rate = meets_count / total_count * 100
        
        print(f"\n  ⏱️  O-RAN Requirement (≤1000ms): {meets_count}/{total_count} ({pass_rate:.1f}%)")
        
        if pass_rate >= 95:
            print(f"  ✅ PASS: {pass_rate:.1f}% queries meet the requirement")
        else:
            print(f"  ❌ FAIL: Only {pass_rate:.1f}% queries meet the requirement")
        
        # 2. 组件分解
        print("\n2. Component Breakdown:")
        print("-" * 80)
        
        components = ['decomposer_ms', 'retriever_ms', 'synthesizer_ms', 'overhead_ms']
        component_names = ['Decomposer', 'Retriever', 'Synthesizer', 'Overhead']
        
        for comp, name in zip(components, component_names):
            mean_ms = df[comp].mean()
            percentage = (mean_ms / df['total_latency_ms'].mean()) * 100
            print(f"  {name:15s}: {mean_ms:7.1f}ms ({percentage:5.1f}%)")
        
        # 3. 瓶颈识别
        print("\n3. Bottleneck Analysis:")
        print("-" * 80)
        
        avg_times = {
            'Decomposer': df['decomposer_ms'].mean(),
            'Retriever': df['retriever_ms'].mean(),
            'Synthesizer': df['synthesizer_ms'].mean(),
            'Overhead': df['overhead_ms'].mean()
        }
        
        bottleneck = max(avg_times.items(), key=lambda x: x[1])
        
        print(f"  Primary Bottleneck: {bottleneck[0]} ({bottleneck[1]:.1f}ms)")
        print(f"  Optimization Priority:")
        for i, (comp, time_ms) in enumerate(sorted(avg_times.items(), key=lambda x: -x[1]), 1):
            print(f"    {i}. {comp:15s}: {time_ms:.1f}ms")
        
        # 保存详细结果
        df.to_csv(os.path.join(output_dir, 'latency_measurements.csv'), index=False)
        print(f"\n✅ Saved detailed results to {output_dir}/latency_measurements.csv")
        
        # 保存汇总统计
        summary = {
            'metric': ['mean', 'median', 'std', 'min', 'max', 'p95', 'p99', 'pass_rate'],
            'total_latency_ms': [
                stats['mean'],
                stats['50%'],
                stats['std'],
                stats['min'],
                stats['max'],
                df['total_latency_ms'].quantile(0.95),
                df['total_latency_ms'].quantile(0.99),
                pass_rate
            ]
        }
        
        for comp in components:
            summary[comp] = [
                df[comp].mean(),
                df[comp].median(),
                df[comp].std(),
                df[comp].min(),
                df[comp].max(),
                df[comp].quantile(0.95),
                df[comp].quantile(0.99),
                np.nan
            ]
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(os.path.join(output_dir, 'latency_summary.csv'), index=False)
        print(f"✅ Saved summary to {output_dir}/latency_summary.csv")
        
        return avg_times
    
    def visualize(self, df: pd.DataFrame, output_dir: str = "results/latency"):
        """
        可视化延迟数据
        
        Args:
            df: 测量结果DataFrame
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(16, 10))
        
        # 2x2 布局
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. 延迟分布直方图
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(df['total_latency_ms'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(1000, color='red', linestyle='--', linewidth=2, label='O-RAN Requirement (1000ms)')
        ax1.set_xlabel('Total Latency (ms)', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title('Latency Distribution', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. 组件耗时堆叠条形图
        ax2 = fig.add_subplot(gs[0, 1])
        
        components = ['decomposer_ms', 'retriever_ms', 'synthesizer_ms', 'overhead_ms']
        component_labels = ['Decomposer', 'Retriever', 'Synthesizer', 'Overhead']
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        avg_times = [df[comp].mean() for comp in components]
        
        ax2.bar(range(len(components)), avg_times, color=colors, edgecolor='black', alpha=0.8)
        ax2.set_xticks(range(len(components)))
        ax2.set_xticklabels(component_labels, rotation=0)
        ax2.set_ylabel('Average Time (ms)', fontsize=11)
        ax2.set_title('Component Breakdown', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for i, (comp, val) in enumerate(zip(component_labels, avg_times)):
            percentage = (val / sum(avg_times)) * 100
            ax2.text(i, val + 50, f'{val:.0f}ms\n({percentage:.1f}%)', 
                    ha='center', fontsize=9, fontweight='bold')
        
        # 3. CDF曲线
        ax3 = fig.add_subplot(gs[1, 0])
        
        sorted_latencies = np.sort(df['total_latency_ms'])
        cdf = np.arange(1, len(sorted_latencies) + 1) / len(sorted_latencies)
        
        ax3.plot(sorted_latencies, cdf * 100, linewidth=2, color='#3498db')
        ax3.axvline(1000, color='red', linestyle='--', linewidth=2, label='1000ms Requirement')
        ax3.set_xlabel('Latency (ms)', fontsize=11)
        ax3.set_ylabel('Cumulative Probability (%)', fontsize=11)
        ax3.set_title('Cumulative Distribution Function (CDF)', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # 4. 箱线图
        ax4 = fig.add_subplot(gs[1, 1])
        
        box_data = [df[comp] for comp in components]
        bp = ax4.boxplot(box_data, labels=component_labels, patch_artist=True, 
                         notch=True, showmeans=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax4.set_ylabel('Time (ms)', fontsize=11)
        ax4.set_title('Component Latency Distribution', fontsize=12, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=0)
        
        plt.savefig(os.path.join(output_dir, 'latency_analysis.png'), dpi=300, bbox_inches='tight')
        print(f"\n✅ Saved visualization to {output_dir}/latency_analysis.png")
        plt.close()


def main():
    """主函数"""
    print("="*80)
    print("ARGO Latency Measurement")
    print("="*80)
    
    # 加载模型
    print("\nLoading model...")
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print(f"✅ Model loaded: {model_name}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        print(f"\n⚠️  Model loading failed: {e}")
        return
    
    # 创建系统
    print("\nInitializing ARGO System...")
    system = ARGO_System(
        model,
        tokenizer,
        use_mdp=True,
        retriever_mode='mock',
        max_steps=6,
        verbose=False
    )
    print("✅ System ready")
    
    # 创建性能分析器
    profiler = LatencyProfiler(system)
    
    # 测试问题（较少以便快速测试）
    test_questions = [
        "What are the latency requirements for O-RAN fronthaul?",
        "How does O-RAN handle network slicing?",
        "What is the role of RIC in O-RAN architecture?",
    ]
    
    try:
        # 测量延迟
        df = profiler.measure_batch(test_questions, verbose=True)
        
        # 分析结果
        profiler.analyze(df)
        
        # 可视化
        profiler.visualize(df)
        
        print("\n" + "="*80)
        print("LATENCY MEASUREMENT COMPLETED! ✅")
        print("="*80)
        print("\nResults saved to: results/latency/")
        print("  - latency_measurements.csv: 详细测量数据")
        print("  - latency_summary.csv: 汇总统计")
        print("  - latency_analysis.png: 可视化图表")
        
    except Exception as e:
        logger.error(f"Measurement failed: {e}", exc_info=True)
        print(f"\n❌ Measurement failed: {e}")


if __name__ == "__main__":
    main()
