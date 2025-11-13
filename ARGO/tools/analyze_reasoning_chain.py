"""
推理链分析工具
用于分析和可视化ARGO系统的完整推理链

功能:
1. 读取实验结果JSON
2. 提取推理链 H_t = {(q_1,r_1), (q_2,r_2), ..., (q_T,r_T)}
3. 可视化U的演化过程
4. 导出子查询-答案对
5. 生成推理链报告
"""

import json
import os
import sys
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ReasoningChainAnalyzer:
    """推理链分析器"""
    
    def __init__(self, results_path: str):
        """
        Args:
            results_path: 实验结果JSON文件路径
        """
        self.results_path = results_path
        self.results = self._load_results()
        
    def _load_results(self) -> Dict:
        """加载实验结果"""
        with open(self.results_path, 'r') as f:
            return json.load(f)
    
    def extract_reasoning_chains(self) -> List[Dict]:
        """提取所有问题的推理链"""
        chains = []
        
        results_data = self.results.get('results', [])
        
        for result in results_data:
            chain = {
                'question_id': result['question_id'],
                'is_correct': result['is_correct'],
                'total_cost': result['total_cost'],
                'iterations': result['iterations'],
                'history': result['history'],
                'qa_pairs': []  # (q_t, r_t) 对
            }
            
            # 提取 (q_t, r_t) 对
            for step in result['history']:
                if step['action'] == 'reason' and step['response']:
                    qa_pair = {
                        'iteration': step['iteration'],
                        'subquery': step['subquery'],
                        'response': step['response'],
                        'answer': step['intermediate_answer'],
                        'confidence': step['confidence']
                    }
                    chain['qa_pairs'].append(qa_pair)
            
            chains.append(chain)
        
        return chains
    
    def visualize_uncertainty_evolution(
        self, 
        question_ids: List[str] = None,
        save_path: str = None
    ):
        """
        可视化U的演化过程
        
        Args:
            question_ids: 要可视化的问题ID列表（None表示所有）
            save_path: 保存图片的路径
        """
        chains = self.extract_reasoning_chains()
        
        if question_ids:
            chains = [c for c in chains if c['question_id'] in question_ids]
        
        # 只选择前10个问题可视化
        chains = chains[:10]
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for idx, chain in enumerate(chains):
            ax = axes[idx]
            
            # 提取U的轨迹
            uncertainties = []
            iterations = []
            actions = []
            
            for step in chain['history']:
                if step['uncertainty'] is not None:
                    uncertainties.append(step['uncertainty'])
                    iterations.append(step['iteration'])
                    actions.append(step['action'])
            
            # 绘制U的演化
            ax.plot(iterations, uncertainties, 'b-o', linewidth=2, markersize=6)
            
            # 标注动作
            for i, (iter_num, unc, action) in enumerate(zip(iterations, uncertainties, actions)):
                color = 'green' if action == 'retrieve' else 'orange'
                ax.scatter(iter_num, unc, c=color, s=100, alpha=0.6, zorder=5)
                ax.text(iter_num, unc + 0.02, action[0].upper(), 
                       ha='center', fontsize=8)
            
            # 设置
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Uncertainty (1-U)')
            ax.set_title(f"Q{chain['question_id'][:6]} - {'✓' if chain['is_correct'] else '✗'}")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 不确定性演化图已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def export_qa_pairs(self, output_path: str):
        """
        导出所有子查询-答案对
        
        Args:
            output_path: 输出JSON文件路径
        """
        chains = self.extract_reasoning_chains()
        
        qa_export = []
        for chain in chains:
            for qa in chain['qa_pairs']:
                qa_export.append({
                    'question_id': chain['question_id'],
                    'iteration': qa['iteration'],
                    'subquery': qa['subquery'],
                    'response': qa['response'],
                    'answer': qa['answer'],
                    'confidence': qa['confidence']
                })
        
        with open(output_path, 'w') as f:
            json.dump(qa_export, f, indent=2)
        
        print(f"✓ 共导出 {len(qa_export)} 个子查询-答案对")
        print(f"✓ 保存到: {output_path}")
    
    def generate_report(self, output_path: str):
        """
        生成推理链分析报告
        
        Args:
            output_path: 输出Markdown报告路径
        """
        chains = self.extract_reasoning_chains()
        
        # 统计信息
        total_questions = len(chains)
        correct_count = sum(1 for c in chains if c['is_correct'])
        accuracy = correct_count / total_questions if total_questions > 0 else 0
        
        avg_cost = np.mean([c['total_cost'] for c in chains])
        avg_iterations = np.mean([c['iterations'] for c in chains])
        
        # 动作统计
        action_counts = {'retrieve': 0, 'reason': 0, 'terminate': 0}
        for chain in chains:
            for step in chain['history']:
                action_counts[step['action']] = action_counts.get(step['action'], 0) + 1
        
        # 生成报告
        report = []
        report.append("# ARGO推理链分析报告\n")
        report.append(f"**生成时间**: {self.results.get('timestamp', 'N/A')}\n")
        report.append(f"**模型**: {self.results.get('model', 'N/A')}\n")
        report.append(f"**策略**: {self.results.get('strategy', 'N/A')}\n\n")
        
        report.append("## 📊 总体统计\n")
        report.append(f"- **问题总数**: {total_questions}")
        report.append(f"- **正确数**: {correct_count}")
        report.append(f"- **准确率**: {accuracy:.2%}")
        report.append(f"- **平均成本**: {avg_cost:.3f}")
        report.append(f"- **平均迭代次数**: {avg_iterations:.1f}\n")
        
        report.append("## 🎯 动作分布\n")
        for action, count in action_counts.items():
            report.append(f"- **{action.capitalize()}**: {count}")
        report.append("\n")
        
        report.append("## 📝 推理链示例\n")
        
        # 显示前3个推理链
        for i, chain in enumerate(chains[:3], 1):
            report.append(f"### 示例 {i}: Question {chain['question_id'][:8]}\n")
            report.append(f"- **结果**: {'✓ 正确' if chain['is_correct'] else '✗ 错误'}")
            report.append(f"- **成本**: {chain['total_cost']:.3f}")
            report.append(f"- **迭代次数**: {chain['iterations']}\n")
            
            report.append("**推理链轨迹**:\n")
            report.append("```")
            for step in chain['history']:
                action_symbol = {
                    'retrieve': 'R',
                    'reason': 'P',
                    'terminate': 'T'
                }.get(step['action'], '?')
                
                unc_str = f"U={1-step['uncertainty']:.2f}" if step['uncertainty'] is not None else "U=N/A"
                report.append(f"  {step['iteration']:2d}. [{action_symbol}] {unc_str}, Cost={step['cost']:.3f}")
                
                if step['action'] == 'reason' and step['intermediate_answer']:
                    report.append(f"      → Answer: {step['intermediate_answer']}")
            
            report.append("```\n")
            
            if chain['qa_pairs']:
                report.append("**子查询-答案对**:\n")
                for qa in chain['qa_pairs']:
                    report.append(f"{qa['iteration']}. Q: {qa['subquery'][:50]}...")
                    report.append(f"   A: {qa['answer']} (conf={qa['confidence']:.2f})\n")
        
        # 保存报告
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"✓ 推理链报告已生成")
        print(f"✓ 保存到: {output_path}")
    
    def compare_strategies(self, other_results_path: str, output_path: str):
        """
        对比两种策略的推理链差异
        
        Args:
            other_results_path: 另一个策略的结果文件
            output_path: 输出对比报告路径
        """
        # 加载另一个结果
        with open(other_results_path, 'r') as f:
            other_results = json.load(f)
        
        chains_1 = self.extract_reasoning_chains()
        
        # 临时保存当前结果，加载另一个
        temp_results = self.results
        self.results = other_results
        chains_2 = self.extract_reasoning_chains()
        self.results = temp_results
        
        # 生成对比
        report = []
        report.append("# 策略对比报告\n")
        report.append(f"## 策略1: {self.results.get('strategy', 'Unknown')}\n")
        report.append(f"- 准确率: {sum(c['is_correct'] for c in chains_1) / len(chains_1):.2%}")
        report.append(f"- 平均成本: {np.mean([c['total_cost'] for c in chains_1]):.3f}")
        report.append(f"- 平均迭代: {np.mean([c['iterations'] for c in chains_1]):.1f}\n")
        
        report.append(f"## 策略2: {other_results.get('strategy', 'Unknown')}\n")
        report.append(f"- 准确率: {sum(c['is_correct'] for c in chains_2) / len(chains_2):.2%}")
        report.append(f"- 平均成本: {np.mean([c['total_cost'] for c in chains_2]):.3f}")
        report.append(f"- 平均迭代: {np.mean([c['iterations'] for c in chains_2]):.1f}\n")
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"✓ 策略对比报告已生成: {output_path}")


def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ARGO推理链分析工具")
    parser.add_argument('results_path', help="实验结果JSON文件路径")
    parser.add_argument('--visualize', action='store_true', help="生成不确定性演化图")
    parser.add_argument('--export-qa', help="导出子查询-答案对到指定路径")
    parser.add_argument('--report', help="生成推理链报告到指定路径")
    parser.add_argument('--compare', help="与另一个结果文件对比")
    parser.add_argument('--output-dir', default='analysis_output', help="输出目录")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化分析器
    analyzer = ReasoningChainAnalyzer(args.results_path)
    
    # 可视化
    if args.visualize:
        fig_path = os.path.join(args.output_dir, 'uncertainty_evolution.png')
        analyzer.visualize_uncertainty_evolution(save_path=fig_path)
    
    # 导出QA对
    if args.export_qa:
        qa_path = args.export_qa if args.export_qa else os.path.join(args.output_dir, 'qa_pairs.json')
        analyzer.export_qa_pairs(qa_path)
    
    # 生成报告
    if args.report:
        report_path = args.report if args.report else os.path.join(args.output_dir, 'reasoning_chain_report.md')
        analyzer.generate_report(report_path)
    
    # 对比策略
    if args.compare:
        compare_path = os.path.join(args.output_dir, 'strategy_comparison.md')
        analyzer.compare_strategies(args.compare, compare_path)
    
    print("\n✓ 分析完成!")


if __name__ == '__main__':
    main()
