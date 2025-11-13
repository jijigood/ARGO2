#!/usr/bin/env python
"""
实验3: 成本-质量权衡影响 (Pareto边界) - 真实LLM版本
============================================
证明ARGO是一个可调框架，可以生成从"快速/便宜"到"慢速/高质量"的
整个最优策略族，追踪Pareto边界。

核心思想:
- 固定环境参数 (δ_r, δ_p, p_s, c_r, c_p)
- 扫描成本权重 μ: 从 0 (只关注质量) 到高值 (关注成本)
- 对每个μ重新求解MDP，得到新的阈值
- 绘制 (Cost, Quality) Pareto边界

预期结果:
- ARGO形成Pareto边界曲线 (任何成本下都是最优质量)
- 基线策略是单点，落在ARGO曲线下方

硬件要求:
- 多张GPU (支持 RTX 3060 x8)
- CUDA环境

模型:
- LLM: Qwen2.5-3B-Instruct (快速版本)
- Embedding: all-MiniLM-L6-v2
- 检索库: Chroma (ORAN规范文档)
"""

import os
import sys
import torch
import numpy as np
import yaml
import json
import random
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# 尝试导入chromadb
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError as e:
    print(f"⚠ ChromaDB不可用: {e}")
    CHROMADB_AVAILABLE = False
    chromadb = None

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oran_benchmark_loader import ORANBenchmark

sys.path.insert(0, '../ARGO_MDP/src')
from mdp_solver import MDPSolver


class RealParetoFrontierExperiment:
    """实验3: Pareto边界 - 真实LLM版本"""
    
    def __init__(
        self,
        config_path: str = "configs/multi_gpu.yaml",
        llm_model_path: str = "/data/user/huangxiaolin/ARGO/RAG_Models/models/Qwen2.5-3B-Instruct",
        embedding_model_path: str = "/data/user/huangxiaolin/ARGO/models/all-MiniLM-L6-v2",
        chroma_db_path: str = "/data/user/huangxiaolin/ARGO2/ARGO/Environments/chroma_store",
        n_test_questions: int = 30,
        difficulty: str = "hard",
        seed: int = 42,
        gpu_ids: List[int] = None
    ):
        """
        Args:
            config_path: MDP配置文件路径
            llm_model_path: Qwen模型本地路径
            embedding_model_path: 嵌入模型本地路径
            chroma_db_path: Chroma数据库路径
            n_test_questions: 测试问题数量
            difficulty: 问题难度
            seed: 随机种子
            gpu_ids: 使用的GPU ID列表
        """
        print(f"\n{'='*80}")
        print(f"实验3: Pareto边界 - 成本质量权衡 (真实LLM版本)")
        print(f"{'='*80}")
        print(f"LLM模型: {llm_model_path}")
        print(f"嵌入模型: {embedding_model_path}")
        print(f"问题难度: {difficulty.upper()}")
        print(f"问题数量: {n_test_questions}")
        print(f"{'='*80}\n")
        
        self.config_path = config_path
        self.llm_model_path = llm_model_path
        self.embedding_model_path = embedding_model_path
        self.chroma_db_path = chroma_db_path
        self.n_test_questions = n_test_questions
        self.difficulty = difficulty
        self.seed = seed
        
        # GPU配置
        if not torch.cuda.is_available():
            raise RuntimeError("需要GPU!")
        
        self.n_gpus = torch.cuda.device_count()
        self.gpu_ids = gpu_ids if gpu_ids else list(range(min(8, self.n_gpus)))
        
        print(f"GPU配置:")
        print(f"  可用GPU: {self.n_gpus}张")
        print(f"  使用GPU: {self.gpu_ids}")
        for i in self.gpu_ids:
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"    GPU {i}: {name} ({mem:.1f}GB)")
        print()
        
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 加载数据集
        print("加载ORAN-Bench-13K数据集...")
        self.benchmark = ORANBenchmark()
        self.test_questions = self.benchmark.sample_questions(
            n=n_test_questions,
            difficulty=difficulty,
            seed=seed
        )
        print(f"✓ 加载了 {len(self.test_questions)} 道 {difficulty.upper()} 问题\n")
        
        # 加载嵌入模型
        print(f"加载嵌入模型: {embedding_model_path}")
        self.embedding_model = SentenceTransformer(embedding_model_path)
        self.embedding_model.to(f'cuda:{self.gpu_ids[0]}')
        print(f"✓ 嵌入模型加载成功 (GPU {self.gpu_ids[0]})\n")
        
        # 连接Chroma数据库
        if CHROMADB_AVAILABLE:
            print(f"连接Chroma数据库: {chroma_db_path}")
            self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
            self.collection = self.chroma_client.get_collection(name="oran_specs")
            print(f"✓ Chroma集合加载成功 (文档数: {self.collection.count()})\n")
        else:
            print("⚠ ChromaDB不可用，使用模拟检索模式\n")
            self.chroma_client = None
            self.collection = None
        
        # 加载LLM模型
        self._load_llm()
        
        print(f"{'='*80}", flush=True)
        print(f"初始化完成!", flush=True)
        print(f"{'='*80}\n", flush=True)
    
    def _load_llm(self):
        """加载LLM模型 (多GPU)"""
        print(f"加载LLM模型: {self.llm_model_path}")
        print(f"  使用 {len(self.gpu_ids)} 张GPU加载模型...")
        
        from accelerate import dispatch_model, infer_auto_device_map
        from accelerate.utils import get_balanced_memory
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.llm_model_path,
            trust_remote_code=True
        )
        
        # 加载模型到CPU
        self.model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_path,
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True
        )
        
        print(f"✓ LLM模型加载成功", flush=True)
        print(f"  Device map: {self.model.hf_device_map}", flush=True)
    
    def retrieve_documents(self, query: str, top_k: int = 5) -> List[str]:
        """检索相关文档"""
        if not CHROMADB_AVAILABLE or self.collection is None:
            # 模拟检索
            return [f"[模拟文档{i+1}] 关于 '{query[:30]}...' 的内容" for i in range(top_k)]
        
        # 真实语义检索
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        if results['documents'] and len(results['documents']) > 0:
            return results['documents'][0]
        return []
    
    def generate_answer(self, question: str, context: str = "") -> str:
        """生成答案"""
        if context:
            prompt = f"""Based on the following context, answer the question.

Context: {context[:1000]}

Question: {question}

Answer:"""
        else:
            prompt = f"""Answer the following question based on your knowledge.

Question: {question}

Answer:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=1.0,
                top_p=0.9,
                top_k=50
            )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 提取Answer:后的内容
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        
        return answer
    
    def evaluate_answer(self, question: str, answer: str, ground_truth: str) -> bool:
        """评估答案正确性 (简单字符串匹配)"""
        answer_lower = answer.lower()
        gt_lower = ground_truth.lower()
        
        # 检查ground_truth是否在answer中
        if gt_lower in answer_lower:
            return True
        
        # 检查关键词匹配
        gt_keywords = set(gt_lower.split())
        answer_keywords = set(answer_lower.split())
        overlap = len(gt_keywords & answer_keywords) / max(len(gt_keywords), 1)
        
        return overlap > 0.5
    
    def solve_mdp(self, mu: float) -> Tuple[float, float]:
        """
        求解MDP (给定成本权重μ)
        
        Args:
            mu: 成本权重 (目标函数: max[Q(O) - μ·C_T])
        
        Returns:
            (theta_cont, theta_star): 检索阈值和终止阈值
        """
        # 创建MDP配置
        mdp_config = self.config['mdp'].copy()
        mdp_config['mu'] = mu
        
        # 添加 U_grid_size (兼容性)
        if 'U_grid_size' not in mdp_config and 'grid_size' in mdp_config:
            mdp_config['U_grid_size'] = mdp_config['grid_size']
        
        solver_config = {
            'mdp': mdp_config,
            'quality': self.config.get('quality', {'mode': 'linear', 'k': 5.0}),
            'solver': {
                'max_iterations': 1000,
                'convergence_threshold': 1e-6,
                'verbose': False
            }
        }
        
        # 求解MDP
        mdp_solver = MDPSolver(solver_config)
        mdp_solver.solve()
        
        return mdp_solver.theta_cont, mdp_solver.theta_star
    
    def simulate_argo_policy(
        self,
        question_data: Dict,
        theta_cont: float,
        theta_star: float,
        max_steps: int = 20
    ) -> Dict:
        """
        模拟ARGO策略
        
        根据MDP阈值动态选择动作:
        - U < theta_cont: Retrieve
        - theta_cont <= U < theta_star: Reason
        - U >= theta_star: Terminate
        """
        question = question_data['question']
        ground_truth = question_data.get('answer', '')
        
        # 初始化
        U = 0.0  # 初始不确定性
        context = ""
        total_cost = 0.0
        retrieval_count = 0
        reason_count = 0
        
        c_r = self.config['mdp']['c_r']
        c_p = self.config['mdp']['c_p']
        delta_r = self.config['mdp']['delta_r']
        delta_p = self.config['mdp']['delta_p']
        
        for step in range(max_steps):
            # 决策
            if U >= theta_star:
                # 终止
                break
            elif U < theta_cont:
                # Retrieve
                docs = self.retrieve_documents(question, top_k=3)
                context += " " + " ".join(docs)
                total_cost += c_r
                retrieval_count += 1
                U += delta_r  # 降低不确定性
            else:
                # Reason
                total_cost += c_p
                reason_count += 1
                U += delta_p  # 降低不确定性
        
        # 生成最终答案
        final_answer = self.generate_answer(question, context)
        correct = self.evaluate_answer(question, final_answer, ground_truth)
        
        # 质量评估 (基于U和正确性)
        quality = U if correct else U * 0.5
        
        return {
            'quality': quality,
            'cost': total_cost,
            'retrieval_count': retrieval_count,
            'reason_count': reason_count,
            'correct': correct,
            'final_answer': final_answer
        }
    
    def simulate_always_retrieve_policy(
        self,
        question_data: Dict,
        theta_star: float = 0.9,
        max_steps: int = 20
    ) -> Dict:
        """Always-Retrieve基线"""
        question = question_data['question']
        ground_truth = question_data.get('answer', '')
        
        U = 0.0
        context = ""
        total_cost = 0.0
        retrieval_count = 0
        
        c_r = self.config['mdp']['c_r']
        delta_r = self.config['mdp']['delta_r']
        
        for step in range(max_steps):
            if U >= theta_star:
                break
            
            docs = self.retrieve_documents(question, top_k=3)
            context += " " + " ".join(docs)
            total_cost += c_r
            retrieval_count += 1
            U += delta_r
        
        final_answer = self.generate_answer(question, context)
        correct = self.evaluate_answer(question, final_answer, ground_truth)
        quality = U if correct else U * 0.5
        
        return {
            'quality': quality,
            'cost': total_cost,
            'retrieval_count': retrieval_count,
            'reason_count': 0,
            'correct': correct,
            'final_answer': final_answer
        }
    
    def simulate_always_reason_policy(
        self,
        question_data: Dict,
        theta_star: float = 0.9,
        max_steps: int = 20
    ) -> Dict:
        """Always-Reason基线"""
        question = question_data['question']
        ground_truth = question_data.get('answer', '')
        
        U = 0.0
        total_cost = 0.0
        reason_count = 0
        
        c_p = self.config['mdp']['c_p']
        delta_p = self.config['mdp']['delta_p']
        
        for step in range(max_steps):
            if U >= theta_star:
                break
            
            total_cost += c_p
            reason_count += 1
            U += delta_p
        
        final_answer = self.generate_answer(question, "")
        correct = self.evaluate_answer(question, final_answer, ground_truth)
        quality = U if correct else U * 0.5
        
        return {
            'quality': quality,
            'cost': total_cost,
            'retrieval_count': 0,
            'reason_count': reason_count,
            'correct': correct,
            'final_answer': final_answer
        }
    
    def run_experiment(
        self,
        mu_min: float = 0.0,
        mu_max: float = 10.0,
        n_mu_steps: int = 10
    ):
        """
        运行Pareto边界实验
        
        Args:
            mu_min: 最小成本权重 (0 = 只关注质量)
            mu_max: 最大成本权重 (高值 = 关注成本)
            n_mu_steps: μ采样点数量
        """
        mu_values = np.linspace(mu_min, mu_max, n_mu_steps)
        
        print(f"\n{'='*80}", flush=True)
        print(f"开始实验 - Pareto边界追踪", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"μ范围: {mu_min:.2f} ~ {mu_max:.2f} (扫描 {n_mu_steps} 个点)", flush=True)
        print(f"问题数量: {len(self.test_questions)}", flush=True)
        print(f"总评估次数: {n_mu_steps}×ARGO + 3×基线 = {n_mu_steps + 3}", flush=True)
        print(f"{'='*80}\n", flush=True)
        
        # 存储ARGO的Pareto点
        argo_pareto_points = []
        
        # 1. 扫描μ值，追踪ARGO的Pareto边界
        for i, mu in enumerate(mu_values, 1):
            print(f"\n[{i}/{n_mu_steps}] μ = {mu:.2f}", flush=True)
            print(f"{'-'*80}", flush=True)
            
            # 求解MDP
            print(f"  求解MDP (μ={mu:.2f})...", end=" ", flush=True)
            theta_cont, theta_star = self.solve_mdp(mu)
            print(f"θ_cont={theta_cont:.3f}, θ*={theta_star:.3f}", flush=True)
            
            # 评估ARGO
            print(f"  评估ARGO策略 ({len(self.test_questions)}道题)...")
            argo_results = []
            for j, q in enumerate(self.test_questions, 1):
                result = self.simulate_argo_policy(q, theta_cont, theta_star)
                argo_results.append(result)
                if j % 10 == 0:
                    print(f"    进度: {j}/{len(self.test_questions)}")
            
            # 聚合结果
            avg_quality = np.mean([r['quality'] for r in argo_results])
            avg_cost = np.mean([r['cost'] for r in argo_results])
            avg_retrievals = np.mean([r['retrieval_count'] for r in argo_results])
            accuracy = np.mean([r['correct'] for r in argo_results])
            
            print(f"  ARGO: Quality={avg_quality:.3f}, Cost={avg_cost:.3f}, "
                  f"Retrievals={avg_retrievals:.1f}, Accuracy={accuracy:.1%}")
            
            argo_pareto_points.append({
                'mu': mu,
                'theta_cont': theta_cont,
                'theta_star': theta_star,
                'quality': avg_quality,
                'cost': avg_cost,
                'retrievals': avg_retrievals,
                'accuracy': accuracy
            })
        
        # 2. 评估基线策略 (单点)
        print(f"\n{'='*80}")
        print(f"评估基线策略 (固定策略)")
        print(f"{'='*80}\n")
        
        baseline_points = {}
        
        # Always-Retrieve
        print("评估 Always-Retrieve...")
        always_retrieve_results = []
        for q in self.test_questions:
            result = self.simulate_always_retrieve_policy(q)
            always_retrieve_results.append(result)
        
        baseline_points['Always-Retrieve'] = {
            'quality': np.mean([r['quality'] for r in always_retrieve_results]),
            'cost': np.mean([r['cost'] for r in always_retrieve_results]),
            'retrievals': np.mean([r['retrieval_count'] for r in always_retrieve_results]),
            'accuracy': np.mean([r['correct'] for r in always_retrieve_results])
        }
        print(f"  Quality={baseline_points['Always-Retrieve']['quality']:.3f}, "
              f"Cost={baseline_points['Always-Retrieve']['cost']:.3f}")
        
        # Always-Reason
        print("\n评估 Always-Reason...")
        always_reason_results = []
        for q in self.test_questions:
            result = self.simulate_always_reason_policy(q)
            always_reason_results.append(result)
        
        baseline_points['Always-Reason'] = {
            'quality': np.mean([r['quality'] for r in always_reason_results]),
            'cost': np.mean([r['cost'] for r in always_reason_results]),
            'retrievals': np.mean([r['retrieval_count'] for r in always_reason_results]),
            'accuracy': np.mean([r['correct'] for r in always_reason_results])
        }
        print(f"  Quality={baseline_points['Always-Reason']['quality']:.3f}, "
              f"Cost={baseline_points['Always-Reason']['cost']:.3f}")
        
        print(f"\n{'='*80}")
        print(f"实验完成!")
        print(f"{'='*80}\n")
        
        self.argo_pareto_points = argo_pareto_points
        self.baseline_points = baseline_points
        
        return {
            'argo_pareto': argo_pareto_points,
            'baselines': baseline_points
        }
    
    def save_results(self, output_dir: str = "draw_figs/data"):
        """保存结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exp3_real_pareto_frontier_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        results = {
            'argo_pareto': self.argo_pareto_points,
            'baselines': self.baseline_points
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ 结果已保存: {filepath}")
        return filepath
    
    def plot_pareto_frontier(self, output_dir: str = "figs"):
        """绘制Pareto边界 (核心图表)"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 提取ARGO的Pareto边界
        argo_costs = [p['cost'] for p in self.argo_pareto_points]
        argo_qualities = [p['quality'] for p in self.argo_pareto_points]
        
        # 创建图表
        plt.figure(figsize=(12, 8))
        
        # 绘制ARGO的Pareto边界曲线
        plt.plot(argo_costs, argo_qualities, 
                marker='o', linewidth=3, markersize=10,
                color='#1f77b4', label='ARGO (Pareto Frontier)',
                zorder=3)
        
        # 绘制基线策略的单点
        colors = {
            'Always-Retrieve': '#ff7f0e',
            'Always-Reason': '#2ca02c'
        }
        markers = {
            'Always-Retrieve': 's',
            'Always-Reason': '^'
        }
        
        for policy_name, point in self.baseline_points.items():
            plt.scatter(point['cost'], point['quality'],
                       s=200, marker=markers[policy_name],
                       color=colors[policy_name], label=policy_name,
                       edgecolors='black', linewidths=2, zorder=4)
        
        plt.xlabel('Average Total Cost ($E[C_T]$)', fontsize=14)
        plt.ylabel('Average Answer Quality ($E[Q(O)]$)', fontsize=14)
        plt.title('Experiment 3: Pareto Frontier - Cost-Quality Tradeoff (Real LLM)', 
                 fontsize=16, fontweight='bold')
        plt.legend(fontsize=12, loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig_path = os.path.join(output_dir, 'exp3_real_pareto_frontier.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Pareto边界图已保存: {fig_path}")
        
        return fig_path


def main():
    """主函数"""
    experiment = RealParetoFrontierExperiment(
        llm_model_path="/data/user/huangxiaolin/ARGO/RAG_Models/models/Qwen2.5-3B-Instruct",
        embedding_model_path="/data/user/huangxiaolin/ARGO/models/all-MiniLM-L6-v2",
        n_test_questions=30,
        difficulty="hard",
        seed=42,
        gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7]
    )
    
    # 运行实验
    results = experiment.run_experiment(
        mu_min=0.0,
        mu_max=10.0,
        n_mu_steps=10
    )
    
    # 保存结果
    experiment.save_results()
    
    # 绘制Pareto边界
    experiment.plot_pareto_frontier()
    
    print("\n实验3完成!")


if __name__ == "__main__":
    main()
