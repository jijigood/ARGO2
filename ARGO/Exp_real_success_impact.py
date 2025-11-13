#!/usr/bin/env python
"""
实验2: 检索成功率影响 (真实LLM版本)
============================================
使用真实的Qwen模型和嵌入模型，多GPU并行

硬件要求:
- 多张GPU (支持 RTX 3060 x8)
- CUDA环境

模型:
- LLM: Qwen2.5-7B-Instruct 或 Qwen2.5-14B-Instruct
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
from typing import Dict, List
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# 尝试导入chromadb (可能失败)
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError as e:
    print(f"⚠ ChromaDB不可用: {e}")
    print(f"  将使用模拟检索模式")
    CHROMADB_AVAILABLE = False
    chromadb = None

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oran_benchmark_loader import ORANBenchmark

sys.path.insert(0, '../ARGO_MDP/src')
from mdp_solver import MDPSolver


class RealSuccessImpactExperiment:
    """实验2: 检索成功率影响 - 真实LLM版本"""
    
    def __init__(
        self,
        config_path: str = "configs/multi_gpu.yaml",
        llm_model_path: str = "/data/user/huangxiaolin/ARGO/RAG_Models/models/Qwen2.5-14B-Instruct",
        embedding_model_path: str = "/data/user/huangxiaolin/ARGO/models/all-MiniLM-L6-v2",
        chroma_db_path: str = "/data/user/huangxiaolin/ARGO2/ARGO/Environments/chroma_store",
        n_test_questions: int = 50,
        difficulty: str = "hard",  # Hard难度
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
        print(f"实验2: 检索成功率影响 (真实LLM版本)")
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
        self.gpu_ids = gpu_ids if gpu_ids else list(range(min(4, self.n_gpus)))
        
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
        self.embedding_model = self.embedding_model.to(f'cuda:{self.gpu_ids[0]}')
        print(f"✓ 嵌入模型加载成功 (GPU {self.gpu_ids[0]})\n")
        
        # 加载Chroma检索库
        print(f"连接Chroma数据库: {chroma_db_path}")
        if CHROMADB_AVAILABLE:
            try:
                self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
                self.collection = self.chroma_client.get_collection("oran_specs")
                print(f"✓ Chroma集合加载成功 (文档数: {self.collection.count()})\n")
            except Exception as e:
                print(f"⚠ Chroma集合加载失败: {e}")
                print(f"  将使用模拟检索模式\n")
                self.collection = None
        else:
            print(f"⚠ ChromaDB不可用，使用模拟检索模式\n")
            self.collection = None
        
        # 加载LLM模型
        print(f"加载LLM模型: {llm_model_path}")
        self._load_llm()
        
        print(f"\n{'='*80}")
        print(f"初始化完成!")
        print(f"{'='*80}\n")
    
    def _load_llm(self):
        """加载LLM (多GPU并行)"""
        print(f"  使用 {len(self.gpu_ids)} 张GPU加载模型...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.llm_model_path,
            trust_remote_code=True,
            padding_side='left'
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 使用Accelerate自动分配到多个GPU
        max_memory = {i: "10GB" for i in self.gpu_ids}
        max_memory["cpu"] = "30GB"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory=max_memory,
            trust_remote_code=True,
            offload_folder="offload"
        )
        
        self.model.eval()
        
        print(f"✓ LLM模型加载成功")
        print(f"  Device map: {self.model.hf_device_map}")
    
    def create_mdp_config(self, p_s: float) -> Dict:
        """创建MDP配置"""
        mdp_config = self.config['mdp'].copy()
        mdp_config['p_s'] = p_s  # 变化检索成功率
        
        # 添加 U_grid_size (兼容性)
        if 'U_grid_size' not in mdp_config and 'grid_size' in mdp_config:
            mdp_config['U_grid_size'] = mdp_config['grid_size']
        
        return {
            'mdp': mdp_config,
            'quality': self.config.get('quality', {'mode': 'linear', 'k': 5.0}),
            'solver': {
                'max_iterations': 1000,
                'convergence_threshold': 1e-6,
                'verbose': False
            }
        }
    
    def solve_mdp(self, p_s: float) -> tuple:
        """求解MDP获取阈值"""
        print(f"  求解MDP (p_s={p_s:.2f})...", end=" ")
        
        config = self.create_mdp_config(p_s)
        solver = MDPSolver(config)
        solver.solve()
        
        theta_cont = solver.theta_cont
        theta_star = solver.theta_star
        
        print(f"θ_cont={theta_cont:.3f}, θ*={theta_star:.3f}")
        return theta_cont, theta_star
    
    def retrieve_documents(self, question: str, top_k: int = 3) -> List[str]:
        """检索相关文档"""
        if self.collection is None:
            return [f"模拟文档 {i+1}" for i in range(top_k)]
        
        query_embedding = self.embedding_model.encode(question, convert_to_tensor=True)
        query_embedding = query_embedding.cpu().numpy().tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents"]
        )
        
        documents = results.get("documents", [[]])[0]
        return documents
    
    def generate_answer(self, question: Dict, context: str = "") -> tuple:
        """使用LLM生成答案"""
        prompt = self._create_prompt(question, context)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        inputs = {k: v.to(f'cuda:{self.gpu_ids[0]}') for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = self._extract_answer(response)
        
        confidence = 0.7 if context else 0.5
        
        return answer, confidence
    
    def _create_prompt(self, question: Dict, context: str = "") -> str:
        """创建提示"""
        context_part = f"\nContext: {context}\n" if context else "\n"
        
        prompt = f"""You are an O-RAN standards expert. Answer the following question.{context_part}
Question: {question['question']}

Options:
1. {question['options'][0]}
2. {question['options'][1]}
3. {question['options'][2]}
4. {question['options'][3]}

Answer with only the number (1, 2, 3, or 4):"""
        
        return prompt
    
    def _extract_answer(self, response: str) -> int:
        """从响应中提取答案"""
        import re
        
        response = response.lower()
        matches = re.findall(r'\b([1-4])\b', response)
        
        if matches:
            return int(matches[-1])
        
        return 1
    
    def simulate_argo_policy(self, question: Dict, theta_cont: float, theta_star: float, p_s: float) -> Dict:
        """执行ARGO策略"""
        U = 0.0
        C = 0.0
        retrieval_count = 0
        reason_count = 0
        
        delta_r = self.config['mdp']['delta_r']
        delta_p = self.config['mdp']['delta_p']
        c_r = self.config['mdp']['c_r']
        c_p = self.config['mdp']['c_p']
        
        max_steps = 30  # 增加步数(低p_s需要更多重试)
        final_answer = None
        
        for step in range(max_steps):
            if U >= theta_star:
                break
            
            if U < theta_cont:
                # Retrieve
                retrieval_count += 1
                C += c_r
                
                docs = self.retrieve_documents(question['question'], top_k=3)
                context = " ".join(docs)
                
                # 使用当前p_s
                if random.random() < p_s:
                    U += delta_r
                    final_answer, _ = self.generate_answer(question, context)
                else:
                    final_answer, _ = self.generate_answer(question, context)
            else:
                # Reason
                reason_count += 1
                C += c_p
                U += delta_p
                
                final_answer, _ = self.generate_answer(question, "")
        
        quality = min(U / 1.0, 1.0)
        correct = (final_answer == question['correct_answer']) if final_answer else False
        
        return {
            'quality': quality,
            'cost': C,
            'retrieval_count': retrieval_count,
            'reason_count': reason_count,
            'steps': step + 1,
            'correct': correct
        }
    
    def simulate_always_retrieve_policy(self, question: Dict, p_s: float) -> Dict:
        """Always-Retrieve基线"""
        U = 0.0
        C = 0.0
        retrieval_count = 0
        
        delta_r = self.config['mdp']['delta_r']
        c_r = self.config['mdp']['c_r']
        theta_star = 0.9
        
        max_steps = 30
        final_answer = None
        
        for step in range(max_steps):
            if U >= theta_star:
                break
            
            retrieval_count += 1
            C += c_r
            
            docs = self.retrieve_documents(question['question'], top_k=3)
            context = " ".join(docs)
            
            if random.random() < p_s:
                U += delta_r
            
            final_answer, _ = self.generate_answer(question, context)
        
        quality = min(U / 1.0, 1.0)
        correct = (final_answer == question['correct_answer']) if final_answer else False
        
        return {
            'quality': quality,
            'cost': C,
            'retrieval_count': retrieval_count,
            'reason_count': 0,
            'steps': step + 1,
            'correct': correct
        }
    
    def simulate_always_reason_policy(self, question: Dict) -> Dict:
        """Always-Reason基线"""
        U = 0.0
        C = 0.0
        reason_count = 0
        
        delta_p = self.config['mdp']['delta_p']
        c_p = self.config['mdp']['c_p']
        theta_star = 0.9
        
        max_steps = 30
        final_answer = None
        
        for step in range(max_steps):
            if U >= theta_star:
                break
            
            reason_count += 1
            C += c_p
            U += delta_p
            
            final_answer, _ = self.generate_answer(question, "")
        
        quality = min(U / 1.0, 1.0)
        correct = (final_answer == question['correct_answer']) if final_answer else False
        
        return {
            'quality': quality,
            'cost': C,
            'retrieval_count': 0,
            'reason_count': reason_count,
            'steps': step + 1,
            'correct': correct
        }
    
    def evaluate_all_policies(self, p_s: float, theta_cont: float, theta_star: float) -> Dict:
        """评估所有策略"""
        results = {
            'ARGO': [],
            'Always-Retrieve': [],
            'Always-Reason': []
        }
        
        print(f"\n  评估 {len(self.test_questions)} 道问题...")
        
        for i, question in enumerate(self.test_questions, 1):
            if i % 10 == 0:
                print(f"    进度: {i}/{len(self.test_questions)}")
            
            # ARGO
            result = self.simulate_argo_policy(question, theta_cont, theta_star, p_s)
            results['ARGO'].append(result)
            
            # Always-Retrieve
            result = self.simulate_always_retrieve_policy(question, p_s)
            results['Always-Retrieve'].append(result)
            
            # Always-Reason
            result = self.simulate_always_reason_policy(question)
            results['Always-Reason'].append(result)
        
        return results
    
    def run_experiment(
        self,
        p_s_min: float = 0.3,
        p_s_max: float = 1.0,
        n_steps: int = 4  # 减少到4个点
    ):
        """运行实验"""
        p_s_values = np.linspace(p_s_min, p_s_max, n_steps)
        
        print(f"\n{'='*80}")
        print(f"开始实验 - 检索成功率影响")
        print(f"{'='*80}")
        print(f"p_s范围: {p_s_values[0]:.2f} ~ {p_s_values[-1]:.2f} (扫描 {n_steps} 个点)")
        print(f"c_r固定: {self.config['mdp']['c_r']:.3f}")
        print(f"问题数量: {len(self.test_questions)}")
        print(f"总评估次数: {n_steps} × 3策略 × {len(self.test_questions)}题 = {n_steps * 3 * len(self.test_questions)}")
        print(f"{'='*80}\n")
        
        all_results = []
        
        for i, p_s in enumerate(p_s_values, 1):
            print(f"\n[{i}/{n_steps}] p_s = {p_s:.2f}")
            print(f"{'-'*80}")
            
            # 求解MDP
            theta_cont, theta_star = self.solve_mdp(p_s)
            
            # 评估所有策略
            results = self.evaluate_all_policies(p_s, theta_cont, theta_star)
            
            # 聚合结果
            aggregated = {
                'p_s': p_s,
                'theta_cont': theta_cont,
                'theta_star': theta_star
            }
            
            for policy_name, policy_results in results.items():
                avg_quality = np.mean([r['quality'] for r in policy_results])
                avg_cost = np.mean([r['cost'] for r in policy_results])
                avg_retrievals = np.mean([r['retrieval_count'] for r in policy_results])
                avg_reasons = np.mean([r['reason_count'] for r in policy_results])
                accuracy = np.mean([r['correct'] for r in policy_results])
                
                aggregated[f'{policy_name}_quality'] = avg_quality
                aggregated[f'{policy_name}_cost'] = avg_cost
                aggregated[f'{policy_name}_retrievals'] = avg_retrievals
                aggregated[f'{policy_name}_reasons'] = avg_reasons
                aggregated[f'{policy_name}_accuracy'] = accuracy
                
                print(f"  {policy_name:20s}: Quality={avg_quality:.3f}, Cost={avg_cost:.3f}, "
                      f"Retrievals={avg_retrievals:.1f}, Accuracy={accuracy:.1%}")
            
            all_results.append(aggregated)
        
        self.results = all_results
        
        print(f"\n{'='*80}")
        print(f"实验完成!")
        print(f"{'='*80}\n")
        
        return all_results
    
    def save_results(self, output_dir: str = "draw_figs/data"):
        """保存结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exp2_real_success_impact_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"✓ 结果已保存: {filepath}")
        return filepath
    
    def plot_results(self, output_dir: str = "figs"):
        """绘制结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        p_s_values = [r['p_s'] for r in self.results]
        
        # 图1: 质量 vs 成功率
        plt.figure(figsize=(10, 6))
        for policy in ['ARGO', 'Always-Retrieve', 'Always-Reason']:
            quality = [r[f'{policy}_quality'] for r in self.results]
            plt.plot(p_s_values, quality, marker='o', label=policy, linewidth=2)
        
        plt.xlabel('Retrieval Success Rate (p_s)', fontsize=12)
        plt.ylabel('Average Quality', fontsize=12)
        plt.title('实验2: 质量 vs 检索成功率 (真实LLM)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig1_path = os.path.join(output_dir, 'exp2_real_ps_vs_quality.png')
        plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 图表已保存: {fig1_path}")
        
        # 图2: 检索次数 vs 成功率
        plt.figure(figsize=(10, 6))
        for policy in ['ARGO', 'Always-Retrieve']:
            retrievals = [r[f'{policy}_retrievals'] for r in self.results]
            plt.plot(p_s_values, retrievals, marker='o', label=policy, linewidth=2)
        
        plt.xlabel('Retrieval Success Rate (p_s)', fontsize=12)
        plt.ylabel('Average Retrievals', fontsize=12)
        plt.title('实验2: 检索次数 vs 检索成功率 (真实LLM)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig2_path = os.path.join(output_dir, 'exp2_real_ps_vs_retrievals.png')
        plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 图表已保存: {fig2_path}")
        
        # 图3: 准确率 vs 成功率
        plt.figure(figsize=(10, 6))
        for policy in ['ARGO', 'Always-Retrieve', 'Always-Reason']:
            accuracy = [r[f'{policy}_accuracy'] for r in self.results]
            plt.plot(p_s_values, accuracy, marker='o', label=policy, linewidth=2)
        
        plt.xlabel('Retrieval Success Rate (p_s)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('实验2: 准确率 vs 检索成功率 (真实LLM)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig3_path = os.path.join(output_dir, 'exp2_real_ps_vs_accuracy.png')
        plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 图表已保存: {fig3_path}")


def main():
    """主函数"""
    experiment = RealSuccessImpactExperiment(
        llm_model_path="/data/user/huangxiaolin/ARGO/RAG_Models/models/Qwen2.5-14B-Instruct",
        embedding_model_path="/data/user/huangxiaolin/ARGO/models/all-MiniLM-L6-v2",
        n_test_questions=50,  # 50道Hard题
        difficulty="hard",    # Hard难度
        seed=42,
        gpu_ids=[0, 1, 2, 3]  # 使用4张GPU
    )
    
    # 运行实验
    results = experiment.run_experiment(
        p_s_min=0.3,
        p_s_max=1.0,
        n_steps=4  # 4个p_s采样点
    )
    
    # 保存结果
    experiment.save_results()
    
    # 绘图
    experiment.plot_results()
    
    print("\n实验2完成!")


if __name__ == "__main__":
    main()
