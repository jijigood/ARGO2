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
        
        # 初始化检索成功检查器
        from Environments.retrieval_success_checker import RetrievalSuccessChecker
        self.success_checker = RetrievalSuccessChecker(
            mode='probabilistic',  # 使用固定p_s进行验证
            p_s=self.config['mdp']['p_s']
        )
        print(f"✓ 检索成功检查器初始化 (mode=probabilistic, p_s={self.config['mdp']['p_s']})\n")
        
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
        """
        Evaluate answer correctness for multiple-choice questions.
        
        Extraction Strategy:
        1. Structured format: <choice>X</choice> (most reliable)
        2. Explicit patterns: "Answer: X", "Option X" (very reliable)
        3. Contextual patterns: "X is correct" (reliable)
        4. Conservative fallback: reject if unclear
        
        Args:
            question: The question text (not used for evaluation)
            answer: LLM-generated answer text
            ground_truth: Correct option number as string (e.g., "2")
        
        Returns:
            True if answer indicates correct option, False otherwise
        """
        import re
        
        # Normalize
        answer = answer.strip()
        gt = ground_truth.strip()
        
        # Validate ground truth
        if gt not in ['1', '2', '3', '4']:
            print(f"⚠️ Invalid ground truth: '{gt}' (expected '1', '2', '3', or '4')")
            return False
        
        # ===================================================================
        # Strategy 1: Structured Output (Highest Priority)
        # ===================================================================
        # Check for XML-style tags: <choice>X</choice>
        structured_match = re.search(r'<choice>\s*([1-4])\s*</choice>', answer, re.IGNORECASE)
        if structured_match:
            prediction = structured_match.group(1)
            return prediction == gt
        
        # ===================================================================
        # Strategy 2: Explicit Option Selection Patterns
        # ===================================================================
        # These patterns strongly indicate the model is selecting an option
        explicit_patterns = [
            # "Answer: 2", "Answer is 2", "The answer: 2"
            r'(?:the\s+)?answer\s*(?:is\s*)?[:\s]+([1-4])\b',
            
            # "Option 2", "Option is 2", "Select option 2"
            r'(?:select\s+)?option\s*(?:is\s*)?[:\s]*([1-4])\b',
            
            # "Choice 2", "Choose 2"
            r'(?:choose\s+)?choice\s*(?:is\s*)?[:\s]*([1-4])\b',
            
            # "2 is correct", "2 is the correct answer"
            r'\b([1-4])\s+is\s+(?:the\s+)?correct',
            
            # "Correct answer is 2", "Correct option is 2"
            r'correct\s+(?:answer|option|choice)\s+is\s+([1-4])\b',
            
            # "Therefore, 2", "So, 2", "Thus, 2"
            r'(?:therefore|thus|so|hence)[,\s]+([1-4])\b',
            
            # Starts with "2." or "2)"
            r'^([1-4])[\.\)]',
            
            # Exact match on single line (when answer is just "2")
            r'^([1-4])$',
            
            # "[2]" or "(2)"
            r'[\[\(]([1-4])[\]\)]',
            
            # "The correct answer: 2" or "Answer → 2"
            r'(?:answer|option|choice)\s*[:\→→]\s*([1-4])\b',
        ]
        
        for pattern in explicit_patterns:
            matches = re.findall(pattern, answer, re.IGNORECASE | re.MULTILINE)
            if matches:
                # Take the LAST match (models often reason first, then conclude)
                prediction = matches[-1]
                return prediction == gt
        
        # ===================================================================
        # Strategy 3: Negative Filtering (Rejection Patterns)
        # ===================================================================
        # Check if model explicitly rejected the ground truth option
        rejection_patterns = [
            rf'\b(?:option|choice)\s*{gt}\s+is\s+(?:incorrect|wrong|false)',
            rf'{gt}\s+is\s+not\s+(?:the\s+)?(?:correct|right)',
            rf'not\s+(?:option|choice)\s*{gt}\b',
        ]
        
        for pattern in rejection_patterns:
            if re.search(pattern, answer, re.IGNORECASE):
                return False  # Model explicitly rejected this option
        
        # ===================================================================
        # Strategy 4: Last Resort - End of Answer Check
        # ===================================================================
        # Only if answer ends with the GT number in isolation
        # Example: "...based on the specifications, the answer is 2"
        # This matches "2" at the very end, possibly with punctuation
        end_pattern = rf'\b{gt}\s*[\.!]?\s*$'
        if re.search(end_pattern, answer):
            # Additional check: answer should be reasonably long (not just "2")
            if len(answer) > 20:
                return True
        
        # ===================================================================
        # Default: Conservative Rejection
        # ===================================================================
        # If we can't confidently extract the choice, mark as incorrect
        # This is better than false positives
        return False
    
    def solve_mdp(self, mu: float) -> Tuple[float, float]:
        """
        求解MDP (给定成本权重μ) - 带诊断输出
        
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
        
        # 从MDP配置构建quality字典 (FIX: 使用配置的quality_function和quality_k)
        quality_config = {
            'mode': mdp_config.get('quality_function', 'linear'),
            'k': mdp_config.get('quality_k', 1.0)
        }
        
        solver_config = {
            'mdp': mdp_config,
            'quality': quality_config,
            'reward_shaping': mdp_config.get('reward_shaping', {'enabled': False, 'k': 1.0}),
            'solver': {
                'max_iterations': 1000,
                'convergence_threshold': 1e-6,
                'verbose': False
            }
        }
        
        # 求解MDP
        mdp_solver = MDPSolver(solver_config)
        mdp_solver.solve()
        
        # FIX: 添加诊断输出
        if hasattr(mdp_solver, 'V') and hasattr(mdp_solver, 'U_grid'):
            n_states = len(mdp_solver.U_grid)
            V_at_0 = mdp_solver.V[0]
            V_at_half = mdp_solver.V[n_states // 2]
            V_at_max = mdp_solver.V[-1]
            
            # 计算终止奖励
            U_max = mdp_config.get('U_max', 1.0)
            k = mdp_config.get('quality_k', 1.0)
            reward_at_09 = k * 0.9
            reward_at_10 = k * U_max
            
            # 估算从0.9到1.0的成本
            steps_needed = 0.1 / mdp_config.get('delta_r', 0.25)
            cost_to_reach_10 = steps_needed * mdp_config.get('c_r', 0.5)
            
            print(f"    [MDP诊断] V(U=0)={V_at_0:.3f}, V(U=0.5)={V_at_half:.3f}, V(U=1.0)={V_at_max:.3f}")
            print(f"    [MDP诊断] σ(0.9)={reward_at_09:.3f}, σ(1.0)={reward_at_10:.3f}, "
                  f"Cost(0.9→1.0)≈{cost_to_reach_10:.3f}")
            print(f"    [MDP诊断] μ×Cost={mu * cost_to_reach_10:.3f}, "
                  f"Reward增益={reward_at_10 - reward_at_09:.3f}")
        
        return mdp_solver.theta_cont, mdp_solver.theta_star
    
    def _compute_quality(self, U: float, correct: bool) -> float:
        """
        Calculate information quality (Option B: pure information metric)
        
        Quality = information completeness = U/U_max
        This aligns with paper's Q(O) = σ(U_T/U_max) and MDP optimization target.
        
        Answer correctness is tracked separately.
        
        Args:
            U: Current progress/uncertainty reduction
            correct: Answer correctness (NOT used in quality computation)
        
        Returns:
            Information quality ∈ [0, 1]
        """
        U_max = self.config['mdp'].get('U_max', 1.0)
        info_quality = U / U_max
        
        return info_quality
    
    def simulate_argo_policy(
        self,
        question_data: Dict,
        theta_cont: float,
        theta_star: float,
        max_steps: int = 20
    ) -> Dict:
        """
        模拟ARGO策略 (修复版本 - 带随机检索成功)
        
        根据MDP阈值动态选择动作:
        - U < theta_cont: Retrieve
        - theta_cont <= U < theta_star: Reason
        - U >= theta_star: Terminate
        
        关键修复:
        1. 检索以概率p_s成功，只有成功时才增加U
        2. 分离信息完整性和答案正确性
        3. 完整的历史追踪
        """
        question = question_data['question']
        ground_truth = question_data.get('answer', '')
        
        # 时间追踪初始化
        import time
        start_time = time.time()
        retrieval_times = []
        reasoning_times = []
        
        # 初始化
        U = 0.0  # 初始不确定性
        context = ""
        history = []  # 动作历史
        total_cost = 0.0
        retrieval_count = 0
        successful_retrieval_count = 0  # 成功的检索次数
        reason_count = 0
        
        c_r = self.config['mdp']['c_r']
        c_p = self.config['mdp']['c_p']
        delta_r = self.config['mdp']['delta_r']
        delta_p = self.config['mdp']['delta_p']
        U_max = self.config['mdp'].get('U_max', 1.0)
        
        for step in range(max_steps):
            # 终止检查
            if U >= theta_star:
                break
            
            # 动作选择
            if U < theta_cont:
                action = 'retrieve'
            else:
                action = 'reason'
            
            # 执行动作
            if action == 'retrieve':
                # 开始计时
                retrieve_start = time.time()
                
                # 1. 检索文档
                docs = self.retrieve_documents(question, top_k=3)
                
                # 2. 检查成功 (关键修复!)
                success = self.success_checker.check_success(
                    docs=docs,
                    query=question
                )
                
                # 3. 只有成功时才更新进度
                if success:
                    context += " " + " ".join(docs)
                    U = min(U + delta_r, U_max)  # 强制U_max上限
                    successful_retrieval_count += 1
                # 否则U保持不变 (检索失败)
                
                # 4. 成本无论成功失败都要付
                total_cost += c_r
                retrieval_count += 1
                
                # 记录检索时间
                retrieval_times.append(time.time() - retrieve_start)
                
                # 5. 记录历史
                history.append({
                    'step': step,
                    'action': 'retrieve',
                    'success': success,
                    'U_after': U,
                    'docs': docs if success else []
                })
            
            else:  # reason
                # 开始计时
                reason_start = time.time()
                
                # 1. 更新进度 (确定性)
                U = min(U + delta_p, U_max)
                
                # 2. 成本
                total_cost += c_p
                reason_count += 1
                
                # 记录推理时间
                reasoning_times.append(time.time() - reason_start)
                
                # 3. 记录历史
                history.append({
                    'step': step,
                    'action': 'reason',
                    'U_after': U
                })
        
        # 计算总延迟
        total_latency = time.time() - start_time
        
        # 生成最终答案
        final_answer = self.generate_answer(question, context)
        correct = self.evaluate_answer(question, final_answer, ground_truth)
        
        # 计算信息质量 (Option B: pure information metric)
        info_quality = self._compute_quality(U, correct)
        
        return {
            'information_quality': info_quality,  # ← Renamed from 'quality'
            'accuracy': correct,                   # ← Renamed from 'answer_correctness'
            'cost': total_cost,
            'retrieval_count': retrieval_count,
            'successful_retrieval_count': successful_retrieval_count,
            'retrieval_success_rate': successful_retrieval_count / retrieval_count if retrieval_count > 0 else 0.0,
            'reason_count': reason_count,
            'final_U': U,
            'num_steps': len(history),
            'correct': correct,  # Keep for compatibility
            'final_answer': final_answer,
            'history': history,
            
            # 延迟度量
            'total_latency': total_latency,
            'avg_retrieval_latency': np.mean(retrieval_times) if retrieval_times else 0.0,
            'avg_reasoning_latency': np.mean(reasoning_times) if reasoning_times else 0.0,
            'retrieval_latency_sum': sum(retrieval_times),
            'reasoning_latency_sum': sum(reasoning_times),
            'within_oran_1s': total_latency <= 1.0,
            'within_oran_100ms': total_latency <= 0.1,
        }
    
    def simulate_always_retrieve_policy(
        self,
        question_data: Dict,
        theta_star: float = 0.9,
        max_steps: int = 20
    ) -> Dict:
        """Always-Retrieve基线 (修复版本)"""
        import time
        question = question_data['question']
        ground_truth = question_data.get('answer', '')
        
        start_time = time.time()
        retrieval_times = []
        
        U = 0.0
        context = ""
        total_cost = 0.0
        retrieval_count = 0
        successful_retrieval_count = 0
        
        c_r = self.config['mdp']['c_r']
        delta_r = self.config['mdp']['delta_r']
        U_max = self.config['mdp'].get('U_max', 1.0)
        
        for step in range(max_steps):
            if U >= theta_star:
                break
            
            retrieve_start = time.time()
            docs = self.retrieve_documents(question, top_k=3)
            success = self.success_checker.check_success(docs, query=question)
            
            if success:
                context += " " + " ".join(docs)
                U = min(U + delta_r, U_max)
                successful_retrieval_count += 1
            
            total_cost += c_r
            retrieval_count += 1
            retrieval_times.append(time.time() - retrieve_start)
        
        total_latency = time.time() - start_time
        final_answer = self.generate_answer(question, context)
        correct = self.evaluate_answer(question, final_answer, ground_truth)
        info_quality = self._compute_quality(U, correct)
        
        return {
            'information_quality': info_quality,
            'accuracy': correct,
            'cost': total_cost,
            'retrieval_count': retrieval_count,
            'successful_retrieval_count': successful_retrieval_count,
            'retrieval_success_rate': successful_retrieval_count / retrieval_count if retrieval_count > 0 else 0.0,
            'reason_count': 0,
            'final_U': U,
            'correct': correct,
            'final_answer': final_answer,
            
            # 延迟度量
            'total_latency': total_latency,
            'avg_retrieval_latency': np.mean(retrieval_times) if retrieval_times else 0.0,
            'avg_reasoning_latency': 0.0,
            'retrieval_latency_sum': sum(retrieval_times),
            'reasoning_latency_sum': 0.0,
            'within_oran_1s': total_latency <= 1.0,
            'within_oran_100ms': total_latency <= 0.1,
        }
    
    def simulate_always_reason_policy(
        self,
        question_data: Dict,
        theta_star: float = 0.9,
        max_steps: int = 20
    ) -> Dict:
        """Always-Reason基线 (修复版本)"""
        import time
        question = question_data['question']
        ground_truth = question_data.get('answer', '')
        
        start_time = time.time()
        reasoning_times = []
        
        U = 0.0
        total_cost = 0.0
        reason_count = 0
        
        c_p = self.config['mdp']['c_p']
        delta_p = self.config['mdp']['delta_p']
        U_max = self.config['mdp'].get('U_max', 1.0)
        
        for step in range(max_steps):
            if U >= theta_star:
                break
            
            reason_start = time.time()
            total_cost += c_p
            reason_count += 1
            U = min(U + delta_p, U_max)
            reasoning_times.append(time.time() - reason_start)
        
        total_latency = time.time() - start_time
        final_answer = self.generate_answer(question, "")
        correct = self.evaluate_answer(question, final_answer, ground_truth)
        info_quality = self._compute_quality(U, correct)
        
        return {
            'information_quality': info_quality,
            'accuracy': correct,
            'cost': total_cost,
            'retrieval_count': 0,
            'successful_retrieval_count': 0,
            'retrieval_success_rate': 0.0,
            'reason_count': reason_count,
            'final_U': U,
            'correct': correct,
            'final_answer': final_answer,
            
            # 延迟度量
            'total_latency': total_latency,
            'avg_retrieval_latency': 0.0,
            'avg_reasoning_latency': np.mean(reasoning_times) if reasoning_times else 0.0,
            'retrieval_latency_sum': 0.0,
            'reasoning_latency_sum': sum(reasoning_times),
            'within_oran_1s': total_latency <= 1.0,
            'within_oran_100ms': total_latency <= 0.1,
        }
    
    def simulate_fixed_threshold_policy(
        self,
        question_data: Dict,
        theta_cont: float = 0.5,
        theta_star: float = 0.9,
        max_steps: int = 20
    ) -> Dict:
        """Fixed-Threshold基线 (非自适应)"""
        question = question_data['question']
        ground_truth = question_data.get('answer', '')
        
        U = 0.0
        context = ""
        total_cost = 0.0
        retrieval_count = 0
        successful_retrieval_count = 0
        reason_count = 0
        
        c_r = self.config['mdp']['c_r']
        c_p = self.config['mdp']['c_p']
        delta_r = self.config['mdp']['delta_r']
        delta_p = self.config['mdp']['delta_p']
        U_max = self.config['mdp'].get('U_max', 1.0)
        
        for step in range(max_steps):
            if U >= theta_star:
                break
            
            # 固定决策规则 (不是MDP优化的)
            if U < theta_cont:
                action = 'retrieve'
            else:
                action = 'reason'
            
            if action == 'retrieve':
                docs = self.retrieve_documents(question, top_k=3)
                success = self.success_checker.check_success(docs, query=question)
                
                if success:
                    context += " " + " ".join(docs)
                    U = min(U + delta_r, U_max)
                    successful_retrieval_count += 1
                
                total_cost += c_r
                retrieval_count += 1
            else:
                U = min(U + delta_p, U_max)
                total_cost += c_p
                reason_count += 1
        
        final_answer = self.generate_answer(question, context)
        correct = self.evaluate_answer(question, final_answer, ground_truth)
        info_quality = self._compute_quality(U, correct)
        
        return {
            'information_quality': info_quality,
            'accuracy': correct,
            'cost': total_cost,
            'retrieval_count': retrieval_count,
            'successful_retrieval_count': successful_retrieval_count,
            'retrieval_success_rate': successful_retrieval_count / retrieval_count if retrieval_count > 0 else 0.0,
            'reason_count': reason_count,
            'final_U': U,
            'correct': correct,
            'final_answer': final_answer
        }
    
    def simulate_random_policy(
        self,
        question_data: Dict,
        p_retrieve: float = 0.5,
        theta_star: float = 0.9,
        max_steps: int = 20
    ) -> Dict:
        """Random基线 (随机选择动作)"""
        import random
        
        question = question_data['question']
        ground_truth = question_data.get('answer', '')
        
        U = 0.0
        context = ""
        total_cost = 0.0
        retrieval_count = 0
        successful_retrieval_count = 0
        reason_count = 0
        
        c_r = self.config['mdp']['c_r']
        c_p = self.config['mdp']['c_p']
        delta_r = self.config['mdp']['delta_r']
        delta_p = self.config['mdp']['delta_p']
        U_max = self.config['mdp'].get('U_max', 1.0)
        
        for step in range(max_steps):
            if U >= theta_star:
                break
            
            # 随机动作选择
            if random.random() < p_retrieve:
                action = 'retrieve'
            else:
                action = 'reason'
            
            if action == 'retrieve':
                docs = self.retrieve_documents(question, top_k=3)
                success = self.success_checker.check_success(docs, query=question)
                
                if success:
                    context += " " + " ".join(docs)
                    U = min(U + delta_r, U_max)
                    successful_retrieval_count += 1
                
                total_cost += c_r
                retrieval_count += 1
            else:
                U = min(U + delta_p, U_max)
                total_cost += c_p
                reason_count += 1
        
        final_answer = self.generate_answer(question, context)
        correct = self.evaluate_answer(question, final_answer, ground_truth)
        info_quality = self._compute_quality(U, correct)
        
        return {
            'information_quality': info_quality,
            'accuracy': correct,
            'cost': total_cost,
            'retrieval_count': retrieval_count,
            'successful_retrieval_count': successful_retrieval_count,
            'retrieval_success_rate': successful_retrieval_count / retrieval_count if retrieval_count > 0 else 0.0,
            'reason_count': reason_count,
            'final_U': U,
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
            
            # 聚合结果 (Option B: separate info quality and accuracy)
            avg_info_quality = np.mean([r['information_quality'] for r in argo_results])
            avg_accuracy = np.mean([r['accuracy'] for r in argo_results])
            avg_cost = np.mean([r['cost'] for r in argo_results])
            avg_retrievals = np.mean([r['retrieval_count'] for r in argo_results])
            
            # 计算置信区间 (95%)
            std_info_quality = np.std([r['information_quality'] for r in argo_results])
            info_quality_ci = 1.96 * std_info_quality / np.sqrt(len(argo_results))
            
            std_accuracy = np.std([r['accuracy'] for r in argo_results])
            accuracy_ci = 1.96 * std_accuracy / np.sqrt(len(argo_results))
            
            std_cost = np.std([r['cost'] for r in argo_results])
            cost_ci = 1.96 * std_cost / np.sqrt(len(argo_results))
            
            # 额外的度量
            avg_latency = np.mean([r.get('total_latency', 0) for r in argo_results])
            
            print(f"  ARGO: Info_Quality={avg_info_quality:.3f}±{info_quality_ci:.3f}, "
                  f"Accuracy={avg_accuracy:.1%}±{accuracy_ci:.1%}, "
                  f"Cost={avg_cost:.3f}±{cost_ci:.3f}, "
                  f"Retrievals={avg_retrievals:.1f}")
            
            argo_pareto_points.append({
                'mu': mu,
                'theta_cont': theta_cont,
                'theta_star': theta_star,
                'information_quality': avg_info_quality,      # ← New name
                'accuracy': avg_accuracy,                      # ← New name  
                'cost': avg_cost,
                'retrievals': avg_retrievals,
                
                # Confidence intervals
                'info_quality_ci': info_quality_ci,
                'accuracy_ci': accuracy_ci,
                'cost_ci': cost_ci,
                
                # Other metrics
                'avg_latency': avg_latency,
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
        
        baseline_points['Always-Retrieve'] = self._aggregate_results(always_retrieve_results)
        print(f"  Info_Quality={baseline_points['Always-Retrieve']['information_quality']:.3f}, "
              f"Accuracy={baseline_points['Always-Retrieve']['accuracy']:.3f}, "
              f"Cost={baseline_points['Always-Retrieve']['cost']:.3f}")
        
        # Always-Reason
        print("\n评估 Always-Reason...")
        always_reason_results = []
        for q in self.test_questions:
            result = self.simulate_always_reason_policy(q)
            always_reason_results.append(result)
        
        baseline_points['Always-Reason'] = self._aggregate_results(always_reason_results)
        print(f"  Info_Quality={baseline_points['Always-Reason']['information_quality']:.3f}, "
              f"Accuracy={baseline_points['Always-Reason']['accuracy']:.3f}, "
              f"Cost={baseline_points['Always-Reason']['cost']:.3f}")
        
        # Fixed-Threshold (NEW)
        print("\n评估 Fixed-Threshold (θ_cont=0.5, θ*=0.9)...")
        fixed_results = []
        for q in self.test_questions:
            result = self.simulate_fixed_threshold_policy(q, theta_cont=0.5, theta_star=0.9)
            fixed_results.append(result)
        
        baseline_points['Fixed-Threshold'] = self._aggregate_results(fixed_results)
        print(f"  Info_Quality={baseline_points['Fixed-Threshold']['information_quality']:.3f}, "
              f"Accuracy={baseline_points['Fixed-Threshold']['accuracy']:.3f}, "
              f"Cost={baseline_points['Fixed-Threshold']['cost']:.3f}")
        
        # Random (NEW)
        print("\n评估 Random (p=0.5)...")
        random_results = []
        for q in self.test_questions:
            result = self.simulate_random_policy(q, p_retrieve=0.5)
            random_results.append(result)
        
        baseline_points['Random'] = self._aggregate_results(random_results)
        print(f"  Info_Quality={baseline_points['Random']['information_quality']:.3f}, "
              f"Accuracy={baseline_points['Random']['accuracy']:.3f}, "
              f"Cost={baseline_points['Random']['cost']:.3f}")
        
        print(f"\n{'='*80}")
        print(f"实验完成!")
        print(f"{'='*80}\n")
        
        self.argo_pareto_points = argo_pareto_points
        self.baseline_points = baseline_points
        
        return {
            'argo_pareto': argo_pareto_points,
            'baselines': baseline_points
        }
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate metrics (Option B: separate info quality and accuracy)"""
        info_qualities = [r['information_quality'] for r in results]
        accuracies = [r['accuracy'] for r in results]
        costs = [r['cost'] for r in results]
        
        # Compute confidence intervals
        info_quality_ci = 1.96 * np.std(info_qualities) / np.sqrt(len(info_qualities)) if len(info_qualities) > 1 else 0.0
        accuracy_ci = 1.96 * np.std(accuracies) / np.sqrt(len(accuracies)) if len(accuracies) > 1 else 0.0
        cost_ci = 1.96 * np.std(costs) / np.sqrt(len(costs)) if len(costs) > 1 else 0.0
        
        return {
            'information_quality': np.mean(info_qualities),    # ← New name
            'accuracy': np.mean(accuracies),                   # ← New name
            'cost': np.mean(costs),
            'retrievals': np.mean([r['retrieval_count'] for r in results]),
            'retrieval_success_rate': np.mean([r.get('retrieval_success_rate', 0) for r in results]),
            
            # Confidence intervals
            'info_quality_ci': info_quality_ci,
            'accuracy_ci': accuracy_ci,
            'cost_ci': cost_ci,
            
            # Other metrics
            'avg_latency': np.mean([r.get('total_latency', 0) for r in results]),
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
        """Plot Pareto frontier: Cost vs Information Quality (what MDP optimizes)"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract ARGO data
        argo_costs = [p['cost'] for p in self.argo_pareto_points]
        argo_info_qualities = [p['information_quality'] for p in self.argo_pareto_points]  # ← Changed
        cost_cis = [p.get('cost_ci', 0) for p in self.argo_pareto_points]
        info_quality_cis = [p.get('info_quality_ci', 0) for p in self.argo_pareto_points]  # ← Changed
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot ARGO Pareto frontier
        plt.errorbar(
            argo_costs, argo_info_qualities,  # ← Changed
            xerr=cost_cis, yerr=info_quality_cis,  # ← Changed
            fmt='o-', linewidth=3, markersize=10,
            color='#1f77b4', label='ARGO (Pareto Frontier)',
            capsize=5, capthick=2, elinewidth=2,
            zorder=3
        )
        
        # Plot baselines
        colors = {
            'Always-Retrieve': '#ff7f0e',
            'Always-Reason': '#2ca02c',
            'Fixed-Threshold': '#d62728',
            'Random': '#9467bd'
        }
        markers = {
            'Always-Retrieve': 's',
            'Always-Reason': '^',
            'Fixed-Threshold': 'D',
            'Random': 'v'
        }
        
        for policy_name, point in self.baseline_points.items():
            plt.errorbar(
                point['cost'], point['information_quality'],  # ← Changed
                xerr=point.get('cost_ci', 0),
                yerr=point.get('info_quality_ci', 0),  # ← Changed
                fmt=markers.get(policy_name, 'o'),
                markersize=12,
                color=colors.get(policy_name, 'gray'),
                label=policy_name,
                capsize=4, capthick=1.5,
                markeredgecolor='black', markeredgewidth=2,
                zorder=4
            )
        
        plt.xlabel('Average Total Cost ($E[C_T]$)', fontsize=14, fontweight='bold')
        plt.ylabel('Information Quality ($E[Q_{info}]$)', fontsize=14, fontweight='bold')  # ← Changed
        plt.title('Experiment 3: Pareto Frontier (Cost vs Information Quality)', 
                 fontsize=16, fontweight='bold')
        plt.legend(fontsize=12, loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig_path = os.path.join(output_dir, 'exp3_pareto_info_quality.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Pareto frontier (Info Quality) saved: {fig_path}")
        
        return fig_path
    
    def plot_pareto_accuracy(self, output_dir: str = "figs"):
        """Plot second Pareto frontier: Cost vs Accuracy (what users care about)"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract ARGO data
        argo_costs = [p['cost'] for p in self.argo_pareto_points]
        argo_accuracies = [p['accuracy'] for p in self.argo_pareto_points]
        cost_cis = [p.get('cost_ci', 0) for p in self.argo_pareto_points]
        accuracy_cis = [p.get('accuracy_ci', 0) for p in self.argo_pareto_points]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot ARGO accuracy frontier
        plt.errorbar(
            argo_costs, argo_accuracies,
            xerr=cost_cis, yerr=accuracy_cis,
            fmt='o-', linewidth=3, markersize=10,
            color='#2ca02c', label='ARGO',
            capsize=5, capthick=2, elinewidth=2,
            zorder=3
        )
        
        # Plot baselines
        colors = {
            'Always-Retrieve': '#ff7f0e',
            'Always-Reason': '#2ca02c',
            'Fixed-Threshold': '#d62728',
            'Random': '#9467bd'
        }
        markers = {
            'Always-Retrieve': 's',
            'Always-Reason': '^',
            'Fixed-Threshold': 'D',
            'Random': 'v'
        }
        
        for policy_name, point in self.baseline_points.items():
            plt.errorbar(
                point['cost'], point['accuracy'],
                xerr=point.get('cost_ci', 0),
                yerr=point.get('accuracy_ci', 0),
                fmt=markers.get(policy_name, 'o'),
                markersize=12,
                color=colors.get(policy_name, 'gray'),
                label=policy_name,
                capsize=4, capthick=1.5,
                markeredgecolor='black', markeredgewidth=2,
                zorder=4
            )
        
        plt.xlabel('Average Total Cost ($E[C_T]$)', fontsize=14, fontweight='bold')
        plt.ylabel('Answer Accuracy', fontsize=14, fontweight='bold')
        plt.title('Experiment 3: Cost vs Accuracy (End-User Performance)', 
                 fontsize=16, fontweight='bold')
        plt.legend(fontsize=12, loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.05])
        plt.tight_layout()
        
        fig_path = os.path.join(output_dir, 'exp3_pareto_accuracy.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Pareto frontier (Accuracy) saved: {fig_path}")
        
        return fig_path
    
    def plot_threshold_evolution(self, output_dir: str = "figs"):
        """可视化阈值如何随μ变化 (验证定理1)"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 提取数据
        mu_values = [p['mu'] for p in self.argo_pareto_points]
        theta_conts = [p['theta_cont'] for p in self.argo_pareto_points]
        theta_stars = [p['theta_star'] for p in self.argo_pareto_points]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 绘制两个阈值
        ax.plot(mu_values, theta_stars, 'o-', 
                linewidth=3, markersize=12,
                color='#d62728', label='θ* (Termination Threshold)',
                markeredgecolor='black', markeredgewidth=2)
        
        ax.plot(mu_values, theta_conts, 's-',
                linewidth=3, markersize=12,
                color='#1f77b4', label='θ_cont (Retrieve/Reason Threshold)',
                markeredgecolor='black', markeredgewidth=2)
        
        # 填充阈值之间的区域
        ax.fill_between(mu_values, theta_conts, theta_stars, 
                         alpha=0.2, color='gray',
                         label='Reasoning Region [θ_cont, θ*)')
        
        ax.set_xlabel('Cost Weight μ', fontsize=14, fontweight='bold')
        ax.set_ylabel('Threshold Value', fontsize=14, fontweight='bold')
        ax.set_title('Two-Level Threshold Structure (Theorem 1 Validation)',
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0, 1.05])
        
        # 添加注释
        ax.annotate('Higher μ → Earlier termination\n(Cost-conscious)',
                    xy=(mu_values[-1], theta_stars[-1]),
                    xytext=(mu_values[-1] - 2, theta_stars[-1] + 0.15),
                    fontsize=11, ha='right',
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        ax.annotate('Lower μ → More information gathering\n(Quality-focused)',
                    xy=(mu_values[0], theta_stars[0]),
                    xytext=(mu_values[0] + 2, theta_stars[0] - 0.15),
                    fontsize=11, ha='left',
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2))
        
        plt.tight_layout()
        fig_path = os.path.join(output_dir, 'exp3_threshold_evolution.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 阈值演化图已保存: {fig_path}")
        return fig_path
    
    def plot_comprehensive_dashboard(self, output_dir: str = "figs"):
        """创建综合 2x2 仪表板 (Option B: separate info quality and accuracy)"""
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 提取数据
        mu_values = [p['mu'] for p in self.argo_pareto_points]
        costs = [p['cost'] for p in self.argo_pareto_points]
        info_qualities = [p['information_quality'] for p in self.argo_pareto_points]  # ← Changed
        retrievals = [p['retrievals'] for p in self.argo_pareto_points]
        accuracies = [p['accuracy'] for p in self.argo_pareto_points]
        
        # (0,0) Pareto边界 - Information Quality
        ax = axes[0, 0]
        ax.plot(costs, info_qualities, 'o-', linewidth=3, markersize=10,  # ← Changed
               color='#1f77b4', label='ARGO', zorder=3)
        
        # 添加基线
        colors = {'Always-Retrieve': '#ff7f0e', 'Always-Reason': '#2ca02c',
                 'Fixed-Threshold': '#d62728', 'Random': '#9467bd'}
        markers = {'Always-Retrieve': 's', 'Always-Reason': '^',
                  'Fixed-Threshold': 'D', 'Random': 'v'}
        
        for name, point in self.baseline_points.items():
            ax.scatter(point['cost'], point['information_quality'],  # ← Changed
                      s=200, marker=markers.get(name, 'o'),
                      color=colors.get(name, 'gray'),
                      label=name, edgecolors='black', linewidths=2, zorder=4)
        
        ax.set_xlabel('Cost', fontsize=12, fontweight='bold')
        ax.set_ylabel('Information Quality', fontsize=12, fontweight='bold')  # ← Changed
        ax.set_title('(a) Pareto Frontier (Info Quality)', fontsize=14, fontweight='bold')  # ← Changed
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # (0,1) 检索次数 vs μ
        ax = axes[0, 1]
        ax.plot(mu_values, retrievals, 's-', linewidth=3, markersize=10,
               color='#ff7f0e', label='ARGO')
        
        # 基线水平线
        for name, point in self.baseline_points.items():
            if 'retrievals' in point:
                linestyles = {'Always-Retrieve': '--', 'Always-Reason': ':',
                            'Fixed-Threshold': '-.', 'Random': (0, (3, 1, 1, 1))}
                ax.axhline(y=point['retrievals'],
                          linestyle=linestyles.get(name, '-'),
                          linewidth=2, label=f'{name}', alpha=0.7)
        
        ax.set_xlabel('Cost Weight μ', fontsize=12, fontweight='bold')
        ax.set_ylabel('Avg Retrieval Count', fontsize=12, fontweight='bold')
        ax.set_title('(b) Retrieval Efficiency', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # (1,0) 准确率 vs 成本 (NEW PANEL)
        ax = axes[1, 0]
        ax.plot(costs, accuracies, 'o-', linewidth=3, markersize=10,
               color='#2ca02c', label='ARGO')
        
        # 基线
        for name, point in self.baseline_points.items():
            ax.scatter(point['cost'], point['accuracy'],
                      s=200, marker=markers.get(name, 'o'),
                      color=colors.get(name, 'gray'),
                      label=name, edgecolors='black', linewidths=2)
        
        ax.set_xlabel('Cost', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('(c) Accuracy vs Cost', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        # (1,1) Dual Quality Metrics vs μ (UPDATED PANEL)
        ax = axes[1, 1]
        ax.plot(mu_values, info_qualities, 'o-', linewidth=3, markersize=10,  # ← Changed
               color='#1f77b4', label='Information Quality')
        ax.plot(mu_values, accuracies, '^--', linewidth=2, markersize=8,
               color='#2ca02c', label='Answer Accuracy', alpha=0.7)
        
        ax.set_xlabel('Cost Weight μ', fontsize=12, fontweight='bold')
        ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
        ax.set_title('(d) Information Quality vs Accuracy', fontsize=14, fontweight='bold')  # ← Changed
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        plt.suptitle('Experiment 3: Comprehensive Analysis Dashboard',
                    fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        fig_path = os.path.join(output_dir, 'exp3_dashboard.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 综合仪表板已保存: {fig_path}")
        return fig_path
    
    def plot_latency_analysis(self, output_dir: str = "figs"):
        """绘制延迟分析图 (O-RAN合规性)"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 提取数据
        mu_values = [p['mu'] for p in self.argo_pareto_points]
        costs = [p['cost'] for p in self.argo_pareto_points]
        latencies = [p.get('avg_latency', 0) for p in self.argo_pareto_points]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左图: 成本 vs 延迟
        ax1.plot(costs, latencies, 'o-', linewidth=3, markersize=10, color='#1f77b4')
        ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='O-RAN 1s limit')
        ax1.axhline(y=0.1, color='orange', linestyle='--', linewidth=2, label='O-RAN 100ms limit')
        ax1.set_xlabel('Cost', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Latency (seconds)', fontsize=14, fontweight='bold')
        ax1.set_title('(a) Latency vs Cost', fontsize=16, fontweight='bold')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 右图: μ vs 延迟
        ax2.plot(mu_values, latencies, 's-', linewidth=3, markersize=10, color='#2ca02c')
        ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='O-RAN 1s limit')
        ax2.axhline(y=0.1, color='orange', linestyle='--', linewidth=2, label='O-RAN 100ms limit')
        ax2.set_xlabel('Cost Weight μ', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Latency (seconds)', fontsize=14, fontweight='bold')
        ax2.set_title('(b) Latency vs Cost Weight μ', fontsize=16, fontweight='bold')
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_path = os.path.join(output_dir, 'exp3_latency_analysis.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 延迟分析图已保存: {fig_path}")
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
    
    # 绘制阈值演化图
    experiment.plot_threshold_evolution()
    
    # 绘制综合仪表板
    experiment.plot_comprehensive_dashboard()
    
    print("\n实验3完成!")


if __name__ == "__main__":
    main()
