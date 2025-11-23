#!/usr/bin/env python
"""
实验1: 检索成本影响 (真实LLM版本 - 修正版)
============================================
使用真实的Qwen模型和嵌入模型，多GPU并行

修正内容:
1. ✓ 添加Random策略
2. ✓ 基线策略使用动态θ*（而非硬编码0.9）
3. ✓ 支持小规模测试模式和大规模实验模式切换
4. ✓ 图表命名与实验设计文档一致

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
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

# 添加src路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from progress import ProgressTracker
from complexity import QuestionComplexityClassifier
from oran_benchmark_loader import ORANBenchmark

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


class RealCostImpactExperiment:
    """实验1: 检索成本影响 - 真实LLM版本 (修正版)"""
    
    def __init__(
        self,
        config_path: str = "configs/multi_gpu_data_calibrated.yaml",
        policy_config_path: Optional[str] = None,
        llm_model_path: str = "/data/user/huangxiaolin/ARGO/RAG_Models/models/Qwen2.5-14B-Instruct",
        embedding_model_path: str = "/data/user/huangxiaolin/ARGO/models/all-MiniLM-L6-v2",
        chroma_db_path: str = "/data/user/huangxiaolin/ARGO2/ARGO/Environments/chroma_store",
        test_mode: str = "small",  # "small" (快速测试), "full" (完整实验), 或 "custom" (自定义)
        n_test_questions: Optional[int] = None,  # 自定义问题数量（仅用于custom模式）
        difficulty: str = "hard",
        seed: int = 42,
        gpu_ids: List[int] = None
    ):
        """
        Args:
            config_path: MDP配置文件路径
            policy_config_path: 自适应策略配置文件路径
            llm_model_path: Qwen模型本地路径
            embedding_model_path: 嵌入模型本地路径
            chroma_db_path: Chroma数据库路径
            test_mode: "small" (10题, 5个c_r点), "full" (全部~12K题, 10个c_r点), 或 "custom" (自定义)
            n_test_questions: 自定义问题数量 (仅当test_mode="custom"时使用)
            difficulty: 问题难度 ("easy", "medium", "hard")
            seed: 随机种子
            gpu_ids: 使用的GPU ID列表，如 [0,1,2,3]
        """
        self.policy_config_path = policy_config_path
        # 根据测试模式设置参数
        if test_mode == "small":
            self.n_test_questions = 10
            self.n_cost_steps = 5
            self.mode_desc = "小规模测试模式 (快速验证)"
        elif test_mode == "full":
            self.n_test_questions = None  # 使用全部数据集
            self.n_cost_steps = 10
            self.mode_desc = "完整实验模式 (全部数据)"
        elif test_mode == "custom":
            if n_test_questions is None:
                raise ValueError("custom模式必须指定 n_test_questions 参数")
            self.n_test_questions = n_test_questions
            self.n_cost_steps = 10
            self.mode_desc = f"自定义模式 ({n_test_questions}题)"
        else:
            raise ValueError(f"test_mode必须是'small', 'full', 或 'custom'，当前值: {test_mode}")
        
        self.test_mode = test_mode
        
        print(f"\n{'='*80}")
        print(f"实验1: 检索成本影响 (真实LLM版本 - 修正版)")
        print(f"{'='*80}")
        print(f"运行模式: {self.mode_desc}")
        print(f"LLM模型: {llm_model_path}")
        print(f"嵌入模型: {embedding_model_path}")
        print(f"问题难度: {difficulty.upper()}")
        print(f"问题数量: {self.n_test_questions if self.n_test_questions else '全部 (~12K)'}")
        print(f"c_r采样点: {self.n_cost_steps}个")
        print(f"{'='*80}\n")
        
        self.config_path = config_path
        self.llm_model_path = llm_model_path
        self.embedding_model_path = embedding_model_path
        self.chroma_db_path = chroma_db_path
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
            
        # 加载策略配置
        self.policy_config = None
        if self.policy_config_path:
            print(f"加载自适应策略配置: {self.policy_config_path}")
            with open(self.policy_config_path, 'r') as f:
                self.policy_config = yaml.safe_load(f)
        
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 加载数据集
        print("加载ORAN-Bench-13K数据集...")
        self.benchmark = ORANBenchmark()
        
        if self.n_test_questions:
            self.test_questions = self.benchmark.sample_questions(
                n=self.n_test_questions,
                difficulty=difficulty,
                seed=seed
            )
        else:
            # 使用全部测试集（传入超大数字，sample_questions会自动限制为实际数量）
            # 从stats中获取该难度的总题数
            total_count = self.benchmark.stats[difficulty]
            self.test_questions = self.benchmark.sample_questions(
                n=total_count,
                difficulty=difficulty,
                seed=seed
            )
        
        print(f"✓ 加载了 {len(self.test_questions)} 道 {difficulty.upper()} 问题\n")
        
        # 加载嵌入模型
        print(f"加载嵌入模型: {embedding_model_path}")
        self.embedding_model = SentenceTransformer(embedding_model_path)
        self.embedding_model = self.embedding_model.to(f'cuda:{self.gpu_ids[0]}')
        print(f"✓ 嵌入模型加载成功 (GPU {self.gpu_ids[0]})\n")
        
        # 初始化自适应组件
        if self.policy_config:
            print("初始化自适应组件 (ComplexityClassifier)...")
            # ProgressTracker 将在每个问题中实例化
            self.classifier = QuestionComplexityClassifier()
            print("✓ 自适应组件已就绪")
        else:
            self.classifier = None

        # 🆕 预计算所有问题的embeddings (优化检索速度)
        print(f"{'='*80}")
        print(f"预计算问题embeddings (优化检索性能)...")
        print(f"{'='*80}")
        self.query_embeddings = {}
        
        import time
        start_time = time.time()
        
        for idx, q in enumerate(self.test_questions):
            question_text = q['question']
            
            # 避免重复计算（虽然理论上不会有重复）
            if question_text not in self.query_embeddings:
                # 直接返回numpy数组，避免GPU转换
                embedding = self.embedding_model.encode(
                    question_text, 
                    convert_to_tensor=False,
                    show_progress_bar=False
                )
                self.query_embeddings[question_text] = embedding.tolist()
            
            # 每100个打印一次进度
            if (idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (idx + 1)
                remaining = avg_time * (len(self.test_questions) - idx - 1)
                print(f"  进度: {idx+1}/{len(self.test_questions)} "
                      f"({(idx+1)/len(self.test_questions)*100:.1f}%) - "
                      f"预计剩余: {remaining:.0f}秒")
        
        elapsed = time.time() - start_time
        print(f"\n✓ 预计算完成!")
        print(f"  - 问题数: {len(self.query_embeddings)}")
        print(f"  - 耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
        
        if len(self.query_embeddings) > 0:
            print(f"  - 平均: {elapsed/len(self.query_embeddings)*1000:.1f}ms/问题")
            print(f"  - 内存占用: ~{len(self.query_embeddings) * 384 * 4 / 1024 / 1024:.2f} MB")
        else:
            print(f"  ⚠️  警告: 没有预计算任何embeddings")
        
        print(f"{'='*80}\n")
        
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
    
    def create_mdp_config(self, c_r: float) -> Dict:
        """创建MDP配置"""
        mdp_config = self.config['mdp'].copy()
        mdp_config['c_r'] = c_r
        
        # 添加 U_grid_size (兼容性)
        if 'U_grid_size' not in mdp_config and 'grid_size' in mdp_config:
            mdp_config['U_grid_size'] = mdp_config['grid_size']
        
        # 加载自适应策略配置
        policy_config = None
        if self.policy_config_path and os.path.exists(self.policy_config_path):
            with open(self.policy_config_path, 'r') as f:
                policy_config = yaml.safe_load(f)
                # Extract the 'policy' section
                policy_config = policy_config.get('policy', policy_config)
        
        return {
            'mdp': mdp_config,
            'policy': policy_config,
            'quality': self.config.get('quality', {'mode': 'linear', 'k': 5.0}),
            'solver': {
                'max_iterations': 1000,
                'convergence_threshold': 1e-6,
                'verbose': False
            }
        }
    
    def solve_mdp(self, c_r: float) -> tuple:
        """求解MDP获取阈值"""
        print(f"  求解MDP (c_r={c_r:.3f})...", end=" ")
        
        config = self.create_mdp_config(c_r)
        solver = MDPSolver(config)
        solver.solve()
        
        theta_cont = solver.theta_cont
        theta_star = solver.theta_star
        
        print(f"θ_cont={theta_cont:.3f}, θ*={theta_star:.3f}")
        return theta_cont, theta_star
    
    def retrieve_documents(self, question: str, top_k: int = 3) -> List[str]:
        """检索相关文档 (使用预计算的embeddings)"""
        if self.collection is None:
            # 模拟检索(如果Chroma不可用)
            return [f"模拟文档 {i+1}: O-RAN specification content related to the query." for i in range(top_k)]
        
        # ✅ 使用预计算的embedding (避免重复编码)
        query_embedding = self.query_embeddings.get(question)
        
        # 如果缓存中没有（理论上不应该发生），才临时计算
        if query_embedding is None:
            print(f"⚠️  缓存未命中，临时计算embedding: {question[:60]}...")
            query_embedding = self.embedding_model.encode(
                question, 
                convert_to_tensor=False,
                show_progress_bar=False
            ).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents"]
        )
        
        documents = results.get("documents", [[]])[0]
        return documents
    
    def generate_answer(self, question: Dict, context: str = "") -> tuple:
        """使用LLM生成答案
        
        Returns:
            (answer_index, confidence, response_text)
        """
        prompt = self._create_prompt(question, context)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        # 移动到第一个GPU (Accelerate会自动处理后续的分布)
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
        
        # 简单的置信度估计
        confidence = 0.7 if context else 0.5
        
        return answer, confidence, response
    
    def _create_prompt(self, question: Dict, context: Optional[str] = None) -> str:
        """创建提示词"""
        context_part = f"\n\nContext:\n{context}\n" if context else ""
        
        # 处理异常情况：确保有4个选项
        options = question.get('options', [])
        if len(options) < 4:
            print(f"⚠️  问题选项数异常: {len(options)}个 - {question['question'][:60]}...")
            # 跳过此问题或填充默认选项
            options = options + ['N/A'] * (4 - len(options))
        elif len(options) > 4:
            print(f"⚠️  问题选项数异常: {len(options)}个 - {question['question'][:60]}...")
            options = options[:4]  # 只取前4个
        
        prompt = f"""You are an O-RAN standards expert. Answer the following question.{context_part}
Question: {question['question']}

Options:
1. {options[0]}
2. {options[1]}
3. {options[2]}
4. {options[3]}

Answer with only the number (1, 2, 3, or 4):"""
        
        return prompt
    
    def _extract_answer(self, response: str) -> int:
        """从响应中提取答案"""
        import re
        
        response = response.lower()
        matches = re.findall(r'\b([1-4])\b', response)
        
        if matches:
            return int(matches[-1])
        
        return 1  # 默认
    
    def simulate_argo_policy(self, question: Dict, theta_cont: float, theta_star: float, c_r: float) -> Dict:
        """执行ARGO策略"""
        # 如果启用了自适应策略配置，使用新逻辑
        if self.policy_config:
            return self._simulate_adaptive_policy(question, c_r)

        U = 0.0
        C = 0.0
        retrieval_count = 0
        reason_count = 0
        
        delta_r = self.config['mdp']['delta_r']
        delta_p = self.config['mdp']['delta_p']
        c_p = self.config['mdp']['c_p']
        p_s = self.config['mdp']['p_s']
        
        max_steps = 20
        final_answer = None
        
        for step in range(max_steps):
            if U >= theta_star:
                break
            
            if U < theta_cont:
                # Retrieve
                retrieval_count += 1
                C += c_r
                
                # 真实检索
                docs = self.retrieve_documents(question['question'], top_k=3)
                context = " ".join(docs)
                
                # 用检索成功率模拟
                if random.random() < p_s:
                    U += delta_r
                    final_answer, _, _ = self.generate_answer(question, context)
                else:
                    final_answer, _, _ = self.generate_answer(question, context)
            else:
                # Reason
                reason_count += 1
                C += c_p
                U += delta_p
                
                # 无检索推理
                final_answer, _, _ = self.generate_answer(question, "")
        
        # 最终质量
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
    
    def _simulate_adaptive_policy(self, question: Dict, c_r: float) -> Dict:
        """使用自适应策略配置执行 (ProgressTracker + ComplexityClassifier)"""
        q_text = question['question']
        
        # 1. 分类复杂度
        complexity = self.classifier.classify(q_text)
        policy_params = self.policy_config['policy'][complexity]
        
        theta_star = policy_params['theta_star']
        theta_cont = policy_params['theta_cont']
        max_steps = policy_params['max_steps']
        
        # 2. 初始化状态
        tracker = ProgressTracker(q_text)
        U = 0.0
        C = 0.0
        retrieval_count = 0
        reason_count = 0
        accumulated_context = ""
        current_answer = ""
        
        c_p = self.config['mdp']['c_p']
        
        # 3. 逐步执行
        final_step = 0
        for step in range(max_steps):
            final_step = step + 1
            
            # 检查终止
            if U >= theta_star:
                break
                
            # 决策
            if U < theta_cont:
                # Action: Retrieve
                retrieval_count += 1
                C += c_r
                
                # 检索
                new_docs = self.retrieve_documents(q_text, top_k=3)
                new_context = " ".join(new_docs)
                # 简单拼接，实际应用可能需要去重或摘要
                accumulated_context = (accumulated_context + " " + new_context).strip()
                
                # 生成答案
                ans_idx, _, ans_text = self.generate_answer(question, accumulated_context)
                current_answer = ans_text
                
                # 更新进度
                step_data = {
                    'intermediate_answer': current_answer,
                    'retrieved_docs': new_docs,
                    'confidence': 0.6
                }
                U = tracker.update('retrieve', step_data)
                
            else:
                # Action: Reason
                reason_count += 1
                C += c_p
                
                # 推理 (使用已有上下文)
                ans_idx, _, ans_text = self.generate_answer(question, accumulated_context)
                current_answer = ans_text
                
                # 更新进度
                step_data = {
                    'intermediate_answer': current_answer,
                    'confidence': 0.7
                }
                U = tracker.update('reason', step_data)
        
        # 最终结果
        final_ans_idx = self._extract_answer(current_answer)
        correct = (final_ans_idx == question['correct_answer'])
        
        return {
            'quality': min(U, 1.0),
            'cost': C,
            'retrieval_count': retrieval_count,
            'reason_count': reason_count,
            'steps': final_step,
            'correct': correct,
            'complexity': complexity
        }
    
    def simulate_always_retrieve_policy(self, question: Dict, c_r: float, theta_star: float) -> Dict:
        """Always-Retrieve基线 (修正: 使用动态θ*)"""
        U = 0.0
        C = 0.0
        retrieval_count = 0
        
        delta_r = self.config['mdp']['delta_r']
        p_s = self.config['mdp']['p_s']
        
        max_steps = 20
        final_answer = None
        
        for step in range(max_steps):
            if U >= theta_star:  # ← 使用传入的theta_star
                break
            
            retrieval_count += 1
            C += c_r
            
            docs = self.retrieve_documents(question['question'], top_k=3)
            context = " ".join(docs)
            
            if random.random() < p_s:
                U += delta_r
            
            final_answer, _, _ = self.generate_answer(question, context)
        
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
    
    def simulate_always_reason_policy(self, question: Dict, theta_star: float) -> Dict:
        """Always-Reason基线 (修正: 使用动态θ*)"""
        U = 0.0
        C = 0.0
        reason_count = 0
        
        delta_p = self.config['mdp']['delta_p']
        c_p = self.config['mdp']['c_p']
        
        max_steps = 20
        final_answer = None
        
        for step in range(max_steps):
            if U >= theta_star:  # ← 使用传入的theta_star
                break
            
            reason_count += 1
            C += c_p
            U += delta_p
            
            final_answer, _, _ = self.generate_answer(question, "")
        
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
    
    def simulate_random_policy(self, question: Dict, c_r: float, theta_star: float) -> Dict:
        """Random基线: 随机选择Retrieve或Reason (新增)"""
        U = 0.0
        C = 0.0
        retrieval_count = 0
        reason_count = 0
        
        delta_r = self.config['mdp']['delta_r']
        delta_p = self.config['mdp']['delta_p']
        c_p = self.config['mdp']['c_p']
        p_s = self.config['mdp']['p_s']
        
        max_steps = 20
        final_answer = None
        
        for step in range(max_steps):
            if U >= theta_star:  # ← 使用传入的theta_star
                break
            
            # 随机选择动作 (50% Retrieve, 50% Reason)
            if random.random() < 0.5:
                # Retrieve
                retrieval_count += 1
                C += c_r
                docs = self.retrieve_documents(question['question'], top_k=3)
                context = " ".join(docs)
                if random.random() < p_s:
                    U += delta_r
                final_answer, _, _ = self.generate_answer(question, context)
            else:
                # Reason
                reason_count += 1
                C += c_p
                U += delta_p
                final_answer, _, _ = self.generate_answer(question, "")
        
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
    
    def evaluate_all_policies(self, c_r: float, theta_cont: float, theta_star: float) -> Dict:
        """评估所有策略 (修正: 添加Random，传入θ*)"""
        results = {
            'ARGO': [],
            'Always-Retrieve': [],
            'Always-Reason': [],
            'Random': []  # ← 新增Random策略
        }
        
        print(f"\n  评估 {len(self.test_questions)} 道问题...")
        
        for i, question in enumerate(self.test_questions, 1):
            if i % 10 == 0:
                print(f"    进度: {i}/{len(self.test_questions)}")
            
            # ARGO
            result = self.simulate_argo_policy(question, theta_cont, theta_star, c_r)
            results['ARGO'].append(result)
            
            # Always-Retrieve (传入theta_star)
            result = self.simulate_always_retrieve_policy(question, c_r, theta_star)
            results['Always-Retrieve'].append(result)
            
            # Always-Reason (传入theta_star)
            result = self.simulate_always_reason_policy(question, theta_star)
            results['Always-Reason'].append(result)
            
            # Random (传入theta_star)
            result = self.simulate_random_policy(question, c_r, theta_star)
            results['Random'].append(result)
        
        return results
    
    def run_experiment(
        self,
        c_r_min_multiplier: float = 1.0,
        c_r_max_multiplier: float = 10.0
    ):
        """运行实验"""
        c_p = self.config['mdp']['c_p']
        c_r_values = np.linspace(
            c_r_min_multiplier * c_p,
            c_r_max_multiplier * c_p,
            self.n_cost_steps  # ← 使用根据test_mode设定的步数
        )
        
        print(f"\n{'='*80}")
        print(f"开始实验 - 检索成本影响")
        print(f"{'='*80}")
        print(f"运行模式: {self.mode_desc}")
        print(f"c_r范围: {c_r_values[0]:.3f} ~ {c_r_values[-1]:.3f} (扫描 {self.n_cost_steps} 个点)")
        print(f"c_p固定: {c_p:.3f}")
        print(f"问题数量: {len(self.test_questions)}")
        print(f"策略数量: 4 (ARGO, Always-Retrieve, Always-Reason, Random)")
        print(f"总评估次数: {self.n_cost_steps} × 4策略 × {len(self.test_questions)}题 = {self.n_cost_steps * 4 * len(self.test_questions)}")
        print(f"{'='*80}\n")
        
        all_results = []
        self.raw_results = []  # 🆕 初始化详细结果列表
        
        for i, c_r in enumerate(c_r_values, 1):
            print(f"\n[{i}/{self.n_cost_steps}] c_r = {c_r:.4f} ({c_r/c_p:.1f}x c_p)")
            print(f"{'-'*80}")
            
            # 求解MDP
            theta_cont, theta_star = self.solve_mdp(c_r)
            
            # 评估所有策略
            results = self.evaluate_all_policies(c_r, theta_cont, theta_star)
            
            # 🆕 保存详细结果
            self.raw_results.append({
                'c_r': c_r,
                'details': results
            })
            
            # 聚合结果
            aggregated = {
                'c_r': c_r,
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
                
                print(f"  {policy_name:20s}: Accuracy={accuracy:.1%}, Quality={avg_quality:.3f}, "
                      f"Cost={avg_cost:.3f}, Retrievals={avg_retrievals:.1f}")
            
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
        
        # 根据test_mode选择文件名后缀
        if self.test_mode == "small":
            mode_suffix = "small"
        elif self.test_mode == "full":
            mode_suffix = "full"
        else:  # custom
            mode_suffix = "custom"
        
        filename = f"exp1_real_cost_impact_{mode_suffix}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # 保存完整结果 + 元数据
        output_data = {
            'metadata': {
                'test_mode': self.test_mode,
                'n_questions': len(self.test_questions),
                'difficulty': self.difficulty,
                'n_cost_steps': self.n_cost_steps,
                'seed': self.seed,  # ← 添加seed到元数据
                'timestamp': timestamp
            },
            'results': self.results,
            'raw_results': self.raw_results
        }
        
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ 结果已保存: {filepath}")
        return filepath
    
    def plot_results(self, output_dir: str = "figs"):
        """绘制结果 (修正: 按实验设计文档要求绘制2张图)"""
        os.makedirs(output_dir, exist_ok=True)
        
        c_r_values = [r['c_r'] for r in self.results]
        mode_suffix = "small" if self.test_mode == "small" else "full"
        
        # ====================================================================
        # 图1.A: Cost vs. Accuracy (按实验设计文档要求)
        # ====================================================================
        plt.figure(figsize=(10, 6))
        
        policies = ['ARGO', 'Always-Retrieve', 'Always-Reason', 'Random']
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#6A994E']
        markers = ['o', 's', '^', 'D']
        
        for policy, color, marker in zip(policies, colors, markers):
            accuracy = [r[f'{policy}_accuracy'] for r in self.results]
            plt.plot(c_r_values, accuracy, marker=marker, label=policy, 
                    linewidth=2.5, markersize=8, color=color, alpha=0.8)
        
        plt.xlabel('Retrieval Cost ($c_r$)', fontsize=13, fontweight='bold')
        plt.ylabel('Average Accuracy', fontsize=13, fontweight='bold')
        plt.title('Graph 1.A: Cost vs. Accuracy', fontsize=15, fontweight='bold', pad=15)
        plt.legend(fontsize=11, loc='best', framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        fig1_path = os.path.join(output_dir, f'exp1_graph1A_cost_vs_accuracy_{mode_suffix}.png')
        plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 图表已保存: {fig1_path}")
        
        # ====================================================================
        # 图1.B: Cost vs. Retrieval Calls (按实验设计文档要求)
        # ====================================================================
        plt.figure(figsize=(10, 6))
        
        # 只绘制有检索行为的策略 (Always-Reason不检索，所以不画)
        retrieval_policies = ['ARGO', 'Always-Retrieve', 'Random']
        retrieval_colors = ['#2E86AB', '#A23B72', '#6A994E']
        retrieval_markers = ['o', 's', 'D']
        
        for policy, color, marker in zip(retrieval_policies, retrieval_colors, retrieval_markers):
            retrievals = [r[f'{policy}_retrievals'] for r in self.results]
            plt.plot(c_r_values, retrievals, marker=marker, label=policy, 
                    linewidth=2.5, markersize=8, color=color, alpha=0.8)
        
        plt.xlabel('Retrieval Cost ($c_r$)', fontsize=13, fontweight='bold')
        plt.ylabel('Average Retrieval Calls ($E[R_T]$)', fontsize=13, fontweight='bold')
        plt.title('Graph 1.B: Cost vs. Retrieval Calls', fontsize=15, fontweight='bold', pad=15)
        plt.legend(fontsize=11, loc='best', framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        fig2_path = os.path.join(output_dir, f'exp1_graph1B_cost_vs_retrievals_{mode_suffix}.png')
        plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 图表已保存: {fig2_path}")
        
        # ====================================================================
        # 额外图: Cost vs. Total Cost (补充分析)
        # ====================================================================
        plt.figure(figsize=(10, 6))
        
        for policy, color, marker in zip(policies, colors, markers):
            total_cost = [r[f'{policy}_cost'] for r in self.results]
            plt.plot(c_r_values, total_cost, marker=marker, label=policy, 
                    linewidth=2.5, markersize=8, color=color, alpha=0.8)
        
        plt.xlabel('Retrieval Cost ($c_r$)', fontsize=13, fontweight='bold')
        plt.ylabel('Average Total Cost', fontsize=13, fontweight='bold')
        plt.title('Supplementary: Cost vs. Total Cost', fontsize=15, fontweight='bold', pad=15)
        plt.legend(fontsize=11, loc='best', framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        fig3_path = os.path.join(output_dir, f'exp1_supplementary_cost_vs_total_{mode_suffix}.png')
        plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 图表已保存: {fig3_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='实验1: 检索成本影响 (真实LLM版本)')
    parser.add_argument('--mode', type=str, default='small', choices=['small', 'full', 'custom'],
                       help='测试模式: small (10题, 快速验证), full (全部~12K题), custom (自定义)')
    parser.add_argument('--n-questions', type=int, default=None,
                       help='自定义问题数量 (仅用于 --mode custom)')
    parser.add_argument('--difficulty', type=str, default='hard', choices=['easy', 'medium', 'hard'],
                       help='问题难度')
    parser.add_argument('--gpus', type=str, default='0,1,2,3',
                       help='使用的GPU ID列表，逗号分隔，例如: 0,1,2,3')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--model-path', type=str, default=None,
                       help='LLM模型路径 (可选，用于覆盖默认的14B模型)')
    parser.add_argument('--config-path', type=str, default='configs/multi_gpu_data_calibrated.yaml',
                       help='MDP配置文件路径 (默认使用data_calibrated版本，c_p=0.02)')
    parser.add_argument('--policy-config-path', type=str, default=None,
                       help='自适应策略配置文件路径')
    
    args = parser.parse_args()
    
    # 验证参数
    if args.mode == 'custom' and args.n_questions is None:
        parser.error("--mode custom 必须指定 --n-questions")
    
    # 解析GPU列表
    gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    
    # 确定模型路径
    if args.model_path:
        llm_model_path = args.model_path
    else:
        llm_model_path = "/data/user/huangxiaolin/ARGO/RAG_Models/models/Qwen2.5-14B-Instruct"
    
    print(f"\n启动参数:")
    print(f"  模式: {args.mode}")
    if args.mode == 'custom':
        print(f"  问题数: {args.n_questions}")
    print(f"  难度: {args.difficulty}")
    print(f"  GPU: {gpu_ids}")
    print(f"  种子: {args.seed}")
    print(f"  模型: {Path(llm_model_path).name}\n")
    
    # 配置
    experiment = RealCostImpactExperiment(
        config_path=args.config_path,
        policy_config_path=args.policy_config_path,
        llm_model_path=llm_model_path,
        embedding_model_path="/data/user/huangxiaolin/ARGO/models/all-MiniLM-L6-v2",
        test_mode=args.mode,
        n_test_questions=args.n_questions,  # ← 传入自定义问题数
        difficulty=args.difficulty,
        seed=args.seed,
        gpu_ids=gpu_ids
    )
    
    # 运行实验
    results = experiment.run_experiment(
        c_r_min_multiplier=1.0,
        c_r_max_multiplier=10.0
    )
    
    # 保存结果
    experiment.save_results()
    
    # 绘图
    experiment.plot_results()
    
    print("\n实验1完成!")
    print(f"\n使用提示:")
    if args.mode == "small":
        print(f"  当前是小规模测试模式，如果运行成功，请使用以下命令运行完整实验:")
        print(f"  python Exp_real_cost_impact_v2.py --mode custom --n-questions 100 --difficulty {args.difficulty} --gpus {args.gpus}")
    elif args.mode == "custom":
        print(f"  ✓ Custom模式运行成功")
        print(f"  如需统计分析，请运行多个种子后使用:")
        print(f"  python Exp1_aggregate_and_analyze.py")


if __name__ == "__main__":
    main()
