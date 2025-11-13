"""
ARGO System - Phase 3.4 (Enhanced with ARGO Prompts V2.0)
==========================================================

完整的4组件架构 MDP-Guided RAG系统（使用标准化高质量Prompts）

组件集成：
1. QueryDecomposer: 动态子查询生成（带进度追踪）
2. Retriever: 向量检索（Chroma）+ 答案生成
3. Reasoner: MDP引导的推理（使用增强prompts）
4. AnswerSynthesizer: 最终答案合成（格式化输出）

核心流程：
    while U_t < θ* and t < T_max:
        action = MDP.get_action(U_t)
        if action == 'retrieve':
            q_t = Decomposer(x, H_t, U_t)
            r_t = Retriever(q_t, k)
            intermediate_answer = Retriever.generate_answer(q_t, r_t)
            U_{t+1} = U_t + delta_r (if success) else U_t
        else:  # reason
            intermediate_answer = Reasoner(x, H_t)
            U_{t+1} = U_t + delta_p
        H_{t+1} = H_t + [(action, ...)]
    
    O = Synthesizer(x, H_T)

Author: ARGO Team
Version: 3.1 (Enhanced Prompts)
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import time
from .prompts import ARGOPrompts, PromptConfig

logger = logging.getLogger(__name__)

# 导入组件
from .decomposer import QueryDecomposer
from .retriever import Retriever, MockRetriever
from .synthesizer import AnswerSynthesizer

# 导入MDP Solver
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../ARGO_MDP/src'))
try:
    from mdp_solver import MDPSolver
    MDP_AVAILABLE = True
except:
    logger.warning("MDPSolver not available, will use fixed strategy")
    MDP_AVAILABLE = False


class ARGO_System:
    """
    ARGO V3.0 - 完整的4组件架构系统
    
    核心改进：
    1. 模块化设计：每个组件独立，易于测试和替换
    2. MDP引导：动态决策Retrieve vs Reason
    3. 渐进式推理：根据历史调整查询策略
    4. 可追溯性：完整记录推理链
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        use_mdp: bool = True,
        mdp_config: Optional[Dict] = None,
        retriever_mode: str = "mock",  # "mock" or "chroma"
        chroma_dir: Optional[str] = None,
        collection_name: str = "oran_specs",
        max_steps: int = 10,
        verbose: bool = True
    ):
        """
        Args:
            model: 预训练的因果语言模型
            tokenizer: 对应的tokenizer
            use_mdp: 是否使用MDP策略（否则使用固定策略）
            mdp_config: MDP配置字典
            retriever_mode: "mock"（测试用）或"chroma"（真实检索）
            chroma_dir: Chroma数据库路径
            collection_name: Chroma集合名称
            max_steps: 最大推理步数
            verbose: 是否打印详细日志
        """
        self.model = model
        self.tokenizer = tokenizer
        self.use_mdp = use_mdp and MDP_AVAILABLE
        self.max_steps = max_steps
        self.verbose = verbose
        
        # 设备
        self.device = next(model.parameters()).device
        
        logger.info(f"Initializing ARGO System on {self.device}")
        
        # 初始化组件
        self._init_components(
            mdp_config,
            retriever_mode,
            chroma_dir,
            collection_name
        )
        
        # 统计信息
        self.stats = {
            'total_queries': 0,
            'total_steps': 0,
            'retrieve_actions': 0,
            'reason_actions': 0,
            'successful_retrievals': 0,
            'total_time': 0.0
        }
        
        logger.info("✅ ARGO System initialized successfully")
    
    def _init_components(
        self,
        mdp_config: Optional[Dict],
        retriever_mode: str,
        chroma_dir: Optional[str],
        collection_name: str
    ):
        """初始化所有组件"""
        
        # 1. Query Decomposer
        self.decomposer = QueryDecomposer(
            self.model,
            self.tokenizer,
            max_subquery_length=128,
            temperature=0.7,
            top_p=0.9
        )
        logger.info("✅ QueryDecomposer initialized")
        
        # 2. Retriever
        if retriever_mode == "mock":
            self.retriever = MockRetriever(p_s_value=0.8)
            logger.info("✅ MockRetriever initialized (test mode)")
        elif retriever_mode == "chroma":
            if chroma_dir is None:
                raise ValueError("chroma_dir must be provided for chroma mode")
            self.retriever = Retriever(
                chroma_dir=chroma_dir,
                collection_name=collection_name,
                similarity_threshold=0.3,
                p_s_mode="threshold",
                p_s_value=0.8
            )
            logger.info(f"✅ Retriever initialized (Chroma: {chroma_dir})")
        else:
            raise ValueError(f"Unknown retriever_mode: {retriever_mode}")
        
        # 3. MDP Solver (if enabled)
        if self.use_mdp:
            # 默认MDP配置（符合MDPSolver的config格式）
            default_mdp_config = {
                'mdp': {
                    'U_max': 1.0,
                    'delta_r': 0.25,
                    'delta_p': 0.08,
                    'c_r': 0.05,
                    'c_p': 0.02,
                    'p_s': 0.8,
                    'mu': 0.0,
                    'gamma': 0.98,
                    'U_grid_size': 1000
                },
                'quality': {
                    'mode': 'linear',
                    'k': 1.0
                },
                'reward_shaping': {
                    'enabled': False,
                    'k': 0.0
                },
                'solver': {
                    'max_iterations': 1000,
                    'convergence_threshold': 1e-6,
                    'verbose': False
                }
            }
            
            # 合并用户配置
            if mdp_config:
                # 深度合并
                for key in mdp_config:
                    if key in default_mdp_config and isinstance(default_mdp_config[key], dict):
                        default_mdp_config[key].update(mdp_config[key])
                    else:
                        default_mdp_config[key] = mdp_config[key]
            
            self.mdp_solver = MDPSolver(default_mdp_config)
            logger.info("✅ MDPSolver initialized")
            
            # 预计算策略（使用solve方法完成value iteration和阈值计算）
            result = self.mdp_solver.solve()
            logger.info(f"MDP策略已收敛: θ*={result['theta_star']:.4f}, θ_cont={result['theta_cont']:.4f}")
        else:
            self.mdp_solver = None
            logger.info("⚠️  MDP disabled, using fixed strategy")
        
        # 4. Answer Synthesizer
        self.synthesizer = AnswerSynthesizer(
            self.model,
            self.tokenizer,
            max_answer_length=512,
            temperature=0.3,
            top_p=0.95,
            include_sources=True
        )
        logger.info("✅ AnswerSynthesizer initialized")
    
    def answer_question(
        self,
        question: str,
        return_history: bool = True,
        options: Optional[List[str]] = None
    ) -> Tuple[str, Optional[str], Optional[List[Dict]], Optional[Dict]]:
        """
        回答问题（主入口）
        
        Args:
            question: 输入问题
            return_history: 是否返回推理历史
            options: 选择题的选项列表（可选，格式为 ["选项1文本", "选项2文本", ...]）
        
        Returns:
            (answer, choice, history, metadata):
                - answer: 最终答案（详细解释）
                - choice: 选择的选项编号（"1"/"2"/"3"/"4"，仅当提供options时）
                - history: 推理历史（如果return_history=True）
                - metadata: 元数据（步数、时间等）
        """
        start_time = time.time()
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"ARGO System Processing Question")
            print(f"{'='*80}")
            print(f"Question: {question}")
            print(f"Max Steps: {self.max_steps}")
            print(f"Strategy: {'MDP-Guided' if self.use_mdp else 'Fixed'}")
            print(f"{'='*80}\n")
        
        # 初始化
        history = []
        U_t = 0.0  # 初始知识完整度
        t = 0
        
        # 获取阈值
        if self.use_mdp:
            theta_star = self.mdp_solver.theta_star
            theta_cont = self.mdp_solver.theta_cont
        else:
            theta_star = 0.9  # 固定策略：达到90%就停止
            theta_cont = 0.5  # 固定策略：50%以下检索，否则推理
        
        # 主循环
        while U_t < theta_star and t < self.max_steps:
            t += 1
            
            if self.verbose:
                print(f"\n--- Step {t} ---")
                print(f"Current U_t: {U_t:.3f}")
            
            # 决定动作
            if self.use_mdp:
                action = self._mdp_get_action(U_t)
            else:
                # 固定策略
                action = 'retrieve' if U_t < theta_cont else 'reason'
            
            if self.verbose:
                print(f"Action: {action.upper()}")
            
            # 执行动作
            if action == 'retrieve':
                step_data = self._execute_retrieve(question, history, U_t)
                
                if step_data['retrieval_success']:
                    U_t += self.mdp_solver.delta_r if self.use_mdp else 0.25
                    if self.verbose:
                        print(f"✓ Retrieval successful, U_t → {U_t:.3f}")
                else:
                    if self.verbose:
                        print(f"✗ Retrieval failed, U_t unchanged")
                
                self.stats['retrieve_actions'] += 1
                if step_data['retrieval_success']:
                    self.stats['successful_retrievals'] += 1
            
            else:  # reason
                step_data = self._execute_reason(question, history, U_t)
                
                U_t += self.mdp_solver.delta_p if self.use_mdp else 0.08
                if self.verbose:
                    print(f"Reasoning complete, U_t → {U_t:.3f}")
                
                self.stats['reason_actions'] += 1
            
            # 记录到历史
            history.append(step_data)
            
            # 确保U_t不超过1
            U_t = min(U_t, 1.0)
        
        # 合成最终答案
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Synthesizing Final Answer")
            print(f"{'='*80}")
            print(f"Final U_t: {U_t:.3f}")
            print(f"Total Steps: {t}")
        
        final_answer, choice, sources = self.synthesizer.synthesize(
            question, 
            history,
            options=options
        )
        
        # 计算统计
        elapsed_time = time.time() - start_time
        self.stats['total_queries'] += 1
        self.stats['total_steps'] += t
        self.stats['total_time'] += elapsed_time
        
        metadata = {
            'total_steps': t,
            'final_uncertainty': U_t,
            'retrieve_count': sum(1 for s in history if s['action'] == 'retrieve'),
            'reason_count': sum(1 for s in history if s['action'] == 'reason'),
            'successful_retrievals': sum(1 for s in history 
                                        if s['action'] == 'retrieve' and s.get('retrieval_success', False)),
            'elapsed_time': elapsed_time,
            'sources': sources
        }
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"COMPLETED")
            print(f"{'='*80}")
            print(f"Answer: {final_answer[:200]}...")
            if choice:
                print(f"Selected Choice: {choice}")
            print(f"Steps: {metadata['total_steps']}")
            print(f"Time: {metadata['elapsed_time']:.2f}s")
            if sources:
                print(f"Sources: {', '.join(sources[:3])}")
            print(f"{'='*80}\n")
        
        if return_history:
            return final_answer, choice, history, metadata
        else:
            return final_answer, choice, None, metadata
    
    def _execute_retrieve(
        self,
        original_question: str,
        history: List[Dict],
        U_t: float
    ) -> Dict:
        """执行检索动作（使用增强的答案生成）"""
        
        # 1. 生成子查询
        uncertainty = 1.0 - U_t
        subquery = self.decomposer.generate_subquery(
            original_question,
            history,
            uncertainty
        )
        
        if self.verbose:
            print(f"Subquery: {subquery}")
        
        # 2. 检索
        docs, success, scores = self.retriever.retrieve(
            subquery,
            k=3,
            return_scores=True
        )
        
        if self.verbose:
            if success:
                print(f"Retrieved {len(docs)} documents")
            else:
                print(f"Retrieval failed")
        
        # 3. 生成基于检索的中间答案（新增）
        intermediate_answer = ""
        confidence = 0.0
        
        if success and docs:
            # 使用检索器的答案生成方法（基于ARGO V2.0 prompts）
            intermediate_answer = self.retriever.generate_answer_from_docs(
                question=subquery,
                docs=docs,
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=PromptConfig.REASONER_MAX_LENGTH,
                temperature=PromptConfig.REASONER_TEMPERATURE,
                top_p=PromptConfig.REASONER_TOP_P
            )
            
            # 估算置信度（基于检索分数和答案质量）
            if intermediate_answer:
                avg_score = np.mean(scores) if scores else 0.5
                confidence = min(0.95, avg_score * 0.8 + 0.2)
            
            if self.verbose and intermediate_answer:
                print(f"Intermediate Answer: {intermediate_answer[:150]}...")
                print(f"Confidence: {confidence:.2f}")
        
        # 4. 构造步骤数据
        step_data = {
            'action': 'retrieve',
            'subquery': subquery,
            'retrieval_success': success,
            'retrieved_docs': docs,
            'retrieval_scores': scores,
            'intermediate_answer': intermediate_answer,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'progress': U_t
        }
        
        return step_data
    
    def _execute_reason(
        self,
        original_question: str,
        history: List[Dict],
        U_t: float
    ) -> Dict:
        """执行推理动作（使用增强的推理prompt）"""
        
        # 构建推理提示词（使用ARGO V2.0标准模板）
        prompt = ARGOPrompts.build_reasoning_prompt(
            original_question=original_question,
            history=history
        )
        
        # 生成中间答案
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096  # 增加以容纳完整历史
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=PromptConfig.REASONER_MAX_LENGTH,
                temperature=PromptConfig.REASONER_TEMPERATURE,
                top_p=PromptConfig.REASONER_TOP_P,
                do_sample=True if PromptConfig.REASONER_TEMPERATURE > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        intermediate_answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        intermediate_answer = intermediate_answer.strip()
        
        # 简单的置信度估计（基于答案长度和U_t）
        confidence = min(0.9, 0.5 + U_t * 0.4)
        
        if self.verbose:
            print(f"Intermediate Answer: {intermediate_answer[:150]}...")
            print(f"Confidence: {confidence:.2f}")
        
        step_data = {
            'action': 'reason',
            'intermediate_answer': intermediate_answer,
            'confidence': confidence,
            'uncertainty': 1.0 - U_t,
            'progress': U_t
        }
        
        return step_data
    
    def _mdp_get_action(self, U_t: float) -> str:
        """
        从MDP的Q函数中获取最优动作
        
        Args:
            U_t: 当前状态（知识完整度）
        
        Returns:
            "retrieve" or "reason"
        """
        # 获取状态索引
        idx = self.mdp_solver.get_state_index(U_t)
        
        # 获取Q值（忽略terminate动作，因为我们在主循环中已经检查了theta_star）
        Q_retrieve = self.mdp_solver.Q[idx, 0]
        Q_reason = self.mdp_solver.Q[idx, 1]
        
        # 选择Q值更高的动作
        if Q_retrieve >= Q_reason:
            return 'retrieve'
        else:
            return 'reason'
    
    def get_statistics(self) -> Dict:
        """获取系统统计信息"""
        stats = self.stats.copy()
        
        if stats['total_queries'] > 0:
            stats['avg_steps_per_query'] = stats['total_steps'] / stats['total_queries']
            stats['avg_time_per_query'] = stats['total_time'] / stats['total_queries']
            stats['retrieval_success_rate'] = (
                stats['successful_retrievals'] / stats['retrieve_actions']
                if stats['retrieve_actions'] > 0 else 0.0
            )
        
        return stats
    
    def reset_statistics(self):
        """重置统计信息"""
        self.stats = {
            'total_queries': 0,
            'total_steps': 0,
            'retrieve_actions': 0,
            'reason_actions': 0,
            'successful_retrievals': 0,
            'total_time': 0.0
        }
        logger.info("Statistics reset")
