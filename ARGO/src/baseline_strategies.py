"""
Baseline Strategies - Phase 4.1
================================

实现基线策略用于对比实验：
1. AlwaysReasonStrategy: 每步都推理，从不检索
2. RandomStrategy: 随机选择retrieve或reason
3. FixedThresholdStrategy: 固定阈值策略（已在ARGO_System中实现）

这些基线策略继承ARGO_System，只覆盖动作选择逻辑。

对比维度：
- 准确性 (Accuracy)
- 效率 (Steps, Time)
- 成本 (Retrieve cost, Reason cost)
- 鲁棒性 (对检索失败的容忍度)
"""

import torch
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
import logging

from .argo_system import ARGO_System

logger = logging.getLogger(__name__)


class AlwaysReasonStrategy(ARGO_System):
    """
    Always-Reason 基线策略
    
    特点：
    - 每步都执行Reason动作
    - 从不执行Retrieve
    - 完全依赖LLM的内部知识
    - 不受检索失败影响
    
    用途：
    - 衡量纯推理的上限
    - 评估检索的必要性
    - 对比检索增强的效果
    """
    
    def __init__(self, *args, **kwargs):
        """
        初始化Always-Reason策略
        
        注意：强制设置use_mdp=False，因为不需要MDP
        """
        # 强制禁用MDP（不需要）
        kwargs['use_mdp'] = False
        
        super().__init__(*args, **kwargs)
        
        logger.info("AlwaysReasonStrategy initialized (never retrieve)")
    
    def answer_question(
        self,
        question: str,
        return_history: bool = True
    ) -> Tuple[str, Optional[List[Dict]], Optional[Dict]]:
        """
        回答问题（Always-Reason策略）
        
        覆盖父类方法，强制每步都推理
        """
        import time
        start_time = time.time()
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Always-Reason Strategy Processing Question")
            print(f"{'='*80}")
            print(f"Question: {question}")
            print(f"Max Steps: {self.max_steps}")
            print(f"Strategy: Always-Reason (never retrieve)")
            print(f"{'='*80}\n")
        
        # 初始化
        history = []
        U_t = 0.0
        t = 0
        
        # 固定终止阈值（不依赖MDP）
        theta_star = 0.9
        delta_p = 0.08  # 每步推理增长
        
        # 主循环：只执行Reason
        while U_t < theta_star and t < self.max_steps:
            t += 1
            
            if self.verbose:
                print(f"\n--- Step {t} ---")
                print(f"Current U_t: {U_t:.3f}")
                print(f"Action: REASON (always)")
            
            # 执行推理
            step_data = self._execute_reason(question, history, U_t)
            
            # 更新U_t
            U_t += delta_p
            U_t = min(U_t, 1.0)
            
            if self.verbose:
                print(f"Reasoning complete, U_t → {U_t:.3f}")
            
            # 记录历史
            history.append(step_data)
            
            self.stats['reason_actions'] += 1
        
        # 合成最终答案
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Synthesizing Final Answer")
            print(f"{'='*80}")
            print(f"Final U_t: {U_t:.3f}")
            print(f"Total Steps: {t}")
        
        final_answer, sources = self.synthesizer.synthesize(question, history)
        
        # 统计
        elapsed_time = time.time() - start_time
        self.stats['total_queries'] += 1
        self.stats['total_steps'] += t
        self.stats['total_time'] += elapsed_time
        
        metadata = {
            'strategy': 'Always-Reason',
            'total_steps': t,
            'final_uncertainty': U_t,
            'retrieve_count': 0,  # 永远是0
            'reason_count': t,
            'successful_retrievals': 0,
            'elapsed_time': elapsed_time,
            'sources': sources
        }
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"COMPLETED")
            print(f"{'='*80}")
            print(f"Answer: {final_answer[:200]}...")
            print(f"Steps: {metadata['total_steps']} (all Reason)")
            print(f"Time: {metadata['elapsed_time']:.2f}s")
            print(f"{'='*80}\n")
        
        if return_history:
            return final_answer, history, metadata
        else:
            return final_answer, None, metadata


class RandomStrategy(ARGO_System):
    """
    Random 基线策略
    
    特点：
    - 随机选择Retrieve或Reason
    - 可配置检索概率
    - 不依赖U_t或历史
    - 完全随机的baseline
    
    用途：
    - 作为最弱基线
    - 评估策略的重要性
    - 统计显著性测试
    """
    
    def __init__(
        self,
        *args,
        retrieve_probability: float = 0.5,  # 检索概率
        **kwargs
    ):
        """
        初始化Random策略
        
        Args:
            retrieve_probability: 选择Retrieve的概率（0-1）
            其他参数同ARGO_System
        """
        # 强制禁用MDP
        kwargs['use_mdp'] = False
        
        super().__init__(*args, **kwargs)
        
        self.retrieve_probability = retrieve_probability
        
        logger.info(f"RandomStrategy initialized (p_retrieve={retrieve_probability})")
    
    def answer_question(
        self,
        question: str,
        return_history: bool = True
    ) -> Tuple[str, Optional[List[Dict]], Optional[Dict]]:
        """
        回答问题（Random策略）
        
        覆盖父类方法，随机选择动作
        """
        import time
        start_time = time.time()
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Random Strategy Processing Question")
            print(f"{'='*80}")
            print(f"Question: {question}")
            print(f"Max Steps: {self.max_steps}")
            print(f"Strategy: Random (p_retrieve={self.retrieve_probability})")
            print(f"{'='*80}\n")
        
        # 初始化
        history = []
        U_t = 0.0
        t = 0
        
        # 固定参数
        theta_star = 0.9
        delta_r = 0.25
        delta_p = 0.08
        
        # 主循环：随机选择动作
        while U_t < theta_star and t < self.max_steps:
            t += 1
            
            if self.verbose:
                print(f"\n--- Step {t} ---")
                print(f"Current U_t: {U_t:.3f}")
            
            # 随机选择动作
            if random.random() < self.retrieve_probability:
                action = 'retrieve'
            else:
                action = 'reason'
            
            if self.verbose:
                print(f"Action: {action.upper()} (random choice)")
            
            # 执行动作
            if action == 'retrieve':
                step_data = self._execute_retrieve(question, history, U_t)
                
                if step_data['retrieval_success']:
                    U_t += delta_r
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
                
                U_t += delta_p
                if self.verbose:
                    print(f"Reasoning complete, U_t → {U_t:.3f}")
                
                self.stats['reason_actions'] += 1
            
            # 记录历史
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
        
        final_answer, sources = self.synthesizer.synthesize(question, history)
        
        # 统计
        elapsed_time = time.time() - start_time
        self.stats['total_queries'] += 1
        self.stats['total_steps'] += t
        self.stats['total_time'] += elapsed_time
        
        metadata = {
            'strategy': 'Random',
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
            print(f"Steps: {metadata['total_steps']}")
            print(f"Retrieve: {metadata['retrieve_count']}, Reason: {metadata['reason_count']}")
            print(f"Time: {metadata['elapsed_time']:.2f}s")
            print(f"{'='*80}\n")
        
        if return_history:
            return final_answer, history, metadata
        else:
            return final_answer, None, metadata


class FixedThresholdStrategy(ARGO_System):
    """
    Fixed-Threshold 基线策略
    
    特点：
    - 使用固定阈值决策
    - U_t < θ_cont: Retrieve
    - U_t >= θ_cont: Reason
    - 不需要MDP求解
    
    用途：
    - 简单启发式baseline
    - 对比MDP的优势
    
    注意：这个策略已经在ARGO_System中实现（use_mdp=False）
    这里提供一个显式的类以便统一接口
    """
    
    def __init__(
        self,
        *args,
        theta_cont: float = 0.5,
        theta_star: float = 0.9,
        **kwargs
    ):
        """
        初始化Fixed-Threshold策略
        
        Args:
            theta_cont: 继续阈值（低于此值检索，否则推理）
            theta_star: 终止阈值
        """
        kwargs['use_mdp'] = False
        
        super().__init__(*args, **kwargs)
        
        self.fixed_theta_cont = theta_cont
        self.fixed_theta_star = theta_star
        
        logger.info(f"FixedThresholdStrategy initialized "
                   f"(θ_cont={theta_cont}, θ_star={theta_star})")
    
    def answer_question(
        self,
        question: str,
        return_history: bool = True
    ) -> Tuple[str, Optional[List[Dict]], Optional[Dict]]:
        """
        回答问题（Fixed-Threshold策略）
        
        使用固定阈值决策
        """
        import time
        start_time = time.time()
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Fixed-Threshold Strategy Processing Question")
            print(f"{'='*80}")
            print(f"Question: {question}")
            print(f"Max Steps: {self.max_steps}")
            print(f"Strategy: Fixed (θ_cont={self.fixed_theta_cont}, θ_star={self.fixed_theta_star})")
            print(f"{'='*80}\n")
        
        # 初始化
        history = []
        U_t = 0.0
        t = 0
        
        delta_r = 0.25
        delta_p = 0.08
        
        # 主循环：基于固定阈值决策
        while U_t < self.fixed_theta_star and t < self.max_steps:
            t += 1
            
            if self.verbose:
                print(f"\n--- Step {t} ---")
                print(f"Current U_t: {U_t:.3f}")
            
            # 固定阈值策略
            if U_t < self.fixed_theta_cont:
                action = 'retrieve'
            else:
                action = 'reason'
            
            if self.verbose:
                print(f"Action: {action.upper()} (U_t {'<' if action=='retrieve' else '>='} θ_cont)")
            
            # 执行动作
            if action == 'retrieve':
                step_data = self._execute_retrieve(question, history, U_t)
                
                if step_data['retrieval_success']:
                    U_t += delta_r
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
                
                U_t += delta_p
                if self.verbose:
                    print(f"Reasoning complete, U_t → {U_t:.3f}")
                
                self.stats['reason_actions'] += 1
            
            history.append(step_data)
            U_t = min(U_t, 1.0)
        
        # 合成最终答案
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Synthesizing Final Answer")
            print(f"{'='*80}")
            print(f"Final U_t: {U_t:.3f}")
            print(f"Total Steps: {t}")
        
        final_answer, sources = self.synthesizer.synthesize(question, history)
        
        # 统计
        elapsed_time = time.time() - start_time
        self.stats['total_queries'] += 1
        self.stats['total_steps'] += t
        self.stats['total_time'] += elapsed_time
        
        metadata = {
            'strategy': 'Fixed-Threshold',
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
            print(f"Steps: {metadata['total_steps']}")
            print(f"Retrieve: {metadata['retrieve_count']}, Reason: {metadata['reason_count']}")
            print(f"Time: {metadata['elapsed_time']:.2f}s")
            print(f"{'='*80}\n")
        
        if return_history:
            return final_answer, history, metadata
        else:
            return final_answer, None, metadata
