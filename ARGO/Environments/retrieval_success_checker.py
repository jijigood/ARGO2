#!/usr/bin/env python
"""
检索成功检查器
================
实现基于检索质量的概率性成功判断机制

两种模式:
1. 'probabilistic': 固定概率 p_s (用于验证MDP)
2. 'similarity': 基于检索分数 (用于实际部署)
"""
import numpy as np
from typing import List, Tuple, Optional


class RetrievalSuccessChecker:
    """
    判断检索是否成功
    
    根据MDP假设：检索以概率p_s成功，只有成功时才降低不确定性U
    """
    
    def __init__(
        self,
        mode: str = 'similarity',
        p_s: float = 0.8,
        similarity_threshold: float = 0.5
    ):
        """
        Args:
            mode: 'probabilistic' (固定p_s) 或 'similarity' (基于相似度)
            p_s: 成功概率 (在probabilistic模式下使用)
            similarity_threshold: 相似度阈值 (在similarity模式下使用)
        """
        assert mode in ['probabilistic', 'similarity'], f"Unknown mode: {mode}"
        assert 0.0 <= p_s <= 1.0, f"p_s must be in [0, 1], got {p_s}"
        assert 0.0 <= similarity_threshold <= 1.0, f"threshold must be in [0, 1]"
        
        self.mode = mode
        self.p_s = p_s
        self.similarity_threshold = similarity_threshold
    
    def check_success(
        self,
        docs: List[str],
        scores: Optional[List[float]] = None,
        query: Optional[str] = None
    ) -> bool:
        """
        判断检索是否成功
        
        Args:
            docs: 检索到的文档列表
            scores: 检索相似度分数 (如果可用)
            query: 原始查询 (用于启发式质量检查)
        
        Returns:
            True 如果检索成功, False 否则
        """
        # 空结果肯定是失败
        if not docs or len(docs) == 0:
            return False
        
        if self.mode == 'probabilistic':
            # 固定概率模式 (符合MDP假设)
            return np.random.random() < self.p_s
        
        elif self.mode == 'similarity':
            # 基于检索质量的模式
            if scores is None or len(scores) == 0:
                # 没有分数时，使用启发式检查
                return self._heuristic_quality_check(docs, query)
            
            # 检查最高分是否超过阈值
            max_score = max(scores)
            return max_score >= self.similarity_threshold
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _heuristic_quality_check(
        self,
        docs: List[str],
        query: Optional[str]
    ) -> bool:
        """
        启发式质量检查 (当没有分数时)
        
        检查文档是否包含查询关键词
        """
        if query is None:
            return True  # 乐观默认值
        
        query_words = set(query.lower().split())
        
        # 检查是否有文档包含足够的关键词
        for doc in docs:
            doc_words = set(doc.lower().split())
            overlap = len(query_words & doc_words)
            
            # 至少匹配3个词或50%的查询词
            if overlap >= min(3, len(query_words) * 0.5):
                return True
        
        return False
    
    def get_expected_progress(self) -> float:
        """
        返回期望进度增量 (用于理论验证)
        
        在概率模式下: E[ΔU] = p_s * δ_r
        """
        if self.mode == 'probabilistic':
            return self.p_s
        else:
            # 相似度模式下，估计成功概率
            return 0.8  # 默认估计值
