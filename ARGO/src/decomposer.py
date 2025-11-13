"""
Query Decomposer - Phase 3.1 (Enhanced with ARGO Prompts V2.0)
===============================================================

基于LLM的动态子查询生成器。根据原问题、历史记录和当前进度U_t，
生成针对性的子查询以填补知识缺口。

Key Features:
- 动态生成：根据推理历史生成渐进式子问题
- 上下文感知：考虑已有信息，避免重复查询
- 进度引导：根据U_t调整查询粒度和深度
- 标准化Prompts：使用ARGO V2.0的高质量提示词模板

Input:
- x: 原始问题（字符串）
- H_t: 历史记录（包含之前的子查询和检索结果）
- U_t: 当前知识完整度 (0到1)

Output:
- q_t: 针对性子查询（字符串）

Example:
    decomposer = QueryDecomposer(model, tokenizer)
    subquery = decomposer.generate_subquery(
        original_question="What is the latency requirement for O-RAN?",
        history=history,
        uncertainty=0.3
    )
    # Output: "What are the specific latency thresholds for O-RAN fronthaul?"
"""

import torch
from typing import Dict, List, Optional, Tuple
import logging
from .prompts import ARGOPrompts, PromptConfig

logger = logging.getLogger(__name__)


class QueryDecomposer:
    """
    基于LLM的查询分解器
    
    核心思想：
    1. 分析原问题和历史，识别知识缺口
    2. 根据U_t调整子查询的具体程度
    3. 生成有针对性的子问题，引导检索
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        max_subquery_length: int = None,
        temperature: float = None,
        top_p: float = None
    ):
        """
        Args:
            model: 预训练的因果语言模型
            tokenizer: 对应的tokenizer
            max_subquery_length: 子查询的最大长度（默认使用PromptConfig）
            temperature: 生成温度（越高越发散，默认使用PromptConfig）
            top_p: nucleus sampling参数（默认使用PromptConfig）
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # 使用配置的默认值（如果未提供）
        self.max_subquery_length = max_subquery_length or PromptConfig.DECOMPOSER_MAX_LENGTH
        self.temperature = temperature if temperature is not None else PromptConfig.DECOMPOSER_TEMPERATURE
        self.top_p = top_p if top_p is not None else PromptConfig.DECOMPOSER_TOP_P
        
        # 设备
        self.device = next(model.parameters()).device
        
        logger.info(f"QueryDecomposer initialized on device: {self.device}")
        logger.info(f"Using ARGO Prompts V2.0 with progress tracking")
    
    def _build_decomposition_prompt(
        self,
        original_question: str,
        history: List[Dict],
        uncertainty: float
    ) -> str:
        """
        构建查询分解的提示词（使用ARGO V2.0标准模板）
        
        Args:
            original_question: 原始问题
            history: 之前的推理历史
            uncertainty: 当前不确定度 (1-U_t)，值越大说明缺口越大
        
        Returns:
            完整的提示词字符串
        """
        # 转换为进度（progress = 1 - uncertainty）
        progress = 1.0 - uncertainty
        
        # 使用标准化的prompt构建方法
        prompt = ARGOPrompts.build_decomposition_prompt(
            original_question=original_question,
            history=history,
            progress=progress
        )
        
        return prompt
    
    def generate_subquery(
        self,
        original_question: str,
        history: List[Dict],
        uncertainty: float
    ) -> str:
        """
        生成子查询（主入口）
        
        Args:
            original_question: 原始问题
            history: 推理历史（每个元素包含 'action', 'progress' 等字段）
            uncertainty: 不确定度 (1 - U_t)
        
        Returns:
            生成的子查询字符串
        """
        # 为历史记录添加进度信息（如果缺失）
        history_with_progress = []
        for i, step in enumerate(history):
            step_copy = step.copy()
            if 'progress' not in step_copy:
                # 估算进度（假设线性增长）
                step_copy['progress'] = (i + 1) / (len(history) + 1)
            history_with_progress.append(step_copy)
        
        # 构建提示词
        prompt = self._build_decomposition_prompt(
            original_question,
            history_with_progress,
            uncertainty
        )
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096  # 增加长度以容纳完整示例
        ).to(self.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_subquery_length,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True if self.temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解码
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]  # 只取新生成的部分
        subquery = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # 后处理
        subquery = self._postprocess_subquery(subquery)
        
        logger.info(f"Generated subquery (U={1-uncertainty:.2f}): {subquery}")
        
        return subquery
    
    def _postprocess_subquery(self, subquery: str) -> str:
        """
        后处理生成的子查询
        
        清理和规范化：
        1. 去除前后空白
        2. 截断到第一个换行或句号
        3. 确保以问号结尾（如果是疑问句）
        4. 长度限制
        
        Args:
            subquery: 原始生成的子查询
        
        Returns:
            清理后的子查询
        """
        # 去除前后空白
        subquery = subquery.strip()
        
        # 截断到第一个换行
        if '\n' in subquery:
            subquery = subquery.split('\n')[0].strip()
        
        # 截断到第一个句号（如果不是问号结尾）
        if '?' not in subquery and '.' in subquery:
            subquery = subquery.split('.')[0].strip()
        
        # 如果包含典型疑问词，确保以问号结尾
        question_words = ['what', 'where', 'when', 'who', 'why', 'how', 'which']
        if any(subquery.lower().startswith(qw) for qw in question_words):
            if not subquery.endswith('?'):
                subquery += '?'
        
        # 长度限制（字符级别）
        max_chars = 256
        if len(subquery) > max_chars:
            # 截断到最后一个完整单词
            subquery = subquery[:max_chars].rsplit(' ', 1)[0] + '...'
        
        return subquery
    
    def batch_generate_subqueries(
        self,
        questions: List[str],
        histories: List[List[Dict]],
        uncertainties: List[float]
    ) -> List[str]:
        """
        批量生成子查询（用于加速）
        
        Args:
            questions: 原始问题列表
            histories: 历史记录列表
            uncertainties: 不确定度列表
        
        Returns:
            子查询列表
        """
        assert len(questions) == len(histories) == len(uncertainties), \
            "Input lists must have the same length"
        
        # 批量构建提示词
        prompts = [
            self._build_decomposition_prompt(q, h, u)
            for q, h, u in zip(questions, histories, uncertainties)
        ]
        
        # Tokenize all prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # 批量生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_subquery_length,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 批量解码
        batch_size = len(questions)
        subqueries = []
        
        for i in range(batch_size):
            input_len = inputs['input_ids'][i].shape[0]
            generated_ids = outputs[i][input_len:]
            subquery = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            subquery = self._postprocess_subquery(subquery)
            subqueries.append(subquery)
        
        return subqueries
