"""
Answer Synthesizer - Phase 3.3 (Enhanced with ARGO Prompts V2.0)
=================================================================

基于完整推理历史合成最终答案。整合所有子查询、检索文档和推理步骤，
生成连贯的最终回答。

Key Features:
- 历史整合：利用所有 (q_t, r_t) 对
- 上下文聚合：智能组合多次检索的信息
- 答案生成：基于LLM生成连贯回答（使用标准化prompts）
- 溯源支持：可追溯答案来源
- 格式化输出：支持长/短答案格式

Input:
- x: 原始问题
- H_T: 完整推理历史（包含所有步骤）

Output:
- O: 最终答案（字符串）
- sources: 答案来源（可选）

Example:
    synthesizer = AnswerSynthesizer(model, tokenizer)
    answer, sources = synthesizer.synthesize(
        original_question="What is O-RAN latency requirement?",
        history=complete_history
    )
"""

import torch
from typing import Dict, List, Optional, Tuple
import logging
from .prompts import ARGOPrompts, PromptConfig

logger = logging.getLogger(__name__)


class AnswerSynthesizer:
    """
    基于完整推理历史的答案合成器
    
    核心思想：
    1. 收集所有检索到的文档
    2. 整理推理步骤的中间结论
    3. 用LLM综合生成最终答案
    4. 提供答案溯源信息
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        max_answer_length: int = None,
        temperature: float = None,
        top_p: float = None,
        include_sources: bool = True
    ):
        """
        Args:
            model: 预训练的因果语言模型
            tokenizer: 对应的tokenizer
            max_answer_length: 答案的最大长度（默认使用PromptConfig）
            temperature: 生成温度（默认使用PromptConfig）
            top_p: nucleus sampling参数（默认使用PromptConfig）
            include_sources: 是否包含答案来源
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # 使用配置的默认值（如果未提供）
        self.max_answer_length = max_answer_length or PromptConfig.SYNTHESIZER_MAX_LENGTH
        self.temperature = temperature if temperature is not None else PromptConfig.SYNTHESIZER_TEMPERATURE
        self.top_p = top_p if top_p is not None else PromptConfig.SYNTHESIZER_TOP_P
        self.include_sources = include_sources
        
        # 设备
        self.device = next(model.parameters()).device
        
        logger.info(f"AnswerSynthesizer initialized on device: {self.device}")
        logger.info(f"Using ARGO Prompts V2.0 with formatted output")
    
    def _build_synthesis_prompt(
        self,
        original_question: str,
        history: List[Dict],
        options: Optional[List[str]] = None
    ) -> Tuple[str, List[str]]:
        """
        构建答案合成的提示词（使用ARGO V2.0标准模板）
        
        Args:
            original_question: 原始问题
            history: 完整推理历史
            options: 选择题的选项列表（可选）
        
        Returns:
            (prompt, sources): 提示词和来源列表
        """
        # 使用标准化的prompt构建方法
        prompt = ARGOPrompts.build_synthesis_prompt(
            original_question=original_question,
            history=history,
            options=options
        )
        
        # 提取来源
        sources = []
        for step in history:
            if step['action'] == 'retrieve' and step.get('retrieval_success', False):
                docs = step.get('retrieved_docs', [])
                for doc in docs:
                    if '[Source:' in doc:
                        source = doc.split('[Source:')[1].split(']')[0].strip()
                        if source not in sources:
                            sources.append(source)
        
        return prompt, sources
    
    def synthesize(
        self,
        original_question: str,
        history: List[Dict],
        options: Optional[List[str]] = None
    ) -> Tuple[str, Optional[str], Optional[List[str]]]:
        """
        合成最终答案（主入口）
        
        Args:
            original_question: 原始问题
            history: 完整推理历史
            options: 选择题的选项列表（可选，格式为 ["选项1文本", "选项2文本", ...]）
        
        Returns:
            (answer, choice, sources):
                - answer: 最终答案字符串（详细解释）
                - choice: 选择的选项编号（"1"/"2"/"3"/"4"，仅当提供options时）
                - sources: 答案来源列表（如果include_sources=True）
        """
        # 构建提示词
        prompt, sources = self._build_synthesis_prompt(
            original_question,
            history,
            options
        )
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096  # 允许较长的上下文
        ).to(self.device)
        
        # 生成答案
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_answer_length,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True if self.temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解码
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        raw_answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # 后处理（提取答案和选项）
        answer, choice = self._postprocess_answer(raw_answer, has_options=options is not None)
        
        logger.info(f"Synthesized answer ({len(answer)} chars) from {len(history)} reasoning steps")
        if choice:
            logger.info(f"Selected choice: {choice}")
        
        if self.include_sources:
            return answer, choice, sources
        else:
            return answer, choice, None
    
    def _postprocess_answer(self, answer: str, has_options: bool = False) -> Tuple[str, Optional[str]]:
        """
        后处理生成的答案
        
        清理和优化：
        1. 提取格式化的长/短答案（如果存在）
        2. 提取选择题的选项编号（如果是选择题）
        3. 去除前后空白
        4. 移除多余的换行
        5. 确保答案以句号结尾
        6. 长度限制
        
        Args:
            answer: 原始生成的答案
            has_options: 是否为选择题格式
        
        Returns:
            (cleaned_answer, choice):
                - cleaned_answer: 清理后的答案文本
                - choice: 选择的选项编号（"1"/"2"/"3"/"4"，仅当has_options=True时）
        """
        import re
        
        choice = None
        
        # 如果是选择题，提取 <choice>X</choice>
        if has_options:
            choice_match = re.search(r'<choice>(\d)</choice>', answer)
            if choice_match:
                choice = choice_match.group(1)
                logger.info(f"Extracted choice: {choice}")
            else:
                # 尝试从文本中提取（例如 "Option 3" 或 "选项3"）
                fallback_match = re.search(r'[Oo]ption\s*(\d)|选项\s*(\d)', answer)
                if fallback_match:
                    choice = fallback_match.group(1) or fallback_match.group(2)
                    logger.warning(f"Fallback choice extraction: {choice}")
                else:
                    logger.warning("Could not extract choice from answer")
        
        # 尝试提取格式化的答案
        # 格式: <answer long>...</answer long><answer short>...</answer short>
        long_match = re.search(r'<answer long>(.*?)</answer long>', answer, re.DOTALL)
        short_match = re.search(r'<answer short>(.*?)</answer short>', answer, re.DOTALL)
        
        if long_match:
            # 使用长答案
            answer = long_match.group(1).strip()
            logger.info("Extracted formatted long answer")
        elif short_match:
            # 如果只有短答案，使用短答案
            answer = short_match.group(1).strip()
            logger.info("Extracted formatted short answer")
        else:
            # 没有格式化标签，使用原始答案
            # 移除 <choice> 标签以免干扰
            answer = re.sub(r'<choice>\d</choice>', '', answer).strip()
        
        # 压缩多个换行为一个
        answer = re.sub(r'\n\n+', '\n\n', answer)
        
        # 如果答案不以标点结尾，添加句号
        if answer and answer[-1] not in '.!?':
            answer += '.'
        
        # 长度限制（字符级别）
        max_chars = 1024
        if len(answer) > max_chars:
            # 截断到最后一个句号
            truncated = answer[:max_chars].rsplit('.', 1)[0] + '.'
            answer = truncated
            logger.warning(f"Answer truncated to {len(answer)} chars")
        
        return answer, choice
    
    def batch_synthesize(
        self,
        questions: List[str],
        histories: List[List[Dict]],
        options_list: Optional[List[List[str]]] = None
    ) -> List[Tuple[str, Optional[str], Optional[List[str]]]]:
        """
        批量合成答案（加速）
        
        Args:
            questions: 原始问题列表
            histories: 历史记录列表
            options_list: 选项列表的列表（可选）
        
        Returns:
            (answer, choice, sources) 列表
        """
        assert len(questions) == len(histories), \
            "Questions and histories must have the same length"
        
        if options_list is not None:
            assert len(options_list) == len(questions), \
                "Options list must have the same length as questions"
        
        # 批量构建提示词
        prompts_and_sources = [
            self._build_synthesis_prompt(
                q, h, 
                options=options_list[i] if options_list else None
            )
            for i, (q, h) in enumerate(zip(questions, histories))
        ]
        
        prompts = [p for p, _ in prompts_and_sources]
        sources_list = [s for _, s in prompts_and_sources]
        
        # Tokenize all prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(self.device)
        
        # 批量生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_answer_length,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True if self.temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 批量解码
        batch_size = len(questions)
        results = []
        has_options = options_list is not None
        
        for i in range(batch_size):
            input_len = inputs['input_ids'][i].shape[0]
            generated_ids = outputs[i][input_len:]
            raw_answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            answer, choice = self._postprocess_answer(raw_answer, has_options=has_options)
            
            if self.include_sources:
                results.append((answer, choice, sources_list[i]))
            else:
                results.append((answer, choice, None))
        
        return results
    
    def generate_summary(self, history: List[Dict]) -> str:
        """
        生成推理历史摘要（用于日志或可视化）
        
        Args:
            history: 推理历史
        
        Returns:
            摘要字符串
        """
        summary = "Reasoning Summary:\n"
        summary += "=" * 60 + "\n"
        
        total_steps = len(history)
        retrieve_count = sum(1 for s in history if s.get('action') == 'retrieve')
        reason_count = sum(1 for s in history if s.get('action') == 'reason')
        
        successful_retrieves = sum(
            1 for s in history 
            if s.get('action') == 'retrieve' and s.get('retrieval_success', False)
        )
        
        summary += f"Total Steps: {total_steps}\n"
        summary += f"Retrieve Actions: {retrieve_count} ({successful_retrieves} successful)\n"
        summary += f"Reason Actions: {reason_count}\n"
        summary += "\nStep-by-step:\n"
        
        for i, step in enumerate(history):
            action = step.get('action', 'unknown')
            
            if action == 'retrieve':
                subq = step.get('subquery', 'N/A')
                success = "✓" if step.get('retrieval_success', False) else "✗"
                summary += f"  {i+1}. [Retrieve {success}] {subq[:60]}...\n"
            
            elif action == 'reason':
                answer = step.get('intermediate_answer', 'N/A')
                conf = step.get('confidence', 0.0)
                summary += f"  {i+1}. [Reason] {answer[:60]}... (conf={conf:.2f})\n"
        
        summary += "=" * 60 + "\n"
        
        return summary
