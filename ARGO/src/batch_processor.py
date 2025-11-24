"""
Batched LLM inference utilities for ARGO.

Centralizes repeated generation calls so we can amortize tokenizer/model
invocations across multiple prompts when processing several questions or
subqueries at once.
"""

from __future__ import annotations

import logging
from collections import deque
from threading import Lock
from typing import Dict, List

import torch

logger = logging.getLogger(__name__)


class LLMBatchProcessor:
    """Simple batching wrapper around a HuggingFace-style model/tokenizer."""

    def __init__(
        self,
        model,
        tokenizer,
        batch_size: int = 4,
        timeout_ms: int = 100,
        max_padding_ratio: float = 2.0,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

        self.batch_size = batch_size
        self.timeout_ms = timeout_ms
        self.max_padding_ratio = max_padding_ratio

        self.queue: deque = deque()
        self.lock = Lock()

        self.total_requests = 0
        self.total_batches = 0
        self.avg_batch_size = 0.0

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> str:
        # For now this routes directly to single generation. Queue-based
        # aggregation can be enabled later when multiple threads submit work.
        return self._generate_single(prompt, max_new_tokens, temperature, top_p, **kwargs)

    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> List[str]:
        if not prompts:
            return []

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(self.device)

        attention_lengths = inputs['attention_mask'].sum(dim=1).cpu().numpy()
        max_len = attention_lengths.max()
        avg_len = attention_lengths.mean() or 1.0
        padding_ratio = max_len / avg_len
        if padding_ratio > self.max_padding_ratio:
            logger.warning(
                "High padding ratio detected (%.2f). Consider splitting batches.",
                padding_ratio,
            )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        responses: List[str] = []
        for idx in range(len(prompts)):
            input_len = inputs['input_ids'][idx].shape[0]
            generated_ids = outputs[idx][input_len:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            responses.append(text.strip())

        self.total_requests += len(prompts)
        self.total_batches += 1
        self.avg_batch_size = self.total_requests / self.total_batches

        return responses

    def get_stats(self) -> Dict[str, float]:
        efficiency = (self.avg_batch_size / self.batch_size) if self.batch_size else 0.0
        return {
            'total_requests': float(self.total_requests),
            'total_batches': float(self.total_batches),
            'avg_batch_size': self.avg_batch_size,
            'efficiency': efficiency,
        }

    def _generate_single(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        **kwargs,
    ) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text.strip()
