"""
Local LLM answer generation utilities.
为本地LLM提供答案生成封装。
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional, Union

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        pipeline,
    )
except ImportError as exc:  # pragma: no cover - hard failure when transformers missing
    raise ImportError(
        "transformers is required to use LocalLLMAnswerGenerator"
    ) from exc

try:  # pragma: no cover - optional dependency for dtype selection
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


DEFAULT_LLM_MODEL = os.getenv("ARGO_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
DEFAULT_DEVICE_MAP = os.getenv("ARGO_LLM_DEVICE_MAP", "auto")
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("ARGO_LLM_MAX_NEW_TOKENS", "512"))
DEFAULT_TEMPERATURE = float(os.getenv("ARGO_LLM_TEMPERATURE", "0.2"))
DEFAULT_TOP_P = float(os.getenv("ARGO_LLM_TOP_P", "0.9"))
DEFAULT_USE_4BIT = os.getenv("ARGO_LLM_4BIT", "true").lower() not in {"0", "false", "no"}
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "ARGO_LLM_SYSTEM_PROMPT",
    "You are an expert telecom assistant. Use the provided O-RAN context to answer questions succinctly.",
)
DEFAULT_CACHE_DIR = os.getenv("ARGO_HF_CACHE")


def _select_compute_dtype() -> Optional["torch.dtype"]:
    if torch is None:
        return None
    if torch.cuda.is_available():
        if hasattr(torch, "bfloat16"):
            return torch.bfloat16
        return torch.float16
    return None


@dataclass
class GenerationConfig:
    """LLM generation configuration."""

    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


class LocalLLMAnswerGenerator:
    """Wrap a local LLM for answer generation within the RAG pipeline."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_LLM_MODEL,
        device_map: str | Dict[str, int] = DEFAULT_DEVICE_MAP,
        use_4bit: bool = DEFAULT_USE_4BIT,
        cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
        generation_config: Optional[GenerationConfig] = None,
        trust_remote_code: bool = False,
        max_memory: Optional[Dict[str, Union[int, str]]] = None,
    ) -> None:
        self.model_name = model_name_or_path
        self.device_map = device_map
        self.use_4bit = use_4bit
        self.cache_dir = cache_dir
        self.trust_remote_code = trust_remote_code
        self.generation_config = generation_config or GenerationConfig()
        self.max_memory = max_memory

        quant_config = None
        compute_dtype = _select_compute_dtype()
        if use_4bit:
            if BitsAndBytesConfig is None:
                logging.warning("bitsandbytes not available; falling back to full precision load.")
            else:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype or None,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )

        logging.info("Loading LLM '%s' with device_map=%s", self.model_name, self.device_map)
        model_kwargs = {
            "device_map": self.device_map,
            "cache_dir": self.cache_dir,
            "trust_remote_code": self.trust_remote_code,
        }
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config
        elif compute_dtype is not None:
            model_kwargs["torch_dtype"] = compute_dtype
        if self.max_memory is not None:
            model_kwargs["max_memory"] = self.max_memory

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=self.trust_remote_code,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs,
            )
        except Exception as exc:  # pragma: no cover - surface quantization failures
            raise RuntimeError(
                f"Failed to load LLM '{self.model_name}'. Check GPU memory, bitsandbytes installation, and model availability."
            ) from exc

        self.generator = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map=self.device_map,
            max_new_tokens=self.generation_config.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # Pipeline copies tokenizer; ensure padding is defined.
        self.generator.tokenizer.pad_token_id = self.tokenizer.pad_token_id

    # prompt formatting -------------------------------------------------
    def _build_prompt(
        self,
        question: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        system = system_prompt or self.generation_config.system_prompt
        prompt = (
            f"<|system|>\n{system}\n"  # support chat-style models
            f"<|user|>\nQuestion: {question}\n\nContext:\n{context}\n\nPlease craft a grounded answer.\n"
            "<|assistant|>\n"
        )
        return prompt

    def generate_answer(
        self,
        question: str,
        context: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, str]:
        """Generate an answer given question and retrieved context."""

        prompt = self._build_prompt(question, context, system_prompt)
        gen_kwargs = {
            "temperature": self.generation_config.temperature,
            "top_p": self.generation_config.top_p,
            "do_sample": self.generation_config.temperature > 0,
            "max_new_tokens": max_new_tokens or self.generation_config.max_new_tokens,
        }
        outputs = self.generator(prompt, **gen_kwargs)
        generated_text = outputs[0]["generated_text"]
        answer = generated_text[len(prompt) :].strip()
        return {
            "answer": answer,
            "full_generation": generated_text,
            "prompt": prompt,
        }


__all__ = ["LocalLLMAnswerGenerator", "GenerationConfig"]
