"""Progress tracking utilities for ARGO.

Provides a lightweight heuristic estimator that maps intermediate
retrieval/reasoning outputs to a normalized progress score in [0, 1].
The estimator rewards coverage of question keywords and penalizes
redundant evidence so that simple questions can terminate earlier
than complex ones.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional


class ProgressTracker:
    """Estimate information progress based on accumulated evidence."""

    # Enhanced pattern to capture O-RAN terms
    TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
    
    # Define important O-RAN short-form terms to preserve
    ORAN_TECHNICAL_TERMS = {
        'oran', 'o-ran', 'ran',
        'e2', 'a1', 'o1', 'o2',
        'ric', 'cu', 'du', 'ru', 'cp', 'up',
        'ue', 'ng', 'f1', 'n1', 'n2', 'n3',
        'phy', 'mac', 'rlc', 'pdcp', 'rrc',
        'smo', 'rapp', 'xapp',
        'qos', 'kpm', 'rc', 'mro',
        '5g', '4g', 'lte', 'nr',
        'api', 'sdk', 'ai', 'ml',
    }

    def __init__(
        self,
        question: str,
        base_retrieval_gain: float = 0.25,
        base_reason_gain: float = 0.1,
        coverage_weight: float = 0.5,
        novelty_weight: float = 0.3,
        confidence_weight: float = 0.4,
        min_gain: float = 0.02,
        max_gain: float = 0.35,
        gain_multiplier: float = 1.0,
    ) -> None:
        self.question_tokens = self._extract_tokens(question)
        if not self.question_tokens:
            self.question_tokens = self._extract_tokens(question + " context")

        self.base_retrieval_gain = max(0.01, base_retrieval_gain) * gain_multiplier
        self.base_reason_gain = max(0.01, base_reason_gain) * gain_multiplier
        self.coverage_weight = max(0.0, coverage_weight)
        self.novelty_weight = max(0.0, novelty_weight)
        self.confidence_weight = max(0.0, min(confidence_weight, 1.0))
        self.min_gain = max(0.0, min_gain)
        self.max_gain = max(self.min_gain, max_gain)

        self.current_progress = 0.0
        self.covered_question_tokens = set()
        self.seen_tokens = set()

    def update(self, action: str, step_data: Dict) -> float:
        """Update progress after executing *action* with metadata *step_data*."""

        if action == 'retrieve' and not step_data.get('retrieval_success', True):
            return self.current_progress

        text_segments: List[str] = []
        answer = step_data.get('intermediate_answer')
        if answer:
            text_segments.append(answer)

        if action == 'retrieve':
            docs: Iterable[str] = step_data.get('retrieved_docs') or []
            text_segments.extend(docs)

        combined_text = " ".join(text_segments).strip()
        if not combined_text:
            return self.current_progress

        tokens = set(self._extract_tokens(combined_text))
        question_tokens = set(self.question_tokens)

        newly_covered = tokens & question_tokens - self.covered_question_tokens
        coverage_gain = 0.0
        if question_tokens:
            coverage_gain = len(newly_covered) / len(question_tokens)

        novel_tokens = tokens - self.seen_tokens
        novelty_gain = 0.0
        denominator = max(len(question_tokens), len(tokens) // 3, 10)
        if denominator:
            novelty_gain = len(novel_tokens) / denominator

        self.covered_question_tokens.update(newly_covered)
        self.seen_tokens.update(tokens)

        confidence = step_data.get('confidence', 0.5)
        confidence = max(0.0, min(1.0, confidence))
        
        confidence_multiplier = 0.5 + 0.8 * (confidence - 0.5)
        confidence_multiplier = max(0.2, min(1.2, confidence_multiplier))

        base_gain = self.base_retrieval_gain if action == 'retrieve' else self.base_reason_gain
        base_gain *= confidence_multiplier

        delta = base_gain
        delta += self.coverage_weight * coverage_gain
        delta += self.novelty_weight * max(0.0, novelty_gain)
        delta = max(self.min_gain, min(delta, self.max_gain))

        self.current_progress = min(1.0, self.current_progress + delta)
        return self.current_progress

    @classmethod
    def _extract_tokens(cls, text: str) -> List[str]:
        raw_tokens = cls.TOKEN_PATTERN.findall(text.lower())
        
        filtered = []
        for token in raw_tokens:
            if token in cls.ORAN_TECHNICAL_TERMS:
                filtered.append(token)
            elif len(token) >= 4 and not token.isdigit():
                filtered.append(token)
        
        return filtered


class FastProgressTracker:
    """Lightweight heuristic tracker that avoids expensive tokenization."""

    def __init__(
        self,
        question: str,
        base_retrieval_gain: float = 0.25,
        base_reason_gain: float = 0.1,
    ) -> None:
        self.question = question
        self.question_length = len(question)
        self.base_retrieval_gain = max(0.01, base_retrieval_gain)
        self.base_reason_gain = max(0.01, base_reason_gain)

        self.current_progress = 0.0
        self.step_count = 0
        self.retrieval_count = 0
        self.accumulated_length = 0

    def update(self, action: str, step_data: Dict) -> float:
        self.step_count += 1

        if action == 'retrieve' and not step_data.get('retrieval_success', True):
            return self.current_progress

        answer = step_data.get('intermediate_answer') or ''
        docs = step_data.get('retrieved_docs') or []

        new_length = len(answer)
        if isinstance(docs, Iterable):
            new_length += sum(len(doc) for doc in docs)
        self.accumulated_length += new_length

        gain = self.base_retrieval_gain if action == 'retrieve' else self.base_reason_gain
        confidence = float(step_data.get('confidence', 0.5))
        confidence = max(0.0, min(1.0, confidence))
        gain *= 0.5 + confidence

        if action == 'retrieve':
            self.retrieval_count += 1
            if self.retrieval_count > 3:
                gain *= 0.9

        if self.step_count > 5:
            gain *= 0.8

        info_budget = max(2000, self.question_length * 5)
        if self.accumulated_length > info_budget:
            gain *= 0.9

        self.current_progress = min(1.0, self.current_progress + gain)
        return self.current_progress

