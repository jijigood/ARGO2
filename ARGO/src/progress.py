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

    TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")

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
        denominator = max(len(question_tokens), 20)
        if denominator:
            novelty_gain = len(novel_tokens) / denominator

        self.covered_question_tokens.update(newly_covered)
        self.seen_tokens.update(tokens)

        confidence = step_data.get('confidence', 0.5)
        confidence = max(0.0, min(1.0, confidence))
        confidence_term = 0.5 + self.confidence_weight * (confidence - 0.5)

        base_gain = self.base_retrieval_gain if action == 'retrieve' else self.base_reason_gain
        base_gain *= max(0.2, confidence_term)

        delta = base_gain
        delta += self.coverage_weight * coverage_gain
        delta += self.novelty_weight * max(0.0, novelty_gain)
        delta = max(self.min_gain, min(delta, self.max_gain))

        self.current_progress = min(1.0, self.current_progress + delta)
        return self.current_progress

    @staticmethod
    def _extract_tokens(text: str) -> List[str]:
        tokens = ProgressTracker.TOKEN_PATTERN.findall(text.lower())
        return [t for t in tokens if len(t) >= 4 and not t.isdigit()]

