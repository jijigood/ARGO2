"""Heuristic question-complexity estimation for ARGO experiments."""

from __future__ import annotations

import re
from typing import Dict, Optional


class QuestionComplexityClassifier:
    """Estimate question complexity using lightweight lexical cues."""

    CONNECTOR_PATTERN = re.compile(
        r"\b(and|also|furthermore|additionally|between|versus|while|whereas|either|both|respectively)\b",
        re.IGNORECASE,
    )
    MULTI_HOP_KEYWORDS = (
        'compare', 'difference', 'relationship', 'sequence', 'impact', 'interaction', 'combination'
    )

    def __init__(self, config: Optional[Dict] = None) -> None:
        config = config or {}
        self.simple_length = config.get('simple_length', 12)
        self.medium_length = config.get('medium_length', 24)
        self.simple_score_max = config.get('simple_score_max', 1)
        self.medium_score_max = config.get('medium_score_max', 3)
        self.use_difficulty_hint = config.get('use_difficulty_hint', True)
        self.difficulty_mapping = config.get(
            'difficulty_mapping',
            {
                'easy': 'simple',
                'e': 'simple',
                'medium': 'medium',
                'm': 'medium',
                'hard': 'complex',
                'h': 'complex',
            },
        )

    def classify(self, question: str, metadata: Optional[Dict] = None) -> str:
        """Return 'simple', 'medium', or 'complex' for *question*."""

        sanitized = (question or '').strip()
        tokens = self._tokenize(sanitized)
        length = len(tokens)
        lower = sanitized.lower()

        connectors = len(self.CONNECTOR_PATTERN.findall(lower))
        clauses = sanitized.count(';') + sanitized.count(':')
        multi_sent = max(0, len([s for s in re.split(r'[.!?]', sanitized) if s.strip()]) - 1)
        enumerations = len(re.findall(r'(?:\d+\.|\([a-d]\)|\([ivx]+\))', sanitized, re.IGNORECASE))
        numerics = len(re.findall(r'\d', sanitized))
        hop_keywords = sum(1 for kw in self.MULTI_HOP_KEYWORDS if kw in lower)
        question_marks = sanitized.count('?')

        score = 0
        if length > self.medium_length:
            score += 2
        elif length > self.simple_length:
            score += 1

        score += max(0, connectors - 1)
        score += hop_keywords
        if enumerations >= 2:
            score += 1
        if numerics >= 3:
            score += 1
        if multi_sent >= 1 or question_marks > 1:
            score += 1
        score += min(clauses, 2)

        if length <= self.simple_length and connectors == 0 and hop_keywords == 0:
            label = 'simple'
        elif score <= self.simple_score_max:
            label = 'simple'
        elif score <= self.medium_score_max:
            label = 'medium'
        else:
            label = 'complex'

        if self.use_difficulty_hint and metadata:
            hint = metadata.get('difficulty')
            if hint:
                hint_label = self.difficulty_mapping.get(str(hint).lower())
                if hint_label:
                    label = self._blend_labels(label, hint_label)

        return label

    @staticmethod
    def _blend_labels(primary: str, hint: str) -> str:
        order = {'simple': 0, 'medium': 1, 'complex': 2}
        if primary not in order or hint not in order:
            return primary
        blended = max(order[primary], order[hint])
        for label, idx in order.items():
            if idx == blended:
                return label
        return primary

    @staticmethod
    def _tokenize(text: str) -> list:
        return [tok for tok in re.findall(r"[A-Za-z0-9]+", text) if tok]