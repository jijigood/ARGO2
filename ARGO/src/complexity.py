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
        'compare', 'difference', 'relationship', 'sequence', 
        'impact', 'interaction', 'combination',
        'architecture', 'deployment', 'optimization',
        'procedure', 'workflow', 'lifecycle',
        'trade-off', 'tradeoff', 'advantage', 'disadvantage',
        'between', 'versus', 'affects', 'enables', 'explain', 'how', 'support'
    )

    TECHNICAL_DEPTH_KEYWORDS = (
        'protocol', 'algorithm', 'mechanism', 'implementation',
        'specification', 'standard', 'architecture', 'framework',
        'latency', 'throughput', 'scalability', 'performance'
    )

    def __init__(self, config: Optional[Dict] = None) -> None:
        config = config or {}
        self.simple_length = config.get('simple_length', 10)
        self.medium_length = config.get('medium_length', 20)
        self.simple_score_max = config.get('simple_score_max', 0)
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

        tech_depth = sum(1 for kw in self.TECHNICAL_DEPTH_KEYWORDS if kw in lower)

        score = 0
        if length > self.medium_length:
            score += 2
        elif length > self.simple_length:
            score += 1

        score += max(0, connectors - 1)
        score += hop_keywords
        if tech_depth >= 2:
            score += 1
        if enumerations >= 2:
            score += 1
        if numerics >= 3:
            score += 1
        if multi_sent >= 1 or question_marks > 1:
            score += 1
        score += min(clauses, 2)

        if length <= self.simple_length and connectors == 0 and hop_keywords == 0 and tech_depth == 0:
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

    def estimate_umax(self, question: str, metadata: Optional[Dict] = None) -> float:
        """Estimate question-specific information requirement U_max(x)."""

        complexity = self.classify(question, metadata)
        base_umax = {
            'simple': 0.40,
            'medium': 0.70,
            'complex': 0.85,
        }[complexity]

        lower = (question or '').lower()
        adjustments = 0.0

        if any(pattern in lower for pattern in ['what is', 'define', 'who is', 'when was', 'where is']):
            adjustments -= 0.12

        if any(pattern in lower for pattern in ['is it', 'does it', 'can it', 'will it', 'should it']):
            adjustments -= 0.08

        if any(pattern in lower for pattern in ['compare', 'difference between', 'versus', 'vs']):
            adjustments += 0.12

        if any(pattern in lower for pattern in ['analyze', 'explain why', 'how does', 'what causes']):
            adjustments += 0.08

        if any(pattern in lower for pattern in ['list all', 'enumerate', 'what are the']):
            adjustments += 0.10

        question_marks = (question or '').count('?')
        if question_marks > 1:
            adjustments += 0.05 * (question_marks - 1)

        conjunction_count = sum(1 for word in ['and', 'also', 'furthermore'] if word in lower)
        if conjunction_count > 1:
            adjustments += 0.04 * (conjunction_count - 1)

        final_umax = base_umax + adjustments
        final_umax = max(0.30, min(0.95, final_umax))
        return final_umax

    def get_adaptive_max_steps(self, question: str, base_max_steps: int = 10) -> int:
        """Estimate appropriate max_steps based on question complexity."""

        complexity = self.classify(question)
        scaling_factor = {
            'simple': 0.5,
            'medium': 0.8,
            'complex': 1.2,
        }[complexity]

        adjusted = int(base_max_steps * scaling_factor)
        return max(3, min(adjusted, base_max_steps + 5))

    @staticmethod
    def _blend_labels(primary: str, hint: str) -> str:
        """
        Blend primary (estimated) complexity with hint (from dataset).
        
        When dataset provides difficulty hint (easy/medium/hard), 
        we TRUST the hint and use it directly since it comes from
        expert annotation.
        
        Args:
            primary: Estimated complexity from question analysis
            hint: Difficulty hint from dataset metadata
            
        Returns:
            Final complexity label
        """
        order = {'simple': 0, 'medium': 1, 'complex': 2}
        if primary not in order:
            return primary
        if hint not in order:
            return primary
        # When hint is available, trust it over estimation
        # This ensures easy questions get classified as simple
        return hint

    @staticmethod
    def _tokenize(text: str) -> list:
        return [tok for tok in re.findall(r"[A-Za-z0-9]+", text) if tok]