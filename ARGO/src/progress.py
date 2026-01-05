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
        question_umax: float = 1.0,
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

        self.question_umax = max(0.3, min(0.95, question_umax))
        umax_scaling = (1.0 / max(0.3, self.question_umax)) ** 0.6

        self.base_retrieval_gain = max(0.01, base_retrieval_gain) * gain_multiplier * umax_scaling
        self.base_reason_gain = max(0.01, base_reason_gain) * gain_multiplier * umax_scaling
        self.coverage_weight = max(0.0, coverage_weight)
        self.novelty_weight = max(0.0, novelty_weight)
        self.confidence_weight = max(0.0, min(confidence_weight, 1.0))
        self.min_gain = max(0.0, min_gain)
        self.max_gain = max(self.min_gain, max_gain) * umax_scaling

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

        coverage_ratio = 0.0
        if self.question_tokens:
            coverage_ratio = len(self.covered_question_tokens) / len(self.question_tokens)

        if self.question_umax < 0.6 and confidence > 0.75 and coverage_ratio > 0.65:
            delta *= 1.6
        elif self.question_umax < 0.5 and confidence > 0.80 and coverage_ratio > 0.70:
            delta *= 2.0

        self.current_progress = min(self.question_umax, self.current_progress + delta)
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



class StationaryProgressTracker:
    """
    Stationary (fixed-gain) progress tracker that satisfies Assumption 1.
    
    Solves Problem 2: Dynamic Progress Tracking Violates Assumption 1
    
    Assumption 1 requires:
        E[R_t | H_t, U_t, a_t] = E[R_t | U_t, a_t]
    
    This means the progress indicator U_t must be a sufficient statistic,
    and the transition dynamics P(U_{t+1} | U_t, a_t) must not depend on
    the content of the history H_t.
    
    This tracker uses FIXED gains (δ_r, δ_p) exactly as assumed in the
    MDP formulation, ensuring the computed thresholds remain optimal.
    
    Theory:
        - Retrieve success: U_{t+1} = U_t + δ_r
        - Retrieve failure: U_{t+1} = U_t (no change)
        - Reason: U_{t+1} = U_t + δ_p
    """
    
    def __init__(
        self,
        question: str,
        delta_r: float = 0.25,
        delta_p: float = 0.08,
        p_s: float = 0.8,
        U_max: float = 1.0,
    ) -> None:
        """
        Initialize stationary tracker.
        
        Args:
            question: The question (stored for reference, not used in updates)
            delta_r: Fixed information gain for successful retrieval
            delta_p: Fixed information gain for reasoning
            p_s: Success probability (for reference only, actual success from step_data)
            U_max: Maximum progress value (question-specific upper bound)
        """
        self.question = question
        self.delta_r = max(0.0, delta_r)
        self.delta_p = max(0.0, delta_p)
        self.p_s = p_s
        self.U_max = max(0.1, U_max)
        
        self.current_progress = 0.0
        self.step_count = 0
        self.retrieve_count = 0
        self.reason_count = 0
        self.successful_retrieves = 0
    
    def update(self, action: str, step_data: Dict) -> float:
        """
        Update progress using fixed gains (MDP-compliant).
        
        The gain depends ONLY on action and success/failure,
        NOT on content (coverage, novelty, etc.)
        
        Args:
            action: 'retrieve' or 'reason'
            step_data: Must contain 'retrieval_success' for retrieve actions
            
        Returns:
            Updated progress value
        """
        self.step_count += 1
        
        if action == 'retrieve':
            self.retrieve_count += 1
            
            # Check retrieval success (from actual system, not simulated)
            success = step_data.get('retrieval_success', True)
            
            if success:
                self.successful_retrieves += 1
                delta = self.delta_r
            else:
                delta = 0.0  # Failed retrieval: no progress gain
        
        elif action == 'reason':
            self.reason_count += 1
            delta = self.delta_p  # Reasoning is deterministic
        
        else:
            delta = 0.0
        
        self.current_progress = min(self.U_max, self.current_progress + delta)
        return self.current_progress
    
    def get_statistics(self) -> Dict:
        """Return tracking statistics."""
        return {
            'current_progress': self.current_progress,
            'step_count': self.step_count,
            'retrieve_count': self.retrieve_count,
            'reason_count': self.reason_count,
            'successful_retrieves': self.successful_retrieves,
            'empirical_p_s': self.successful_retrieves / max(1, self.retrieve_count),
            'delta_r': self.delta_r,
            'delta_p': self.delta_p,
            'U_max': self.U_max
        }
    
    def reset(self) -> None:
        """Reset tracker state."""
        self.current_progress = 0.0
        self.step_count = 0
        self.retrieve_count = 0
        self.reason_count = 0
        self.successful_retrieves = 0


class HybridProgressTracker:
    """
    Hybrid tracker that combines stationary MDP dynamics with optional
    content-based confidence estimation.
    
    This tracker:
    1. Uses FIXED gains for progress updates (preserving Assumption 1)
    2. Optionally computes content-based confidence for early termination heuristics
    
    The confidence signal does NOT affect progress updates, but can be used
    by the policy for additional termination conditions.
    """
    
    TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
    
    def __init__(
        self,
        question: str,
        delta_r: float = 0.25,
        delta_p: float = 0.08,
        U_max: float = 1.0,
        track_confidence: bool = True
    ) -> None:
        """
        Initialize hybrid tracker.
        
        Args:
            question: The question being answered
            delta_r: Fixed retrieval gain
            delta_p: Fixed reasoning gain
            U_max: Maximum progress
            track_confidence: Whether to compute content-based confidence
        """
        self.question = question
        self.question_tokens = self._extract_tokens(question)
        self.delta_r = delta_r
        self.delta_p = delta_p
        self.U_max = U_max
        self.track_confidence = track_confidence
        
        # Progress state (stationary)
        self.current_progress = 0.0
        
        # Confidence tracking (optional, does not affect progress)
        self.covered_tokens: set = set()
        self.total_tokens_seen = 0
        self.current_confidence = 0.0
        
        # Statistics
        self.step_count = 0
    
    def update(self, action: str, step_data: Dict) -> float:
        """
        Update progress using FIXED gains.
        
        Confidence is updated separately and does not affect progress.
        
        Returns:
            Updated progress value (based on fixed gains only)
        """
        self.step_count += 1
        
        # === STATIONARY PROGRESS UPDATE (Preserves Assumption 1) ===
        if action == 'retrieve':
            if step_data.get('retrieval_success', True):
                delta = self.delta_r
            else:
                delta = 0.0
        elif action == 'reason':
            delta = self.delta_p
        else:
            delta = 0.0
        
        self.current_progress = min(self.U_max, self.current_progress + delta)
        
        # === OPTIONAL CONFIDENCE TRACKING (Separate from progress) ===
        if self.track_confidence:
            self._update_confidence(step_data)
        
        return self.current_progress
    
    def _update_confidence(self, step_data: Dict) -> None:
        """Update content-based confidence estimate (does not affect progress)."""
        text_segments = []
        
        answer = step_data.get('intermediate_answer')
        if answer:
            text_segments.append(answer)
        
        docs = step_data.get('retrieved_docs') or []
        text_segments.extend(docs)
        
        combined_text = " ".join(text_segments)
        tokens = set(self._extract_tokens(combined_text))
        
        # Track coverage
        newly_covered = tokens & set(self.question_tokens) - self.covered_tokens
        self.covered_tokens.update(newly_covered)
        self.total_tokens_seen += len(tokens)
        
        # Compute confidence
        if self.question_tokens:
            coverage = len(self.covered_tokens) / len(self.question_tokens)
        else:
            coverage = 0.5
        
        # Blend with step_data confidence if available
        step_confidence = step_data.get('confidence', 0.5)
        self.current_confidence = 0.6 * coverage + 0.4 * step_confidence
    
    def get_confidence(self) -> float:
        """Get current confidence estimate (for early termination heuristics)."""
        return self.current_confidence
    
    def should_early_terminate(self, threshold: float = 0.85) -> bool:
        """
        Suggest early termination based on high confidence.
        
        This is a HEURISTIC that can be used alongside the MDP policy,
        but does not replace the MDP threshold comparison.
        """
        return self.current_confidence >= threshold
    
    @classmethod
    def _extract_tokens(cls, text: str) -> list:
        """Extract tokens from text."""
        raw_tokens = cls.TOKEN_PATTERN.findall(text.lower())
        return [t for t in raw_tokens if len(t) >= 3]
