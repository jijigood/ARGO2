"""
Fixed-Gain Progress Tracker

Implements Equation (2) from the paper exactly:
- U_{t+1} = min(U_t + δ_r, U_max)  if a_t=0 and retrieval succeeds
- U_{t+1} = U_t                     if a_t=0 and retrieval fails  
- U_{t+1} = min(U_t + δ_p, U_max)  if a_t=1 (reason)
- U_{t+1} = U_t                     if a_t=2 (terminate)

This preserves Assumption 1: U_t is a sufficient statistic for decision-making.

Solves Problem 2: Dynamic Progress Tracking Violates Assumption 1
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class FixedProgressTracker:
    """
    Theory-aligned progress tracker with fixed gains.
    
    Unlike the dynamic tracker, this does NOT consider:
    - Coverage of question tokens
    - Novelty of retrieved information  
    - Confidence scores
    
    This ensures E[R_t | H_t, U_t, a_t] = E[R_t | U_t, a_t]
    (Assumption 1 from the paper)
    """
    
    def __init__(
        self,
        delta_r: float,
        delta_p: float,
        u_max: float = 1.0,
        initial_progress: float = 0.0
    ):
        """
        Args:
            delta_r: Retrieval gain (applied on success only)
            delta_p: Reasoning gain (deterministic)
            u_max: Maximum progress for this question
            initial_progress: Starting progress (default 0)
            
        Raises:
            ValueError: If gains are not positive or u_max invalid
        """
        if delta_r <= 0 or delta_p <= 0:
            raise ValueError(f"Gains must be positive: delta_r={delta_r}, delta_p={delta_p}")
        if not (0 < u_max <= 1.0):
            raise ValueError(f"u_max must be in (0, 1]: got {u_max}")
        
        self.delta_r = delta_r
        self.delta_p = delta_p
        self.u_max = u_max
        self.current_progress = initial_progress
        
        # Statistics
        self.retrieve_attempts = 0
        self.retrieve_successes = 0
        self.reason_steps = 0
        self.history: List[Dict] = []
    
    def update(self, action: str, step_data: Dict) -> float:
        """
        Update progress using fixed gains.
        
        Implements Equation (2) exactly:
        - Retrieve success: U_{t+1} = min(U_t + δ_r, U_max)
        - Retrieve failure: U_{t+1} = U_t
        - Reason: U_{t+1} = min(U_t + δ_p, U_max)
        
        Args:
            action: 'retrieve' or 'reason'
            step_data: Must contain 'retrieval_success' for retrieve actions
            
        Returns:
            Updated progress value
        """
        old_progress = self.current_progress
        gain = 0.0
        success = True
        
        if action == 'retrieve':
            self.retrieve_attempts += 1
            success = step_data.get('retrieval_success', False)
            
            if success:
                self.retrieve_successes += 1
                gain = self.delta_r
                self.current_progress = min(
                    self.current_progress + gain,
                    self.u_max
                )
                logger.debug(
                    f"Retrieve SUCCESS: U_t += δ_r ({self.delta_r:.4f}) → {self.current_progress:.4f}"
                )
            else:
                logger.debug(
                    f"Retrieve FAILED: U_t unchanged at {self.current_progress:.4f}"
                )
        
        elif action == 'reason':
            self.reason_steps += 1
            gain = self.delta_p
            self.current_progress = min(
                self.current_progress + gain,
                self.u_max
            )
            logger.debug(
                f"Reason: U_t += δ_p ({self.delta_p:.4f}) → {self.current_progress:.4f}"
            )
        
        elif action == 'terminate':
            logger.debug(f"Terminate: U_t unchanged at {self.current_progress:.4f}")
        
        else:
            logger.warning(f"Unknown action: {action}")
        
        # Record transition for debugging/analysis
        self.history.append({
            'action': action,
            'success': success,
            'progress_before': old_progress,
            'progress_after': self.current_progress,
            'gain': self.current_progress - old_progress
        })
        
        return self.current_progress
    
    def get_statistics(self) -> Dict:
        """Return tracker statistics for metadata."""
        empirical_ps = (
            self.retrieve_successes / self.retrieve_attempts
            if self.retrieve_attempts > 0 else 0.0
        )
        
        return {
            'current_progress': self.current_progress,
            'u_max': self.u_max,
            'delta_r': self.delta_r,
            'delta_p': self.delta_p,
            'retrieve_attempts': self.retrieve_attempts,
            'retrieve_successes': self.retrieve_successes,
            'reason_steps': self.reason_steps,
            'empirical_p_s': empirical_ps,
            'total_steps': self.retrieve_attempts + self.reason_steps
        }
    
    def reset(self, u_max: Optional[float] = None) -> None:
        """Reset tracker for new question."""
        self.current_progress = 0.0
        self.retrieve_attempts = 0
        self.retrieve_successes = 0
        self.reason_steps = 0
        self.history = []
        if u_max is not None:
            if not (0 < u_max <= 1.0):
                raise ValueError(f"u_max must be in (0, 1]: got {u_max}")
            self.u_max = u_max
    
    def get_history(self) -> List[Dict]:
        """Return full transition history."""
        return self.history.copy()


class BoundedConfidenceTracker(FixedProgressTracker):
    """
    Fixed gains with bounded confidence scaling (practical compromise).
    
    Gain formula:
        δ_effective = δ_base × (1 + scale × (confidence - 0.5))
    
    With scale=0.3 and confidence in [0, 1]:
        - confidence=0.0: multiplier = 0.85
        - confidence=0.5: multiplier = 1.00  
        - confidence=1.0: multiplier = 1.15
    
    This keeps gains within [0.85×δ, 1.15×δ] range, providing
    a bounded deviation from theory while potentially improving
    empirical performance.
    
    Note: This APPROXIMATELY satisfies Assumption 1 but is NOT exact.
    """
    
    def __init__(
        self,
        delta_r: float,
        delta_p: float,
        u_max: float = 1.0,
        confidence_scale: float = 0.3,
        initial_progress: float = 0.0
    ):
        """
        Args:
            delta_r: Base retrieval gain
            delta_p: Base reasoning gain
            u_max: Maximum progress
            confidence_scale: How much confidence affects gain (default ±15%)
            initial_progress: Starting progress
        """
        super().__init__(delta_r, delta_p, u_max, initial_progress)
        self.confidence_scale = max(0.0, min(0.5, confidence_scale))  # Cap at ±25%
    
    def update(self, action: str, step_data: Dict) -> float:
        """Update with bounded confidence adjustment."""
        old_progress = self.current_progress
        
        # Get confidence (default 0.5 = neutral)
        confidence = step_data.get('confidence', 0.5)
        confidence = max(0.0, min(1.0, confidence))
        
        # Bounded multiplier
        multiplier = 1.0 + self.confidence_scale * (confidence - 0.5)
        
        if action == 'retrieve':
            self.retrieve_attempts += 1
            success = step_data.get('retrieval_success', False)
            
            if success:
                self.retrieve_successes += 1
                effective_delta = self.delta_r * multiplier
                self.current_progress = min(
                    self.current_progress + effective_delta,
                    self.u_max
                )
        
        elif action == 'reason':
            self.reason_steps += 1
            effective_delta = self.delta_p * multiplier
            self.current_progress = min(
                self.current_progress + effective_delta,
                self.u_max
            )
        
        # Record transition
        self.history.append({
            'action': action,
            'success': step_data.get('retrieval_success', True) if action == 'retrieve' else True,
            'progress_before': old_progress,
            'progress_after': self.current_progress,
            'gain': self.current_progress - old_progress,
            'confidence': confidence,
            'multiplier': multiplier
        })
        
        return self.current_progress
