"""
ARGO V3.2 - Theory-Aligned MDP-Guided RAG System
=================================================

Components:
1. QueryDecomposer: LLM-based dynamic subquery generation
2. Retriever: Chroma vector database integration
3. Reasoner: LLM-based reasoning with MDP guidance
4. Synthesizer: Final answer synthesis from history

Progress Trackers (Problem 2 Solutions):
- FixedProgressTracker: Fixed gains, implements Eq(2) exactly (RECOMMENDED)
- BoundedConfidenceTracker: Fixed gains + bounded confidence scaling
- StationaryProgressTracker: Fixed gains (legacy alias)
- HybridProgressTracker: Fixed gains + confidence tracking
- ProgressTracker: Dynamic (legacy, violates Assumption 1)

Threshold Management (Problem 1 Solution):
- ThresholdTable: Pre-computed optimal thresholds for question-adaptive U_max

Complexity Classification (Problem 3 Solution):
- ORANComplexityClassifier: Domain-aware classifier with discrete U_max buckets (V2)
- QuestionComplexityClassifier: Generic heuristic classifier (V1, legacy)

Strategies:
- ARGO_System: MDP-guided strategy
- AlwaysReasonStrategy: Never retrieve baseline
- RandomStrategy: Random action baseline
- FixedThresholdStrategy: Fixed threshold baseline

Author: ARGO Team
Version: 3.3
"""

from .decomposer import QueryDecomposer
from .retriever import Retriever, MockRetriever
from .synthesizer import AnswerSynthesizer
from .argo_system import ARGO_System

# Progress trackers
from .fixed_progress import FixedProgressTracker, BoundedConfidenceTracker
from .progress import (
    ProgressTracker,
    FastProgressTracker,
    StationaryProgressTracker,
    HybridProgressTracker
)

# Threshold management
from .threshold_table import ThresholdTable, precompute_threshold_grid

# Complexity classifiers
from .complexity import QuestionComplexityClassifier
from .complexity_v2 import ORANComplexityClassifier, ComplexityProfile

# Baseline strategies
from .baseline_strategies import (
    AlwaysReasonStrategy,
    RandomStrategy,
    FixedThresholdStrategy
)

__all__ = [
    # Core components
    'QueryDecomposer',
    'Retriever',
    'MockRetriever',
    'AnswerSynthesizer',
    'ARGO_System',
    
    # Progress trackers (theory-aligned)
    'FixedProgressTracker',       # RECOMMENDED: Eq(2) exact
    'BoundedConfidenceTracker',   # Practical: bounded scaling
    
    # Progress trackers (legacy)
    'StationaryProgressTracker',  # Alias for fixed
    'HybridProgressTracker',      # Legacy hybrid
    'ProgressTracker',            # Legacy dynamic
    'FastProgressTracker',        # Legacy fast
    
    # Threshold management
    'ThresholdTable',
    'precompute_threshold_grid',
    
    # Complexity classifiers
    'ORANComplexityClassifier',       # RECOMMENDED: V2 domain-aware
    'ComplexityProfile',              # Structured output
    'QuestionComplexityClassifier',   # V1 legacy
    
    # Baseline strategies
    'AlwaysReasonStrategy',
    'RandomStrategy',
    'FixedThresholdStrategy'
]
