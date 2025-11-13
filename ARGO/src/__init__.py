"""
ARGO V3.0 - 4-Component Architecture
====================================

Components:
1. QueryDecomposer: LLM-based dynamic subquery generation
2. Retriever: Chroma vector database integration
3. Reasoner: LLM-based reasoning with MDP guidance
4. Synthesizer: Final answer synthesis from history

Strategies:
- ARGO_System: MDP-guided strategy
- AlwaysReasonStrategy: Never retrieve baseline
- RandomStrategy: Random action baseline
- FixedThresholdStrategy: Fixed threshold baseline

Author: ARGO Team
Version: 3.0
"""

from .decomposer import QueryDecomposer
from .retriever import Retriever, MockRetriever
from .synthesizer import AnswerSynthesizer
from .argo_system import ARGO_System
from .baseline_strategies import (
    AlwaysReasonStrategy,
    RandomStrategy,
    FixedThresholdStrategy
)

__all__ = [
    'QueryDecomposer',
    'Retriever',
    'MockRetriever',
    'AnswerSynthesizer',
    'ARGO_System',
    'AlwaysReasonStrategy',
    'RandomStrategy',
    'FixedThresholdStrategy'
]
