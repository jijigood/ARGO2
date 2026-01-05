"""
Improved Question Complexity Classifier for O-RAN Domain

Uses domain-specific patterns without LLM overhead.
Maps to discrete U_max buckets that align with pre-computed thresholds.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ComplexityProfile:
    """Structured complexity assessment."""
    label: str  # 'simple', 'medium', 'complex'
    umax: float  # Estimated U_max
    confidence: float  # Confidence in estimate
    features: Dict  # Extracted features for debugging


class ORANComplexityClassifier:
    """
    O-RAN domain-aware complexity classifier.
    
    Design Principles:
    1. Lightweight (no LLM) - aligns with paper's "lightweight profiler"
    2. Domain-specific - uses O-RAN terminology patterns
    3. Discrete outputs - maps to pre-computed threshold buckets
    4. Transparent - returns interpretable features
    """
    
    # Question type patterns with expected U_max
    QUESTION_PATTERNS = {
        'definitional': {
            'patterns': [
                r'^what\s+is\s+(?:the\s+)?(?:an?\s+)?',
                r'^define\s+',
                r'^what\s+does\s+\w+\s+stand\s+for',
                r'^what\s+is\s+meant\s+by',
                r'(?:acronym|abbreviation)\s+for',
            ],
            'umax': 0.35,
            'label': 'simple',
            'confidence': 0.9
        },
        'binary': {
            'patterns': [
                r'^(?:is|does|can|will|should|are)\s+',
                r'^(?:true|false)[:\s]',
                r'\?$.*(?:yes|no)',
            ],
            'umax': 0.40,
            'label': 'simple',
            'confidence': 0.85
        },
        'factual': {
            'patterns': [
                r'^(?:what|which)\s+(?:is|are)\s+the\s+(?:main|primary|key)',
                r'^how\s+many\s+',
                r'^what\s+(?:protocol|interface|layer)',
                r'^which\s+(?:component|module|entity)',
            ],
            'umax': 0.45,
            'label': 'simple',
            'confidence': 0.85
        },
        'procedural': {
            'patterns': [
                r'^how\s+(?:does|do|to)\s+',
                r'^what\s+(?:happens|occurs)\s+when',
                r'^describe\s+(?:the\s+)?(?:process|procedure|steps)',
                r'(?:sequence|flow|workflow)\s+of',
            ],
            'umax': 0.65,
            'label': 'medium',
            'confidence': 0.8
        },
        'explanatory': {
            'patterns': [
                r'^explain\s+',
                r'^why\s+(?:does|do|is|are)',
                r'^what\s+(?:causes|enables|prevents)',
                r'(?:purpose|role|function)\s+of',
            ],
            'umax': 0.70,
            'label': 'medium',
            'confidence': 0.75
        },
        'comparative': {
            'patterns': [
                r'(?:compare|contrast|difference|distinguish)',
                r'(?:versus|vs\.?)\s+',
                r'(?:between|among)\s+\w+\s+and\s+',
                r'(?:advantages?|disadvantages?)\s+of',
            ],
            'umax': 0.80,
            'label': 'complex',
            'confidence': 0.85
        },
        'analytical': {
            'patterns': [
                r'^analyze\s+',
                r'(?:trade-?off|tradeoff)s?\s+',
                r'(?:impact|effect|influence)\s+(?:of|on)',
                r'(?:relationship|interaction)\s+between',
                r'(?:implications?|consequences?)\s+of',
            ],
            'umax': 0.85,
            'label': 'complex',
            'confidence': 0.8
        },
        'multi_hop': {
            'patterns': [
                r'(?:and|also|additionally)\s+(?:how|what|why)',
                r'\?\s*(?:and|also)\s*(?:how|what|why)',
                r'(?:first|then|finally|subsequently)',
                r'(?:if|when|assuming)\s+.*(?:what|how)',
            ],
            'umax': 0.90,
            'label': 'complex',
            'confidence': 0.75
        }
    }
    
    # O-RAN technical depth indicators
    TECHNICAL_TERMS = {
        'high_complexity': [
            'architecture', 'deployment', 'optimization', 'algorithm',
            'latency budget', 'synchronization', 'interference',
            'handover', 'load balancing', 'slice', 'policy',
            'xapp', 'rapp', 'ric', 'near-rt', 'non-rt',
            'e2', 'o1', 'a1', 'o2', 'open fronthaul'
        ],
        'medium_complexity': [
            'interface', 'protocol', 'message', 'procedure',
            'configuration', 'parameter', 'function', 'service',
            'cu', 'du', 'ru', 'smo', 'o-cloud'
        ],
        'low_complexity': [
            'definition', 'meaning', 'acronym', 'name', 'type', 'layer',
            'abbreviation', 'stands for'
        ]
    }
    
    # Multi-part question indicators
    MULTI_PART_PATTERNS = [
        r'\d+\)',  # 1) 2) 3)
        r'[a-d]\)',  # a) b) c)
        r'\([ivx]+\)',  # (i) (ii) (iii)
        r'(?:first|second|third|finally)',
        r';\s*(?:and|also)',
    ]
    
    # Default U_max buckets (should match ThresholdTable)
    DEFAULT_UMAX_BUCKETS = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    
    def __init__(
        self,
        umax_buckets: Optional[List[float]] = None,
        config: Optional[Dict] = None
    ):
        """
        Args:
            umax_buckets: Discrete U_max values to map to (must match ThresholdTable)
            config: Optional configuration overrides
        """
        self.umax_buckets = umax_buckets or self.DEFAULT_UMAX_BUCKETS
        self.config = config or {}
        
        # Compile patterns for efficiency
        self._compiled_patterns = {}
        for qtype, info in self.QUESTION_PATTERNS.items():
            self._compiled_patterns[qtype] = [
                re.compile(p, re.IGNORECASE) for p in info['patterns']
            ]
        
        self._multi_part_compiled = [
            re.compile(p, re.IGNORECASE) for p in self.MULTI_PART_PATTERNS
        ]
        
        logger.debug(f"ORANComplexityClassifier initialized with buckets: {self.umax_buckets}")
    
    def classify(self, question: str, metadata: Optional[Dict] = None) -> ComplexityProfile:
        """
        Classify question complexity.
        
        Args:
            question: Input question text
            metadata: Optional metadata (e.g., difficulty hints)
            
        Returns:
            ComplexityProfile with label, umax, confidence, and features
        """
        question = (question or '').strip()
        lower_q = question.lower()
        
        # Extract features
        features = self._extract_features(question, lower_q)
        
        # Determine base complexity from question type
        qtype, qtype_conf = self._detect_question_type(lower_q)
        base_info = self.QUESTION_PATTERNS.get(qtype, {
            'umax': 0.65,
            'label': 'medium',
            'confidence': 0.5
        })
        
        # Adjust based on features
        umax = base_info['umax']
        label = base_info['label']
        confidence = base_info['confidence'] * qtype_conf
        
        # Technical depth adjustment
        umax += features['tech_depth_score'] * 0.1
        
        # Multi-part adjustment
        if features['multi_part_count'] > 0:
            umax += 0.05 * features['multi_part_count']
            if label == 'simple':
                label = 'medium'
        
        # Length-based adjustment
        if features['word_count'] > 30:
            umax += 0.05
        elif features['word_count'] < 10:
            umax -= 0.05
        
        # Use difficulty hint if available
        if metadata and 'difficulty' in metadata:
            hint = str(metadata['difficulty']).lower()
            hint_umax = self._difficulty_to_umax(hint)
            if hint_umax is not None:
                # Blend with hint (60% pattern, 40% hint)
                umax = 0.6 * umax + 0.4 * hint_umax
        
        # Clamp and bucket
        umax = max(0.30, min(0.95, umax))
        umax = self._map_to_bucket(umax)
        
        # Update label based on final umax
        if umax <= 0.45:
            label = 'simple'
        elif umax <= 0.70:
            label = 'medium'
        else:
            label = 'complex'
        
        features['detected_type'] = qtype
        features['raw_umax'] = base_info['umax']
        
        return ComplexityProfile(
            label=label,
            umax=umax,
            confidence=confidence,
            features=features
        )
    
    def estimate_umax(self, question: str, metadata: Optional[Dict] = None) -> float:
        """
        Compatibility method: returns U_max directly.
        
        For backward compatibility with existing code that calls estimate_umax().
        """
        profile = self.classify(question, metadata)
        return profile.umax
    
    def _extract_features(self, question: str, lower_q: str) -> Dict:
        """Extract complexity-relevant features."""
        features = {}
        
        # Basic stats
        features['word_count'] = len(question.split())
        features['char_count'] = len(question)
        features['question_marks'] = question.count('?')
        
        # Technical depth
        high_count = sum(1 for t in self.TECHNICAL_TERMS['high_complexity'] if t in lower_q)
        med_count = sum(1 for t in self.TECHNICAL_TERMS['medium_complexity'] if t in lower_q)
        low_count = sum(1 for t in self.TECHNICAL_TERMS['low_complexity'] if t in lower_q)
        
        features['high_tech_terms'] = high_count
        features['med_tech_terms'] = med_count
        features['low_tech_terms'] = low_count
        features['tech_depth_score'] = (high_count * 2 + med_count - low_count) / 5.0
        features['tech_depth_score'] = max(-0.3, min(0.3, features['tech_depth_score']))
        
        # Multi-part detection
        multi_count = sum(
            1 for pattern in self._multi_part_compiled
            if pattern.search(question)
        )
        features['multi_part_count'] = multi_count
        
        # Connectors
        connectors = len(re.findall(
            r'\b(and|also|furthermore|additionally|moreover)\b',
            lower_q
        ))
        features['connector_count'] = connectors
        
        return features
    
    def _detect_question_type(self, lower_q: str) -> Tuple[str, float]:
        """Detect question type from patterns."""
        best_type = 'explanatory'  # default
        best_confidence = 0.5
        
        for qtype, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(lower_q):
                    info = self.QUESTION_PATTERNS[qtype]
                    if info['confidence'] > best_confidence:
                        best_type = qtype
                        best_confidence = info['confidence']
        
        return best_type, best_confidence
    
    def _map_to_bucket(self, umax: float) -> float:
        """Map continuous umax to nearest discrete bucket."""
        return min(self.umax_buckets, key=lambda x: abs(x - umax))
    
    def _difficulty_to_umax(self, hint: str) -> Optional[float]:
        """Convert difficulty hint to U_max estimate."""
        mapping = {
            'easy': 0.40,
            'e': 0.40,
            'simple': 0.40,
            'medium': 0.65,
            'm': 0.65,
            'hard': 0.85,
            'h': 0.85,
            'complex': 0.85,
        }
        return mapping.get(hint)
    
    def get_adaptive_max_steps(
        self,
        question: str,
        base_max_steps: int = 10,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Compute max steps based on complexity.
        
        Scaling:
        - simple: 50% of base
        - medium: 80% of base
        - complex: 100-120% of base
        """
        profile = self.classify(question, metadata)
        
        scale = {
            'simple': 0.5,
            'medium': 0.8,
            'complex': 1.2
        }
        
        factor = scale.get(profile.label, 1.0)
        steps = int(base_max_steps * factor)
        return max(3, min(steps, base_max_steps + 3))
    
    def get_full_profile(
        self,
        question: str,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Get complete classification with all derived values.
        
        Useful for debugging and logging.
        """
        profile = self.classify(question, metadata)
        
        return {
            'label': profile.label,
            'umax': profile.umax,
            'confidence': profile.confidence,
            'features': profile.features,
            'max_steps_10': self.get_adaptive_max_steps(question, 10, metadata),
            'max_steps_15': self.get_adaptive_max_steps(question, 15, metadata),
        }


# Backward compatibility alias
QuestionComplexityClassifierV2 = ORANComplexityClassifier
