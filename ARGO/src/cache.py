"""
Semantic caching utilities for ARGO.

Provides a similarity-aware cache that prevents redundant LLM calls by storing
query/decomposition/retrieval outputs. The implementation trades sophistication
for speed, using lightweight embeddings while keeping a persistence hook for
long-running experiments.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SemanticCache:
    """Similarity-aware cache with TTL + optional persistence."""

    def __init__(
        self,
        cache_dir: str = "cache",
        similarity_threshold: float = 0.92,
        max_cache_size: int = 10_000,
        ttl_hours: int = 24,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.ttl = timedelta(hours=ttl_hours)

        self.cache_index: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.hits = 0
        self.misses = 0

        self._load_cache()
        logger.info("SemanticCache ready with %d entries", len(self.cache_index))

    def get(self, query: str, context: Optional[str] = None, use_semantic: bool = True) -> Optional[Any]:
        key = self._compute_key(query, context)
        entry = self.cache_index.get(key)

        if entry and not self._is_expired(entry['timestamp']):
            self.hits += 1
            logger.debug("Cache HIT (exact): %.50s", query)
            return entry['value']

        if entry:
            self._remove(key)

        if use_semantic:
            match_key = self._semantic_match(query)
            if match_key:
                entry = self.cache_index[match_key]
                if not self._is_expired(entry['timestamp']):
                    self.hits += 1
                    logger.debug("Cache HIT (semantic): %.50s", query)
                    return entry['value']
                self._remove(match_key)

        self.misses += 1
        return None

    def put(self, query: str, value: Any, context: Optional[str] = None) -> None:
        key = self._compute_key(query, context)

        if len(self.cache_index) >= self.max_cache_size:
            oldest_key = min(self.cache_index.keys(), key=lambda k: self.cache_index[k]['timestamp'])
            self._remove(oldest_key)

        entry = {
            'query': query,
            'value': value,
            'timestamp': datetime.now(),
        }
        self.cache_index[key] = entry
        self.embeddings[key] = self._compute_embedding(query)
        logger.debug("Cache PUT: %.50s", query)

    def get_stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        hit_rate = self.hits / total if total else 0.0
        return {
            'size': len(self.cache_index),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
        }

    def save_cache(self) -> None:
        cache_file = self.cache_dir / "cache.pkl"
        try:
            with open(cache_file, 'wb') as fh:
                pickle.dump({'index': self.cache_index, 'embeddings': self.embeddings}, fh)
            logger.info("Semantic cache persisted to %s", cache_file)
        except Exception as exc:  # pragma: no cover - filesystem errors
            logger.warning("Failed to persist semantic cache: %s", exc)

    def clear(self) -> None:
        self.cache_index.clear()
        self.embeddings.clear()
        self.hits = 0
        self.misses = 0
        logger.info("Semantic cache cleared")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_cache(self) -> None:
        cache_file = self.cache_dir / "cache.pkl"
        if not cache_file.exists():
            return
        try:
            with open(cache_file, 'rb') as fh:
                data = pickle.load(fh)
            self.cache_index = data.get('index', {})
            self.embeddings = data.get('embeddings', {})
            logger.info("Loaded semantic cache from %s", cache_file)
        except Exception as exc:  # pragma: no cover - filesystem errors
            logger.warning("Failed to load semantic cache: %s", exc)

    def _compute_key(self, query: str, context: Optional[str]) -> str:
        combined = query if context is None else f"{query}|{context}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()

    def _compute_embedding(self, text: str) -> np.ndarray:
        words = text.lower().split()
        vocab = list(dict.fromkeys(words))  # preserve order, drop duplicates
        vec = np.zeros(100, dtype=np.float32)
        for idx, word in enumerate(vocab[:100]):
            vec[idx] = (hash(word) % 1000) / 1000.0
        norm = np.linalg.norm(vec) or 1.0
        return vec / norm

    def _semantic_match(self, query: str) -> Optional[str]:
        if not self.embeddings:
            return None

        query_embedding = self._compute_embedding(query)
        best_key = None
        best_similarity = 0.0

        for key, emb in self.embeddings.items():
            similarity = float(np.dot(query_embedding, emb))
            if similarity > best_similarity:
                best_similarity = similarity
                best_key = key

        if best_key and best_similarity >= self.similarity_threshold:
            logger.debug("Semantic similarity %.3f for cache key %s", best_similarity, best_key)
            return best_key
        return None

    def _is_expired(self, timestamp: datetime) -> bool:
        return datetime.now() - timestamp > self.ttl

    def _remove(self, key: str) -> None:
        self.cache_index.pop(key, None)
        self.embeddings.pop(key, None)
