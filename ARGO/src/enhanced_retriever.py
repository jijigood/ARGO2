"""
Enhanced 4-Stage Retrieval Pipeline for ARGO.

Adds query rewriting, dynamic routing, hybrid retrieval with Reciprocal Rank
Fusion (RRF), and optional cross-encoder reranking on top of the baseline
`Retriever`. Each stage is designed to reduce redundant LLM calls by improving
recall/precision per retrieval step.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np

try:  # Optional heavy dependencies
    import torch
except ImportError:  # pragma: no cover - torch should exist but guard anyway
    torch = None

try:  # pragma: no cover - rank_bm25 may be optional at runtime
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    BM25Okapi = None

try:  # pragma: no cover - reranker is optional
    from sentence_transformers import CrossEncoder
except ImportError:  # pragma: no cover
    CrossEncoder = None

from .retriever import Retriever

logger = logging.getLogger(__name__)


class QueryRewriter:
    """Stage 1: Rule-based + optional LLM query rewriting."""

    ORAN_ACRONYMS: Dict[str, str] = {
        'ric': 'RAN Intelligent Controller',
        'o-du': 'O-RAN Distributed Unit',
        'o-cu': 'O-RAN Central Unit',
        'o-ru': 'O-RAN Radio Unit',
        'smo': 'Service Management and Orchestration',
        'xapp': 'Near-RT RIC Application',
        'rapp': 'Non-RT RIC Application',
        'e2': 'E2 interface Near-RT RIC',
        'a1': 'A1 interface Non-RT RIC',
        'o1': 'O1 management interface',
        'f1': 'F1 interface CU-DU',
        'kpm': 'Key Performance Monitoring',
        'rc': 'RAN Control',
        '7-2x': 'functional split option 7-2x fronthaul',
        'cpri': 'Common Public Radio Interface',
        'ecpri': 'enhanced Common Public Radio Interface',
    }

    def __init__(
        self,
        model=None,
        tokenizer=None,
        max_length: int = 128,
        enable_llm: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.enable_llm = enable_llm and model is not None and tokenizer is not None
        self.device = None

        if self.enable_llm:
            if torch is None:
                raise ImportError("torch is required for LLM-based rewriting.")
            self.device = next(model.parameters()).device

    def rewrite(self, query: str) -> str:
        expanded = self._expand_acronyms(query)

        if not self.enable_llm:
            return expanded

        rewritten = self._llm_rewrite(expanded)
        return rewritten if rewritten else expanded

    def _expand_acronyms(self, query: str) -> str:
        expanded = query.lower()
        for acronym, full_form in self.ORAN_ACRONYMS.items():
            pattern = r"\b" + re.escape(acronym) + r"\b"
            if re.search(pattern, expanded):
                replacement = f"{acronym} ({full_form})"
                expanded = re.sub(pattern, replacement, expanded, count=1)

        if 'o-ran' not in expanded and 'oran' not in expanded:
            expanded = f"O-RAN {expanded}"

        logger.debug("Query rewritten (rule-based): %s -> %s", query, expanded)
        return expanded

    def _llm_rewrite(self, query: str) -> str:
        prompt = (
            "Rewrite this O-RAN technical query to include specific interfaces, "
            "protocols, and architectural terminology when relevant.\n"
            f"Original query: {query}\n\nRewritten query:"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():  # pragma: no cover - heavy model inference
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_length,
                temperature=0.3,
                top_p=0.9,
                do_sample=False,
            )

        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        rewritten = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return rewritten.strip()


class RetrievalSupervisor:
    """Stage 2: Heuristic router that selects keyword/vector/hybrid search."""

    def route(self, query: str) -> str:
        query_lower = query.lower()

        keyword_patterns = [
            r'section\s+\d+',
            r'\d+\.\d+\.\d+',
            r'\d+\s*khz',
            r'\d+\s*mhz',
            r'\d+\s*ms',
            r'\d+\s*μs',
            r'table\s+\d+',
            r'figure\s+\d+',
            r'clause\s+\d+',
        ]
        keyword_score = sum(1 for pattern in keyword_patterns if re.search(pattern, query_lower))

        vector_keywords = [
            'explain', 'describe', 'what is', 'how does', 'architecture',
            'principle', 'concept', 'overview', 'compare',
            'difference between', 'relationship'
        ]
        vector_score = sum(1 for kw in vector_keywords if kw in query_lower)

        if keyword_score >= 2:
            strategy = 'keyword'
        elif vector_score >= 1 and keyword_score == 0:
            strategy = 'vector'
        else:
            strategy = 'hybrid'

        logger.debug(
            "Routing query using strategy=%s (keyword_score=%d, vector_score=%d)",
            strategy,
            keyword_score,
            vector_score,
        )
        return strategy


class EnhancedRetriever(Retriever):
    """4-stage retrieval pipeline with optional reranking."""

    def __init__(
        self,
        model=None,
        tokenizer=None,
        chroma_dir: str = "Environments/chroma_store",
        collection_name: str = "oran_specs",
        use_pipeline: bool = True,
        use_reranker: bool = True,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L12-v2",
        cross_encoder_local_dir: Optional[str] = "/data/user/huangxiaolin/ARGO/models",
        **kwargs,
    ) -> None:
        super().__init__(
            chroma_dir=chroma_dir,
            collection_name=collection_name,
            **kwargs,
        )

        self.use_pipeline = use_pipeline
        self.use_reranker = use_reranker

        if not use_pipeline:
            logger.info("Enhanced pipeline disabled; falling back to baseline Retriever")
            return

        self.query_rewriter = QueryRewriter(model, tokenizer)
        self.supervisor = RetrievalSupervisor()

        self.bm25 = None
        self.bm25_docs: List[str] = []
        self._build_bm25_index()

        self.cross_encoder = None
        if use_reranker:
            if CrossEncoder is None:
                logger.warning("sentence-transformers not installed; disabling reranker")
                self.use_reranker = False
            else:
                try:
                    model_location = cross_encoder_model
                    if cross_encoder_local_dir:
                        local_path = Path(cross_encoder_local_dir).expanduser()
                        if local_path.exists():
                            model_location = str(local_path)
                        else:
                            logger.warning(
                                "Cross-encoder local dir %s not found; falling back to HF repo",
                                cross_encoder_local_dir,
                            )

                    self.cross_encoder = CrossEncoder(model_location)
                    logger.info("Loaded cross-encoder model from %s", model_location)
                except Exception as exc:  # pragma: no cover - external resource
                    logger.warning("Failed to load cross-encoder (%s); disabling reranker", exc)
                    self.use_reranker = False

        logger.info("EnhancedRetriever initialized (pipeline=%s, reranker=%s)", use_pipeline, self.use_reranker)

    def _build_bm25_index(self) -> None:
        if BM25Okapi is None:
            logger.warning("rank-bm25 not installed; keyword routing disabled")
            return

        try:
            all_docs = self.collection.get(include=['documents', 'metadatas'])
            documents = all_docs.get('documents', [])
            metadatas = all_docs.get('metadatas', [{}] * len(documents))
            if not documents:
                logger.warning("No documents found for BM25 index build")
                return

            tokenized_docs = [doc.lower().split() for doc in documents]
            self.bm25 = BM25Okapi(tokenized_docs)

            self.bm25_docs = []
            for doc, meta in zip(documents, metadatas):
                meta = meta or {}
                source = meta.get('source') or meta.get('doc_id') or 'unknown'
                self.bm25_docs.append(f"[Source: {source}] {doc}")

            logger.info("BM25 index built with %d documents", len(documents))
        except Exception as exc:  # pragma: no cover - chroma I/O
            logger.warning("Failed to build BM25 index: %s", exc)
            self.bm25 = None

    def retrieve(
        self,
        query: str,
        k: int = 3,
        return_scores: bool = False,
    ) -> Tuple[List[str], bool, Optional[List[float]]]:
        if not self.use_pipeline:
            return super().retrieve(query, k, return_scores)

        rewritten_query = self.query_rewriter.rewrite(query)
        strategy = self.supervisor.route(rewritten_query)
        candidates_k = max(k * 3, k)

        if strategy == 'keyword' and self.bm25 is not None:
            candidates, scores = self._keyword_search(rewritten_query, candidates_k)
        elif strategy == 'vector':
            candidates, scores = self._vector_search(rewritten_query, candidates_k)
        else:
            candidates, scores = self._hybrid_search_rrf(rewritten_query, candidates_k)

        if self.use_reranker and candidates:
            candidates, scores = self._rerank(rewritten_query, candidates, k)
        else:
            candidates = candidates[:k]
            scores = scores[:k] if scores else None

        score_list = scores if scores is not None else [0.5] * len(candidates)
        success = self._check_success(score_list)

        if return_scores:
            return candidates, success, scores
        return candidates, success, None

    def _keyword_search(self, query: str, k: int) -> Tuple[List[str], List[float]]:
        if self.bm25 is None:
            return self._vector_search(query, k)

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_idx = np.argsort(scores)[::-1][:k]

        documents = [self.bm25_docs[i] for i in top_idx]
        doc_scores = [float(scores[i]) for i in top_idx]
        max_score = max(doc_scores) if doc_scores else 1.0
        if max_score > 0:
            doc_scores = [s / max_score for s in doc_scores]
        return documents, doc_scores

    def _vector_search(self, query: str, k: int) -> Tuple[List[str], List[float]]:
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=['documents', 'distances', 'metadatas'],
        )
        documents = results['documents'][0] if results['documents'] else []
        distances = results['distances'][0] if results['distances'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []

        formatted_docs: List[str] = []
        for doc, meta in zip(documents, metadatas):
            meta = meta or {}
            source = meta.get('source') or meta.get('doc_id') or 'unknown'
            formatted_docs.append(f"[Source: {source}] {doc}")

        similarities = [1.0 / (1.0 + d) for d in distances]
        return formatted_docs, similarities

    def _hybrid_search_rrf(self, query: str, k: int) -> Tuple[List[str], List[float]]:
        bm25_docs, bm25_scores = self._keyword_search(query, k * 2)
        vector_docs, vector_scores = self._vector_search(query, k * 2)

        doc_scores: Dict[str, float] = {}
        rrf_k = 60

        for rank, doc in enumerate(bm25_docs):
            score = 1.0 / (rrf_k + rank + 1)
            doc_scores[doc] = doc_scores.get(doc, 0.0) + score

        for rank, doc in enumerate(vector_docs):
            score = 1.0 / (rrf_k + rank + 1)
            doc_scores[doc] = doc_scores.get(doc, 0.0) + score

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        docs = [doc for doc, _ in sorted_docs[:k]]
        scores = [score for _, score in sorted_docs[:k]]

        max_score = max(scores) if scores else 1.0
        if max_score > 0:
            scores = [s / max_score for s in scores]
        return docs, scores

    def _rerank(self, query: str, candidates: List[str], k: int) -> Tuple[List[str], List[float]]:
        if not self.use_reranker or not candidates or self.cross_encoder is None:
            truncated = candidates[:k]
            return truncated, [1.0] * len(truncated)

        pairs = [[query, doc] for doc in candidates]
        try:
            scores = self.cross_encoder.predict(pairs)
        except Exception as exc:  # pragma: no cover - external model
            logger.warning("Cross-encoder failed (%s); skipping rerank", exc)
            truncated = candidates[:k]
            return truncated, [1.0] * len(truncated)

        sorted_idx = np.argsort(scores)[::-1][:k]
        reranked_docs = [candidates[i] for i in sorted_idx]
        reranked_scores = [float(scores[i]) for i in sorted_idx]

        min_score = min(reranked_scores) if reranked_scores else 0.0
        max_score = max(reranked_scores) if reranked_scores else 1.0
        if max_score > min_score:
            reranked_scores = [
                (score - min_score) / (max_score - min_score)
                for score in reranked_scores
            ]
        return reranked_docs, reranked_scores
