"""ChromaDB-backed vector store implementation for ARGO RAG."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
import numpy as np


class ChromaVectorStore:
    """Wrapper around a Chroma persistent collection."""

    def __init__(
        self,
        persist_directory: str | Path,
        collection_name: str = "oran_docs",
        embedding_dim: int = 384,
    ) -> None:
        self.persist_directory = Path(persist_directory).expanduser().resolve()
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self._client = chromadb.PersistentClient(path=str(self.persist_directory))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def has_data(self) -> bool:
        return self._collection.count() > 0

    def reset(self) -> None:
        try:
            self._client.delete_collection(self.collection_name)
        except ValueError:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        documents: List[Dict],
        embeddings: np.ndarray,
        batch_size: int = 4096,
    ) -> None:
        if len(documents) == 0:
            return

        total = len(documents)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_docs = documents[start:end]
            batch_vectors = embeddings[start:end]

            ids: List[str] = []
            metadatas: List[Dict] = []
            texts: List[str] = []
            vectors: List[List[float]] = []

            for doc, vector in zip(batch_docs, batch_vectors, strict=True):
                chunk_id = doc.get("chunk_id") or doc.get("doc_id")
                if not chunk_id:
                    raise ValueError("Each document must define a chunk_id or doc_id.")

                ids.append(str(chunk_id))
                metadatas.append(
                    {
                        "doc_id": doc.get("doc_id"),
                        "title": doc.get("title"),
                        "category": doc.get("category"),
                        "complexity": doc.get("complexity"),
                        "chunk_index": doc.get("chunk_index"),
                        "total_chunks": doc.get("total_chunks"),
                        "source_path": doc.get("source_path"),
                        "section_id": doc.get("section_id"),
                        "section_title": doc.get("section_title"),
                        "work_group": doc.get("work_group"),
                    }
                )
                texts.append(doc.get("content", ""))
                vectors.append(vector.tolist())

            self._collection.upsert(
                ids=ids,
                metadatas=metadatas,
                documents=texts,
                embeddings=vectors,
            )

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Dict, float]]:
        if query_embedding.ndim != 1:
            raise ValueError("query_embedding must be a 1-D vector")

        if not self.has_data():
            return []

        result = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["metadatas", "documents", "distances", "ids"],
        )

        docs: List[Tuple[Dict, float]] = []
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        contents = result.get("documents", [[]])[0]

        for idx, chunk_id in enumerate(ids):
            metadata = metadatas[idx] or {}
            distance = distances[idx] if distances else None
            similarity = 1.0 - distance if distance is not None else 0.0
            doc = {
                "doc_id": metadata.get("doc_id"),
                "chunk_id": chunk_id,
                "title": metadata.get("title"),
                "category": metadata.get("category", "unknown"),
                "complexity": metadata.get("complexity", 2),
                "chunk_index": metadata.get("chunk_index"),
                "total_chunks": metadata.get("total_chunks"),
                "source_path": metadata.get("source_path"),
                "section_id": metadata.get("section_id"),
                "section_title": metadata.get("section_title"),
                "work_group": metadata.get("work_group"),
                "content": contents[idx],
            }
            docs.append((doc, float(similarity)))

        return docs

    def get_statistics(self) -> Dict:
        return {
            "backend": "chroma",
            "collection": self.collection_name,
            "persist_directory": str(self.persist_directory),
            "total_documents": self._collection.count(),
            "embedding_dim": self.embedding_dim,
        }


__all__ = ["ChromaVectorStore"]
