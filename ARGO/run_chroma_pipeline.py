"""Build a ChromaDB vector store from O-RAN specification documents."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import chromadb
import numpy as np

from RAG_Models.embeddings import EmbeddingModel, TextChunker
from RAG_Models.text_extractor import convert_documents_to_text, iter_text_files

logger = logging.getLogger(__name__)

DEFAULT_DOCS_DIR = Path("ORAN_Docs")
DEFAULT_TEXT_DIR = DEFAULT_DOCS_DIR / "processed_text"
DEFAULT_CHROMA_DIR = Path("Environments") / "chroma_store"
DEFAULT_COLLECTION = "oran_specs"


def load_text_documents(text_dir: Path) -> List[Dict]:
    documents: List[Dict] = []
    for text_path in iter_text_files(text_dir):
        content = text_path.read_text(encoding="utf-8", errors="ignore")
        if not content.strip():
            logger.debug("Skipping empty text file %s", text_path.name)
            continue
        documents.append(
            {
                "doc_id": text_path.stem,
                "title": text_path.name,
                "content": content,
                "category": "unknown",
                "complexity": 2,
                "source_path": str(text_path),
            }
        )
    return documents


def upsert_chunks(
    collection,
    chunked_docs: List[Dict],
    embeddings: np.ndarray,
    batch_size: int = 128,
) -> None:
    total = len(chunked_docs)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_docs = chunked_docs[start:end]
        batch_embeddings = embeddings[start:end].astype(np.float32)
        collection.upsert(
            ids=[doc["chunk_id"] for doc in batch_docs],
            documents=[doc["content"] for doc in batch_docs],
            embeddings=[emb.tolist() for emb in batch_embeddings],
            metadatas=
            [
                {
                    "doc_id": doc["doc_id"],
                    "title": doc.get("title", ""),
                    "chunk_index": int(doc.get("chunk_index", 0)),
                    "total_chunks": int(doc.get("total_chunks", 1)),
                    "source_path": doc.get("source_path", ""),
                }
                for doc in batch_docs
            ],
        )
        logger.info("Upserted chunks %s-%s/%s", start + 1, end, total)


def build_client(persist_dir: Path):
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert O-RAN docs to ChromaDB store")
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR, help="Directory with original documents")
    parser.add_argument("--text-dir", type=Path, default=DEFAULT_TEXT_DIR, help="Directory to store processed text")
    parser.add_argument("--persist-dir", type=Path, default=DEFAULT_CHROMA_DIR, help="Chroma persistence directory")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection name")
    parser.add_argument("--chunk-size", type=int, default=600, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="Chunk overlap in characters")
    parser.add_argument("--batch-size", type=int, default=256, help="Number of chunks per Chroma upsert")
    parser.add_argument("--embedding-model", default=None, help="Embedding model name or local path")
    parser.add_argument("--embedding-device", default=None, help="Device for embeddings (e.g., cuda, cpu)")
    parser.add_argument("--embedding-cache", default=None, help="Cache directory for embeddings")
    parser.add_argument("--force-extract", action="store_true", help="Re-extract text even if it exists")
    parser.add_argument("--rebuild", action="store_true", help="Drop existing Chroma collection before insert")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    logger.info("Starting text extraction from %s", args.docs_dir)
    convert_documents_to_text(args.docs_dir, args.text_dir, force=args.force_extract)

    logger.info("Loading processed text from %s", args.text_dir)
    documents = load_text_documents(args.text_dir)
    if not documents:
        logger.error("No documents found after text extraction. Aborting.")
        return 1
    logger.info("Loaded %s documents for chunking", len(documents))

    chunker = TextChunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunked_docs = chunker.chunk_documents(documents)
    if not chunked_docs:
        logger.error("Chunking produced no documents. Aborting.")
        return 1
    logger.info("Generated %s chunks", len(chunked_docs))

    embedding_kwargs: Dict[str, str] = {}
    if args.embedding_model:
        embedding_kwargs["model_name_or_path"] = args.embedding_model
    if args.embedding_device:
        embedding_kwargs["device"] = args.embedding_device
    if args.embedding_cache:
        embedding_kwargs["cache_folder"] = args.embedding_cache

    embedder = EmbeddingModel(**embedding_kwargs)
    chunked_docs, embeddings = embedder.encode_documents(chunked_docs)
    logger.info("Computed embeddings with shape %s", embeddings.shape)

    client = build_client(args.persist_dir)
    if args.rebuild:
        try:
            client.delete_collection(args.collection)
            logger.info("Deleted existing collection '%s'", args.collection)
        except Exception:  # pragma: no cover - collection may not exist
            logger.debug("Collection '%s' did not exist prior to rebuild", args.collection)
    collection = client.get_or_create_collection(args.collection)

    upsert_chunks(collection, chunked_docs, embeddings, batch_size=args.batch_size)
    if hasattr(client, "persist"):
        client.persist()
        logger.debug("Chroma client persisted explicitly")
    else:  # pragma: no cover - depends on chromadb version
        logger.debug("Chroma client relies on automatic persistence")

    summary = {
        "docs_dir": str(args.docs_dir.resolve()),
        "text_dir": str(args.text_dir.resolve()),
        "persist_dir": str(args.persist_dir.resolve()),
        "collection": args.collection,
        "chunks": len(chunked_docs),
        "embedding_model": embedder.model_name,
        "embedding_dim": embedder.embedding_dim,
    }
    logger.info("ChromaDB build summary: %s", json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
