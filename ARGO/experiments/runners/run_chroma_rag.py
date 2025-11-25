#!/usr/bin/env python
"""Run the RAG pipeline against the persistent Chroma knowledge base."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
import numpy as np

from RAG_Models.answer_generator import LocalLLMAnswerGenerator
from RAG_Models.embeddings import EmbeddingModel

DEFAULT_LLM_PATH = Path("RAG_Models") / "models" / "mistral-7b-instruct-v0.3"

logger = logging.getLogger(__name__)

# Focus the assistant on O-RAN grounding by default.
DEFAULT_SYSTEM_PROMPT = (
    "You are an O-RAN standards expert. Answer strictly using the retrieved context. "
    "If the context does not cover the question, admit that explicitly."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the Chroma knowledge base with a local LLM.")
    parser.add_argument("question", nargs="?", help="Question for the RAG system.")
    parser.add_argument("--persist-dir", type=Path, default=Path("Environments") / "chroma_store",
                        help="Directory containing the persistent Chroma database.")
    parser.add_argument("--collection", default="oran_specs", help="Chroma collection name.")
    parser.add_argument("--top-k", dest="top_k", type=int, default=3, help="Number of chunks to retrieve.")
    parser.add_argument("--embedding-model", default=None, help="Embedding model name or local path override.")
    parser.add_argument("--embedding-device", default=None, help="Embedding device (e.g. cuda, cpu).")
    parser.add_argument("--embedding-cache", default=None, help="Embedding model cache directory.")
    parser.add_argument("--llm-model", default=None, help="Local LLM model name or path.")
    parser.add_argument("--device-map", default="auto", help="Device map for transformers pipeline.")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantised loading for the LLM.")
    parser.add_argument("--cache-dir", default=None, help="Shared cache directory for Hugging Face models.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override max new tokens for generation.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature override.")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p override.")
    parser.add_argument("--system-prompt", default=None, help="Override system prompt for the LLM.")
    parser.add_argument("--show-context", action="store_true", help="Print retrieved chunks before generation.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow loading models with custom code.")
    parser.add_argument("--spread-gpus", action="store_true", help="Distribute the LLM across all CUDA devices.")
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=0.85,
        help="Fraction of each GPU memory to allocate when --spread-gpus is enabled (0 < f ≤ 1).",
    )
    return parser.parse_args()


def build_embedder(args: argparse.Namespace) -> EmbeddingModel:
    kwargs: Dict[str, Optional[str]] = {}
    if args.embedding_model:
        kwargs["model_name_or_path"] = args.embedding_model
    if args.embedding_device:
        kwargs["device"] = args.embedding_device
    if args.embedding_cache:
        kwargs["cache_folder"] = args.embedding_cache
    return EmbeddingModel(**kwargs)


def load_collection(persist_dir: Path, collection_name: str):
    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        return client.get_collection(collection_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to open collection '{collection_name}' in {persist_dir}. "
            "Ensure run_chroma_pipeline.py finished successfully."
        ) from exc


def query_collection(collection, embedder: EmbeddingModel, question: str, top_k: int) -> List[Dict]:
    query_vec = embedder.encode_text(question)
    results = collection.query(
        query_embeddings=[query_vec.astype(np.float32).tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]

    chunks: List[Dict] = []
    for idx, chunk_text in enumerate(documents):
        metadata = metadatas[idx] if idx < len(metadatas) and metadatas[idx] is not None else {}
        candidate_vec = embedder.encode_text(chunk_text)
        similarity = embedder.cosine_similarity(query_vec, candidate_vec)
        chunks.append(
            {
                "chunk_id": ids[idx] if idx < len(ids) else f"chunk_{idx}",
                "content": chunk_text,
                "similarity": similarity,
                "metadata": metadata,
            }
        )
    return chunks


def build_context(chunks: List[Dict]) -> str:
    sections = []
    for rank, chunk in enumerate(chunks, start=1):
        meta = chunk["metadata"]
        title = meta.get("title") or meta.get("doc_id") or "unknown"
        chunk_index = meta.get("chunk_index")
        total_chunks = meta.get("total_chunks")
        segment = ""
        if chunk_index is not None and total_chunks is not None:
            segment = f" segment {int(chunk_index) + 1}/{int(total_chunks)}"
        header = f"[Rank {rank} | score {chunk['similarity']:.3f} | doc {title}{segment}]"
        sections.append(f"{header}\n{chunk['content']}")
    return "\n\n".join(sections)


def print_sources(chunks: List[Dict]) -> List[Dict]:
    records: List[Dict] = []
    for chunk in chunks:
        meta = chunk["metadata"]
        record = {
            "chunk_id": chunk.get("chunk_id"),
            "doc_id": meta.get("doc_id"),
            "title": meta.get("title"),
            "chunk_index": meta.get("chunk_index"),
            "total_chunks": meta.get("total_chunks"),
            "source_path": meta.get("source_path"),
            "score": chunk.get("similarity"),
        }
        records.append(record)
    return records


def compute_max_memory(fraction: float) -> Optional[Dict[str, str]]:
    """Estimate per-GPU memory budgets for multi-GPU loading."""
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not available; cannot spread across GPUs.")
        return None

    if not torch.cuda.is_available():
        logger.warning("CUDA unavailable; ignoring --spread-gpus.")
        return None

    fraction = max(0.1, min(float(fraction), 1.0))
    max_memory: Dict[str, str] = {}
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        total_mb = props.total_memory // (1024 * 1024)
        budget = max(512, int(total_mb * fraction))
        max_memory[idx] = f"{budget}MiB"
        logger.info("GPU %s budget set to %s MiB", idx, budget)
    max_memory["cpu"] = "0MiB"
    return max_memory


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    question = args.question or input("Enter your question: ")
    embedder = build_embedder(args)
    collection = load_collection(args.persist_dir, args.collection)
    chunks = query_collection(collection, embedder, question, args.top_k)
    if not chunks:
        raise SystemExit("No chunks retrieved; consider rebuilding the store or lowering top-k.")

    if args.show_context:
        logging.info("Retrieved %s chunks", len(chunks))
        for idx, chunk in enumerate(chunks, start=1):
            meta = chunk["metadata"]
            logging.info(
                "[%s] score=%.3f doc=%s chunk=%s source=%s",
                idx,
                chunk["similarity"],
                meta.get("doc_id"),
                meta.get("chunk_index"),
                meta.get("source_path"),
            )
            logging.info(chunk["content"])

    context = build_context(chunks)

    llm_kwargs = {
        "model_name_or_path": args.llm_model or str(DEFAULT_LLM_PATH),
        "device_map": args.device_map,
        "use_4bit": not args.no_4bit,
        "cache_dir": args.cache_dir,
        "trust_remote_code": args.trust_remote_code,
    }

    if args.spread_gpus:
        max_memory = compute_max_memory(args.gpu_memory_fraction)
        if max_memory:
            llm_kwargs["device_map"] = "balanced"
            llm_kwargs["max_memory"] = max_memory
        else:
            logger.warning("Falling back to device_map=%s", args.device_map)

    if any(value is not None for value in (args.max_new_tokens, args.temperature, args.top_p, args.system_prompt)):
        from RAG_Models.answer_generator import GenerationConfig

        gen_config = GenerationConfig()
        if args.max_new_tokens is not None:
            gen_config.max_new_tokens = args.max_new_tokens
        if args.temperature is not None:
            gen_config.temperature = args.temperature
        if args.top_p is not None:
            gen_config.top_p = args.top_p
        if args.system_prompt is not None:
            gen_config.system_prompt = args.system_prompt
        llm_kwargs["generation_config"] = gen_config

    answer_generator = LocalLLMAnswerGenerator(**llm_kwargs)

    system_prompt = args.system_prompt or DEFAULT_SYSTEM_PROMPT
    output = answer_generator.generate_answer(question=question, context=context, system_prompt=system_prompt,
                                              max_new_tokens=args.max_new_tokens)

    print("\nAnswer:\n" + "=" * 60)
    print(output["answer"])

    sources = print_sources(chunks)
    if sources:
        print("\nSources:\n" + "-" * 60)
        for src in sources:
            print(
                f"doc_id={src['doc_id']} | chunk={src['chunk_index']} | score={src['score']:.3f} | title={src['title']}"
            )
            if src.get("source_path"):
                print(f"path: {src['source_path']}")
            print("-" * 60)

    metadata = {
        "question": question,
        "top_k": args.top_k,
        "persist_dir": str(args.persist_dir),
        "collection": args.collection,
        "embedding_model": args.embedding_model,
        "llm_model": llm_kwargs["model_name_or_path"],
        "sources": sources,
    }
    print("\nMetadata:\n" + json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
