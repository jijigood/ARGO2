#!/usr/bin/env python
"""CLI helper to run the ARGO RAG pipeline with local models."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from RAG_Models.answer_generator import LocalLLMAnswerGenerator
from RAG_Models.retrieval import build_vector_store
from RAG_Models.text_extraction import extract_documents_to_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local RAG query using ARGO components.")
    parser.add_argument("question", nargs="?", help="Question to ask the RAG system.")
    parser.add_argument(
        "--top-k",
        dest="top_k",
        type=int,
        default=3,
        help="Number of chunks to feed into the LLM context (default: 3).",
    )
    parser.add_argument(
        "--vector-store",
        type=str,
        default="./Environments/vector_store.pkl",
        help="Path to persist/reuse the vector store (default: ./Environments/vector_store.pkl).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild of the vector store even if the file exists.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Override embedding model name or local path.",
    )
    parser.add_argument(
        "--embedding-device",
        type=str,
        default=None,
        help="Device for embedding model (e.g. cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default=None,
        help="Directory containing raw documents (defaults to $ARGO_DOCS_DIR or ../ORAN_Docs).",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Directory containing pre-extracted text documents (defaults to $ARGO_PROCESSED_DIR).",
    )
    parser.add_argument(
        "--auto-extract",
        action="store_true",
        help="Convert supported raw documents (PDF/DOCX) to text before building the vector store.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Force document re-extraction even if the target text file already exists.",
    )
    parser.add_argument(
        "--source-docs",
        type=str,
        default=None,
        help="Override source directory for auto extraction (defaults to --docs-dir or $ARGO_DOCS_DIR).",
    )
    parser.add_argument(
        "--source-extensions",
        type=str,
        nargs="+",
        default=None,
        help="Extensions to include when auto extracting (default: .pdf .docx).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Chunk size for sentence chunker (default: 200).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap for sentence chunker (default: 50).",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Override local LLM model name or path.",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device map passed to transformers (default: auto).",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantized loading for the LLM.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Shared cache directory for Hugging Face models.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Maximum new tokens for generation (defaults to config value).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (defaults to config value).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p nucleus sampling (defaults to config value).",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="System prompt override for the LLM.",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Display retrieved chunks before generation.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow loading models that require custom code (use with caution).",
    )
    parser.add_argument(
        "--vector-backend",
        type=str,
        choices=["numpy", "chroma"],
        default="numpy",
        help="Vector store backend to use (default: numpy).",
    )
    parser.add_argument(
        "--chroma-dir",
        type=str,
        default=None,
        help="Persistence directory for ChromaDB when using the chroma backend.",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="oran_docs",
        help="Collection name to use when persisting to ChromaDB.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    question = args.question or input("Enter your question: ")

    vector_backend = args.vector_backend.lower()
    vector_store_path: Path | None = None
    if vector_backend == "numpy":
        vector_store_path = Path(args.vector_store)
        vector_store_path.parent.mkdir(parents=True, exist_ok=True)

    docs_dir = args.docs_dir or os.getenv("ARGO_DOCS_DIR")
    processed_dir = args.processed_dir or os.getenv("ARGO_PROCESSED_DIR")
    source_docs_dir = args.source_docs or docs_dir

    if args.auto_extract:
        if source_docs_dir is None:
            source_docs_dir = os.getenv("ARGO_DOCS_DIR", "../ORAN_Docs")
        source_docs_dir = str(Path(source_docs_dir).expanduser().resolve())
        processed_root = processed_dir or str(
            Path(source_docs_dir).joinpath("processed_text").resolve()
        )
        include_extensions = args.source_extensions
        if include_extensions:
            include_extensions = [
                ext if ext.startswith(".") else f".{ext}" for ext in include_extensions
            ]
        summary = extract_documents_to_text(
            source_dir=source_docs_dir,
            output_dir=processed_root,
            force=args.force_extract,
            include_extensions=include_extensions,
        )
        print("\nDocument extraction summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        processed_dir = processed_root

    docs_dir_for_builder = processed_dir or docs_dir

    vector_store, retriever = build_vector_store(
        save_path=str(vector_store_path) if vector_store_path else None,
        embedding_model_name=args.embedding_model,
        docs_dir=docs_dir_for_builder,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        rebuild=args.rebuild,
        embedding_device=args.embedding_device,
        embedding_cache=args.cache_dir,
        vector_backend=args.vector_backend,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection_name,
    )

    results = retriever.retrieve(question, top_k=args.top_k)

    structured_chunks = []
    sources = []
    for idx, (doc, score) in enumerate(results, start=1):
        source_title = doc.get("title") or doc.get("doc_id") or "unknown"
        chunk_index = doc.get("chunk_index")
        total_chunks = doc.get("total_chunks")
        segment_label = ""
        if chunk_index is not None and total_chunks is not None:
            segment_label = f" segment {chunk_index + 1}/{total_chunks}"

        header = (
            f"[Chunk {idx} | score {score:.3f} | doc {source_title}{segment_label}]"
        )
        structured_chunks.append(f"{header}\n{doc['content']}")

        sources.append(
            {
                "doc_id": doc.get("doc_id"),
                "chunk_id": doc.get("chunk_id"),
                "score": score,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "source_path": doc.get("source_path"),
                "title": source_title,
            }
        )

    context = "\n\n".join(structured_chunks)

    if args.show_context:
        print("\nRetrieved context:\n" + "-" * 60)
        for idx, (doc, score) in enumerate(results, start=1):
            print(
                f"[{idx}] score={score:.3f} doc={doc.get('doc_id', 'n/a')} "
                f"chunk={doc.get('chunk_index', 'n/a')} source={doc.get('source_path', 'n/a')}"
            )
            print(doc["content"])
            print("-" * 60)

    llm_kwargs = {
        "model_name_or_path": args.llm_model or os.getenv("ARGO_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        "device_map": args.device_map,
        "use_4bit": not args.no_4bit,
        "cache_dir": args.cache_dir,
        "trust_remote_code": args.trust_remote_code,
    }

    generation_config = None
    if any(value is not None for value in [args.max_new_tokens, args.temperature, args.top_p, args.system_prompt]):
        from RAG_Models.answer_generator import GenerationConfig

        generation_config = GenerationConfig()
        if args.max_new_tokens is not None:
            generation_config.max_new_tokens = args.max_new_tokens
        if args.temperature is not None:
            generation_config.temperature = args.temperature
        if args.top_p is not None:
            generation_config.top_p = args.top_p
        if args.system_prompt is not None:
            generation_config.system_prompt = args.system_prompt
        llm_kwargs["generation_config"] = generation_config

    answer_generator = LocalLLMAnswerGenerator(**llm_kwargs)

    output = answer_generator.generate_answer(
        question=question,
        context=context,
        system_prompt=args.system_prompt,
        max_new_tokens=args.max_new_tokens,
    )

    print("\nAnswer:\n" + "=" * 60)
    print(output["answer"])

    if sources:
        print("\nSources:\n" + "-" * 60)
        for entry in sources:
            print(
                f"doc_id={entry['doc_id']} | chunk={entry['chunk_index']} | "
                f"score={entry['score']:.3f} | title={entry['title']}"
            )
            if entry.get("source_path"):
                print(f"path: {entry['source_path']}")
            print("-" * 60)

    vector_store_identifier = (
        str(vector_store_path)
        if vector_store_path
        else args.chroma_dir
        or os.getenv("ARGO_CHROMA_DIR")
        or "Chroma(in-memory)"
    )
    if vector_backend == "chroma":
        vector_store_identifier = getattr(vector_store, "persist_directory", vector_store_identifier)

    metadata = {
        "question": question,
        "top_k": args.top_k,
        "vector_backend": args.vector_backend,
        "vector_store": vector_store_identifier,
        "embedding_model": args.embedding_model or os.getenv("ARGO_EMBEDDING_MODEL"),
        "llm_model": llm_kwargs["model_name_or_path"],
        "sources": sources,
    }
    print("\nMetadata:\n" + json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
