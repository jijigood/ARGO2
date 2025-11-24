"""Build ChromaDB with section-aware chunking."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RAG_Models.document_loader import ORANDocumentLoader
from RAG_Models.embeddings import SectionAwareChunker, EmbeddingModel
from RAG_Models.chroma_backend import ChromaVectorStore
from RAG_Models.text_extractor import convert_documents_to_text


def build_optimized_chroma(
    source_dir: str = "ORAN_Docs",
    text_dir: str | None = None,
    output_dir: str = "../Environments/chroma_store_v2",
    collection_name: str = "oran_specs_semantic",
    force_extract: bool = False,
):
    """Build ChromaDB with semantic chunking."""

    print("=" * 80)
    print("Building Optimized O-RAN Knowledge Base (Semantic Chunking)")
    print("=" * 80)

    source_path = Path(source_dir)
    if not source_path.is_absolute():
        source_path = (PROJECT_ROOT / source_path).resolve()
    else:
        source_path = source_path.resolve()

    if text_dir:
        text_path = Path(text_dir)
        if not text_path.is_absolute():
            text_path = (PROJECT_ROOT / text_path).resolve()
        else:
            text_path = text_path.resolve()
    else:
        text_path = (source_path / "processed_text").resolve()
    text_path.mkdir(parents=True, exist_ok=True)

    print(f"\nExtracting text from: {source_path}")
    extract_results = convert_documents_to_text(source_path, text_path, force=force_extract)
    if not extract_results:
        raise RuntimeError("No documents were extracted; please check the source directory")
    print(f"[1/4] Extracted {len(extract_results)} text files -> {text_path}")

    loader = ORANDocumentLoader(str(text_path))
    docs = loader.load_from_directory(file_extension=".txt")
    print(f"\n[2/5] Loaded {len(docs)} documents")

    chunker = SectionAwareChunker(chunk_size=200, chunk_overlap=50)
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunker.chunk_document_with_sections(doc))
    print(f"[3/5] Created {len(all_chunks)} semantic chunks")

    embedder = EmbeddingModel()
    texts = [chunk["content"] for chunk in all_chunks]
    embeddings = embedder.encode_batch(texts)
    print(f"[4/5] Generated embeddings {embeddings.shape}")

    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = (PROJECT_ROOT / output_path).resolve()
    else:
        output_path = output_path.resolve()

    vector_store = ChromaVectorStore(
        persist_directory=output_path,
        collection_name=collection_name,
        embedding_dim=embedder.embedding_dim,
    )
    vector_store.reset()
    vector_store.add_documents(all_chunks, embeddings)
    print(f"[5/5] Stored in ChromaDB: {output_path}")

    stats = vector_store.get_statistics()
    print("\n" + "=" * 80)
    print("Knowledge Base Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("=" * 80)

    return vector_store


if __name__ == "__main__":
    build_optimized_chroma()
