"""
Text Chunking and Embedding Module
文本分块和向量嵌入
"""
import os
import logging
try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore

from typing import List, Dict, Tuple, Optional
import re

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

try:  # pragma: no cover - optional dependency for device detection
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

from .oran_parser import ORANSectionParser


DEFAULT_EMBEDDING_MODEL = os.getenv(
    "ARGO_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
DEFAULT_EMBEDDING_DEVICE = os.getenv("ARGO_EMBEDDING_DEVICE")
DEFAULT_EMBEDDING_CACHE = os.getenv("ARGO_HF_CACHE")


def _get_fallback_device() -> str:
    """Return best available device for embedding model."""
    if DEFAULT_EMBEDDING_DEVICE:
        return DEFAULT_EMBEDDING_DEVICE
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class TextChunker:
    """文本分块器"""
    
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_by_sentences(self, text: str) -> List[str]:
        """按句子分块"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                # 保留overlap
                overlap_sentences = current_chunk[-1:] if current_chunk else []
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """对文档列表进行分块"""
        chunked_docs = []
        
        for doc in documents:
            content = doc.get("content", "")
            chunks = self.chunk_by_sentences(content)
            
            for i, chunk in enumerate(chunks):
                chunked_doc = {
                    "doc_id": doc["doc_id"],
                    "chunk_id": f"{doc['doc_id']}_chunk_{i}",
                    "title": doc.get("title", ""),
                    "content": chunk,
                    "category": doc.get("category", "unknown"),
                    "complexity": doc.get("complexity", 2),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "source_path": doc.get("source_path")
                }
                chunked_docs.append(chunked_doc)
        
        return chunked_docs


class SectionAwareChunker(TextChunker):
    """Enhanced chunker that respects section boundaries."""

    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 50):
        super().__init__(chunk_size, chunk_overlap)
        self.parser = ORANSectionParser()

    def chunk_document_with_sections(
        self,
        doc: Dict,
        preserve_boundaries: bool = True,
    ) -> List[Dict]:
        """Chunk a document while preserving section structure."""

        work_group = self.parser.extract_work_group(doc.get("doc_id", ""))

        if not preserve_boundaries:
            fallback_chunks = super().chunk_documents([doc])
            for chunk in fallback_chunks:
                chunk.setdefault("section_id", "unknown")
                chunk.setdefault("section_title", doc.get("title", ""))
                chunk.setdefault("work_group", work_group)
            return fallback_chunks

        sections = self.parser.parse_document(
            doc.get("content", ""),
            doc_id=doc.get("doc_id", "unknown"),
            work_group=work_group,
        )

        chunked_docs: List[Dict] = []

        for section_idx, section in enumerate(sections):
            section_chunks = self.chunk_by_sentences(section.get("content", ""))
            if not section_chunks:
                continue

            for i, chunk_text in enumerate(section_chunks):
                chunked_doc = {
                    "doc_id": doc.get("doc_id"),
                    "chunk_id": f"{doc.get('doc_id')}_sec{section_idx}_chunk_{i}",
                    "title": doc.get("title", ""),
                    "content": chunk_text,
                    "category": doc.get("category", "unknown"),
                    "complexity": doc.get("complexity", 2),
                    "chunk_index": i,
                    "total_chunks": len(section_chunks),
                    "source_path": doc.get("source_path"),
                    "section_id": section.get("section_id", "unknown"),
                    "section_title": section.get("section_title", ""),
                    "work_group": section.get("work_group", work_group),
                }
                chunked_docs.append(chunked_doc)

        return chunked_docs


class EmbeddingModel:
    """向量嵌入模型"""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_EMBEDDING_MODEL,
    device: Optional[str] = DEFAULT_EMBEDDING_DEVICE,
    cache_folder: Optional[str] = DEFAULT_EMBEDDING_CACHE,
    ):
        """初始化嵌入模型."""

        self.model_name = model_name_or_path
        resolved_device = device or _get_fallback_device()
        self.device = resolved_device
        self.cache_folder = cache_folder

        if SentenceTransformer is None:
            logging.warning(
                "sentence-transformers is not installed; falling back to mock embeddings."
            )
            self.model = None
            self.embedding_dim = 384
            return

        load_kwargs = {"device": resolved_device}
        if cache_folder:
            load_kwargs["cache_folder"] = cache_folder

        try:
            self.model = SentenceTransformer(self.model_name, **load_kwargs)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logging.info(
                "Loaded embedding model '%s' on device '%s'", self.model_name, resolved_device
            )
        except Exception as exc:  # pragma: no cover - runtime fallback
            logging.warning(
                "Failed to load embedding model %s (%s); using mock embeddings.",
                self.model_name,
                exc,
            )
            self.model = None
            self.embedding_dim = 384  # 默认维度用于回退模式
    
    def encode_text(self, text: str) -> np.ndarray:
        """将文本编码为向量"""
        if self.model is None:
            # Mock embedding for testing
            np.random.seed(hash(text) % (2**32))
            return np.random.randn(self.embedding_dim).astype(np.float32)
        
        return self.model.encode(text, convert_to_numpy=True)
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """批量编码文本"""
        if self.model is None:
            # Mock embeddings
            embeddings = []
            for text in texts:
                np.random.seed(hash(text) % (2**32))
                embeddings.append(np.random.randn(self.embedding_dim).astype(np.float32))
            return np.array(embeddings)
        
        return self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
    
    def encode_documents(self, documents: List[Dict]) -> Tuple[List[Dict], np.ndarray]:
        """
        对文档列表进行编码
        Returns:
            documents: 文档列表
            embeddings: 对应的嵌入向量数组
        """
        texts = [doc["content"] for doc in documents]
        embeddings = self.encode_batch(texts)
        
        # 将embedding添加到文档中
        for i, doc in enumerate(documents):
            doc["embedding"] = embeddings[i]
        
        return documents, embeddings
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def batch_cosine_similarity(self, query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        """批量计算余弦相似度"""
        # 归一化
        query_norm = query_vec / np.linalg.norm(query_vec)
        doc_norms = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
        
        # 计算相似度
        similarities = np.dot(doc_norms, query_norm)
        return similarities


def test_chunking_and_embedding():
    """测试分块和嵌入功能"""
    from .document_loader import ORANDocumentLoader
    
    print("=" * 60)
    print("Text Chunking and Embedding Test")
    print("=" * 60)
    
    # 加载文档
    loader = ORANDocumentLoader()
    docs = loader.load_sample_documents()
    print(f"\nLoaded {len(docs)} documents")
    
    # 分块
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    chunked_docs = chunker.chunk_documents(docs)
    print(f"Created {len(chunked_docs)} chunks")
    
    # 嵌入
    embedder = EmbeddingModel()
    print(f"\nEmbedding model: {embedder.model_name}")
    print(f"Embedding dimension: {embedder.embedding_dim}")
    
    chunked_docs, embeddings = embedder.encode_documents(chunked_docs)
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # 测试相似度
    query = "What is the A1 interface?"
    query_vec = embedder.encode_text(query)
    similarities = embedder.batch_cosine_similarity(query_vec, embeddings)
    
    print(f"\nQuery: {query}")
    print(f"Top 3 most similar chunks:")
    top_indices = np.argsort(similarities)[-3:][::-1]
    for idx in top_indices:
        print(f"  Similarity: {similarities[idx]:.3f}")
        print(f"  Doc: {chunked_docs[idx]['doc_id']}")
        print(f"  Content: {chunked_docs[idx]['content'][:80]}...")
        print()


if __name__ == "__main__":
    test_chunking_and_embedding()
