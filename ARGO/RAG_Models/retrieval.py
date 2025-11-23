"""
Vector Store and Retrieval Module
向量存储和检索模块
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
import pickle
import os
from pathlib import Path


class VectorStore:
    """向量存储库"""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.documents = []
        self.embeddings = None
        self.index_map = {}  # chunk_id -> index
    
    def add_documents(self, documents: List[Dict], embeddings: np.ndarray):
        """添加文档和对应的嵌入向量"""
        start_idx = len(self.documents)
        
        for i, doc in enumerate(documents):
            self.documents.append(doc)
            chunk_id = doc.get("chunk_id", f"doc_{start_idx + i}")
            self.index_map[chunk_id] = start_idx + i
        
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        检索最相似的文档
        Returns:
            List of (document, similarity_score) tuples
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []
        
        # 计算余弦相似度
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        similarities = np.dot(doc_norms, query_norm)
        
        # 获取top-k
        top_k = min(top_k, len(self.documents))
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((self.documents[idx], float(similarities[idx])))
        
        return results
    
    def search_with_filter(self, 
                          query_embedding: np.ndarray, 
                          top_k: int = 5,
                          category: Optional[str] = None,
                          min_similarity: float = 0.0) -> List[Tuple[Dict, float]]:
        """
        带过滤条件的检索
        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量
            category: 文档类别过滤
            min_similarity: 最小相似度阈值
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []
        
        # 计算相似度
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        similarities = np.dot(doc_norms, query_norm)
        
        # 应用过滤条件
        valid_indices = []
        for i, (doc, sim) in enumerate(zip(self.documents, similarities)):
            if sim < min_similarity:
                continue
            if category is not None and doc.get("category") != category:
                continue
            valid_indices.append(i)
        
        if not valid_indices:
            return []
        
        # 排序并获取top-k
        valid_similarities = similarities[valid_indices]
        sorted_local_indices = np.argsort(valid_similarities)[-top_k:][::-1]
        sorted_global_indices = [valid_indices[i] for i in sorted_local_indices]
        
        results = []
        for idx in sorted_global_indices:
            results.append((self.documents[idx], float(similarities[idx])))
        
        return results
    
    def save(self, filepath: str):
        """保存向量库到文件"""
        data = {
            "documents": self.documents,
            "embeddings": self.embeddings,
            "index_map": self.index_map,
            "embedding_dim": self.embedding_dim
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Vector store saved to {filepath}")
    
    def load(self, filepath: str):
        """从文件加载向量库"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.documents = data["documents"]
        self.embeddings = data["embeddings"]
        self.index_map = data["index_map"]
        self.embedding_dim = data["embedding_dim"]
        print(f"Vector store loaded from {filepath}")
    
    def get_statistics(self) -> Dict:
        """获取向量库统计信息"""
        categories = {}
        for doc in self.documents:
            cat = doc.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            "total_documents": len(self.documents),
            "embedding_dim": self.embedding_dim,
            "categories": categories,
            "embeddings_shape": self.embeddings.shape if self.embeddings is not None else None
        }


class Retriever:
    """检索器 - 封装检索逻辑"""
    
    def __init__(self, vector_store: VectorStore, embedding_model):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.query_history = []
    
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Tuple[Dict, float]]:
        """
        检索相关文档
        Args:
            query: 查询文本
            top_k: 返回结果数量
            **kwargs: 其他过滤参数(category, min_similarity等)
        """
        # 编码查询
        query_embedding = self.embedding_model.encode_text(query)
        
        # 检索
        if kwargs:
            results = self.vector_store.search_with_filter(
                query_embedding, top_k=top_k, **kwargs
            )
        else:
            results = self.vector_store.search(query_embedding, top_k=top_k)
        
        # 记录查询历史
        self.query_history.append({
            "query": query,
            "top_k": top_k,
            "num_results": len(results),
            "avg_similarity": np.mean([score for _, score in results]) if results else 0.0
        })
        
        return results
    
    def get_context(self, query: str, top_k: int = 3) -> str:
        """获取查询的上下文字符串（用于LLM输入）"""
        results = self.retrieve(query, top_k=top_k)
        
        context_parts = []
        for i, (doc, score) in enumerate(results, 1):
            context_parts.append(
                f"[Document {i}] (Similarity: {score:.3f})\n"
                f"Title: {doc.get('title', 'Unknown')}\n"
                f"Content: {doc['content']}\n"
            )
        
        return "\n".join(context_parts)
    
    def get_query_statistics(self) -> Dict:
        """获取查询统计信息"""
        if not self.query_history:
            return {"total_queries": 0}
        
        return {
            "total_queries": len(self.query_history),
            "avg_results_per_query": np.mean([q["num_results"] for q in self.query_history]),
            "avg_similarity": np.mean([q["avg_similarity"] for q in self.query_history]),
            "unique_queries": len(set(q["query"] for q in self.query_history))
        }


from typing import Optional


def build_vector_store(
    save_path: Optional[str] = "../vector_store.pkl",
    embedding_model_name: Optional[str] = None,
    docs_dir: Optional[str] = None,
    chunk_size: int = 200,
    chunk_overlap: int = 50,
    rebuild: bool = False,
    embedding_device: Optional[str] = None,
    embedding_cache: Optional[str] = None,
    file_extension: str = ".txt",
) -> Tuple[VectorStore, Retriever]:
    """构建或加载向量存储库."""
    from .document_loader import ORANDocumentLoader
    from .embeddings import TextChunker, EmbeddingModel
    
    print("=" * 60)
    print("Building Vector Store")
    print("=" * 60)
    
    # 解析路径
    resolved_save_path: Optional[Path] = None
    if save_path:
        resolved_save_path = Path(save_path).expanduser().resolve()

    # 设置文档目录
    docs_dir = docs_dir or os.getenv("ARGO_DOCS_DIR", "../ORAN_Docs")
    docs_dir_path = Path(docs_dir).expanduser().resolve()

    # 初始化嵌入模型
    embedding_kwargs = {}
    if embedding_model_name:
        embedding_kwargs["model_name_or_path"] = embedding_model_name
    if embedding_device:
        embedding_kwargs["device"] = embedding_device
    if embedding_cache:
        embedding_kwargs["cache_folder"] = embedding_cache

    embedder = EmbeddingModel(**embedding_kwargs)

    # 如果已有向量库且无需重建，直接加载
    if resolved_save_path and resolved_save_path.exists() and not rebuild:
        vector_store = VectorStore()
        vector_store.load(str(resolved_save_path))
        retriever = Retriever(vector_store, embedder)

        print(f"\n[0/4] Loaded existing vector store from {resolved_save_path}")
        print(f"Embedding model: {embedder.model_name} (device={embedder.device})")
        return vector_store, retriever

    # 1. 加载文档
    loader = ORANDocumentLoader(str(docs_dir_path))
    docs = loader.load_from_directory(file_extension=file_extension)
    print(f"\n[1/4] Loaded {len(docs)} documents from {docs_dir_path}")
    
    # 2. 分块
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_docs = chunker.chunk_documents(docs)
    print(f"[2/4] Created {len(chunked_docs)} chunks")
    
    # 3. 嵌入
    chunked_docs, embeddings = embedder.encode_documents(chunked_docs)
    print(f"[3/4] Generated embeddings with shape {embeddings.shape}")
    
    # 4. 构建向量库
    vector_store = VectorStore(embedding_dim=embedder.embedding_dim)
    vector_store.add_documents(chunked_docs, embeddings)
    print(f"[4/4] Built vector store with {len(vector_store.documents)} documents")
    
    # 保存
    if resolved_save_path:
        resolved_save_path.parent.mkdir(parents=True, exist_ok=True)
        vector_store.save(str(resolved_save_path))
    
    # 创建检索器
    retriever = Retriever(vector_store, embedder)
    
    return vector_store, retriever


def test_retrieval():
    """测试检索功能"""
    print("=" * 60)
    print("Retrieval Test")
    print("=" * 60)
    
    vector_store, retriever = build_vector_store(save_path=None)
    
    # 测试查询
    queries = [
        "What is O-RAN architecture?",
        "Explain the A1 interface",
        "How does E2 interface work?",
        "What is the fronthaul interface?",
    ]
    
    print("\n" + "=" * 60)
    print("Testing Queries")
    print("=" * 60)
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = retriever.retrieve(query, top_k=3)
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n  Result {i} (Similarity: {score:.3f})")
            print(f"    Doc ID: {doc['doc_id']}")
            print(f"    Category: {doc['category']}")
            print(f"    Content: {doc['content'][:100]}...")
    
    # 统计信息
    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)
    print(f"\nVector Store: {vector_store.get_statistics()}")
    print(f"\nQuery Statistics: {retriever.get_query_statistics()}")


if __name__ == "__main__":
    test_retrieval()
