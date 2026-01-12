"""
Retriever - Phase 3.2 (Enhanced with ARGO Prompts V2.0)
========================================================

基于Chroma向量数据库的检索器。接受子查询，返回相关文档。
模拟检索成功率p_s机制，并支持基于检索文档的答案生成。

Key Features:
- 向量检索：使用embedding模型进行语义检索
- Top-k返回：返回最相关的k个文档
- 失败模拟：基于相似度阈值模拟p_s
- 批量检索：支持批量查询加速
- 答案生成：基于检索文档生成中间答案（新增）

Input:
- q_t: 子查询（字符串）
- k: 返回文档数量

Output:
- r_t: 检索到的文档列表（或空列表∅）
- success: 是否成功检索到相关文档
- answer: 基于检索文档的答案（可选）

Example:
    retriever = Retriever(
        chroma_dir="Environments/chroma_store",
        collection_name="oran_specs"
    )
    docs, success = retriever.retrieve("What is O-RAN latency?", k=3)
    
    # 可选：生成基于检索的答案
    answer = retriever.generate_answer_from_docs(
        question="What is O-RAN latency?",
        docs=docs,
        model=model,
        tokenizer=tokenizer
    )
"""

# Chroma作为可选依赖
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    chromadb = None

import torch
import numpy as np
import os
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
from .prompts import ARGOPrompts

logger = logging.getLogger(__name__)


class Retriever:
    """
    基于Chroma的向量检索器
    
    核心功能：
    1. 向量检索：将查询编码为embedding，检索最相似文档
    2. 成功率模拟：基于相似度阈值模拟p_s
    3. 批量处理：支持批量查询提升效率
    """
    
    def __init__(
        self,
        chroma_dir: str = "Environments/chroma_store",
        collection_name: str = "oran_specs",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.3,  # 低于此阈值视为检索失败
        p_s_mode: str = "threshold",  # "threshold" or "random"
        p_s_value: float = 0.8,  # 仅在random模式下使用
        use_reranker: bool = True,  # 是否使用reranker
        reranker_model_path: Optional[str] = None,  # Reranker模型路径
        reranker_device: Optional[str] = None  # Reranker设备：'cuda', 'cpu', 或None（自动选择）
    ):
        """
        Args:
            chroma_dir: Chroma数据库存储目录
            collection_name: 集合名称
            embedding_model_name: Embedding模型名称
            similarity_threshold: 相似度阈值（余弦相似度）
            p_s_mode: 成功率模式
                - "threshold": 基于相似度阈值
                - "random": 固定概率p_s（用于实验）
            p_s_value: random模式下的成功概率
            use_reranker: 是否使用reranker重新排序检索结果
            reranker_model_path: Reranker模型路径（如果为None，使用默认路径）
            reranker_device: Reranker设备
                - 'cuda': 强制使用GPU（如果可用）
                - 'cpu': 强制使用CPU（较慢但内存占用小）
                - None: 自动选择（优先GPU，失败时回退到CPU）
        """
        self.chroma_dir = Path(chroma_dir)
        self.collection_name = collection_name
        self.similarity_threshold = similarity_threshold
        self.p_s_mode = p_s_mode
        self.p_s_value = p_s_value
        self.use_reranker = use_reranker
        # 默认使用更小的bge-reranker-base模型（278M参数，1.1GB）而不是v2-m3（568M参数，2.3GB）
        # 如果指定了路径，使用指定路径；否则使用base模型
        self.reranker_model_name = "BAAI/bge-reranker-base"  # 更小的模型，适合GPU内存受限的情况
        self.reranker_model_path = reranker_model_path
        self.reranker_device_preference = reranker_device  # 设备偏好：'cuda', 'cpu', 或None（自动）
        
        # 初始化Reranker（如果启用）
        self.reranker_model = None
        self.reranker_tokenizer = None
        # 使用WARNING级别确保输出（即使没有配置logging.basicConfig）
        logger.warning(f"🔵 About to initialize reranker. use_reranker={self.use_reranker}")
        if self.use_reranker:
            logger.warning("✅ use_reranker is True, calling _init_reranker()")
            self._init_reranker()
        else:
            logger.warning("⚠️ use_reranker is False, skipping reranker initialization")
        
        # 检查Chroma是否可用
        if not CHROMA_AVAILABLE:
            raise ImportError(
                "chromadb is not installed. Please install it with: pip install chromadb"
            )
        
        # 初始化Chroma客户端
        logger.info(f"Initializing Chroma client from {self.chroma_dir}")
        
        try:
            self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
            self.collection = self.client.get_collection(name=collection_name)
            
            # 获取集合统计
            count = self.collection.count()
            logger.info(f"Loaded collection '{collection_name}' with {count} documents")
            
        except Exception as e:
            logger.error(f"Failed to load Chroma collection: {e}")
            raise RuntimeError(
                f"Cannot load collection '{collection_name}' from {chroma_dir}. "
                "Please run run_chroma_pipeline.py first."
            ) from e
        
        # 使用WARNING级别确保日志输出（即使没有配置logging.basicConfig）
        reranker_status = "ENABLED" if (self.use_reranker and self.reranker_model) else "DISABLED"
        reranker_details = ""
        if self.use_reranker:
            if self.reranker_model:
                reranker_details = f" (model exists: True, device: {getattr(self, 'reranker_device', 'unknown')})"
            else:
                reranker_details = f" (model exists: False - initialization may have failed)"
        
        logger.warning(
            f"Retriever initialized: mode={p_s_mode}, "
            f"threshold={similarity_threshold:.3f}, p_s={p_s_value:.2f}, "
            f"reranker={reranker_status}{reranker_details}"
        )
    
    def _init_reranker(self):
        """初始化Reranker模型"""
        # 使用WARNING级别确保输出（即使没有配置logging.basicConfig）
        logger.warning("="*80)
        logger.warning("RERANKER INITIALIZATION START")
        logger.warning(f"use_reranker: {self.use_reranker}")
        logger.warning(f"reranker_model_path: {self.reranker_model_path}")
        logger.warning(f"reranker_device_preference: {self.reranker_device_preference}")
        
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            logger.warning("✅ transformers imported successfully")
            
            logger.warning(f"Loading reranker model...")
            
            # 优先使用本地路径（如果提供），否则使用HuggingFace Hub
            if self.reranker_model_path and os.path.exists(self.reranker_model_path):
                logger.warning(f"✅ reranker_model_path exists: {self.reranker_model_path}")
                
                # 智能处理HuggingFace缓存路径
                # 如果路径是缓存根目录（包含snapshots子目录），自动查找最新的snapshot
                actual_model_path = self.reranker_model_path
                snapshots_dir = os.path.join(self.reranker_model_path, "snapshots")
                
                logger.warning(f"Checking for snapshots directory: {snapshots_dir}")
                if os.path.exists(snapshots_dir) and os.path.isdir(snapshots_dir):
                    # 这是HuggingFace缓存根目录，需要找到snapshots下的实际模型路径
                    logger.warning(f"✅ Detected HuggingFace cache directory, looking for snapshots...")
                    snapshots = [d for d in os.listdir(snapshots_dir) 
                               if os.path.isdir(os.path.join(snapshots_dir, d))]
                    logger.warning(f"Found {len(snapshots)} snapshots: {snapshots}")
                    
                    if snapshots:
                        # 使用最新的snapshot（按修改时间排序）
                        snapshots.sort(key=lambda x: os.path.getmtime(os.path.join(snapshots_dir, x)), reverse=True)
                        actual_model_path = os.path.join(snapshots_dir, snapshots[0])
                        logger.warning(f"✅ Found snapshot: {snapshots[0]}")
                        logger.warning(f"✅ Using model path: {actual_model_path}")
                    else:
                        logger.warning(f"⚠️ No snapshots found in {snapshots_dir}, using original path")
                else:
                    logger.warning(f"Not a HuggingFace cache directory, using path directly: {actual_model_path}")
                
                try:
                    logger.warning(f"🔵 Trying to load from local path: {actual_model_path}")
                    logger.warning(f"   Checking if path exists: {os.path.exists(actual_model_path)}")
                    logger.warning(f"   Checking for config.json: {os.path.exists(os.path.join(actual_model_path, 'config.json'))}")
                    
                    self.reranker_tokenizer = AutoTokenizer.from_pretrained(
                        actual_model_path,
                        trust_remote_code=True
                    )
                    logger.warning("✅ Tokenizer loaded successfully")
                    
                    self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
                        actual_model_path,
                        trust_remote_code=True
                    )
                    logger.warning("✅ Model loaded successfully from local path")
                except Exception as e1:
                    logger.warning(f"❌ Failed to load from local path: {e1}")
                    import traceback
                    logger.warning(traceback.format_exc())
                    # 如果本地路径失败，尝试Hub
                    logger.warning(f"🔄 Falling back to HuggingFace Hub: {self.reranker_model_name}")
                    try:
                        self.reranker_tokenizer = AutoTokenizer.from_pretrained(
                            self.reranker_model_name,
                            trust_remote_code=True
                        )
                        logger.warning("✅ Tokenizer loaded from Hub")
                        
                        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
                            self.reranker_model_name,
                            trust_remote_code=True
                        )
                        logger.warning("✅ Model loaded from HuggingFace Hub")
                    except Exception as e2:
                        logger.warning(f"❌ Hub loading also failed: {e2}")
                        logger.warning(traceback.format_exc())
                        raise Exception(f"Both local path and Hub failed. Local error: {e1}, Hub error: {e2}")
            else:
                # 没有本地路径，直接从Hub加载
                logger.warning(f"Trying to load from HuggingFace Hub: {self.reranker_model_name}")
                logger.warning(f"  Model size: ~278M parameters, ~1.1GB (smaller than v2-m3)")
                try:
                    self.reranker_tokenizer = AutoTokenizer.from_pretrained(
                        self.reranker_model_name,
                        trust_remote_code=True
                    )
                    self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
                        self.reranker_model_name,
                        trust_remote_code=True
                    )
                    logger.warning("✅ Loaded from HuggingFace Hub")
                except Exception as e:
                    raise Exception(f"Failed to load from Hub: {e}")
            
            # 设置为评估模式
            self.reranker_model.eval()
            
            # 移动到合适的设备
            if self.reranker_device_preference == 'cpu':
                # 强制使用CPU
                self.reranker_device = 'cpu'
                logger.warning("Using CPU for reranker (as specified)")
            elif self.reranker_device_preference == 'cuda' and torch.cuda.is_available():
                # 强制使用GPU
                try:
                    self.reranker_model = self.reranker_model.cuda()
                    self.reranker_device = 'cuda'
                    logger.warning("Using GPU for reranker (as specified)")
                except RuntimeError as e:
                    logger.warning(f"Failed to move reranker to GPU: {e}")
                    logger.warning("Falling back to CPU")
                    self.reranker_device = 'cpu'
            elif torch.cuda.is_available():
                # 自动选择：尝试GPU，失败时回退到CPU
                try:
                    self.reranker_model = self.reranker_model.cuda()
                    self.reranker_device = 'cuda'
                    logger.warning("Using GPU for reranker (auto-selected)")
                except RuntimeError as e:
                    logger.warning(f"Failed to move reranker to GPU (out of memory?): {e}")
                    logger.warning("Falling back to CPU reranker")
                    self.reranker_device = 'cpu'
            else:
                # 没有GPU可用
                self.reranker_device = 'cpu'
                logger.warning("Using CPU for reranker (no GPU available)")
            
            logger.warning(f"✅ Reranker model loaded successfully on {self.reranker_device}")
            logger.warning("="*80)
            logger.warning("✅ RERANKER INITIALIZATION SUCCESS")
            logger.warning(f"  Model: {self.reranker_model_name}")
            logger.warning(f"  Device: {self.reranker_device}")
            logger.warning(f"  Model exists: {self.reranker_model is not None}")
            logger.warning(f"  Tokenizer exists: {self.reranker_tokenizer is not None}")
            logger.warning("="*80)
            
        except Exception as e:
            # 使用WARNING级别确保输出（即使没有配置logging.basicConfig）
            logger.warning("="*80)
            logger.warning("❌ RERANKER INITIALIZATION FAILED")
            logger.warning(f"Error type: {type(e).__name__}")
            logger.warning(f"Error message: {str(e)}")
            logger.warning("="*80)
            # 输出完整堆栈到WARNING级别
            import traceback
            logger.warning(traceback.format_exc())
            logger.warning("Continuing without reranker - using original retrieval scores")
            logger.warning("="*80)
            self.use_reranker = False
            self.reranker_model = None
            self.reranker_tokenizer = None
    
    def _rerank_documents(
        self,
        query: str,
        documents: List[str],
        original_scores: Optional[List[float]] = None
    ) -> Tuple[List[str], List[float]]:
        """
        使用Reranker对检索结果重新排序
        
        Args:
            query: 查询字符串
            documents: 文档列表
            original_scores: 原始相似度分数（可选）
        
        Returns:
            (reranked_docs, reranked_scores): 重新排序后的文档和分数
        """
        logger.warning(f"🔵 _rerank_documents called: use_reranker={self.use_reranker}, model exists={self.reranker_model is not None}, num_docs={len(documents)}")
        
        if not self.use_reranker or self.reranker_model is None:
            # 如果没有reranker，返回原始结果
            logger.warning(f"⚠️ Reranker not available: use_reranker={self.use_reranker}, model is None={self.reranker_model is None}")
            scores = original_scores if original_scores else [1.0] * len(documents)
            return documents, scores
        
        if not documents:
            logger.warning("⚠️ No documents to rerank")
            return [], []
        
        try:
            logger.warning(f"🔄 Starting reranking for {len(documents)} documents...")
            # 构建query-document对
            pairs = [[query, doc] for doc in documents]
            
            # Tokenize
            with torch.no_grad():
                inputs = self.reranker_tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=512
                )
                
                # 移动到正确的设备
                if self.reranker_device == 'cuda':
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # 计算reranking分数
                outputs = self.reranker_model(**inputs, return_dict=True)
                rerank_scores = outputs.logits.view(-1).float().cpu().numpy().tolist()
            
            # 按分数降序排序
            indexed_docs = list(zip(documents, rerank_scores))
            indexed_docs.sort(key=lambda x: x[1], reverse=True)
            
            reranked_docs = [doc for doc, _ in indexed_docs]
            reranked_scores = [score for _, score in indexed_docs]
            
            logger.warning(
                f"✅ Reranked {len(documents)} documents. "
                f"Score range: [{min(reranked_scores):.3f}, {max(reranked_scores):.3f}]"
            )
            
            return reranked_docs, reranked_scores
            
        except Exception as e:
            logger.warning(f"❌ Reranking error: {e}")
            import traceback
            logger.warning(traceback.format_exc())
            # 如果reranking失败，返回原始结果
            scores = original_scores if original_scores else [1.0] * len(documents)
            logger.warning(f"⚠️ Returning original results due to reranking error")
            return documents, scores
    
    def _retrieve_internal(
        self,
        query: str,
        k: int = 3,
        return_scores: bool = False,
        where_filter: Optional[Dict] = None,
    ) -> Tuple[List[str], bool, Optional[List[float]]]:
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                where=where_filter or None,
                include=['documents', 'distances', 'metadatas']
            )

            documents = results['documents'][0] if results['documents'] else []
            distances = results['distances'][0] if results['distances'] else []
            metadatas = results['metadatas'][0] if results['metadatas'] else []

            similarities = [1.0 / (1.0 + d) for d in distances]
            
            # 注意：在应用reranking之前先检查基本成功条件
            # 如果初始检索失败，reranking也不会帮助
            success = self._check_success(similarities)

            if not success:
                logger.info(
                    "Retrieval failed for query: '%s...' (max similarity: %.3f)",
                    query[:50],
                    max(similarities) if similarities else 0.0,
                )
                return ([], False, [] if return_scores else None)

            formatted_docs = []
            doc_metadata = []
            for doc, meta in zip(documents, metadatas):
                meta = meta or {}
                source = meta.get('source') or meta.get('doc_id') or 'unknown'
                formatted_doc = f"[Source: {source}] {doc}"
                formatted_docs.append(formatted_doc)
                doc_metadata.append(meta)
            
            # 应用Reranking（如果启用）
            logger.warning(f"🔵 Reranker check: use_reranker={self.use_reranker}, model exists={self.reranker_model is not None}")
            if self.use_reranker and self.reranker_model is not None:
                logger.warning(f"✅ Applying reranker for query: '{query[:50]}...'")
                # 提取原始文档文本（不含Source标记）用于reranking
                original_docs = documents
                try:
                    reranked_docs, rerank_scores = self._rerank_documents(
                        query, original_docs, similarities
                    )
                    logger.warning(f"✅ Reranking completed: {len(reranked_docs)} documents reranked")
                except Exception as rerank_error:
                    logger.warning(f"❌ Reranking failed: {rerank_error}")
                    import traceback
                    logger.warning(traceback.format_exc())
                    # 如果reranking失败，使用原始结果
                    reranked_docs = original_docs
                    rerank_scores = similarities
                
                # 重新格式化文档（保持Source信息）
                reranked_formatted = []
                doc_map = {doc: (formatted, meta) for doc, formatted, meta in 
                          zip(original_docs, formatted_docs, doc_metadata)}
                
                for doc in reranked_docs:
                    if doc in doc_map:
                        reranked_formatted.append(doc_map[doc][0])
                
                formatted_docs = reranked_formatted
                similarities = rerank_scores
                
                # 使用WARNING级别确保日志输出
                logger.warning(
                    "✅ Retrieved and reranked %s documents for query: '%s...'",
                    len(formatted_docs),
                    query[:50],
                )
            else:
                # 使用WARNING级别输出，方便调试
                if not self.use_reranker:
                    logger.warning(f"⚠️ Reranker disabled (use_reranker=False) for query: '{query[:50]}...'")
                elif self.reranker_model is None:
                    logger.warning(f"⚠️ Reranker model is None (use_reranker={self.use_reranker}) for query: '{query[:50]}...' - reranking skipped")
                else:
                    logger.warning(f"⚠️ Reranker check failed (use_reranker={self.use_reranker}, model is None={self.reranker_model is None}) for query: '{query[:50]}...'")
                    
                logger.warning(
                    "Retrieved %s documents (NO reranking) for query: '%s...'",
                    len(formatted_docs),
                    query[:50],
                )

            return formatted_docs, True, similarities if return_scores else None

        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return ([], False, [] if return_scores else None)

    def retrieve(
        self,
        query: str,
        k: int = 3,
        return_scores: bool = False
    ) -> Tuple[List[str], bool, Optional[List[float]]]:
        """
        检索相关文档（单查询）
        
        Args:
            query: 查询字符串
            k: 返回文档数量
            return_scores: 是否返回相似度分数
        
        Returns:
            (docs, success, scores):
                - docs: 文档列表（如果失败则为空列表）
                - success: 是否成功检索
                - scores: 相似度分数列表（仅当return_scores=True）
        """
        return self._retrieve_internal(query, k, return_scores)

    def retrieve_with_filter(
        self,
        query: str,
        k: int = 3,
        section_filter: Optional[str] = None,
        work_group_filter: Optional[str] = None,
        return_scores: bool = False,
    ) -> Tuple[List[str], bool, Optional[List[float]]]:
        """检索时应用section/work group元数据过滤"""

        where_filter: Dict[str, str] = {}
        if section_filter:
            where_filter["section_id"] = section_filter
        if work_group_filter:
            where_filter["work_group"] = work_group_filter

        if where_filter:
            logger.debug(
                "Applying metadata filter: %s",
                {k: v for k, v in where_filter.items()},
            )

        return self._retrieve_internal(query, k, return_scores, where_filter or None)
    
    def _check_success(self, similarities: List[float]) -> bool:
        """
        判断检索是否成功
        
        两种模式：
        1. threshold: 基于最大相似度是否超过阈值
        2. random: 固定概率p_s（用于实验对比）
        
        Args:
            similarities: 相似度列表
        
        Returns:
            是否成功
        """
        if not similarities:
            return False
        
        if self.p_s_mode == "threshold":
            # 基于阈值
            max_sim = max(similarities)
            return max_sim >= self.similarity_threshold
        
        elif self.p_s_mode == "random":
            # 固定概率（用于实验）
            import random
            return random.random() < self.p_s_value
        
        else:
            raise ValueError(f"Unknown p_s_mode: {self.p_s_mode}")
    
    def batch_retrieve(
        self,
        queries: List[str],
        k: int = 3,
        return_scores: bool = False
    ) -> List[Tuple[List[str], bool, Optional[List[float]]]]:
        """
        批量检索（加速）
        
        Args:
            queries: 查询列表
            k: 每个查询返回的文档数
            return_scores: 是否返回分数
        
        Returns:
            结果列表，每个元素为 (docs, success, scores)
        """
        try:
            # 批量查询Chroma
            results = self.collection.query(
                query_texts=queries,
                n_results=k,
                include=['documents', 'distances', 'metadatas']
            )
            
            # 解析结果
            batch_results = []
            
            for i in range(len(queries)):
                documents = results['documents'][i] if results['documents'] else []
                distances = results['distances'][i] if results['distances'] else []
                metadatas = results['metadatas'][i] if results['metadatas'] else []
                
                # 转换为相似度
                similarities = [1.0 / (1.0 + d) for d in distances]
                
                # 判断成功
                success = self._check_success(similarities)
                
                if not success:
                    if return_scores:
                        batch_results.append(([], False, []))
                    else:
                        batch_results.append(([], False, None))
                    continue
                
                # 格式化文档
                formatted_docs = []
                doc_metadata = []
                for doc, meta in zip(documents, metadatas):
                    meta = meta or {}
                    source = meta.get('source', 'unknown')
                    formatted_doc = f"[Source: {source}] {doc}"
                    formatted_docs.append(formatted_doc)
                    doc_metadata.append(meta)
                
                # 应用Reranking（如果启用）
                if self.use_reranker and self.reranker_model is not None:
                    # 提取原始文档文本用于reranking
                    original_docs = documents
                    reranked_docs, rerank_scores = self._rerank_documents(
                        queries[i], original_docs, similarities
                    )
                    
                    # 重新格式化文档（保持Source信息）
                    reranked_formatted = []
                    doc_map = {doc: (formatted, meta) for doc, formatted, meta in 
                              zip(original_docs, formatted_docs, doc_metadata)}
                    
                    for doc in reranked_docs:
                        if doc in doc_map:
                            reranked_formatted.append(doc_map[doc][0])
                    
                    formatted_docs = reranked_formatted
                    similarities = rerank_scores
                
                if return_scores:
                    batch_results.append((formatted_docs, True, similarities))
                else:
                    batch_results.append((formatted_docs, True, None))
            
            logger.info(f"Batch retrieved for {len(queries)} queries")
            
            return batch_results
            
        except Exception as e:
            logger.error(f"Batch retrieval error: {e}")
            # 返回所有失败
            if return_scores:
                return [([], False, [])] * len(queries)
            else:
                return [([], False, None)] * len(queries)
    
    def get_statistics(self) -> Dict[str, any]:
        """
        获取检索器统计信息
        
        Returns:
            统计字典
        """
        stats = {
            'collection_name': self.collection_name,
            'total_documents': self.collection.count(),
            'similarity_threshold': self.similarity_threshold,
            'p_s_mode': self.p_s_mode,
            'p_s_value': self.p_s_value
        }
        
        return stats
    
    def search_by_metadata(
        self,
        metadata_filter: Dict,
        k: int = 10
    ) -> List[Dict]:
        """
        基于元数据过滤搜索（高级功能）
        
        Args:
            metadata_filter: 元数据过滤条件，例如 {"source": "O-RAN.WG4"}
            k: 返回数量
        
        Returns:
            匹配的文档列表
        """
        try:
            results = self.collection.get(
                where=metadata_filter,
                limit=k,
                include=['documents', 'metadatas']
            )
            
            docs = []
            for doc, meta in zip(results['documents'], results['metadatas']):
                docs.append({
                    'document': doc,
                    'metadata': meta
                })
            
            logger.info(f"Found {len(docs)} documents matching metadata filter")
            
            return docs
            
        except Exception as e:
            logger.error(f"Metadata search error: {e}")
            return []
    
    def generate_answer_from_docs(
        self,
        question: str,
        docs: List[str],
        model,
        tokenizer,
        max_length: int = 256,
        temperature: float = 0.3,
        top_p: float = 0.95,
        original_question: Optional[str] = None,
        options: Optional[List[str]] = None
    ) -> str:
        """
        基于检索文档生成答案（使用ARGO V2.0 Prompts）
        
        这个方法在检索成功后被调用，用于生成基于检索内容的中间答案。
        
        Args:
            question: 子查询问题
            docs: 检索到的文档列表
            model: LLM模型
            tokenizer: 对应的tokenizer
            max_length: 最大答案长度
            temperature: 生成温度
            top_p: nucleus sampling参数
            original_question: 原始问题（数据集全部是选择题，总是传递）
            options: 选择题选项列表（数据集全部是选择题，总是传递）
        
        Returns:
            生成的答案字符串
        """
        # 构建提示词（直接传递原始题目和选项，prompt内部会判断是否使用）
        prompt = ARGOPrompts.build_retrieval_answer_prompt(
            question=question,
            retrieved_docs=docs,
            original_question=original_question,
            options=options
        )
        
        # Tokenize
        device = next(model.parameters()).device
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(device)
        
        # 生成答案
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True if temperature > 0 else False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # 解码
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # 检查是否返回了"未找到信息"（支持多种格式）
        no_info_patterns = [
            "[No information found in O-RAN specs]",
            "No information found in O-RAN specs",
            "no information found in O-RAN specs",
            "```no information found in O-RAN specs```",
            "[No information found",
            "No information found"
        ]
        
        answer_lower = answer.lower()
        if any(pattern.lower() in answer_lower for pattern in no_info_patterns):
            # 检查是否只有"No information found"而没有其他有用信息
            # 如果答案主要是"No information found"，认为检索无效
            if len(answer.strip()) < 200:  # 如果答案很短，可能是纯"No information found"
                logger.warning(f"LLM indicated no information found for: {question[:50]}...")
                return ""
            # 如果答案较长，可能包含一些解释，保留但标记为低质量
            logger.warning(f"LLM indicated no information found (but answer has some content): {question[:50]}...")
            # 仍然返回，但会被标记为低质量
        
        logger.info(f"Generated answer from {len(docs)} docs: {answer[:100]}...")
        
        return answer


class MockRetriever(Retriever):
    """
    模拟检索器（用于测试，不依赖Chroma）
    
    模拟检索行为，返回固定内容，用于：
    1. 单元测试
    2. 无数据库环境的开发
    3. 快速原型验证
    """
    
    def __init__(
        self,
        p_s_value: float = 0.8,
        mock_docs: Optional[List[str]] = None
    ):
        """
        Args:
            p_s_value: 成功概率
            mock_docs: 模拟返回的文档列表
        """
        self.p_s_value = p_s_value
        self.mock_docs = mock_docs or [
            "O-RAN specifies latency requirements for different network segments.",
            "The fronthaul latency budget is typically 100-200 microseconds.",
            "Control plane latency should not exceed 10ms for RRC procedures."
        ]
        
        logger.info(f"MockRetriever initialized with p_s={p_s_value}")
    
    def retrieve(
        self,
        query: str,
        k: int = 3,
        return_scores: bool = False
    ) -> Tuple[List[str], bool, Optional[List[float]]]:
        """模拟检索"""
        import random
        
        # 随机成功/失败
        success = random.random() < self.p_s_value
        
        if not success:
            if return_scores:
                return [], False, []
            else:
                return [], False, None
        
        # 返回模拟文档
        docs = self.mock_docs[:k]
        scores = [0.9 - i*0.1 for i in range(len(docs))] if return_scores else None
        
        if return_scores:
            return docs, True, scores
        else:
            return docs, True, None
    
    def batch_retrieve(
        self,
        queries: List[str],
        k: int = 3,
        return_scores: bool = False
    ) -> List[Tuple[List[str], bool, Optional[List[float]]]]:
        """批量模拟检索"""
        return [self.retrieve(q, k, return_scores) for q in queries]
    
    def get_statistics(self) -> Dict[str, any]:
        """返回模拟统计"""
        return {
            'type': 'MockRetriever',
            'p_s_value': self.p_s_value,
            'num_mock_docs': len(self.mock_docs)
        }
