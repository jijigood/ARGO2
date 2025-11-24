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
        p_s_value: float = 0.8  # 仅在random模式下使用
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
        """
        self.chroma_dir = Path(chroma_dir)
        self.collection_name = collection_name
        self.similarity_threshold = similarity_threshold
        self.p_s_mode = p_s_mode
        self.p_s_value = p_s_value
        
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
        
        logger.info(
            f"Retriever initialized: mode={p_s_mode}, "
            f"threshold={similarity_threshold:.3f}, p_s={p_s_value:.2f}"
        )
    
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
            success = self._check_success(similarities)

            if not success:
                logger.info(
                    "Retrieval failed for query: '%s...' (max similarity: %.3f)",
                    query[:50],
                    max(similarities) if similarities else 0.0,
                )
                return ([], False, [] if return_scores else None)

            formatted_docs = []
            for doc, meta in zip(documents, metadatas):
                meta = meta or {}
                source = meta.get('source') or meta.get('doc_id') or 'unknown'
                formatted_doc = f"[Source: {source}] {doc}"
                formatted_docs.append(formatted_doc)

            logger.info(
                "Retrieved %s documents for query: '%s...'",
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
                for doc, meta in zip(documents, metadatas):
                    source = meta.get('source', 'unknown')
                    formatted_doc = f"[Source: {source}] {doc}"
                    formatted_docs.append(formatted_doc)
                
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
        top_p: float = 0.95
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
        
        Returns:
            生成的答案字符串
        """
        # 构建提示词
        prompt = ARGOPrompts.build_retrieval_answer_prompt(
            question=question,
            retrieved_docs=docs
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
        
        # 检查是否返回了"未找到信息"
        if "[No information found in O-RAN specs]" in answer:
            logger.warning(f"LLM indicated no information found for: {question[:50]}...")
            return ""
        
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
