"""
Document Loader for O-RAN Standards
加载和预处理O-RAN文档
"""
import os
import re
from typing import List, Dict, Optional
import json

class ORANDocumentLoader:
    """O-RAN标准文档加载器"""
    
    def __init__(self, docs_dir: str = "../ORAN_Docs"):
        self.docs_dir = docs_dir
        self.documents = []
        
    def load_sample_documents(self) -> List[Dict]:
        """加载示例O-RAN文档（用于初始测试）"""
        sample_docs = [
            {
                "doc_id": "ORAN_WG1_001",
                "title": "O-RAN Architecture Description",
                "content": "The O-RAN architecture consists of Non-Real Time RIC (Non-RT RIC), Near-Real Time RIC (Near-RT RIC), O-CU (O-RAN Central Unit), O-DU (O-RAN Distributed Unit) and O-RU (O-RAN Radio Unit). The RIC provides intelligent control and optimization of RAN functions.",
                "category": "architecture",
                "complexity": 2  # 1=简单, 2=中等, 3=复杂
            },
            {
                "doc_id": "ORAN_WG2_002",
                "title": "A1 Interface Specification",
                "content": "The A1 interface is defined between Non-RT RIC and Near-RT RIC. It is used for policy-based guidance, ML model management, and enrichment information. The A1 interface supports both policy and enrichment information delivery.",
                "category": "interface",
                "complexity": 3
            },
            {
                "doc_id": "ORAN_WG3_003",
                "title": "E2 Interface and Service Model",
                "content": "E2 interface connects Near-RT RIC with E2 nodes (O-CU-CP, O-CU-UP, O-DU). E2 Service Models (E2SM) define specific functionalities like KPM (Key Performance Measurement), RC (RAN Control), and NI (Network Interface).",
                "category": "interface",
                "complexity": 3
            },
            {
                "doc_id": "ORAN_WG4_004",
                "title": "Fronthaul Interface Specifications",
                "content": "The fronthaul interface connects O-DU and O-RU. It uses eCPRI protocol for user plane and C-plane. Split option 7-2x is commonly used, where O-DU handles MAC/RLC and O-RU handles PHY low and RF.",
                "category": "fronthaul",
                "complexity": 3
            },
            {
                "doc_id": "ORAN_WG5_005",
                "title": "Open F1 Interface",
                "content": "F1 interface is defined between O-CU and O-DU. F1-C carries control plane signaling, while F1-U carries user plane data. It follows 3GPP specifications with O-RAN enhancements.",
                "category": "interface",
                "complexity": 2
            },
            {
                "doc_id": "ORAN_WG6_006",
                "title": "Cloudification and Orchestration",
                "content": "O-RAN supports cloud-native deployment using containerization (Docker/Kubernetes). SMO (Service Management and Orchestration) manages the lifecycle of O-RAN components. CNF (Cloud Native Functions) deployment is supported.",
                "category": "deployment",
                "complexity": 2
            },
            {
                "doc_id": "ORAN_WG7_007",
                "title": "xApps and rApps Development",
                "content": "xApps run on Near-RT RIC with latency requirements < 1 second. rApps run on Non-RT RIC with latency > 1 second. xApps use E2 interface for RAN control. rApps use A1 interface for policy management.",
                "category": "application",
                "complexity": 2
            },
            {
                "doc_id": "ORAN_WG8_008",
                "title": "Security Architecture",
                "content": "O-RAN security includes authentication, authorization, encryption, and integrity protection. Zero-trust principles are applied. Security interfaces include TLS for E2, A1, and O1 interfaces.",
                "category": "security",
                "complexity": 2
            },
            {
                "doc_id": "ORAN_WG9_009",
                "title": "O1 Interface for Management",
                "content": "O1 interface is used for configuration management, fault management, and performance management. It uses NETCONF/YANG or REST APIs. Supports FCAPS (Fault, Configuration, Accounting, Performance, Security) management.",
                "category": "management",
                "complexity": 2
            },
            {
                "doc_id": "ORAN_WG10_010",
                "title": "Testing and Integration",
                "content": "O-RAN ALLIANCE provides test specifications and conformance testing. Plugfests validate interoperability between vendors. OTIC (O-RAN Testing and Integration Centers) provide testing facilities.",
                "category": "testing",
                "complexity": 1
            },
            {
                "doc_id": "3GPP_TS_38.300",
                "title": "5G NR Overall Architecture",
                "content": "5G NR architecture includes gNB (5G base station) with CU-DU split. CU handles PDCP and RRC, while DU handles RLC, MAC, and PHY. The architecture supports NSA and SA deployment modes.",
                "category": "3gpp",
                "complexity": 2
            },
            {
                "doc_id": "3GPP_TS_38.401",
                "title": "NG-RAN Architecture",
                "content": "NG-RAN consists of gNBs connected to 5GC via NG interface. Xn interface connects gNBs for inter-cell coordination. Supports dual connectivity and carrier aggregation.",
                "category": "3gpp",
                "complexity": 2
            }
        ]
        
        self.documents = sample_docs
        return sample_docs
    
    def load_from_directory(self, file_extension: str = ".txt") -> List[Dict]:
        """从目录加载文档文件"""
        documents = []
        if not os.path.exists(self.docs_dir):
            print(f"Warning: {self.docs_dir} not found. Using sample documents.")
            return self.load_sample_documents()
        
        for filename in os.listdir(self.docs_dir):
            if filename.endswith(file_extension):
                filepath = os.path.join(self.docs_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append({
                        "doc_id": filename.replace(file_extension, ""),
                        "title": filename,
                        "content": content,
                        "category": "unknown",
                        "complexity": 2,
                        "source_path": filepath
                    })
        
        self.documents = documents if documents else self.load_sample_documents()
        return self.documents
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """根据ID获取文档"""
        for doc in self.documents:
            if doc["doc_id"] == doc_id:
                return doc
        return None
    
    def get_documents_by_category(self, category: str) -> List[Dict]:
        """根据类别获取文档"""
        return [doc for doc in self.documents if doc["category"] == category]
    
    def get_statistics(self) -> Dict:
        """获取文档统计信息"""
        categories = {}
        complexity_dist = {1: 0, 2: 0, 3: 0}
        
        for doc in self.documents:
            cat = doc.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
            complexity = doc.get("complexity", 2)
            complexity_dist[complexity] = complexity_dist.get(complexity, 0) + 1
        
        return {
            "total_documents": len(self.documents),
            "categories": categories,
            "complexity_distribution": complexity_dist,
            "avg_content_length": sum(len(doc["content"]) for doc in self.documents) / len(self.documents) if self.documents else 0
        }


def test_loader():
    """测试文档加载器"""
    loader = ORANDocumentLoader()
    docs = loader.load_sample_documents()
    
    print("=" * 60)
    print("O-RAN Document Loader Test")
    print("=" * 60)
    print(f"\nLoaded {len(docs)} documents")
    
    stats = loader.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Categories: {stats['categories']}")
    print(f"  Complexity distribution: {stats['complexity_distribution']}")
    print(f"  Avg content length: {stats['avg_content_length']:.1f} chars")
    
    print(f"\nSample document:")
    print(f"  ID: {docs[0]['doc_id']}")
    print(f"  Title: {docs[0]['title']}")
    print(f"  Category: {docs[0]['category']}")
    print(f"  Content: {docs[0]['content'][:100]}...")


if __name__ == "__main__":
    test_loader()
