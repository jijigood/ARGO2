"""
Query Generator for RAG Experiments
Generates actual O-RAN domain queries for testing
"""
import random
from typing import List, Tuple


class QueryGenerator:
    """Generate O-RAN domain queries with complexity labels"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        
        # Simple fact-based queries
        self.simple_queries = [
            "What is O-RAN?",
            "Define the A1 interface.",
            "What does RIC stand for?",
            "List the main O-RAN Alliance workgroups.",
            "What is the role of SMO in O-RAN?",
            "Explain the Near-RT RIC.",
            "What is the O1 interface used for?",
            "Define xApp in O-RAN context.",
            "What is the E2 interface?",
            "Name the O-RAN fronthaul specifications."
        ]
        
        # Multi-concept queries
        self.medium_queries = [
            "How do A1 and E2 interfaces interact in O-RAN architecture?",
            "Compare Near-RT RIC and Non-RT RIC functionalities.",
            "What is the relationship between SMO and RIC in O-RAN?",
            "Explain how xApps and rApps differ in deployment.",
            "Describe the O-RAN protocol stack for fronthaul.",
            "How does O-RAN support network slicing?",
            "What are the security considerations for O1 interface?",
            "Explain RAN intelligent control loop in O-RAN.",
            "How does O-RAN enable multi-vendor interoperability?",
            "Describe the role of O-Cloud in O-RAN architecture."
        ]
        
        # Complex reasoning queries
        self.complex_queries = [
            "How can AI/ML models be deployed via A1 interface to optimize radio resource management in a multi-vendor O-RAN network?",
            "Analyze the end-to-end latency requirements for Near-RT RIC control loops and their impact on 5G URLLC services.",
            "What are the trade-offs between centralized and distributed RAN intelligence in O-RAN, and how do they affect network performance?",
            "Propose an xApp design for interference mitigation in dense urban scenarios using E2 interface measurements.",
            "How can O-RAN architecture support dynamic spectrum sharing between 4G and 5G networks?",
            "Evaluate the security implications of open interfaces in O-RAN and recommend mitigation strategies.",
            "Design a closed-loop automation workflow using SMO, Non-RT RIC, and Near-RT RIC for energy efficiency optimization.",
            "How does O-RAN's disaggregated architecture impact the deployment and orchestration of network functions?",
            "Analyze the challenges of implementing O-RAN in rural areas with limited backhaul capacity.",
            "Propose a multi-vendor integration testing strategy for O-RAN components ensuring E2 interface compatibility."
        ]
    
    def generate_query(self, complexity: int = None) -> Tuple[str, int]:
        """
        Generate a query with specified or random complexity
        
        Args:
            complexity: 1=simple, 2=medium, 3=complex, None=random
            
        Returns:
            (query_text, complexity_level)
        """
        if complexity is None:
            # Random with distribution: 40% simple, 40% medium, 20% complex
            complexity = random.choices([1, 2, 3], weights=[0.4, 0.4, 0.2], k=1)[0]
        
        if complexity == 1:
            query = random.choice(self.simple_queries)
        elif complexity == 2:
            query = random.choice(self.medium_queries)
        elif complexity == 3:
            query = random.choice(self.complex_queries)
        else:
            raise ValueError(f"Invalid complexity: {complexity}")
        
        return query, complexity
    
    def generate_batch(self, n: int = 10) -> List[Tuple[str, int]]:
        """Generate a batch of queries"""
        return [self.generate_query() for _ in range(n)]
    
    def get_query_by_id(self, query_id: int) -> Tuple[str, int]:
        """Get specific query by ID for reproducibility"""
        all_queries = (
            [(q, 1) for q in self.simple_queries] +
            [(q, 2) for q in self.medium_queries] +
            [(q, 3) for q in self.complex_queries]
        )
        
        if 0 <= query_id < len(all_queries):
            return all_queries[query_id]
        else:
            raise ValueError(f"Query ID {query_id} out of range [0, {len(all_queries)-1}]")


# Example usage
if __name__ == "__main__":
    generator = QueryGenerator(seed=42)
    
    print("=" * 80)
    print("Query Generator Test")
    print("=" * 80)
    
    # Test different complexity levels
    for complexity in [1, 2, 3]:
        complexity_names = {1: "Simple", 2: "Medium", 3: "Complex"}
        print(f"\n{complexity_names[complexity]} Query Examples:")
        for i in range(3):
            query, level = generator.generate_query(complexity)
            print(f"  {i+1}. {query}")
    
    # Test random generation
    print("\n" + "=" * 80)
    print("Random Batch Generation (10 queries):")
    print("=" * 80)
    batch = generator.generate_batch(10)
    for i, (query, complexity) in enumerate(batch):
        complexity_label = {1: "[SIMPLE]", 2: "[MEDIUM]", 3: "[COMPLEX]"}[complexity]
        print(f"{i+1}. {complexity_label} {query}")
