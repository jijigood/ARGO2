# Comparison: Current Exp_RAG.py vs. Query-Based Approach

## 🔄 Key Differences

### Current `Exp_RAG.py` (Simulation-Only)

**What it does:**
- ✗ NO actual text queries
- ✗ NO real document retrieval  
- ✗ NO LLM inference
- ✓ Simulates query complexity (1, 2, 3)
- ✓ Tests retrieval strategies in MDP environment
- ✓ Compares policies based on simulated accuracy

**Query "Generation":**
```python
# From Env_RAG.py
def generate_query(self):
    complexity = random.choices([1, 2, 3], weights=[0.4, 0.4, 0.2], k=1)[0]
    query_types = {
        1: "simple_fact",    # Just a label!
        2: "multi_concept", 
        3: "reasoning"       
    }
    return complexity, query_types[complexity]  # Returns (1, "simple_fact")
```

**Problem**: `"simple_fact"` is not a real query! It's just a label for logging.

---

### New `Exp_RAG_with_queries.py` (Query-Based)

**What it does:**
- ✓ Real O-RAN text queries
- ✓ Can integrate with actual retrieval (if RAG_Models available)
- ✓ Maps query complexity to retrieval strategy
- ✓ Provides traceable query-level results

**Query Generation:**
```python
# From query_generator.py
query_gen = QueryGenerator(seed=42)
query, complexity = query_gen.generate_query()

# Returns actual query text:
# ("What is O-RAN?", 1)
# ("How do A1 and E2 interfaces interact?", 2)
# ("Analyze latency requirements for Near-RT RIC...", 3)
```

---

## 📊 File Comparison Table

| Feature | `Exp_RAG.py` (Current) | `Exp_RAG_with_queries.py` (New) |
|---------|------------------------|----------------------------------|
| **Query Type** | Complexity labels only | Actual O-RAN text queries |
| **Query Examples** | "simple_fact", "reasoning" | "What is O-RAN?", "Explain A1 interface" |
| **Document Retrieval** | Simulated | Can use real vector store |
| **LLM Inference** | Not supported | Ready for integration |
| **Reproducibility** | By complexity distribution | By specific query text |
| **Analysis** | Policy-level only | Query-level + complexity-level |
| **Output** | Aggregate statistics | Per-query results + aggregates |

---

## 🔍 Where Queries Are Used

### In Current System (Simulation Flow)

```
Exp_RAG.py
  └─> Env_RAG.generate_query()
        └─> Returns: (2, "multi_concept")  # No actual text!
              └─> env.step(action)
                    └─> Simulated accuracy based on:
                          - Query complexity (2)
                          - Action (top_k=5, rerank=1)
                          - Random success
```

### In Query-Based System (Real Flow)

```
Exp_RAG_with_queries.py
  └─> QueryGenerator.generate_query()
        └─> Returns: ("How does O-RAN support network slicing?", 2)
              └─> RAG_Models.retrieval.retrieve(query, top_k=5)
                    └─> Vector search in document chunks
                          └─> LLM generates answer
                                └─> Evaluate accuracy (BLEU/ROUGE)
```

---

## 🎯 Where to Initialize RAG Queries

### Option 1: Use `query_generator.py` (Created)

```python
from query_generator import QueryGenerator

generator = QueryGenerator(seed=42)

# Single query
query, complexity = generator.generate_query()
print(query)  # "What is O-RAN?"

# Batch generation
queries = generator.generate_batch(100)
for query_text, complexity in queries:
    # Run RAG pipeline
    result = rag_system.query(query_text)
```

### Option 2: Load from File (Best for Real Evaluation)

```python
# queries.jsonl format
{"id": 1, "query": "What is O-RAN?", "complexity": 1, "ground_truth": "..."}
{"id": 2, "query": "Explain A1 interface", "complexity": 1, "ground_truth": "..."}
...

# Load and use
with open('queries.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        query = data['query']
        ground_truth = data['ground_truth']
        # Run RAG and evaluate
```

### Option 3: Interactive Mode

```python
# As shown in README.md
python run_local_rag.py "What is O-RAN architecture?" --show-context
```

---

## 🚀 How to Run with Real Queries

### Step 1: Test Query Generator

```bash
cd /home/data2/huangxiaolin2/ARGO
python query_generator.py
```

Output:
```
Simple Query Examples:
  1. What is O-RAN?
  2. Define the A1 interface.
  ...
```

### Step 2: Run Query-Based Experiment

```bash
python Exp_RAG_with_queries.py
```

This will:
1. Generate 100 real O-RAN queries
2. Test each policy on actual query text
3. Provide per-query results
4. Save to `draw_figs/data/query_based_experiment.json`

### Step 3: Compare with Simulation

```bash
# Old simulation-based
python Exp_RAG.py

# New query-based
python Exp_RAG_with_queries.py
```

---

## 📈 Output Comparison

### Current Output (Exp_RAG.py)

```json
{
  "optimal": {
    "avg_reward": 54.23,
    "avg_accuracy": 0.742,
    "avg_cost": 8.45,
    "success_rate": 0.856
  }
}
```

**Problem**: No traceability to specific queries!

### New Output (Exp_RAG_with_queries.py)

```json
{
  "queries": [
    {"text": "What is O-RAN?", "complexity": 1},
    {"text": "Explain A1 interface", "complexity": 1},
    ...
  ],
  "results": {
    "optimal": {
      "avg_reward": 54.23,
      "queries": [
        {
          "query_id": 0,
          "query_text": "What is O-RAN?",
          "complexity": 1,
          "action": [3, 0, 0],
          "accuracy": 0.85,
          "cost": 3.0,
          "reward": 82.0
        },
        ...
      ]
    }
  }
}
```

**Benefit**: Can analyze which queries succeeded/failed!

---

## 🎓 Summary

### Current State (Exp_RAG.py)
- **Mode**: Pure simulation
- **Queries**: Complexity labels only ("simple_fact", "reasoning")
- **Evaluation**: Statistical aggregates over random scenarios
- **Use Case**: Policy comparison in abstract MDP

### What's Missing
- ❌ Actual query text
- ❌ Real document retrieval
- ❌ LLM answer generation
- ❌ Per-query analysis

### Solution Provided
- ✅ `query_generator.py` - 30 predefined O-RAN queries
- ✅ `Exp_RAG_with_queries.py` - Query-based experiment framework
- ✅ Ready for real RAG integration
- ✅ Per-query result tracking

### Next Steps to Complete RAG Pipeline

1. **Implement actual retrieval**:
   ```python
   from RAG_Models.retrieval import build_vector_store
   vector_store, retriever = build_vector_store()
   docs = retriever.retrieve(query_text, top_k=5)
   ```

2. **Add LLM inference**:
   ```python
   from RAG_Models.answer_generator import LocalLLMAnswerGenerator
   llm = LocalLLMAnswerGenerator()
   answer = llm.generate_answer(query_text, docs)
   ```

3. **Evaluate accuracy**:
   ```python
   from evaluate import load
   rouge = load('rouge')
   score = rouge.compute(predictions=[answer], references=[ground_truth])
   ```

---

**The key insight**: `Exp_RAG.py` tests **retrieval strategies** but doesn't test **retrieval** itself. It's like testing a car's GPS algorithm without ever driving the car!
