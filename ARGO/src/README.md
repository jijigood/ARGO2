# ARGO Core Modules

`src/` contains the production implementation of the ARGO stack.

Key entry points:
- `argo_system.py` – wires QueryDecomposer, Retriever, Reasoner, and AnswerSynthesizer with MDP thresholds and progress tracking.
- `retriever.py` / `enhanced_retriever.py` – knowledge base accessors (Chroma, mock, or hybrid modes).
- `decomposer.py`, `synthesizer.py`, `complexity.py`, `progress.py` – supporting components.
- `Env_RAG.py` (root) – RL-style environment for experimentation.

When creating subpackages (e.g., `src/argo/retrieval/`), add `__init__.py` and update this README to describe ownership and dependencies.
