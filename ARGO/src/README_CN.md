# ARGO 核心模块

`src/` 包含 ARGO 堆栈的生产实现。

主要入口点：
- `argo_system.py` – 将 QueryDecomposer、Retriever、Reasoner 和 AnswerSynthesizer 与 MDP 阈值和进度跟踪连接起来。
- `retriever.py` / `enhanced_retriever.py` – 知识库访问器（Chroma、模拟或混合模式）。
- `decomposer.py`、`synthesizer.py`、`complexity.py`、`progress.py` – 支持组件。
- `Env_RAG.py`（根目录）– 用于实验的强化学习风格环境。

创建子包（例如 `src/argo/retrieval/`）时，添加 `__init__.py` 并更新此 README 以描述所有权和依赖关系。
