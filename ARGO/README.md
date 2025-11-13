# ARGO: Adaptive Retrieval with Guided Optimization

ARGO 项目将 O-RAN 领域的 RAG（Retrieval-Augmented Generation）系统建模为一个马尔可夫决策过程（MDP），
对比多种检索策略在检索准确率与计算成本之间的权衡。项目结构与 `TAoI_jour` 相呼应，提供从文档加载、向量化、检索到策略仿真和可视化的完整流程。

## 项目结构

```
ARGO/
├── Env_RAG.py                # RAG 检索策略的 MDP 环境与策略测试函数
├── Exp_RAG.py                # 实验脚本：策略对比、成本敏感性与多随机种子实验
├── requirements.txt          # 项目依赖
├── README.md                 # 项目说明
├── RAG_Models/
│   ├── __init__.py           # RAG 模块包声明
│   ├── document_loader.py    # O-RAN 文档加载与统计
│   ├── embeddings.py         # 文本分块与嵌入模型封装

## 本地模型配置

针对 8×8GB GPU 的离线环境，推荐选择 15B 以下、可 4-bit 量化的模型：

- **嵌入模型**：`sentence-transformers/all-MiniLM-L6-v2`（默认，~120MB）；如需中文增强可用 `BAAI/bge-base-en-v1.5`。
- **LLM**：`Qwen/Qwen2.5-7B-Instruct`（4-bit ≈ 5GB），或 `Meta-Llama-3-8B-Instruct`（4-bit ≈ 6GB）。

### 1. 准备运行环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

确保 `nvidia-smi` 能看到 8 张 8GB GPU，安装 `bitsandbytes` 后运行 `python -c "import bitsandbytes as bnb; print(bnb.__version__)"` 验证。

### 2. 预下载模型（可选）

```bash
huggingface-cli login   # 若需，从本地HF镜像下载可跳过

# 嵌入模型
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --local-dir ./models/all-MiniLM-L6-v2

# LLM（示例：Qwen 2.5 7B 指令）
│   ├── retrieval.py          # 向量检索与检索器封装
```

设置共享缓存目录（可放在高速 SSD）：

```bash
export ARGO_HF_CACHE=$(pwd)/.hf_cache
```

### 3. 配置环境变量（可写入 `.env` 或 shell）

```bash
# 嵌入模型
export ARGO_EMBEDDING_MODEL=$(pwd)/models/all-MiniLM-L6-v2
export ARGO_EMBEDDING_DEVICE=cuda

# LLM
export ARGO_LLM_MODEL=$(pwd)/models/qwen2.5-7b-instruct
export ARGO_LLM_DEVICE_MAP=auto    # 让 transformers 自动划分到多卡
export ARGO_LLM_4BIT=true          # 默认开启4-bit量化
export ARGO_LLM_MAX_NEW_TOKENS=512
```

若使用 huggingface 缓存，可将 `ARGO_EMBEDDING_MODEL`/`ARGO_LLM_MODEL` 设置为模型名称（无需本地路径）。

### 4. 构建向量库并运行本地 RAG

```bash
python run_local_rag.py "What is O-RAN architecture?" --show-context
```

- 首次运行会在 `Environments/vector_store.pkl` 下生成向量库；如需重新构建，添加 `--rebuild`。
- `--docs-dir` 可指向真实文档目录，默认读取 `../ORAN_Docs` 中的 `.txt` 文件。
- `--llm-model`、`--embedding-model` 可覆盖环境变量配置；`--device-map balanced` 可在 8 卡之间均衡分配 VRAM。
- 若希望直连 GPU ID，例如仅用前两卡，可设置 `--device-map "{'': 'cuda:0'}"` 或在环境变量中传入 JSON 字符串。

### 5. 生成结果

脚本会打印：

1. 检索到的 chunk（可选 `--show-context`）。
2. LLM 答案与元数据（使用的模型、top-k 等）。

若需将该流程嵌入其他实验，可在 Python 代码中直接调用：

```python
from RAG_Models.retrieval import build_vector_store
from RAG_Models.answer_generator import LocalLLMAnswerGenerator

vector_store, retriever = build_vector_store(save_path="./Environments/vector_store.pkl")
llm = LocalLLMAnswerGenerator()
context = retriever.get_context("Explain the A1 interface", top_k=3)
result = llm.generate_answer("Explain the A1 interface", context)
print(result["answer"])
```

### 6. 故障排查

- 若加载 LLM 提示 VRAM 不足，可降低 `ARGO_LLM_MAX_NEW_TOKENS`、`--top-k` 或选择更小的 4-bit 模型（如 3B-7B）。
- 如果 `bitsandbytes` 编译失败，可改用 `pip install bitsandbytes==0.43.1 --extra-index-url=https://download.pytorch.org/whl/cu124`（按 CUDA 版本调整）。
- 对需自定义代码的模型（如 Qwen）若报错，可附加 `--trust-remote-code`。
├── ORAN_Docs/                # 预留的 O-RAN 文档目录
├── draw_figs/
│   └── data/                 # 实验输出数据（JSON/CSV）
└── figs/                     # 图表输出（待生成人）
```

## RAG 流程概述

1. **文档加载（`RAG_Models/document_loader.py`）**
   - 模拟 O-RAN 标准文档（架构、接口、安全、测试等）并提供统计分析
   - 可扩展到真实的 `.txt`/`.pdf` 文档
2. **文本分块与嵌入（`RAG_Models/embeddings.py`）**
   - 自定义 `TextChunker` 进行句子级分块
   - 通过 `EmbeddingModel` 调用 `sentence-transformers` 生成向量（提供 mock 回退）
3. **向量存储与检索（`RAG_Models/retrieval.py`）**
   - 封装向量库、相似度检索、过滤与查询统计
   - `build_vector_store()` 一键构建向量库并返回 `Retriever`
4. **MDP 环境（`Env_RAG.py`）**
   - 状态：`(查询复杂度, 历史检索质量, 剩余预算)`
   - 动作：`(top_k, use_rerank, use_filter)`
   - 奖励：`accuracy * 100 - cost_weight * cost`
   - 预置策略：最优阈值策略、固定 top-k、自适应策略、随机策略
5. **实验脚本（`Exp_RAG.py`）**
   - **策略对比实验**：最优 vs 固定 top-k vs 自适应 vs 随机
   - **成本敏感性实验**：不同成本权重下的性能变化
   - **多随机种子实验**：稳健性分析并写入 `draw_figs/data/`

## 快速开始

1. **安装依赖**
   ```powershell
   cd e:\VS\ARGO
   pip install -r requirements.txt
   ```
2. **构建向量库 & 检索测试**
   ```powershell
   python -m RAG_Models.retrieval
   ```
3. **运行 MDP 实验**
   ```powershell
   python Exp_RAG.py
   ```
   输出结果将保存到 `draw_figs/data/` 目录。

## 技术亮点

- **MDP 建模**：利用状态-动作空间捕获检索策略的动态调整
- **可扩展的 RAG 模块**：支持真实 O-RAN 标准库与 LLM 接入
- **策略对比**：量化不同检索策略在准确率、成本和成功率上的差异
- **实验自动化**：适配论文/报告的数据导出与可视化脚本

## 下一步计划

1. 接入真实 O-RAN 文档（PDF/HTML）并完善解析逻辑
2. 增加基于 LLM 的回答模块，实现完整的问答链路
3. 引入强化学习算法，学习最优检索策略
4. 扩展绘图脚本，自动生成横向对比图表

如需进一步定制或扩展，请告知具体需求！
