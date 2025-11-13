# ORAN-Bench-13K RAG 评估系统 - 文件索引

## 📁 核心文件

### 1. 基准数据加载器
**文件**: `oran_benchmark_loader.py`  
**状态**: ✅ 完成并测试  
**行数**: ~200 行  
**功能**: 加载和管理 ORAN-Bench-13K 数据集  

**核心类/函数**:
- `ORANBenchmark()` - 主类
  - `_load_questions(filename)` - 加载 JSONL 文件
  - `sample_questions(n, difficulty, seed)` - 采样问题
  - `format_question_for_llm(question)` - 格式化为提示
  - `check_answer(question, predicted)` - 验证答案

**使用示例**:
```python
from oran_benchmark_loader import ORANBenchmark
benchmark = ORANBenchmark()
questions = benchmark.sample_questions(n=50, difficulty='medium', seed=42)
```

---

### 2. RAG 评估框架
**文件**: `Exp_RAG_benchmark.py`  
**状态**: ✅ 完成（模拟模式）  
**行数**: ~340 行  
**功能**: 评估不同检索策略在基准上的表现  

**核心函数**:
- `extract_answer_number(llm_output)` - 提取答案数字
- `evaluate_rag_on_benchmark(benchmark, questions, config)` - 评估 RAG 系统
- `run_benchmark_experiment(n_questions, difficulty, seed)` - 运行完整实验
- `analyze_by_difficulty(results, questions)` - 难度级别分析
- `save_results(results, questions, filename)` - 保存结果

**使用示例**:
```python
from Exp_RAG_benchmark import run_benchmark_experiment, save_results
results, questions = run_benchmark_experiment(n_questions=100, seed=42)
save_results(results, questions, 'my_results.json')
```

**支持的策略**:
1. `optimal` - MDP 最优策略
2. `fixed_k3` - 固定检索 3 个文档
3. `fixed_k5` - 固定检索 5 个文档
4. `fixed_k7` - 固定检索 7 个文档
5. `adaptive` - 自适应策略

---

### 3. 结果可视化
**文件**: `plot_benchmark_results.py`  
**状态**: ✅ 完成  
**行数**: ~250 行  
**功能**: 生成评估结果的可视化图表  

**生成的图表**:
1. `benchmark_strategy_comparison.png` - 策略对比（柱状图）
2. `benchmark_difficulty_breakdown.png` - 难度级别分解（分组柱状图）
3. `benchmark_confusion_fixed_k5.png` - 混淆矩阵（热力图）
4. `benchmark_retrieval_impact.png` - 检索深度影响（折线图）
5. `benchmark_summary.txt` - 文本摘要

**使用示例**:
```bash
python plot_benchmark_results.py
```

---

### 4. 真实 RAG 集成模板
**文件**: `integrate_real_rag.py`  
**状态**: 📝 模板（待测试）  
**行数**: ~300 行  
**功能**: 集成 Qwen2.5-14B-Instruct 和检索器的示例代码  

**核心函数**:
- `load_qwen_model(model_path)` - 加载 LLM 模型
- `load_retriever()` - 加载向量检索器
- `rag_inference(model, tokenizer, retriever, question, top_k)` - RAG 推理
- `evaluate_with_real_rag(model, tokenizer, retriever, questions)` - 批量评估

**使用示例**:
```python
from integrate_real_rag import load_qwen_model, load_retriever, rag_inference

# 加载模型
model, tokenizer = load_qwen_model()
retriever = load_retriever()

# 推理
llm_output = rag_inference(model, tokenizer, retriever, question, top_k=5)
```

---

### 5. 快速启动脚本
**文件**: `run_benchmark_eval.sh`  
**状态**: ✅ 完成  
**类型**: Bash 脚本  
**功能**: 一键运行所有评估步骤  

**执行步骤**:
1. 检查基准数据
2. 测试加载器
3. 运行评估实验
4. 生成可视化
5. 显示结果摘要

**使用方法**:
```bash
cd /home/data2/huangxiaolin2/ARGO
./run_benchmark_eval.sh
```

---

## 📚 文档文件

### 1. 完整使用指南
**文件**: `ORAN_BENCHMARK_README.md`  
**行数**: ~400 行  
**内容**:
- 系统概述
- 核心组件详解
- 实验结果分析
- 使用指南（API + 命令行）
- 集成真实 RAG 的步骤
- 文件结构说明

---

### 2. 项目总结
**文件**: `ORAN_BENCHMARK_SUMMARY.md`  
**行数**: ~300 行  
**内容**:
- 项目目标
- 已完成工作清单
- 实验结果和关键发现
- 技术实现细节
- 下一步计划
- 与 ARGO_MDP 的关系

---

### 3. 文件索引（本文件）
**文件**: `ORAN_BENCHMARK_INDEX.md`  
**功能**: 快速导航所有项目文件  

---

## 📊 数据文件

### 1. ORAN-Bench-13K 数据集
**位置**: `ORAN-Bench-13K/Benchmark/`  
**格式**: JSONL（每行一个 JSON 数组）  

**文件列表**:
- `fin_E.json` - 1,139 Easy 问题
- `fin_M.json` - 9,570 Medium 问题
- `fin_H.json` - 3,243 Hard 问题

**数据格式**:
```json
[
  "问题文本",
  ["1. 选项1", "2. 选项2", "3. 选项3", "4. 选项4"],
  "正确答案索引 (1-4)"
]
```

---

### 2. 评估结果
**位置**: `draw_figs/data/`  
**格式**: JSON  

**文件示例**:
- `oran_benchmark_mixed.json` - 混合难度评估结果
- `oran_benchmark_easy.json` - Easy 难度评估结果
- `oran_benchmark_medium.json` - Medium 难度评估结果
- `oran_benchmark_hard.json` - Hard 难度评估结果

**结果结构**:
```json
{
  "timestamp": "2025-10-28T11:20:35.843380",
  "benchmark": "ORAN-Bench-13K",
  "num_questions": 100,
  "results": {
    "optimal": {
      "correct": 74,
      "total": 100,
      "accuracy": 0.74,
      "details": [...]
    },
    ...
  }
}
```

---

## 🖼️ 可视化输出

**位置**: `draw_figs/`  

| 文件名 | 类型 | 描述 | 大小 |
|-------|------|------|------|
| `benchmark_strategy_comparison.png` | 图表 | 策略准确率对比柱状图 | 154 KB |
| `benchmark_difficulty_breakdown.png` | 图表 | 难度级别分组柱状图 | 165 KB |
| `benchmark_confusion_fixed_k5.png` | 图表 | 答案混淆矩阵热力图 | 129 KB |
| `benchmark_retrieval_impact.png` | 图表 | 检索深度影响折线图 | 158 KB |
| `benchmark_summary.txt` | 文本 | 策略排名摘要 | 712 B |

---

## 🔧 依赖配置

### Python 环境
**路径**: `/root/miniconda/envs/ARGO/bin/python`  
**版本**: Python 3.11  

### 必需包（当前）
```
numpy
matplotlib
json (内置)
pathlib (内置)
datetime (内置)
```

### 可选包（真实 RAG）
```
torch
transformers
langchain-community
sentence-transformers
chromadb
```

---

## 🚀 快速开始

### 方案 1: 一键运行（推荐）
```bash
cd /home/data2/huangxiaolin2/ARGO
./run_benchmark_eval.sh
```

### 方案 2: 分步执行
```bash
# 1. 测试加载器
/root/miniconda/envs/ARGO/bin/python oran_benchmark_loader.py

# 2. 运行评估
/root/miniconda/envs/ARGO/bin/python Exp_RAG_benchmark.py

# 3. 生成可视化
/root/miniconda/envs/ARGO/bin/python plot_benchmark_results.py
```

### 方案 3: Python API
```python
from oran_benchmark_loader import ORANBenchmark
from Exp_RAG_benchmark import run_benchmark_experiment, save_results

# 加载基准
benchmark = ORANBenchmark()

# 运行评估
results, questions = run_benchmark_experiment(n_questions=100, seed=42)

# 保存结果
save_results(results, questions, 'my_results.json')
```

---

## 📈 实验结果速览

### 最佳配置（模拟）
- **策略**: Fixed K=5
- **准确率**: 85.0% (85/100)
- **难度分布**: Easy: 85.7%, Medium: 88.7%, Hard: 72.7%

### 策略排名
1. Fixed K=5 - 85.0%
2. Fixed K=7 - 81.0%
3. Optimal - 74.0%
4. Fixed K=3 - 73.0%
5. Adaptive - 68.0%

---

## 🔄 工作流程

```
1. 数据加载
   oran_benchmark_loader.py
   ↓
2. RAG 评估
   Exp_RAG_benchmark.py
   ↓
3. 结果保存
   draw_figs/data/*.json
   ↓
4. 可视化生成
   plot_benchmark_results.py
   ↓
5. 输出图表
   draw_figs/*.png
```

---

## 🎯 下一步行动

### 立即可做
- [x] 测试基准加载器
- [x] 运行模拟评估
- [x] 生成可视化图表
- [ ] 集成真实 LLM (integrate_real_rag.py)

### 进阶任务
- [ ] 多 GPU 并行推理
- [ ] 错误案例分析
- [ ] 检索质量评估
- [ ] 领域适应研究

---

## 📞 帮助与支持

| 需求 | 参考文件 |
|-----|---------|
| 快速上手 | `run_benchmark_eval.sh` |
| 详细文档 | `ORAN_BENCHMARK_README.md` |
| 项目总结 | `ORAN_BENCHMARK_SUMMARY.md` |
| API 使用 | 各文件顶部的 docstring |
| 真实 RAG 集成 | `integrate_real_rag.py` |

---

**最后更新**: 2025-10-28  
**项目版本**: 1.0  
**状态**: ✅ 核心功能完成，可投入使用
