# 文件清单 - ORAN QA提取工具

## 📁 目录结构

```
/data/user/huangxiaolin/ARGO2/ARGO/TeleQnA/
├── TeleQnA.txt                      # [输入] 原始数据集 (106,324个问题)
├── extract_oran_qa.py               # [脚本] 基础版提取脚本
├── extract_oran_qa_enhanced.py      # [脚本] 增强版提取脚本 (推荐)
├── test_extraction.py               # [脚本] 快速测试脚本
├── run_extraction.sh                # [脚本] 运行完整提取
├── run_test.sh                      # [脚本] 运行快速测试
├── requirements.txt                 # [配置] Python依赖
├── README.md                        # [文档] 项目说明
├── QUICKSTART.md                    # [文档] 快速开始指南
├── FILES.md                         # [文档] 本文件清单
├── TeleQnA_ORAN_only.json          # [输出] 提取的ORAN问题
├── extraction_log.txt               # [输出] 提取日志
├── checkpoint.json                  # [临时] 检查点文件 (断点续传)
└── progress.json                    # [临时] 进度文件
```

## 📄 文件说明

### 输入文件

#### `TeleQnA.txt`
- **类型**: 输入数据集
- **格式**: JSON
- **内容**: 106,324个电信领域问答
- **来源**: TeleQnA数据集
- **大小**: ~数百MB

### 核心脚本

#### `extract_oran_qa.py` ⭐
- **类型**: Python脚本 (基础版)
- **功能**: 从TeleQnA提取ORAN问题
- **特性**:
  - vLLM 8卡并行推理
  - 批处理 (batch_size=32)
  - 进度条显示
- **适用场景**: 一次性完整提取

#### `extract_oran_qa_enhanced.py` ⭐⭐⭐ (推荐)
- **类型**: Python脚本 (增强版)
- **功能**: 同上,但增加了:
  - ✅ 断点续传 (checkpoint)
  - ✅ 错误处理和重试
  - ✅ 进度保存和恢复
  - ✅ 定期保存检查点
- **适用场景**: 长时间运行,可能中断的情况

#### `test_extraction.py`
- **类型**: Python脚本 (测试)
- **功能**: 测试前10个问题
- **特性**:
  - 快速验证功能
  - 检查prompt效果
  - 验证模型加载
- **适用场景**: 首次使用前的测试

### Shell脚本

#### `run_extraction.sh`
- **类型**: Bash脚本
- **功能**: 运行完整提取
- **用法**: `./run_extraction.sh`
- **环境**: 自动设置CUDA_VISIBLE_DEVICES=0-7

#### `run_test.sh`
- **类型**: Bash脚本
- **功能**: 运行快速测试
- **用法**: `./run_test.sh`
- **环境**: 自动设置CUDA_VISIBLE_DEVICES=0-7

### 配置文件

#### `requirements.txt`
- **类型**: Python依赖配置
- **内容**:
  ```
  vllm>=0.2.7
  torch>=2.0.0
  transformers>=4.36.0
  tqdm>=4.65.0
  numpy>=1.24.0
  ```
- **用法**: `pip install -r requirements.txt`

### 文档

#### `README.md`
- **类型**: Markdown文档
- **内容**:
  - 项目概述
  - 运行方法
  - 配置说明
  - ORAN判定标准
  - 输出格式
  - 性能预估

#### `QUICKSTART.md`
- **类型**: Markdown文档 (详细指南)
- **内容**:
  - 准备工作
  - 快速开始
  - 完整提取流程
  - 结果分析
  - 高级配置
  - 常见问题
  - 性能优化

#### `FILES.md` (本文件)
- **类型**: Markdown文档
- **内容**: 所有文件的清单和说明

### 输出文件

#### `TeleQnA_ORAN_only.json`
- **类型**: 输出文件 (JSON)
- **内容**: 仅包含ORAN知识的问题
- **格式**:
  ```json
  {
    "question_id": {
      "question": "...",
      "option 1": "...",
      "answer": "...",
      "explanation": "...",
      "category": "..."
    }
  }
  ```
- **预计大小**: 根据提取结果而定 (估计5-10%的原始数据)

#### `extraction_log.txt`
- **类型**: 输出文件 (文本日志)
- **内容**: 每个问题的详细判断过程
- **格式**:
  ```
  ================================================================================
  Question ID: question_0
  Question: ...
  Is ORAN: True/False
  Reason: ...
  LLM Response: ...
  ================================================================================
  ```
- **用途**: 质量检查、调试、分析

### 临时文件

#### `checkpoint.json` (仅增强版)
- **类型**: 临时文件 (检查点)
- **内容**:
  ```json
  {
    "processed_count": 1000,
    "oran_questions": {...},
    "extraction_log": [...],
    "timestamp": "2025-10-29T..."
  }
  ```
- **用途**: 断点续传
- **生命周期**: 提取完成后自动删除

#### `progress.json` (仅增强版)
- **类型**: 临时文件 (进度)
- **内容**:
  ```json
  {
    "current_batch": 100,
    "total_batches": 3323,
    "progress_percent": 3.01,
    "elapsed_time": 456.78,
    "estimated_remaining": 14821.22,
    "timestamp": "2025-10-29T..."
  }
  ```
- **用途**: 监控进度

## 🚀 推荐使用流程

### 1️⃣ 首次使用

```bash
# Step 1: 安装依赖
pip install -r requirements.txt

# Step 2: 快速测试
./run_test.sh

# Step 3: 运行完整提取 (推荐使用增强版)
python extract_oran_qa_enhanced.py
```

### 2️⃣ 断点续传 (增强版)

```bash
# 如果中途中断,直接重新运行即可自动续传
python extract_oran_qa_enhanced.py
```

### 3️⃣ 结果检查

```bash
# 查看ORAN问题数量
cat TeleQnA_ORAN_only.json | jq 'length'

# 查看提取日志
less extraction_log.txt

# 随机抽样检查
cat TeleQnA_ORAN_only.json | jq -r '.[] | .question' | shuf | head -5
```

## 🔧 脚本对比

| 功能 | 基础版 | 增强版 | 测试版 |
|------|--------|--------|--------|
| 文件名 | extract_oran_qa.py | extract_oran_qa_enhanced.py | test_extraction.py |
| 8卡GPU并行 | ✅ | ✅ | ✅ |
| 批处理 | ✅ | ✅ | ✅ (小批次) |
| 进度条 | ✅ | ✅ | ✅ |
| 断点续传 | ❌ | ✅ | ❌ |
| 错误处理 | 基础 | 增强 (重试) | 基础 |
| 进度保存 | ❌ | ✅ | ❌ |
| 检查点 | ❌ | ✅ | ❌ |
| 数据量 | 全部 | 全部 | 前10个 |
| 推荐场景 | 稳定环境 | 生产环境 | 测试验证 |

## 📊 性能参考

### 测试环境
- GPU: 8x NVIDIA A100/V100
- 模型: Qwen2.5-14B-Instruct
- 数据: 106,324个问题

### 预计耗时
| 批处理大小 | GPU数量 | 预计时间 |
|-----------|---------|---------|
| 32 | 8 | 3-5小时 |
| 16 | 4 | 6-8小时 |
| 8 | 2 | 12-16小时 |

### 内存占用
- 每张GPU: ~12-16GB
- 总内存: ~100-128GB (8卡)

## 🎯 下一步

1. **运行测试**: `./run_test.sh`
2. **检查输出**: 确认LLM判断准确
3. **运行完整提取**: `python extract_oran_qa_enhanced.py`
4. **质量检查**: 随机抽样验证结果
5. **使用数据**: 将提取的ORAN数据用于后续任务

## 📞 支持

- 详细指南: 参见 `QUICKSTART.md`
- 常见问题: 参见 `QUICKSTART.md` 第7节
- 技术文档: 参见 `README.md`

---

**更新时间**: 2025-10-29  
**版本**: 1.0
