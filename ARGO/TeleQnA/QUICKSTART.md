# ORAN QA提取工具 - 快速开始指南

## 📋 目录

1. [工具概述](#工具概述)
2. [准备工作](#准备工作)
3. [快速开始](#快速开始)
4. [完整提取](#完整提取)
5. [结果分析](#结果分析)
6. [高级配置](#高级配置)
7. [常见问题](#常见问题)

---

## 🎯 工具概述

本工具从TeleQnA数据集(106,324个电信领域问答)中提取**仅涉及O-RAN知识**的问答对,生成专门的O-RAN问答数据集。

**核心技术:**
- 模型: Qwen2.5-14B-Instruct
- 推理引擎: vLLM (高性能)
- GPU: 8卡并行 (Tensor Parallelism)
- 批处理: 每批32个问题

---

## ⚙️ 准备工作

### 1. 环境要求

```bash
# Python 3.8+
python --version

# CUDA 11.8+
nvidia-smi
```

### 2. 安装依赖

```bash
cd /data/user/huangxiaolin/ARGO2/ARGO/TeleQnA

# 方法1: 使用requirements.txt
pip install -r requirements.txt

# 方法2: 手动安装
pip install vllm>=0.2.7 torch>=2.0.0 transformers>=4.36.0 tqdm
```

### 3. 验证模型路径

```bash
ls -lh /data/user/huangxiaolin/ARGO/RAG_Models/models/Qwen2.5-14B-Instruct
```

确认模型文件存在且完整。

### 4. 验证数据集

```bash
ls -lh /data/user/huangxiaolin/ARGO2/ARGO/TeleQnA/TeleQnA.txt
wc -l /data/user/huangxiaolin/ARGO2/ARGO/TeleQnA/TeleQnA.txt
```

---

## 🚀 快速开始

### Step 1: 快速测试 (推荐首次使用)

在处理全部数据前,先测试前10个问题以验证功能:

```bash
cd /data/user/huangxiaolin/ARGO2/ARGO/TeleQnA

# 运行快速测试
./run_test.sh
```

**预期输出:**

```
=========================================
Quick Test: ORAN QA Extraction
=========================================

Testing with first 10 questions

Loading sample questions from: ...
✓ Loaded 10 sample questions

Initializing vLLM model...
✓ vLLM model loaded

Processing 10 questions...

================================================================================

Question ID: question 0
Question: What is the purpose of the Nmfaf_3daDataManagement_Deconfigure service operation? [3GPP Rele...
Category: Standards specifications
Is ORAN: ✗ NO
Reason: This is a general 3GPP specification, not specific to O-RAN.
LLM Response: NO - This is a general 3GPP specification, not specific to O-RAN.
--------------------------------------------------------------------------------

...

================================================================================
Test Summary:
  Total questions: 10
  ORAN questions: 2 (20.0%)
  Non-ORAN questions: 8 (80.0%)
================================================================================

✓ Quick test completed!
```

### Step 2: 检查测试结果

查看测试输出,确认:
- ✅ 模型加载成功
- ✅ LLM响应格式正确 (YES/NO + 理由)
- ✅ 判断结果合理

如果测试通过,继续下一步。如果有问题,参考[常见问题](#常见问题)。

---

## 🏃 完整提取

### Step 1: 运行完整提取

```bash
cd /data/user/huangxiaolin/ARGO2/ARGO/TeleQnA

# 运行完整提取 (预计3-5小时)
./run_extraction.sh
```

### Step 2: 监控进度

脚本会显示实时进度:

```
============================================================
Starting ORAN extraction with vLLM
Total questions: 106324
Batch size: 32
GPU parallelism: 8
============================================================

Processing batches: 100%|██████████| 3323/3323 [2:34:15<00:00, 2.78s/it]
```

### Step 3: 等待完成

**预计时间:** 3-5小时 (取决于GPU性能)

**GPU使用情况:** 可在另一个终端监控:

```bash
watch -n 1 nvidia-smi
```

---

## 📊 结果分析

### 1. 查看输出文件

```bash
# ORAN问题集
cat TeleQnA_ORAN_only.json | jq '.' | head -50

# 提取日志
head -100 extraction_log.txt

# 统计ORAN问题数量
cat TeleQnA_ORAN_only.json | jq 'length'
```

### 2. 输出文件说明

#### `TeleQnA_ORAN_only.json`

仅包含O-RAN相关问题的JSON文件:

```json
{
  "question 156": {
    "question": "Which deployment scenario in O-RAN Town involves vO-CU and vO-DU at the aggregation location?",
    "option 1": "Scenario 1",
    "option 2": "Scenario 2",
    "option 3": "Scenario 3",
    "option 4": "Scenario 4",
    "answer": "option 3: Scenario 3",
    "explanation": "Scenario 3 in O-RAN Town involves vO-CU and vO-DU at the aggregation location...",
    "category": "Standards specifications"
  }
}
```

#### `extraction_log.txt`

详细的提取日志,记录每个问题的判断过程:

```
================================================================================
Question ID: question 156
Question: Which deployment scenario in O-RAN Town involves vO-CU and vO-DU at the aggregation location?
Is ORAN: True
Reason: This question is about O-RAN deployment scenarios and components.
LLM Response: YES - This question is about O-RAN deployment scenarios and components.
================================================================================
```

### 3. 统计分析示例

```bash
# 总问题数
total=$(cat TeleQnA.txt | grep -c '"question"')
echo "Total questions: $total"

# ORAN问题数
oran=$(cat TeleQnA_ORAN_only.json | jq 'length')
echo "ORAN questions: $oran"

# 计算百分比
echo "scale=2; $oran * 100 / $total" | bc
```

---

## 🔧 高级配置

### 1. 调整GPU数量

如果只有4张GPU:

```python
# 编辑 extract_oran_qa.py
TENSOR_PARALLEL_SIZE = 4  # 改为4

# 或修改环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

### 2. 调整批处理大小

如果遇到GPU内存不足:

```python
# 编辑 extract_oran_qa.py
BATCH_SIZE = 16  # 减小批处理大小(默认32)
```

### 3. 调整内存使用

```python
# 编辑 extract_oran_qa.py
llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    max_model_len=MAX_MODEL_LEN,
    trust_remote_code=True,
    dtype="float16",
    gpu_memory_utilization=0.85,  # 降低到85%(默认90%)
)
```

### 4. 自定义Prompt

如需更严格或更宽松的ORAN判定标准,编辑`EXTRACTION_PROMPT`模板:

```python
# 在 extract_oran_qa.py 中修改 EXTRACTION_PROMPT
```

### 5. 只处理部分数据

```python
# 编辑 extract_oran_qa.py 的 load_teleqna_dataset 函数
# 添加限制条件
def load_teleqna_dataset(file_path: str, max_questions: int = 1000) -> Dict:
    # ... 
    # 只加载前max_questions个问题
```

---

## ❓ 常见问题

### Q1: `ModuleNotFoundError: No module named 'vllm'`

**解决:**
```bash
pip install vllm --upgrade
```

### Q2: `CUDA out of memory`

**解决:**
1. 减小批处理大小: `BATCH_SIZE = 16`
2. 减小序列长度: `MAX_MODEL_LEN = 2048`
3. 降低GPU内存占用: `gpu_memory_utilization=0.8`
4. 减少GPU数量: `TENSOR_PARALLEL_SIZE = 4`

### Q3: 模型加载失败

**解决:**
```bash
# 检查模型路径
ls /data/user/huangxiaolin/ARGO/RAG_Models/models/Qwen2.5-14B-Instruct

# 检查模型文件完整性
du -sh /data/user/huangxiaolin/ARGO/RAG_Models/models/Qwen2.5-14B-Instruct
```

### Q4: vLLM版本不兼容

**解决:**
```bash
# 卸载旧版本
pip uninstall vllm

# 安装最新版本
pip install vllm --upgrade
```

### Q5: 提取结果不准确

**解决:**
1. 检查prompt设计是否清晰
2. 调整temperature参数(当前为0.0)
3. 人工抽样检查并优化prompt
4. 考虑使用更大的模型

### Q6: 处理速度太慢

**优化:**
1. 增加批处理大小: `BATCH_SIZE = 64`
2. 使用更多GPU: `TENSOR_PARALLEL_SIZE = 16`
3. 检查GPU利用率: `nvidia-smi`

### Q7: JSON解析错误

**解决:**
```bash
# 检查输入文件格式
python -m json.tool TeleQnA.txt > /dev/null

# 如果格式有问题,手动修复
```

---

## 📈 性能优化建议

### 1. 最佳GPU配置

| GPU数量 | 批处理大小 | 预计速度 | 推荐场景 |
|---------|-----------|---------|---------|
| 8 | 32 | 最快(3-4h) | 生产环境 |
| 4 | 16 | 中等(6-8h) | 资源受限 |
| 2 | 8 | 较慢(12-16h) | 测试环境 |

### 2. 内存优化

```python
# 低内存配置
gpu_memory_utilization=0.7
BATCH_SIZE = 8
MAX_MODEL_LEN = 2048

# 高内存配置
gpu_memory_utilization=0.95
BATCH_SIZE = 64
MAX_MODEL_LEN = 8192
```

---

## 📝 后续处理

### 1. 质量检查

```bash
# 随机抽样10个ORAN问题
cat TeleQnA_ORAN_only.json | jq -r 'to_entries | .[].value.question' | shuf | head -10
```

### 2. 数据统计

```python
import json

with open('TeleQnA_ORAN_only.json', 'r') as f:
    oran_data = json.load(f)

# 按类别统计
categories = {}
for q in oran_data.values():
    cat = q.get('category', 'Unknown')
    categories[cat] = categories.get(cat, 0) + 1

print("ORAN问题类别分布:")
for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
    print(f"  {cat}: {count}")
```

### 3. 格式转换

如需转换为其他格式(如CSV):

```python
import json
import csv

with open('TeleQnA_ORAN_only.json', 'r') as f:
    oran_data = json.load(f)

with open('TeleQnA_ORAN_only.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['ID', 'Question', 'Answer', 'Category'])
    
    for qid, qdata in oran_data.items():
        writer.writerow([
            qid,
            qdata['question'],
            qdata['answer'],
            qdata.get('category', '')
        ])
```

---

## 📞 技术支持

如遇到问题:

1. 查看日志文件: `extraction_log.txt`
2. 参考本文档的[常见问题](#常见问题)部分
3. 检查GPU状态: `nvidia-smi`
4. 验证环境配置

---

## 📄 许可证

本工具遵循项目主仓库的许可证。

---

**祝使用愉快! 🎉**
