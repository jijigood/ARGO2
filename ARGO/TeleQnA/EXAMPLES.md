# 使用示例 - ORAN QA提取工具

## 📝 示例1: 快速测试

```bash
# 进入目录
cd /data/user/huangxiaolin/ARGO2/ARGO/TeleQnA

# 运行快速测试 (测试前10个问题)
./run_test.sh
```

**预期输出:**
```
=========================================
Quick Test: ORAN QA Extraction
=========================================

Testing with first 10 questions

Loading sample questions from: /data/user/huangxiaolin/ARGO2/ARGO/TeleQnA/TeleQnA.txt
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
...

================================================================================
Test Summary:
  Total questions: 10
  ORAN questions: 2 (20.0%)
  Non-ORAN questions: 8 (80.0%)
================================================================================

✓ Quick test completed!
```

---

## 📝 示例2: 使用交互式菜单 (推荐)

```bash
cd /data/user/huangxiaolin/ARGO2/ARGO/TeleQnA

# 运行交互式菜单
./run_menu.sh
```

**菜单界面:**
```
╔════════════════════════════════════════════════════════════╗
║     ORAN QA Extraction Tool - TeleQnA Dataset             ║
╚════════════════════════════════════════════════════════════╝

📁 当前目录: /data/user/huangxiaolin/ARGO2/ARGO/TeleQnA

请选择操作:

  1) 快速测试 (前10个问题)
  2) 完整提取 - 基础版
  3) 完整提取 - 增强版 (推荐, 支持断点续传)
  4) 查看提取进度
  5) 检查结果统计
  6) 安装依赖
  0) 退出

请输入选项 [0-6]: 
```

选择 **3** 运行增强版提取。

---

## 📝 示例3: 直接运行Python脚本

```bash
cd /data/user/huangxiaolin/ARGO2/ARGO/TeleQnA

# 设置GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 运行增强版 (推荐)
python extract_oran_qa_enhanced.py
```

**输出示例:**
```
############################################################
# ORAN QA Extraction from TeleQnA Dataset (Enhanced)
# Using: /data/user/huangxiaolin/ARGO/RAG_Models/models/Qwen2.5-14B-Instruct
# GPUs: 8
# Features: Checkpoint, Error Handling, Progress Tracking
############################################################

Loading TeleQnA dataset from: /data/user/huangxiaolin/ARGO2/ARGO/TeleQnA/TeleQnA.txt
✓ Loaded 106324 questions

Initializing vLLM model...
  Model: /data/user/huangxiaolin/ARGO/RAG_Models/models/Qwen2.5-14B-Instruct
  Tensor Parallel Size: 8
✓ vLLM model loaded successfully

============================================================
Starting ORAN extraction with vLLM
Total questions: 106324
Remaining questions: 106324
Batch size: 32
GPU parallelism: 8
============================================================

Processing batches: 100%|██████████████| 3323/3323 [2:34:15<00:00, 2.78s/it]

✓ Checkpoint saved: 106324/106324 questions processed

Saving 8456 ORAN questions to: /data/user/huangxiaolin/ARGO2/ARGO/TeleQnA/TeleQnA_ORAN_only.json
Saving extraction log to: /data/user/huangxiaolin/ARGO2/ARGO/TeleQnA/extraction_log.txt

============================================================
Extraction Summary:
  Total questions: 106324
  ORAN questions: 8456 (7.95%)
  Non-ORAN questions: 97868 (92.05%)
============================================================
✓ Checkpoint file removed (extraction completed)

✓ Extraction completed successfully!
```

---

## 📝 示例4: 断点续传

如果提取过程中中断了,直接重新运行即可:

```bash
cd /data/user/huangxiaolin/ARGO2/ARGO/TeleQnA
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 重新运行,自动从上次中断的地方继续
python extract_oran_qa_enhanced.py
```

**输出示例:**
```
Loading TeleQnA dataset from: ...
✓ Loaded 106324 questions

✓ Loaded checkpoint: 50000 questions processed

Initializing vLLM model...
✓ vLLM model loaded successfully

============================================================
Starting ORAN extraction with vLLM
Total questions: 106324
Remaining questions: 56324  # ← 从第50000个继续
Batch size: 32
GPU parallelism: 8
============================================================

✓ Resuming from checkpoint: starting at question 50000

Processing batches: 100%|████████████| 1761/1761 [1:28:42<00:00, 3.02s/it]
...
```

---

## 📝 示例5: 查看结果

### 5.1 查看ORAN问题数量

```bash
cd /data/user/huangxiaolin/ARGO2/ARGO/TeleQnA

# 使用jq查看
cat TeleQnA_ORAN_only.json | jq 'length'
```

输出:
```
8456
```

### 5.2 查看随机ORAN问题

```bash
# 随机抽取5个ORAN问题
cat TeleQnA_ORAN_only.json | jq -r '.[] | .question' | shuf | head -5
```

输出示例:
```
Which deployment scenario in O-RAN Town involves vO-CU and vO-DU at the aggregation location?
Which components are responsible for embedding intelligence in the O-RAN architecture?
How does O-RAN enable interchangeability of components?
Which node controls the O-DUs in the O-RAN architecture?
What does O-RAN define functional blocks that are used in CF (Cell-free) mMIMO networks?
```

### 5.3 查看某个具体问题

```bash
# 查看第一个ORAN问题
cat TeleQnA_ORAN_only.json | jq 'to_entries | .[0]'
```

输出示例:
```json
{
  "key": "question 156",
  "value": {
    "question": "Which deployment scenario in O-RAN Town involves vO-CU and vO-DU at the aggregation location?",
    "option 1": "Scenario 1",
    "option 2": "Scenario 2",
    "option 3": "Scenario 3",
    "option 4": "Scenario 4",
    "answer": "option 3: Scenario 3",
    "explanation": "Scenario 3 in O-RAN Town involves vO-CU and vO-DU at the aggregation location, with user traffic carried over OFH and encryption not needed between the cell site and aggregation site.",
    "category": "Standards specifications"
  }
}
```

### 5.4 查看提取日志

```bash
# 查看日志前50行
head -50 extraction_log.txt

# 或使用less浏览
less extraction_log.txt
```

---

## 📝 示例6: 统计分析

### 6.1 按类别统计

```python
import json
from collections import Counter

# 加载数据
with open('TeleQnA_ORAN_only.json', 'r') as f:
    oran_data = json.load(f)

# 统计类别
categories = [q.get('category', 'Unknown') for q in oran_data.values()]
category_counts = Counter(categories)

print("ORAN问题类别分布:")
for cat, count in category_counts.most_common():
    print(f"  {cat}: {count}")
```

**输出示例:**
```
ORAN问题类别分布:
  Standards specifications: 6234
  Research overview: 1523
  Research publications: 699
```

### 6.2 计算提取率

```bash
# 使用交互式菜单的选项5
./run_menu.sh
# 选择 5) 检查结果统计
```

输出:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
总问题数:      106324
ORAN问题数:    8456
ORAN占比:      7.95%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📝 示例7: 自定义配置

### 7.1 修改GPU数量

```python
# 编辑 extract_oran_qa_enhanced.py

# 改为使用4张GPU
TENSOR_PARALLEL_SIZE = 4

# 或在运行时设置
export CUDA_VISIBLE_DEVICES=0,1,2,3
python extract_oran_qa_enhanced.py
```

### 7.2 修改批处理大小

```python
# 编辑 extract_oran_qa_enhanced.py

# 减小批处理大小 (节省内存)
BATCH_SIZE = 16
```

### 7.3 修改保存频率

```python
# 编辑 extract_oran_qa_enhanced.py

# 每5个batch保存一次检查点
SAVE_FREQUENCY = 5
```

---

## 📝 示例8: 故障排查

### 8.1 检查GPU状态

```bash
# 实时监控GPU
watch -n 1 nvidia-smi

# 或在另一个终端运行
nvidia-smi
```

### 8.2 查看进度文件

```bash
# 查看当前进度
cat progress.json | jq '.'
```

输出:
```json
{
  "current_batch": 1500,
  "total_batches": 3323,
  "progress_percent": 45.14,
  "elapsed_time": 5432.67,
  "estimated_remaining": 6598.33,
  "timestamp": "2025-10-29T14:23:45.123456"
}
```

### 8.3 内存不足时的处理

```python
# 编辑 extract_oran_qa_enhanced.py

# 降低配置
BATCH_SIZE = 8
MAX_MODEL_LEN = 2048

llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    max_model_len=MAX_MODEL_LEN,
    trust_remote_code=True,
    dtype="float16",
    gpu_memory_utilization=0.7,  # 降低到70%
)
```

---

## 📝 示例9: 数据格式转换

### 9.1 转换为CSV

```python
import json
import csv

with open('TeleQnA_ORAN_only.json', 'r') as f:
    oran_data = json.load(f)

with open('TeleQnA_ORAN_only.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['ID', 'Question', 'Options', 'Answer', 'Explanation', 'Category'])
    
    for qid, qdata in oran_data.items():
        options = ', '.join([v for k, v in qdata.items() if k.startswith('option')])
        writer.writerow([
            qid,
            qdata['question'],
            options,
            qdata['answer'],
            qdata.get('explanation', ''),
            qdata.get('category', '')
        ])

print("✓ 已转换为CSV格式")
```

### 9.2 提取问答对

```python
import json

with open('TeleQnA_ORAN_only.json', 'r') as f:
    oran_data = json.load(f)

qa_pairs = []
for qid, qdata in oran_data.items():
    qa_pairs.append({
        'id': qid,
        'question': qdata['question'],
        'answer': qdata['answer']
    })

with open('ORAN_QA_pairs.json', 'w', encoding='utf-8') as f:
    json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

print(f"✓ 提取了 {len(qa_pairs)} 个问答对")
```

---

## 🎯 最佳实践建议

1. **首次使用**: 先运行 `./run_test.sh` 测试
2. **完整提取**: 使用增强版 `python extract_oran_qa_enhanced.py`
3. **监控进度**: 在另一终端运行 `watch -n 10 "cat progress.json | jq '.'"`
4. **质量检查**: 提取完成后随机抽样验证
5. **保存备份**: 定期备份 `TeleQnA_ORAN_only.json`

---

**祝使用顺利! 🎉**
