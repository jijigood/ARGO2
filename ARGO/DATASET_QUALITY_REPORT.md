# ORAN-Bench-13K 数据集质量报告

**检查日期**: 2025-10-30  
**检查范围**: Easy, Medium, Hard 三个难度等级

---

## 📊 总体概况

| 难度 | 总题数 | 异常数量 | 状态 |
|------|--------|----------|------|
| **Easy** | 1,139 | 0 | ✅ 完全正常 |
| **Medium** | 9,570 | 119 | ⚠️ 1.24% 异常 |
| **Hard** | 3,243 | 21 | ⚠️ 0.65% 异常 |
| **总计** | 13,952 | 140 | ⚠️ 1.00% 异常 |

---

## ⚠️ Hard 难度详细问题 (你当前实验使用的数据集)

### 问题分类统计

| 问题类型 | 数量 | 占比 |
|----------|------|------|
| 缺少选项 | 13 | 61.9% |
| 缺少问题文本 | 4 | 19.0% |
| 选项数=5 (应为4) | 2 | 9.5% |
| 答案异常 (5而非1-4) | 2 | 9.5% |

### 异常题目完整列表

#### 1. 选项数量异常 (2题)

**行 186** - 有5个选项，答案是5
- 问题: "Which of the following is NOT a potential field within the 'UeId' object..."
- 选项: 1-5 (应该只有4个)
- 答案: 5 (超出范围)

**行 2598** - 有5个选项，答案是5
- 问题: "Which of the following is NOT a component defined by the Performance Measurement..."
- 选项: 1-5 (应该只有4个)
- 答案: 5 (超出范围)

#### 2. 缺少选项 (13题)

以下题目**有问题文本和答案，但选项列表为空**：

| 行号 | 问题片段 | 答案 |
|------|----------|------|
| 412 | "In the context of a shared O-RU for Multi-MNO configuration..." | 1 |
| 453 | "Which of the following is a type of measurement metric..." | 2 |
| 464 | "Which component is responsible for bridging between OFH and HDLC..." | 3 |
| 465 | "In the context of a shared O-RU for Multi-MNO configuration..." | 4 |
| 701 | "What is the purpose of the Policy feedback –callback URI not supported..." | 1 |
| 702 | "What is the expected HTTP response code when a spelling mistake..." | 2 |
| 703 | "What is the expected outcome of the Query EI type test case..." | 1 |
| 710 | "What is the expected HTTP response code for a successful Update EI job..." | 1 |
| 717 | "What is the primary objective of the Update single policy..." | 3 |
| 720 | "What is a mandatory requirement for the DUT to be able to perform..." | 1 |
| 726 | "What is the purpose of the Notify EI job status..." | 1 |
| 730 | "What is the expected outcome of the A1-P Query single policy..." | 1 |
| 1150 | "Which of the following approaches is considered the most challenging..." | 3 |

#### 3. 缺少问题文本 (4题)

以下题目**有选项和答案，但问题文本为空/null**：

| 行号 | 选项数 | 答案 | 第一个选项片段 |
|------|--------|------|----------------|
| 731 | 4 | 2 | "To specify the URI of the Near-RTR-DUT for receiving policy status updates." |
| 1253 | 4 | 2 | "To verify the radio's ability to transmit a 3GPP test frame..." |
| 1552 | 4 | 1 | "To represent the scaling factor for modulation compression." |
| 1701 | 4 | 2 | "To determine the power level of each transmitted signal." |

---

## ⚠️ Medium 难度问题概况

| 问题类型 | 数量 |
|----------|------|
| 缺少问题文本 | 84 |
| 缺少选项 | 32 |
| 选项数=6 | 1 |
| 选项数=3 | 1 |
| 答案异常 | 1 |

---

## 💡 对实验的影响

### 当前实验 (Hard 难度)

你的实验脚本已经做了**保护性处理**：

```python
# 处理异常情况：确保有4个选项
options = question.get('options', [])
if len(options) < 4:
    print(f"⚠️  问题选项数异常: {len(options)}个 - {question['question'][:60]}...")
    # 填充默认选项
    options = options + ['N/A'] * (4 - len(options))
elif len(options) > 4:
    print(f"⚠️  问题选项数异常: {len(options)}个 - {question['question'][:60]}...")
    options = options[:4]  # 只取前4个
```

### 影响评估

1. **21个异常题目 (0.65%)**
   - 缺少选项的13题：会被填充为 `['N/A', 'N/A', 'N/A', 'N/A']`，LLM几乎不可能答对
   - 缺少问题的4题：会使用空字符串作为问题，LLM无法理解
   - 5个选项的2题：会截取前4个选项，但正确答案(5)会丢失
   
2. **对总体准确率的影响**
   - 这21题基本会答错
   - 影响准确率: 21/3243 ≈ **0.65%**
   - 对实验结论影响: **很小**（误差在统计范围内）

3. **建议**
   - ✅ 当前实验可以继续运行（已有容错处理）
   - 📝 在论文中注明数据集存在0.65%的异常样本
   - 🔧 如果需要完美数据集，可以后续清洗这21个样本

---

## 📁 详细数据

完整的异常数据已导出到:
```
/data/user/huangxiaolin/ARGO2/ARGO/dataset_issues_report.json
```

包含所有异常题目的完整信息（问题、选项、答案、行号等）。

---

## ✅ 结论

1. **Easy难度**: 完全正常，无任何异常 ✨
2. **Medium难度**: 1.24%异常率，主要是缺少问题文本
3. **Hard难度**: 0.65%异常率，对当前实验影响可控

**总体评价**: 数据集质量良好，异常率仅1%，且已有容错处理，不影响实验进行。

