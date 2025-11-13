# 选择题支持说明

**更新时间**: 2024年11月3日  
**版本**: ARGO Prompts V2.1

---

## ✨ 新增功能

ARGO系统现已支持**多选一选择题**（Multiple Choice Questions）格式，专为O-RAN Benchmark数据集（fin_H_clean.json）优化。

---

## 📋 数据集格式

O-RAN Benchmark数据集 (`fin_H_clean.json`) 格式：

```json
[
  "What is a key function of the O-RAN Fronthaul CUS Plane specification?",
  [
    "1. Support for slice differentiation to meet specific SLAs.",
    "2. Optimizing power consumption for the gNB DU system.",
    "3. Managing network security protocols.",
    "4. Determining the optimal frequency band for transmission."
  ],
  "1"
]
```

- **问题**: 字符串
- **选项**: 4个选项的列表（标号1-4）
- **正确答案**: "1"、"2"、"3" 或 "4"

---

## 🔧 使用方法

### 方法1: 基础用法

```python
from src.argo_system import ARGOSystem

# 初始化系统
argo = ARGOSystem(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    retriever_mode="chroma",
    chroma_dir="chroma_db",
    use_mdp=True,
    verbose=True
)

# 准备问题和选项
question = "What is a key function of the O-RAN Fronthaul CUS Plane?"
options = [
    "Support for slice differentiation to meet specific SLAs.",
    "Optimizing power consumption for the gNB DU system.",
    "Managing network security protocols.",
    "Determining the optimal frequency band for transmission."
]

# 回答选择题
answer, choice, history, metadata = argo.answer_question(
    question=question,
    options=options,
    return_history=True
)

print(f"详细答案: {answer}")
print(f"选择的选项: {choice}")  # 输出: "1", "2", "3", 或 "4"
```

### 方法2: 使用Benchmark Loader

```python
import json
from src.argo_system import ARGOSystem

# 加载数据集
with open('ORAN-Bench-13K/Benchmark/fin_H_clean.json', 'r') as f:
    dataset = json.load(f)

# 初始化系统
argo = ARGOSystem(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    retriever_mode="chroma",
    chroma_dir="chroma_db"
)

# 处理每个问题
results = []
for item in dataset[:10]:  # 处理前10题
    question_text = item[0]
    options = item[1]  # ["1. ...", "2. ...", "3. ...", "4. ..."]
    correct_answer = item[2]  # "1", "2", "3", 或 "4"
    
    # 清理选项（移除 "1. ", "2. " 等前缀）
    clean_options = [opt.split('. ', 1)[1] for opt in options]
    
    # ARGO推理
    answer, predicted_choice, _, metadata = argo.answer_question(
        question=question_text,
        options=clean_options
    )
    
    # 评估
    is_correct = (predicted_choice == correct_answer)
    results.append({
        'question': question_text,
        'predicted': predicted_choice,
        'correct': correct_answer,
        'is_correct': is_correct,
        'steps': metadata['total_steps']
    })

# 计算准确率
accuracy = sum(r['is_correct'] for r in results) / len(results)
print(f"准确率: {accuracy*100:.2f}%")
```

### 方法3: 批量处理

```python
from src.argo_system import ARGOSystem
import json

# 加载数据
with open('ORAN-Bench-13K/Benchmark/fin_H_clean.json', 'r') as f:
    dataset = json.load(f)

# 初始化系统
argo = ARGOSystem(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    retriever_mode="chroma",
    chroma_dir="chroma_db",
    verbose=False  # 关闭详细输出加快速度
)

# 批量推理
predictions = []
ground_truth = []

for item in dataset:
    question_text = item[0]
    options = [opt.split('. ', 1)[1] for opt in item[1]]
    correct_answer = item[2]
    
    # 推理（简化版，不返回历史）
    _, choice, _, _ = argo.answer_question(
        question=question_text,
        options=options,
        return_history=False
    )
    
    predictions.append(choice if choice else "1")  # 默认选项1
    ground_truth.append(correct_answer)

# 评估
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(ground_truth, predictions)
print(f"\n整体准确率: {accuracy*100:.2f}%")
print("\n详细报告:")
print(classification_report(ground_truth, predictions, 
                          target_names=["Option 1", "Option 2", "Option 3", "Option 4"]))
```

---

## 🔍 输出格式

### LLM生成格式

LLM会生成以下格式的输出：

```xml
<answer long>
Based on the retrieved information from O-RAN specifications, the Control-User-Synchronization (CUS) Plane specification for fronthaul interface provides support for slice differentiation to meet specific Service Level Agreements (SLAs). This is mentioned in [O-RAN.WG4] specification which describes how different network slices can be configured with distinct QoS parameters...
</answer long>

<answer short>
Option 1 is correct because O-RAN fronthaul CUS Plane specification includes slice differentiation capabilities for meeting specific SLAs.
</answer short>

<choice>1</choice>
```

### 解析后返回

```python
answer, choice, history, metadata = argo.answer_question(...)

# answer (str): 详细解释
"Based on the retrieved information from O-RAN specifications..."

# choice (str): "1", "2", "3", 或 "4"
"1"

# history (List[Dict]): 推理历史
[
    {
        'action': 'retrieve',
        'subquery': 'What is the CUS Plane in O-RAN fronthaul?',
        'retrieval_success': True,
        'retrieved_docs': [...],
        'intermediate_answer': '...',
        'confidence': 0.85,
        'progress': 0.35
    },
    ...
]

# metadata (Dict): 元数据
{
    'total_steps': 3,
    'final_uncertainty': 0.15,
    'retrieve_count': 2,
    'reason_count': 1,
    'successful_retrievals': 2,
    'elapsed_time': 5.23,
    'sources': ['O-RAN.WG4', 'O-RAN Security']
}
```

---

## 🎯 Prompt工程

### Synthesis Instruction

新的synthesis instruction专门针对选择题优化：

```python
SYNTHESIS_INSTRUCTION = """You are an expert at synthesizing comprehensive answers from multi-step reasoning for O-RAN multiple-choice questions.

Task: Generate a complete, accurate answer to the original question based on the reasoning history, and select the correct option.

Guidelines:
1. Integrate ALL retrieved information
2. Use insights from intermediate reasoning steps
3. Analyze each option carefully based on gathered evidence
4. Provide a coherent, well-structured reasoning process
5. Cite sources when possible (e.g., O-RAN.WG4)
6. If information is insufficient, state what's missing
7. Clearly indicate the correct option number (1, 2, 3, or 4)

Format for Multiple Choice Questions:
<answer long>Detailed reasoning and explanation for why the correct option is chosen...</answer long>
<answer short>Option X is correct because [brief justification]</answer short>
<choice>X</choice>

where X is the option number (1, 2, 3, or 4).
"""
```

### 选项显示

在synthesis prompt中，选项会自动格式化：

```
Original Question: What is a key function of the O-RAN Fronthaul CUS Plane?

Options:
1. Support for slice differentiation to meet specific SLAs.
2. Optimizing power consumption for the gNB DU system.
3. Managing network security protocols.
4. Determining the optimal frequency band for transmission.

Retrieved Information:
[1] [O-RAN.WG4] The fronthaul CUS-Plane specification defines...
...

Analyze each option based on the evidence above and select the correct answer:
```

---

## 🔬 鲁棒性处理

### 回退机制

如果LLM没有生成 `<choice>X</choice>` 标签，系统会尝试从文本中提取：

```python
# 提取逻辑（在 synthesizer._postprocess_answer 中）
choice_match = re.search(r'<choice>(\d)</choice>', answer)
if choice_match:
    choice = choice_match.group(1)
else:
    # 回退：查找 "Option 3" 或 "选项3"
    fallback_match = re.search(r'[Oo]ption\s*(\d)|选项\s*(\d)', answer)
    if fallback_match:
        choice = fallback_match.group(1) or fallback_match.group(2)
```

### 默认值

如果完全无法提取选项，`choice` 会返回 `None`：

```python
answer, choice, _, _ = argo.answer_question(question, options)
if choice is None:
    print("警告: 无法从LLM输出中提取选项")
    choice = "1"  # 使用默认值
```

---

## 📊 性能优化建议

### 1. 调整生成参数

```python
from src.synthesizer import AnswerSynthesizer

synthesizer = AnswerSynthesizer(
    model=model,
    tokenizer=tokenizer,
    max_answer_length=256,  # 选择题不需要太长的答案
    temperature=0.3,        # 较低温度提高确定性
    top_p=0.95
)
```

### 2. 使用固定策略加速

```python
# 对于选择题，可以使用固定策略减少推理步数
argo = ARGOSystem(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    retriever_mode="chroma",
    use_mdp=False,  # 禁用MDP，使用固定策略
    max_steps=3     # 限制最大步数
)
```

### 3. 批量处理

```python
# 使用 batch_synthesize 加速（需要手动构建history）
questions = [...]
histories = [...]
options_list = [...]

results = synthesizer.batch_synthesize(
    questions=questions,
    histories=histories,
    options_list=options_list
)
```

---

## 🧪 测试示例

完整测试脚本：`test_multiple_choice.py`

```python
"""测试选择题功能"""
import json
from src.argo_system import ARGOSystem

def test_single_question():
    """测试单个选择题"""
    argo = ARGOSystem(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        retriever_mode="chroma",
        chroma_dir="chroma_db",
        verbose=True
    )
    
    question = "What is the role of the SM Fanout module in an O-DU when an E2 message is received?"
    options = [
        "It interacts with the E2 handler module to send the message to the appropriate internal module.",
        "It consults the SM Catalog module to identify the relevant SM specific modules and APIs.",
        "It maps E2 messages to their corresponding receiver modules and message contents.",
        "It sends the E2 message through the E2 Sender module."
    ]
    correct = "2"
    
    answer, choice, history, metadata = argo.answer_question(
        question=question,
        options=options
    )
    
    print(f"\n问题: {question}")
    print(f"\n答案: {answer}")
    print(f"\n选择: {choice}")
    print(f"正确答案: {correct}")
    print(f"结果: {'✅ 正确' if choice == correct else '❌ 错误'}")
    print(f"\n推理步数: {metadata['total_steps']}")
    print(f"耗时: {metadata['elapsed_time']:.2f}秒")

if __name__ == "__main__":
    test_single_question()
```

---

## 📝 注意事项

### ✅ 推荐做法

1. **清理选项格式**: 移除 "1. ", "2. " 等前缀
2. **使用fin_H_clean.json**: 已移除异常数据的清洗版本
3. **设置合理的max_steps**: 选择题通常2-4步即可
4. **记录推理历史**: 便于分析错误原因

### ⚠️ 注意事项

1. **选项顺序**: 确保选项列表顺序与数据集一致
2. **答案格式**: 正确答案必须是 "1", "2", "3", "4"（字符串）
3. **LLM能力**: 小模型可能难以准确理解复杂的O-RAN技术问题
4. **检索质量**: 答案准确性高度依赖于检索到的文档质量

### 🐛 已知限制

1. **仅支持单选题**: 不支持多选题或判断题
2. **固定4个选项**: 不支持2-3个选项的题目
3. **语言限制**: 主要针对英文O-RAN术语优化

---

## 🔄 向后兼容

### 普通问答仍然支持

```python
# 不提供 options 参数，正常工作
answer, choice, history, metadata = argo.answer_question(
    question="Explain the O-RAN E2 interface"
)

# choice 将为 None
assert choice is None
```

### 返回值变化

| 版本 | 返回格式 |
|------|----------|
| V2.0 | `(answer, history, metadata)` |
| V2.1 | `(answer, choice, history, metadata)` |

**迁移建议**: 在现有代码中添加 `choice` 接收变量即可。

---

## 📚 相关文件

| 文件 | 说明 |
|------|------|
| `src/prompts.py` | 更新SYNTHESIS_INSTRUCTION和build_synthesis_prompt |
| `src/synthesizer.py` | 添加options参数和choice提取逻辑 |
| `src/argo_system.py` | 更新answer_question方法签名 |
| `ORAN-Bench-13K/Benchmark/fin_H_clean.json` | 清洗后的3224题O-RAN选择题数据集 |
| `DATA_CLEANING_SUMMARY.md` | 数据清洗详细报告 |
| `ORAN_TERMINOLOGY_CHECK.md` | O-RAN术语使用检查 |

---

**版权**: ARGO Team  
**许可**: MIT License  
**更新日期**: 2024年11月3日

---

## 🎉 快速开始

```bash
# 1. 准备环境
cd /data/user/huangxiaolin/ARGO2/ARGO

# 2. 运行测试
python test_multiple_choice.py

# 3. 在数据集上评估
python run_benchmark_evaluation.py --dataset fin_H_clean.json --max_samples 100
```

祝您使用愉快！🚀
