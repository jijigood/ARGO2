# 🚀 ARGO选择题快速参考

## 📋 一行代码回答选择题

```python
from src.argo_system import ARGOSystem

argo = ARGOSystem(model_name="Qwen/Qwen2.5-1.5B-Instruct", retriever_mode="chroma", chroma_dir="chroma_db")
answer, choice, _, _ = argo.answer_question(question, options=["选项1", "选项2", "选项3", "选项4"])
```

---

## 🎯 核心API

### 初始化

```python
argo = ARGOSystem(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",  # LLM模型
    retriever_mode="chroma",                   # 检索模式: "chroma" 或 "mock"
    chroma_dir="chroma_db",                    # Chroma数据库路径
    use_mdp=True,                              # 是否使用MDP策略
    max_steps=5,                               # 最大推理步数
    verbose=True                               # 是否显示详细过程
)
```

### 回答问题

```python
answer, choice, history, metadata = argo.answer_question(
    question="问题文本",                        # 必需
    options=["选项1", "选项2", "选项3", "选项4"], # 可选，提供则为选择题
    return_history=True                        # 是否返回推理历史
)
```

### 返回值

| 变量 | 类型 | 说明 |
|------|------|------|
| `answer` | `str` | 详细解释 |
| `choice` | `str` or `None` | 选项编号 ("1"/"2"/"3"/"4") |
| `history` | `List[Dict]` or `None` | 推理历史 |
| `metadata` | `Dict` | 元数据（步数、耗时等） |

---

## 📊 数据集格式

### fin_H_clean.json

```json
[
  "问题文本",
  ["1. 选项1", "2. 选项2", "3. 选项3", "4. 选项4"],
  "2"  // 正确答案
]
```

### 使用示例

```python
import json

with open('data/benchmark/ORAN-Bench-13K/Benchmark/fin_H_clean.json', 'r') as f:
    dataset = json.load(f)

for item in dataset:
    question = item[0]
    options = [opt.split('. ', 1)[1] for opt in item[1]]  # 去掉 "1. " 前缀
    correct = item[2]
    
    _, choice, _, _ = argo.answer_question(question, options=options)
    print(f"预测={choice}, 正确={correct}, {'✅' if choice==correct else '❌'}")
```

---

## 🔍 输出格式

### LLM生成

```xml
<answer long>详细推理过程...</answer long>
<answer short>Option X is correct because...</answer short>
<choice>X</choice>
```

### 提取逻辑

1. **主提取**: `<choice>(\d)</choice>`
2. **回退提取**: `Option (\d)` 或 `选项(\d)`
3. **默认**: 返回 `None`

---

## ⚙️ 性能优化

### 加快速度

```python
argo = ARGOSystem(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    use_mdp=False,       # 禁用MDP，使用固定策略
    max_steps=3,         # 减少最大步数
    verbose=False        # 关闭详细输出
)
```

### 提升准确率

```python
argo = ARGOSystem(
    model_name="Qwen/Qwen2.5-7B-Instruct",  # 使用更大模型
    use_mdp=True,                            # 启用MDP智能决策
    max_steps=5,                             # 允许更多推理步数
)

# 调整生成参数
from src.synthesizer import AnswerSynthesizer
synthesizer = AnswerSynthesizer(
    model=model,
    tokenizer=tokenizer,
    temperature=0.3,    # 降低温度提高确定性
    max_answer_length=256
)
```

---

## 🧪 快速测试

### 测试脚本

```bash
# 运行所有测试
python test_multiple_choice.py

# 运行示例
python example_mcq.py 1  # 单题示例
python example_mcq.py 2  # 批量示例
```

### 单元测试

```python
def test_choice_extraction():
    from src.synthesizer import AnswerSynthesizer
    
    # Mock对象
    class MockModel: pass
    class MockTokenizer: pass
    
    synth = AnswerSynthesizer(MockModel(), MockTokenizer())
    
    # 测试提取
    _, choice = synth._postprocess_answer(
        '<choice>3</choice>',
        has_options=True
    )
    assert choice == '3', "提取失败"
```

---

## 📈 评估指标

### 计算准确率

```python
from sklearn.metrics import accuracy_score, classification_report

predictions = [...]  # 预测结果
ground_truth = [...]  # 正确答案

# 准确率
acc = accuracy_score(ground_truth, predictions)
print(f"准确率: {acc*100:.2f}%")

# 详细报告
print(classification_report(
    ground_truth, 
    predictions,
    target_names=["Option 1", "Option 2", "Option 3", "Option 4"]
))
```

### 元数据统计

```python
total_steps = []
total_times = []

for item in dataset:
    _, _, _, metadata = argo.answer_question(...)
    total_steps.append(metadata['total_steps'])
    total_times.append(metadata['elapsed_time'])

print(f"平均步数: {np.mean(total_steps):.1f}")
print(f"平均耗时: {np.mean(total_times):.2f}s")
```

---

## ⚠️ 常见问题

### Q1: choice返回None怎么办？

**A**: LLM可能没有生成正确格式。检查：
1. 模型是否太小（建议7B+）
2. Prompt是否正确传递options
3. 使用回退默认值

```python
if choice is None:
    choice = "1"  # 默认选项1
```

### Q2: 准确率很低怎么办？

**A**: 尝试以下优化：
1. 使用更大的模型
2. 改善检索质量（更多文档、更好的embedding）
3. 增加推理步数
4. 调整MDP参数

### Q3: 运行很慢怎么办？

**A**: 性能优化：
1. 使用GPU (`device="cuda"`)
2. 减少max_steps
3. 使用mock retriever测试
4. 关闭verbose模式

### Q4: 如何调试推理过程？

**A**: 查看推理历史：

```python
answer, choice, history, _ = argo.answer_question(..., return_history=True)

for i, step in enumerate(history):
    print(f"Step {i+1}: {step['action']}")
    if step['action'] == 'retrieve':
        print(f"  Query: {step['subquery']}")
        print(f"  Success: {step['retrieval_success']}")
    print(f"  Answer: {step['intermediate_answer'][:100]}...")
```

---

## 🔗 相关文件

| 文件 | 用途 |
|------|------|
| `MULTIPLE_CHOICE_SUPPORT.md` | 📚 完整使用文档 |
| `MCQ_UPDATE_SUMMARY.md` | 📝 更新总结 |
| `CHANGELOG.md` | 📋 版本历史 |
| `test_multiple_choice.py` | 🧪 测试脚本 |
| `example_mcq.py` | 💡 示例代码 |

---

## 🎓 最佳实践

### ✅ 推荐

- 使用 `fin_H_clean.json` (清洗后的数据集)
- 清理选项前缀 (`split('. ', 1)[1]`)
- 记录推理历史用于分析
- 批量处理时关闭verbose

### ❌ 避免

- 直接使用 `fin_H.json` (含19个异常)
- 忘记清理选项格式
- 在小模型上期望高准确率
- 同时运行多个ARGO实例（内存不足）

---

## 💡 示例代码片段

### 完整评估流程

```python
import json
from src.argo_system import ARGOSystem
from sklearn.metrics import accuracy_score

# 1. 加载数据
with open('data/benchmark/ORAN-Bench-13K/Benchmark/fin_H_clean.json', 'r') as f:
    dataset = json.load(f)

# 2. 初始化系统
argo = ARGOSystem(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    retriever_mode="chroma",
    chroma_dir="chroma_db",
    verbose=False
)

# 3. 批量推理
predictions, ground_truth = [], []

for item in dataset[:100]:  # 前100题
    q, opts, ans = item[0], item[1], item[2]
    clean_opts = [o.split('. ', 1)[1] for o in opts]
    
    _, choice, _, _ = argo.answer_question(q, options=clean_opts)
    predictions.append(choice if choice else "1")
    ground_truth.append(ans)

# 4. 计算准确率
acc = accuracy_score(ground_truth, predictions)
print(f"Accuracy: {acc*100:.2f}%")
```

---

**版本**: V2.1  
**更新**: 2024-11-03  
**快速帮助**: 查看 `MULTIPLE_CHOICE_SUPPORT.md`
