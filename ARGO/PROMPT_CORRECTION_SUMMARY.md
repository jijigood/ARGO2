# 重要修正：检索答案 vs 参数化推理

## 问题发现

用户指出了一个**关键问题**：在ARGO系统中，检索后的答案生成和纯推理的答案生成应该使用**不同的prompts**！

## ✅ 已修正

现在系统正确区分了两种中间答案生成方式：

### 1. 检索答案生成（Retrieve动作后）

**使用场景**: 检索成功，需要基于检索文档生成答案

**Prompt**: `ARGOPrompts.build_retrieval_answer_prompt()`

**格式**:
```
Question: What is the maximum latency for E2 interface?
Context: 
[O-RAN.WG3.E2AP] The E2 interface supports near-real-time control...

Answer: The E2 interface supports near-real-time operations...
```

**特点**:
- 基于**外部检索文档**
- 强调准确引用文档内容
- 支持"[No information found in O-RAN specs]"
- 4个Few-shot示例

**代码位置**: `src/retriever.py` → `generate_answer_from_docs()`

---

### 2. 参数化知识推理（Reason动作）

**使用场景**: 不检索新文档，基于LLM预训练知识推理

**Prompt**: `ARGOPrompts.build_reasoning_prompt()`

**格式**:
```
Question: How are xApps packaged for deployment?

[Previous context]
[Progress: 30%] Follow up: What is the Near-RT RIC?
Context: [...previous retrieval...]
Intermediate answer: ...

[Progress: 50%] Follow up: (current reasoning)
Intermediate answer: xApps are packaged as Docker containers...
```

**特点**:
- 基于**LLM参数化知识**（预训练知识）
- 不依赖新检索的文档
- 整合历史上下文
- 强调知识连接和推理
- 3个Few-shot示例

**代码位置**: `src/argo_system.py` → `_execute_reason()`

---

## 修改内容

### 修改1: `src/prompts.py`

添加了专门的 `REASONING_EXAMPLES`：

```python
# 旧版本（错误）
REASONING_INSTRUCTION = """Based on the following information, 
provide intermediate reasoning..."""
# 没有Few-shot示例！

# 新版本（正确）
REASONING_INSTRUCTION = """Provide intermediate reasoning based on 
your domain knowledge about O-RAN.

Requirements:
1. Use your parametric knowledge (pre-trained knowledge)
2. DO NOT claim to search or retrieve documents
3. Provide reasoning based on what you already know
...
"""

REASONING_EXAMPLES = """
Examples:
#############
Question: What are the security mechanisms in O-RAN?

[Previous context]
[Progress: 15%] ... (previous retrieval)

[Progress: 50%] Follow up: How are these domains secured?
Intermediate answer: Each domain uses mutual TLS authentication...
#############
"""
```

### 修改2: `src/prompts.py` → `build_reasoning_prompt()`

更新为使用Few-shot示例格式：

```python
# 现在包含：
# 1. REASONING_INSTRUCTION（明确说明基于参数化知识）
# 2. REASONING_EXAMPLES（3个完整示例）
# 3. 历史上下文（模仿示例格式）
# 4. 当前推理步骤提示
```

### 修改3: `test_enhanced_prompts.py`

更新测试以强调区别：

```python
def test_reasoning_prompt(model, tokenizer):
    """测试推理Prompt（基于参数化知识，无需检索文档）"""
    
    print("关键特征:")
    print("✓ 包含Few-shot示例（参数化知识推理）")
    print("✓ 显示之前的检索上下文")
    print("✓ 不要求检索新文档")
    print("✓ 基于LLM预训练知识推理")
```

---

## 对比表格

| 维度 | 检索答案生成 | 参数化推理 |
|------|-------------|-----------|
| **触发** | Retrieve动作成功 | Reason动作 |
| **输入** | Question + **Retrieved Docs** | Question + **History** |
| **Prompt** | `build_retrieval_answer_prompt()` | `build_reasoning_prompt()` |
| **示例数** | 4个 | 3个 |
| **知识源** | 外部文档 | LLM参数化知识 |
| **重点** | 文档引用准确性 | 知识连接推理 |
| **格式** | Question→Context→Answer | Context→Reasoning→Answer |

---

## 为什么这很重要？

### 1. **知识来源明确**
- 检索答案：明确告诉LLM"基于这些文档回答"
- 推理答案：明确告诉LLM"基于你的知识推理"

### 2. **Few-shot示例不同**
- 检索示例：展示如何从文档提取信息
- 推理示例：展示如何连接已知信息进行推理

### 3. **避免混淆**
- 之前：可能让LLM混淆何时使用文档、何时使用知识
- 现在：每种场景有明确的提示和示例

### 4. **性能提升**
- 检索后答案更准确（紧密基于文档）
- 推理更连贯（基于领域知识连接）

---

## 验证方法

运行测试脚本会看到两种不同的prompt：

```bash
python test_enhanced_prompts.py --mode quick
```

**测试2**: 检索答案生成
- 显示带Context的prompt
- 示例强调文档引用

**测试3**: 参数化推理
- 显示带历史上下文的prompt
- 示例强调知识推理

---

## 总结

✅ **修正前**: 只有一个通用的"推理"prompt  
✅ **修正后**: 两个专门的prompts
- `build_retrieval_answer_prompt()` - 检索后答案生成（带文档）
- `build_reasoning_prompt()` - 参数化知识推理（无新文档）

这个修正**至关重要**，因为它确保了：
1. LLM知道何时应该严格基于文档回答
2. LLM知道何时应该基于预训练知识推理
3. Few-shot示例精准引导两种不同的行为
4. 符合ARGO论文中的设计原则

---

*修正日期: 2024年11月3日*  
*感谢用户的细心发现！*
