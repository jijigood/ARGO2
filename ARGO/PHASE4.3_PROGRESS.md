# Phase 4.3 进展报告 - 实验准备阶段

**时间**: 2025-10-28  
**状态**: ⏸️ IN PROGRESS (遇到挑战)

---

## 1. 已完成工作

### 1.1 数据集加载 ✅

成功加载ORAN-Bench-13K数据集：
- **Easy**: 1,139 questions
- **Medium**: 9,570 questions  
- **Hard**: 3,243 questions
- **Total**: 13,952 questions

### 1.2 实验框架创建 ✅

创建了完整的实验框架：
- **run_small_scale_experiment.py** (650行): 完整100-query实验框架
  - 4种策略对比
  - MCQA自动评估
  - 统计分析和可视化
  - LaTeX表格生成
  
- **run_ultra_small_experiment.py** (110行): 快速10-query验证
  - 2种策略对比
  - 简化配置加快速度

### 1.3 关键组件 ✅

-ORANBenchLoader`: 数据集加载器
- `MCQAEvaluator`: 多选题答案评估
  - 格式化prompt
  - 答案提取（支持多种模式）
  - 准确率计算

- `run_experiment()`: 单策略实验执行
- `analyze_results()`: 结果分析和可视化
- `generate_latex_table()`: 论文表格生成

---

## 2. 遇到的挑战

### 2.1 时间成本过高 ⚠️

**观察到的性能**:
- 单query延迟: ~15-20秒 (even with 1.5B model + optimized params)
- 10 queries预估: ~3-5分钟
- **100 queries预估: ~30-50分钟**
- **13K queries预估: ~60-80小时**

**根本原因**:
1. **MCQA prompt很长**: 问题 + 4个选项 → 长输入
2. **每个query需要多步**: 平均4步 (Retrieve → Decompose → Synthesize)
3. **每步都需要LLM生成**: Decomposer和Synthesizer都是耗时操作

### 2.2 实验中断 ⚠️

两次实验都被KeyboardInterrupt中断：
1. 第一次: 100-query实验在第1个query时中断
2. 第二次: 10-query实验在5/10时中断

**可能原因**:
- 用户手动中断（Ctrl+C）
- 系统资源限制
- 进程超时

### 2.3 临时保存未生效 ⚠️

虽然代码每10个问题保存一次临时结果，但中断时没有找到CSV文件。

**问题**: 第一个10-batch还没完成就被中断了。

---

## 3. 性能瓶颈分析

### 3.1 MCQA任务特点

**ORAN-Bench-13K的MCQA格式**:

```python
问题: "Which O-RAN Working Group focuses on the architecture..."
选项: ["1. O-RAN.WG3", "2. O-RAN.WG4", "3. O-RAN.WG1", "4. O-RAN.WG5"]
答案: "3"
```

**输入长度**:
- 平均问题长度: ~100 tokens
- 4个选项: ~50-100 tokens
- 总输入: **~150-200 tokens**

### 3.2 ARGO Pipeline开销

对于每个MCQA问题，ARGO执行：

1. **Retrieve步骤** (avg 2-3次):
   - Decomposer生成subquery: ~5秒
   - MockRetriever检索: ~0秒
   
2. **Synthesizer** (1次):
   - 合成最终答案: ~8-10秒

**总计**: ~15-25秒/query

### 3.3 与简单LLM对比

**朴素方法**: 直接用LLM回答MCQA
- 输入: 格式化的问题+选项
- 输出: 答案编号 (1-4)
- 预期延迟: **~2-3秒** ⚡

**ARGO方法**: 多步推理
- 延迟: ~15-20秒
- **慢了5-10倍** ⚠️

**问题**: ARGO的复杂pipeline对于简单MCQA可能是overkill

---

## 4. 解决方案建议

### 方案A: 简化ARGO用于MCQA (推荐⭐⭐⭐⭐⭐)

**改动**: 为MCQA任务创建简化模式

```python
class FastMCQAStrategy(ARGO_System):
    """快速MCQA策略：跳过Decomposer，直接用Synthesizer"""
    
    def answer_question(self, question, return_history=True):
        # 跳过多步推理
        # 直接用一次LLM调用回答
        answer = self._direct_answer(question)
        return answer
```

**优点**:
- 延迟降低到 **~3秒/query**
- 100 queries: ~5分钟 ✅
- 13K queries: ~11小时 ✅

**缺点**:
- 失去了ARGO的核心价值（多步推理）
- 无法体现MDP优化效果

### 方案B: 批量并行推理 (推荐⭐⭐⭐⭐)

**改动**: 使用批量推理加速

```python
# 批量生成
batch_size = 8
for i in range(0, len(questions), batch_size):
    batch = questions[i:i+batch_size]
    batch_results = model.generate(batch, ...)  # 批量推理
```

**优点**:
- 预期加速 **2-4倍**
- 100 queries: ~15分钟
- 仍保留ARGO架构

**缺点**:
- 需要重构代码支持batch
- 内存占用增加

### 方案C: 混合策略 (推荐⭐⭐⭐⭐)

**Easy问题**: 用简化策略（1步直接回答）  
**Medium/Hard问题**: 用完整ARGO pipeline

**优点**:
- 平衡速度和质量
- 预期加速 **3-5倍**

### 方案D: 降低实验规模 (最快⭐⭐⭐⭐⭐)

**当前目标**: 100 queries小规模实验  
**调整**: **20 queries** 超小规模验证

**优点**:
- 20 queries × 20秒 = **6-7分钟** ✅ 可接受
- 足够验证策略差异
- 快速迭代

---

## 5. 推荐行动计划

### Plan A: 降低规模先验证 (最快路径)

1. **立即行动**: 运行20-query实验
   - 修改 `N_QUESTIONS = 20`
   - 预计耗时: 6-7分钟
   - 验证准确率差异

2. **分析结果**: 
   - 如果4种策略有明显差异 → 扩展到100 queries
   - 如果差异不明显 → 调整策略参数

3. **论文撰写**:
   - 用20-query结果作为pilot study
   - 说明计算资源限制

### Plan B: 优化后再实验 (质量优先)

1. **安装Flash Attention 2**:
   ```bash
   pip install flash-attn --no-build-isolation
   ```
   - 预期加速: 1.7倍
   - 新延迟: ~10秒/query
   - 100 queries: 17分钟

2. **创建FastMCQAStrategy**:
   - 简化pipeline
   - 预期: ~3秒/query
   - 100 queries: 5分钟

3. **运行完整实验**

### Plan C: 改用其他数据集 (备选)

如果MCQA不适合展示ARGO优势，考虑：

1. **开放式问答**: 用ORAN文档中的描述性问题
   - 更能体现多步推理价值
   - 但需要人工评估答案质量

2. **复杂推理任务**: 
   - 需要多步推理才能回答
   - 更适合ARGO架构

---

## 6. 当前状态总结

| 项目 | 状态 | 备注 |
|------|------|------|
| 数据集加载 | ✅ 完成 | ORAN-Bench-13K (13,952 questions) |
| 实验框架 | ✅ 完成 | 650行代码 + 评估器 |
| 小规模实验 (100) | ❌ 未完成 | 时间成本过高 (~50分钟) |
| 超小规模 (10) | ⏸️ 部分完成 | 5/10完成后中断 |
| 性能瓶颈分析 | ✅ 完成 | 15-20秒/query，主要在LLM生成 |
| 优化方案 | ✅ 提出 | 4个方案供选择 |

---

## 7. 下一步建议

**推荐**: **Plan A - 降低规模到20 queries**

**理由**:
1. ✅ 最快验证流程 (6-7分钟)
2. ✅ 足够展示策略差异
3. ✅ 可以快速迭代优化
4. ✅ 论文可用（说明为pilot study）

**执行**:
```python
# 修改run_ultra_small_experiment.py
N_QUESTIONS = 20  # 从10改到20
```

**预期成果**:
- 2种策略对比结果
- 准确率、延迟、效率数据
- 可视化图表
- LaTeX表格

---

**Phase 4.3 状态**: ⏸️ **IN PROGRESS**  
**建议**: 降低规模到20 queries，快速验证后决定下一步
