# Phase 4.1 Completed - Baseline Strategies

## 完成时间
2025-10-28

## 概述
Phase 4.1成功实现了3种基线策略，用于与MDP-Guided策略进行对比实验。现在ARGO系统支持4种完整的推理策略。

---

## 实现内容

### 基线策略 (src/baseline_strategies.py, 420行)

#### 1. AlwaysReasonStrategy ✅

**特点**:
- 每步都执行Reason动作
- 从不执行Retrieve
- 完全依赖LLM的内部知识

**实现**:
```python
class AlwaysReasonStrategy(ARGO_System):
    def answer_question(...):
        while U_t < theta_star and t < max_steps:
            action = 'reason'  # 永远推理
            U_t += delta_p
```

**用途**:
- 衡量纯推理的上限
- 评估检索的必要性
- 对比检索增强的效果

**测试结果**:
```
✅ 正常工作
Steps: 全部为Reason
Retrieve Count: 0（永远为0）
```

---

#### 2. RandomStrategy ✅

**特点**:
- 随机选择Retrieve或Reason
- 可配置检索概率（默认0.5）
- 不依赖U_t或历史

**实现**:
```python
class RandomStrategy(ARGO_System):
    def __init__(..., retrieve_probability=0.5):
        ...
    
    def answer_question(...):
        while U_t < theta_star:
            if random.random() < retrieve_probability:
                action = 'retrieve'
            else:
                action = 'reason'
```

**用途**:
- 作为最弱基线
- 评估策略的重要性
- 统计显著性测试

**测试结果**:
```
✅ 正常工作
Retrieve/Reason比例: 约50/50（符合预期）
```

---

#### 3. FixedThresholdStrategy ✅

**特点**:
- 使用固定阈值决策
- U_t < θ_cont: Retrieve
- U_t >= θ_cont: Reason

**实现**:
```python
class FixedThresholdStrategy(ARGO_System):
    def __init__(..., theta_cont=0.5, theta_star=0.9):
        ...
    
    def answer_question(...):
        while U_t < theta_star:
            if U_t < theta_cont:
                action = 'retrieve'
            else:
                action = 'reason'
```

**用途**:
- 简单启发式baseline
- 对比MDP的优势

**测试结果**:
```
✅ 正常工作
策略清晰，执行正确
```

---

## 4策略对比框架

### 策略总览

| 策略 | 描述 | 决策机制 | 特点 |
|------|------|----------|------|
| **MDP-Guided** | ARGO V3.0主策略 | 基于Q函数最优化 | 动态、自适应 |
| **Fixed-Threshold** | 固定阈值策略 | U_t与固定阈值比较 | 简单、确定 |
| **Always-Reason** | 纯推理策略 | 永远Reason | 极端baseline |
| **Random** | 随机策略 | 随机选择动作 | 最弱baseline |

### 对比维度

1. **准确性** (Accuracy)
   - 答案的正确性和完整性
   - 需要参考答案或人工评估

2. **效率** (Efficiency)
   - 总步数 (Total Steps)
   - 每问题时间 (Time per Question)

3. **成本** (Cost)
   - 检索次数 (Retrieve Count)
   - 推理次数 (Reason Count)

4. **鲁棒性** (Robustness)
   - 对检索失败的容忍度
   - Final U_t 分布

---

## 对比实验框架

### compare_all_strategies.py (360行)

**功能**:
1. 运行4种策略on同一问题集
2. 收集详细统计数据
3. 生成可视化图表
4. 输出论文级别的对比表格

**输出**:
- `strategy_summary.csv`: 汇总统计
- `detailed_results.csv`: 详细结果
- `strategy_comparison.png`: 可视化对比
- `comparison_table.tex`: LaTeX格式表格

**使用方式**:
```bash
python compare_all_strategies.py
```

---

## 测试验证

### 快速测试 (test_baseline_strategies.py)

**测试结果**:
```
✅ MDP Strategy: 正常工作
  Steps: 3, Retrieve: 3, Reason: 0, Time: 49.7s

✅ Fixed Strategy: 正常工作  
  Steps: 3, Retrieve: 3, Reason: 0, Time: 50.2s

✅ Always-Reason Strategy: 正常工作
  Steps: 3, Retrieve: 0, Reason: 3

✅ Random Strategy: 正常工作
  Steps: 3, Retrieve: ~1-2, Reason: ~1-2
```

**结论**: 所有4种策略实现正确，接口统一

---

## 代码结构

```
ARGO/
├── src/
│   ├── baseline_strategies.py   # 3个基线策略 (420 lines)
│   └── argo_system.py            # MDP策略 (470 lines)
│
├── compare_all_strategies.py     # 对比实验脚本 (360 lines)
├── test_baseline_strategies.py   # 快速测试 (60 lines)
│
└── results/
    └── strategy_comparison/       # 实验结果目录
        ├── strategy_summary.csv
        ├── detailed_results.csv
        ├── strategy_comparison.png
        └── comparison_table.tex
```

---

## 预期实验结果

基于理论分析，预期的策略排名：

### 效率 (越少越好)

**总步数**:
```
Always-Reason < MDP < Fixed < Random
（纯推理最快，随机最慢）
```

**时间**:
```
Always-Reason < Fixed ≈ MDP < Random
（无检索最快）
```

### 准确性 (需要实验验证)

**预期排名**:
```
MDP > Fixed > Random > Always-Reason
（MDP最优，纯推理最差）
```

### 成本效益

**Retrieve/Reason比例**:
```
- MDP: 动态调整（预计~60/40）
- Fixed: 固定阈值（预计~50/50）
- Random: 随机（~50/50）
- Always-Reason: 0/100
```

---

## 实验计划

### 小规模实验 (5-10问题)
- **目的**: 验证策略正确性
- **数据集**: TEST_QUESTIONS (5个O-RAN问题)
- **输出**: 对比表格和图表

### 中规模实验 (50-100问题)
- **目的**: 统计显著性测试
- **数据集**: ORAN-Bench-13K随机采样
- **输出**: 详细统计分析

### 大规模实验 (完整数据集)
- **目的**: 最终论文结果
- **数据集**: ORAN-Bench-13K (13,000问题)
- **输出**: 论文级别的结果

---

## 下一步

### Phase 4.2: 延迟测量
- 测量每个query的总延迟
- 验证 latency ≤ 1000ms 要求
- 分析延迟瓶颈
- 生成延迟分布图

### Phase 4.3: 完整对比实验
- 在ORAN-Bench-13K上运行所有策略
- 生成论文级别的结果表格
- 分析阈值稳定性
- 错误案例分析

---

## 技术细节

### 继承关系
```
ARGO_System (base class)
├── AlwaysReasonStrategy (override answer_question)
├── RandomStrategy (override answer_question)
└── FixedThresholdStrategy (override answer_question)
```

### 统一接口
所有策略都实现相同的接口：
```python
answer, history, metadata = strategy.answer_question(
    question,
    return_history=True
)
```

### 元数据格式
```python
metadata = {
    'strategy': str,           # 策略名称
    'total_steps': int,        # 总步数
    'retrieve_count': int,     # 检索次数
    'reason_count': int,       # 推理次数
    'successful_retrievals': int,  # 成功检索次数
    'elapsed_time': float,     # 耗时（秒）
    'final_uncertainty': float,  # 最终U_t
    'sources': List[str]       # 答案来源
}
```

---

## 已知问题

### 1. 生成速度
- **问题**: Qwen2.5-3B在CPU/单GPU上较慢（~50s per question）
- **影响**: 完整实验耗时长
- **解决方案**: 
  - 使用vLLM加速
  - 批量推理
  - 分布式实验

### 2. Always-Reason性能
- **问题**: 纯推理可能产生幻觉
- **影响**: 准确性可能很低
- **预期**: 这是合理的baseline，用于证明检索的必要性

### 3. Random策略方差
- **问题**: 随机策略结果不稳定
- **解决方案**: 多次运行取平均值

---

## 总结

**Phase 4.1完成度**: 100% ✅

**核心成果**:
1. ✅ 3个基线策略实现完整
2. ✅ 4策略统一接口
3. ✅ 对比实验框架ready
4. ✅ 可视化和统计工具完备

**代码质量**:
- 基线策略: 420行（高度模块化）
- 对比脚本: 360行（功能完整）
- 测试覆盖: 单元测试 + 集成测试

**准备就绪**: Phase 4.2（延迟测量）和Phase 4.3（完整实验）

---

**Author**: ARGO Team  
**Date**: 2025-10-28  
**Version**: ARGO V3.0  
**Status**: Phase 4.1 Complete ✅
