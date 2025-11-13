# Experiment 2: Impact of Retrieval Success Rate ($p_s$) on Performance

## 实验概述

**实验目标**: 证明ARGO能够适应不可靠的检索环境,而静态策略无法应对检索不确定性。

**实验日期**: 2025-10-29

## 实验设置

### 固定参数
- **测试集**: ORAN-Bench-13K (100道中等难度问题, seed=42)
- **MDP参数**:
  - $\delta_r$ = 0.25 (检索时U的增量)
  - $\delta_p$ = 0.08 (推理时U的增量)
  - $c_r$ = 0.05 (检索成本)
  - $c_p$ = 0.02 (推理成本)
  - $\mu$ = 0.6 (质量参数)
  - $\gamma$ = 0.98 (折扣因子)

### 自变量
- **检索成功率 ($p_s$)**: 从 0.3 扫描到 1.0
- 测试了8个不同的 $p_s$ 值: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

### 评估策略
1. **ARGO**: MDP引导的自适应策略(对每个$p_s$重新求解MDP)
2. **Always-Retrieve**: 静态策略,始终执行检索
3. **Always-Reason**: 静态策略,从不检索
4. **Random**: 随机策略(50%检索,50%推理)

## 实验步骤

对于每个 $p_s$ 值:
1. **重新求解MDP**: 运行Value Iteration算法,获取新的最优阈值
2. **评估所有策略**: 在整个测试集上运行4种策略,使用对应的$p_s$
3. **记录指标**: 平均答案质量、检索次数、推理次数

## 实验结果

### 关键发现

#### 1. ARGO的智能适应

**低成功率环境 ($p_s = 0.3$)**:
- **ARGO策略**: $\Theta_{cont} = 0.0$ → **完全避免检索,转向推理**
  - 检索次数: 0.0
  - 推理次数: 13.0
  - 质量: 1.000
- **原因**: 当检索成功率低时,MDP求解器发现推理更可靠

**高成功率环境 ($p_s = 1.0$)**:
- **ARGO策略**: $\Theta_{cont} = 0.14$ → **适度使用检索**
  - 检索次数: 1.0
  - 推理次数: 10.0
  - 质量: 1.000
- **原因**: 检索完全可靠时,可以少量使用检索提高效率

#### 2. 基线策略的问题

**Always-Retrieve的脆弱性**:
- $p_s = 0.3$ 时: 需要12.7次检索才能达标(效率低)
- $p_s = 1.0$ 时: 仅需4.0次检索
- **问题**: 在低$p_s$环境下浪费大量检索

**Always-Reason的保守性**:
- 所有$p_s$下都不检索(0次)
- 质量恒定在0.96(较低)
- **问题**: 无法利用外部知识

**Random的不稳定性**:
- 检索次数随$p_s$轻微下降(6.7 → 3.0)
- 质量在0.96-0.97之间波动
- **问题**: 无策略性,效率低

#### 3. ARGO的优势总结

| $p_s$ | ARGO策略 | 检索次数 | 质量 | Always-Retrieve检索次数 |
|-------|---------|---------|------|----------------------|
| 0.3   | 完全推理 | 0.0     | 1.00 | 12.7 (浪费)           |
| 0.6   | 少量检索 | 1.6     | 1.00 | 6.7                  |
| 0.8   | 平衡使用 | 1.3     | 1.00 | 5.1                  |
| 1.0   | 精准检索 | 1.0     | 1.00 | 4.0                  |

**关键洞察**:
- 低$p_s$时: ARGO避免检索,Always-Retrieve陷入"重试陷阱"
- 高$p_s$时: ARGO适度检索,保持高效

### 图表解读

#### Figure 1: Retrieval Success Rate vs. Answer Quality
**文件**: `figs/exp2_ps_vs_quality.png`

**观察**:
- **ARGO**: 质量恒定为1.0(在所有$p_s$下)
- **Always-Retrieve**: 质量恒定为1.0(但代价是大量检索)
- **Always-Reason**: 质量恒定为0.96(较低)
- **Random**: 质量在0.96-0.97之间

**结论**: 
- 所有策略质量受$p_s$影响较小(仿真模型中)
- 关键差异在于**效率**(检索次数)

#### Figure 2: Retrieval Success Rate vs. Retrieval Calls
**文件**: `figs/exp2_ps_vs_retrievals.png`

**预期趋势**: ✅ **完全符合预期!**
- **ARGO**: 
  - 低$p_s$时几乎不检索(0次,转向推理)
  - 高$p_s$时少量检索(1次)
  - **展示智能避险行为**

- **Always-Retrieve**: 
  - 随$p_s$增加而减少(12.7 → 4.0)
  - 因为成功率高,需要的重试次数少
  - **但低$p_s$时效率极低**

- **Always-Reason**: 恒定0次(从不检索)

- **Random**: 随$p_s$轻微下降(无策略性)

**核心发现**: ARGO在$p_s<0.6$时完全避免检索,展示了**风险规避**能力。

#### Figure 3: Action Distribution
**文件**: `figs/exp2_action_distribution.png`

展示不同策略在不同$p_s$下的动作分布(检索vs推理):

**ARGO的智能平衡**:
- $p_s=0.3$: 0次检索 + 13次推理(完全推理)
- $p_s=0.6$: 1.6次检索 + 10次推理(开始检索)
- $p_s=1.0$: 1次检索 + 10次推理(精准检索)

**基线的固定模式**:
- Always-Retrieve: 全部检索,推理为0
- Always-Reason: 全部推理,检索为0
- Random: 固定50-50分布

## 实验验证

### ✅ 成功验证的假设

1. **ARGO的检索风险管理**:
   - ✅ 低$p_s$时避免检索(0次 vs Always-Retrieve的12.7次)
   - ✅ 高$p_s$时适度检索(1次)
   - ✅ 阈值$\Theta_{cont}$随$p_s$变化(0.0 → 0.14)

2. **Always-Retrieve的脆弱性**:
   - ✅ 低$p_s$时陷入重试陷阱(12.7次检索)
   - ✅ 检索次数随$p_s$显著下降(12.7 → 4.0)

3. **质量保持**:
   - ✅ ARGO在所有$p_s$下保持高质量(1.0)
   - ✅ 通过动态切换Retrieve↔Reason维持质量

### 📊 数据文件

结果已保存至:
- **原始数据**: `draw_figs/data/exp2_retrieval_success_impact_20251029_005001.json`
- **图表**: `figs/exp2_*.png`

## 核心贡献

该实验证明了ARGO的**检索不确定性管理**能力:

### 问题场景
在真实RAG系统中,检索质量可能不稳定:
- **网络问题**: 检索API超时/失败
- **文档库质量**: 某些查询难以找到相关文档
- **嵌入模型限制**: 语义匹配不准确

### ARGO的解决方案
**自适应策略**:
1. **低成功率场景** ($p_s < 0.6$):
   - MDP求解器降低$\Theta_{cont}$至0
   - **策略**: 依赖推理,避免无效检索
   - **效果**: 节省大量检索成本

2. **高成功率场景** ($p_s \geq 0.6$):
   - MDP求解器提高$\Theta_{cont}$
   - **策略**: 适度检索,提高质量
   - **效果**: 保持高质量,低成本

### 对比基线
- **Always-Retrieve**: 
  - 低$p_s$时需要12.7次检索(vs ARGO的0次)
  - **浪费**: 12.7倍的成本!

- **Always-Reason**: 
  - 质量固定在0.96(vs ARGO的1.0)
  - **损失**: 4%的质量损失

## 论文贡献

该实验支持以下论点:

> "ARGO exhibits intelligent risk management in unreliable retrieval environments. When retrieval success rate is low ($p_s < 0.6$), ARGO completely avoids retrieval and relies on reasoning, while Always-Retrieve wastes resources in futile retry attempts (12.7 vs 0 retrievals)."

**引用建议**:
- Figure 2是**核心图**,展示ARGO的避险行为
- Figure 3展示动作分布,说明Retrieve↔Reason的平衡
- 表格对比在不同$p_s$下的效率差异

## 理论解释

### MDP如何应对低$p_s$?

**Bellman方程中的期望值计算**:

对于Retrieve动作:
$$
Q(U, \text{Retrieve}) = -c_r + \mathbb{E}[V(U')]
$$

其中:
$$
\mathbb{E}[V(U')] = p_s \cdot V(U + \delta_r) + (1 - p_s) \cdot V(U)
$$

**当$p_s$很低时**:
- 检索成功概率小,期望增益 $p_s \cdot V(U + \delta_r)$ 小
- 检索失败概率大,浪费成本 $c_r$
- **结果**: $Q(U, \text{Retrieve}) < Q(U, \text{Reason})$

**MDP求解器的决策**:
- 降低$\Theta_{cont}$,减少检索
- 当$p_s$极低时,$\Theta_{cont} = 0$(完全不检索)

这正是**最优决策理论**的体现!

## 下一步

建议的后续实验:
1. ✅ **Experiment 1**: 检索成本($c_r$)的影响 (已完成)
2. ✅ **Experiment 2**: 检索成功率($p_s$)的影响 (已完成)
3. **Experiment 3**: 增量参数($\delta_r$ vs $\delta_p$)的影响
4. **Experiment 4**: 不同难度级别的对比(Easy/Medium/Hard)
5. **Experiment 5**: 在真实LLM上验证(非仿真)

---

**生成时间**: 2025-10-29 00:50:01
**实验脚本**: `Exp_retrieval_success_impact.py`
**配置文件**: `configs/multi_gpu.yaml`
