# ARGO实验总索引

**项目**: ARGO - Adaptive Retrieval with Guided Optimization  
**日期**: 2025-10-29  
**环境**: ARGO conda环境

---

## 实验概览

本项目包含多个系统性实验,用于验证ARGO的核心优势:自适应检索策略优于静态基线。

### 已完成的实验

#### ✅ Experiment 1: 检索成本 ($c_r$) 的影响
**文件**: `Exp_retrieval_cost_impact.py`  
**报告**: `EXPERIMENT1_REPORT.md`  
**数据**: `draw_figs/data/exp1_retrieval_cost_impact_*.json`  
**图表**: `figs/exp1_*.png`

**核心发现**:
- ARGO的检索次数随$c_r$增加而急剧下降(5.1 → 0次)
- 基线策略检索次数恒定(无适应性)
- 证明了**成本敏感的自适应性**

**关键图表**:
- `exp1_cost_vs_quality.png`: 成本vs质量
- `exp1_cost_vs_retrievals.png`: 成本vs检索次数 ⭐ **核心图**
- `exp1_threshold_evolution.png`: 阈值演化

---

#### ✅ Experiment 2: 检索成功率 ($p_s$) 的影响
**文件**: `Exp_retrieval_success_impact.py`  
**报告**: `EXPERIMENT2_REPORT.md`  
**数据**: `draw_figs/data/exp2_retrieval_success_impact_*.json`  
**图表**: `figs/exp2_*.png`

**核心发现**:
- 低$p_s$时,ARGO避免检索(0次 vs Always-Retrieve的12.7次)
- 高$p_s$时,ARGO适度检索(1次)
- 证明了**检索不确定性管理**能力

**关键图表**:
- `exp2_ps_vs_quality.png`: 成功率vs质量
- `exp2_ps_vs_retrievals.png`: 成功率vs检索次数 ⭐ **核心图**
- `exp2_action_distribution.png`: 动作分布

---

## 实验参数总结

### 共同测试集
- **数据集**: ORAN-Bench-13K
- **问题数量**: 100道
- **难度**: Medium
- **随机种子**: 42

### 评估策略
所有实验都对比以下4种策略:
1. **ARGO**: MDP引导的自适应策略
2. **Always-Retrieve**: 静态检索策略
3. **Always-Reason**: 静态推理策略
4. **Random**: 随机策略(50-50)

### MDP基础参数
```yaml
delta_r: 0.25    # 检索增量
delta_p: 0.08    # 推理增量
c_p: 0.02        # 推理成本
mu: 0.6          # 质量参数
gamma: 0.98      # 折扣因子
U_max: 1.0       # 信息进度上限
```

### 实验自变量

| 实验 | 自变量 | 范围 | 步数 | 固定值 |
|-----|-------|------|------|--------|
| Exp 1 | $c_r$ | $c_p$ ~ $10c_p$ | 10 | $p_s=0.8$ |
| Exp 2 | $p_s$ | 0.3 ~ 1.0 | 8 | $c_r=0.05$ |

---

## 快速运行指南

### 环境准备
```bash
# 激活conda环境
source activate ARGO

# 进入项目目录
cd /data/user/huangxiaolin/ARGO2/ARGO
```

### 运行实验1
```bash
python Exp_retrieval_cost_impact.py
```
**耗时**: ~2分钟  
**输出**:
- 结果文件: `draw_figs/data/exp1_*.json`
- 图表: `figs/exp1_*.png` (3张)

### 运行实验2
```bash
python Exp_retrieval_success_impact.py
```
**耗时**: ~2分钟  
**输出**:
- 结果文件: `draw_figs/data/exp2_*.json`
- 图表: `figs/exp2_*.png` (3张)

### 查看结果
```bash
# 查看图表
ls -lh figs/exp*.png

# 查看数据
cat draw_figs/data/exp1_*.json | jq '.results.policies.ARGO'
```

---

## 实验结果对比

### 关键指标

#### Experiment 1: 成本适应性

| $c_r/c_p$ | ARGO检索次数 | Always-Retrieve检索次数 | 差异 |
|-----------|------------|----------------------|------|
| 1.0x      | 5.1        | 5.1                  | 0%   |
| 2.0x      | 1.3        | 5.1                  | -75% |
| 4.0x+     | 0.0        | 5.1                  | -100% |

**结论**: ARGO在高成本下完全停止检索,Always-Retrieve无适应性。

#### Experiment 2: 不确定性管理

| $p_s$ | ARGO检索次数 | Always-Retrieve检索次数 | 效率提升 |
|-------|------------|----------------------|---------|
| 0.3   | 0.0        | 12.7                 | **无限** |
| 0.6   | 1.6        | 6.7                  | 76%     |
| 1.0   | 1.0        | 4.0                  | 75%     |

**结论**: 低$p_s$环境下,ARGO避免无效检索,Always-Retrieve陷入重试陷阱。

---

## 核心贡献总结

### 1. 成本自适应 (Exp 1)
**问题**: 检索API费用上涨怎么办?  
**ARGO解决方案**: 动态降低检索,转向推理  
**效果**: 高成本下检索减少100%,质量不变

### 2. 不确定性管理 (Exp 2)
**问题**: 检索质量不稳定怎么办?  
**ARGO解决方案**: 低成功率时避免检索  
**效果**: 节省12.7倍的检索成本

### 3. MDP的理论优势
- **最优性**: Value Iteration求解Bellman方程
- **自适应性**: 策略根据环境参数变化
- **可解释性**: 阈值变化符合理论预期

---

## 论文贡献

这两个实验为论文提供了关键实证支持:

### Section 6.1: Cost Sensitivity (Exp 1)
> "Figure X shows that ARGO intelligently reduces retrieval operations from 5.1 to 0 as retrieval cost increases from $c_p$ to $4c_p$, while Always-Retrieve maintains a constant 5.1 retrievals regardless of cost, demonstrating ARGO's cost-adaptive behavior."

### Section 6.2: Reliability Under Uncertainty (Exp 2)
> "In unreliable retrieval environments ($p_s = 0.3$), ARGO avoids retrieval entirely (0 retrievals), while Always-Retrieve wastes resources in futile retry attempts (12.7 retrievals). This 12.7x efficiency gain demonstrates ARGO's intelligent risk management."

### 推荐引用图表
- **Figure 6.1** (Exp 1): `exp1_cost_vs_retrievals.png`
- **Figure 6.2** (Exp 2): `exp2_ps_vs_retrievals.png`

---

## 建议的后续实验

### Experiment 3: 增量参数比较
**目标**: 比较$\delta_r$ vs $\delta_p$的影响  
**方法**: 固定其他参数,扫描$\delta_r/\delta_p$比率  
**预期**: ARGO偏好高$\delta_r$(检索增益大)

### Experiment 4: 难度分级评估
**目标**: 对比Easy/Medium/Hard问题的性能  
**方法**: 分别在3个难度上运行实验  
**预期**: Hard问题需要更多检索,ARGO能自适应

### Experiment 5: 真实LLM验证
**目标**: 在真实LLM上验证(非仿真)  
**方法**: 集成Qwen2.5-7B,真实检索ORAN文档  
**预期**: 实证验证仿真结果

### Experiment 6: 大规模评估
**目标**: 在完整13K问题集上评估  
**方法**: 使用多GPU并行加速  
**预期**: 统计显著性验证

---

## 技术细节

### 仿真模型

#### 质量函数
```python
def quality_function(U):
    # Linear (default)
    return U / U_max
```

#### 检索成功模拟
```python
if np.random.random() < p_s:
    U = min(U + delta_r, 1.0)  # Success
else:
    U = U  # Failure
```

#### 推理模拟
```python
U = min(U + delta_p, 1.0)  # Deterministic
```

### MDP求解

#### Value Iteration
```python
for iteration in range(max_iterations):
    for U in U_grid:
        Q[U, retrieve] = -c_r + p_s * V[U + delta_r] + (1-p_s) * V[U]
        Q[U, reason] = -c_p + V[U + delta_p]
        Q[U, terminate] = quality_function(U)
        V[U] = max(Q[U, :])
```

#### 阈值提取
```python
# Termination threshold
theta_star = argmax_U { Q[U, terminate] >= max(Q[U, retrieve], Q[U, reason]) }

# Continuation threshold
theta_cont = argmax_U { Q[U, retrieve] < Q[U, reason] }
```

---

## 文件结构

```
ARGO2/ARGO/
├── Exp_retrieval_cost_impact.py      # 实验1脚本
├── Exp_retrieval_success_impact.py   # 实验2脚本
├── EXPERIMENT1_REPORT.md             # 实验1报告
├── EXPERIMENT2_REPORT.md             # 实验2报告
├── EXPERIMENTS_INDEX.md              # 本文件
├── configs/
│   └── multi_gpu.yaml                # MDP配置
├── draw_figs/
│   └── data/
│       ├── exp1_*.json               # 实验1数据
│       └── exp2_*.json               # 实验2数据
└── figs/
    ├── exp1_cost_vs_quality.png
    ├── exp1_cost_vs_retrievals.png   ⭐
    ├── exp1_threshold_evolution.png
    ├── exp2_ps_vs_quality.png
    ├── exp2_ps_vs_retrievals.png     ⭐
    └── exp2_action_distribution.png
```

---

## 常见问题 (FAQ)

### Q1: 为什么质量都是1.0或0.96?
A: 这是仿真模型的简化。在真实实验中,质量会更复杂地依赖于检索内容和LLM推理。

### Q2: 实验耗时多久?
A: 每个实验约2分钟(100问题,10步参数扫描)。使用仿真模拟,无需加载LLM。

### Q3: 如何修改参数?
A: 编辑`configs/multi_gpu.yaml`或直接修改实验脚本中的参数。

### Q4: 如何增加问题数量?
A: 修改`n_test_questions`参数:
```python
exp = CostImpactExperiment(n_test_questions=1000)
```

### Q5: 可以在其他数据集上运行吗?
A: 可以!修改`ORANBenchmark`加载器,或替换为自定义数据集。

---

## 引用

如果使用这些实验结果,请引用:

```bibtex
@article{argo2025,
  title={ARGO: Adaptive Retrieval with Guided Optimization for Open Radio Access Networks},
  author={ARGO Team},
  journal={arXiv preprint},
  year={2025}
}
```

---

**最后更新**: 2025-10-29 00:50  
**维护者**: ARGO Team  
**联系方式**: huangxiaolin@labi3c
