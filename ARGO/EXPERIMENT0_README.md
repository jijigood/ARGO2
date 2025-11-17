# Experiment 0: Threshold Structure Validation

## 概述

这是 ARGO 项目的**第一个基础实验**，用于验证 MDP 解的理论基础。在运行任何真实 LLM 实验之前，必须先验证理论框架的正确性。

## 实验目标

验证 **Theorem 1: Two-Level Threshold Structure**

理论声明:
```
最优策略 π* 具有两级阈值结构:
- π*(U) = Retrieve,    if U < Θ_cont (继续阈值)
- π*(U) = Reason,      if Θ_cont ≤ U < Θ* (终止阈值)
- π*(U) = Terminate,   if U ≥ Θ*
```

## 验证内容

### Part 1: 阈值存在性验证

在4个不同参数集下验证:
1. **Baseline**: 标准参数 (c_r=0.05, p_s=0.8)
2. **High c_r**: 高检索成本 (c_r=0.10)
3. **Low p_s**: 低检索成功率 (p_s=0.6)
4. **High p_s**: 高检索成功率 (p_s=0.9)

每个参数集验证:
- ✓ 终止阈值 Θ* 的唯一性
- ✓ 继续阈值 Θ_cont 的存在性
- ✓ 策略单调性: Retrieve → Reason → Terminate

### Part 2: Q 函数性质验证

1. **单调性 (Monotonicity)**
   - V*(U) 应该是非递减的
   - 更高的进展 U 应该有更高的价值

2. **优势函数 (Advantage Function)**
   - A(U) = V_cont(U) - V_term(U)
   - 应该是递减的
   - 表示继续的优势随 U 增加而减少

3. **单交叉性质 (Single-Crossing Property)**
   - A(U) 应该只与零点相交一次
   - 这是最优停止理论的核心性质

### Part 3: 敏感性分析

测试阈值如何响应参数变化:
- **独立变量**: 检索成本 c_r (0.02 → 0.20)
- **观测指标**: Θ_cont 和 Θ* 的变化

**预期结果**:
- Θ_cont 应该随 c_r 增加而减少 (避免昂贵的检索)
- Θ* 应该相对稳定 (质量阈值独立于成本)

## 运行实验

```bash
cd /data/user/huangxiaolin/ARGO2/ARGO
python run_exp0.py
```

**运行时间**: ~2-3 分钟 (纯数值计算，无需 LLM)

## 输出文件

### 图表 (figs/)
1. `exp0_threshold_structure_0_baseline.png` - 基线参数的策略结构
2. `exp0_threshold_structure_1_high_c_r.png` - 高检索成本情况
3. `exp0_threshold_structure_2_low_p_s.png` - 低成功率情况
4. `exp0_threshold_structure_3_high_p_s.png` - 高成功率情况
5. `exp0_threshold_sensitivity.png` - 阈值敏感性分析

### 数据文件 (results/exp0_threshold_validation/)
1. `threshold_validation_summary.csv` - 验证结果摘要
2. `threshold_sensitivity_analysis.csv` - 敏感性分析数据

## 理解输出图表

### 图 1-4: 策略结构验证 (每个参数集)

每个图包含4个子图:

**左上: 最优策略 π*(U)**
- 蓝点: Retrieve 动作
- 绿点: Reason 动作  
- 红点: Terminate 动作
- 蓝色虚线: Θ_cont (继续阈值)
- 红色虚线: Θ* (终止阈值)

✅ **正确的图应该显示**:
- 在低 U 值时蓝点 (Retrieve)
- 中等 U 值时绿点 (Reason)
- 高 U 值时红点 (Terminate)
- 清晰的阈值分界

**右上: Q 函数**
- 蓝线: Q(U, Retrieve)
- 绿线: Q(U, Reason)
- 红线: Q(U, Terminate)

✅ **正确的图应该显示**:
- 在不同 U 区域，不同 Q 函数占优
- Q(U, Terminate) 在高 U 时最大

**左下: 价值函数 V*(U)**
- 黑色实线: 最优价值函数

✅ **正确的图应该显示**:
- 单调递增 (或至少非递减)
- 绿色标签 "✓ Monotonic"

**右下: 优势函数 A(U)**
- 紫色线: A(U) = V_cont(U) - V_term(U)
- 橙色虚线: 零交叉点

✅ **正确的图应该显示**:
- 递减趋势
- 只有一个零交叉点
- 绿色标签 "✓ Single Crossing"

### 图 5: 阈值敏感性分析

两个子图显示阈值如何随 c_r 变化:

**左图: Θ_cont vs c_r**
- 蓝色线: 继续阈值随检索成本的变化

✅ **预期行为**:
- Θ_cont 随 c_r 增加而减少
- 表示系统理性地避免昂贵的检索

**右图: Θ* vs c_r**  
- 红色线: 终止阈值随检索成本的变化

✅ **预期行为**:
- Θ* 相对稳定
- 质量要求不应因成本而改变

## 验证清单

运行实验后，检查以下项目:

- [ ] 所有4个参数集的策略图都显示清晰的三区域结构
- [ ] 价值函数 V*(U) 在所有情况下都是单调的
- [ ] 优势函数 A(U) 只有一个零交叉点
- [ ] Θ_cont 随 c_r 增加而减少 (敏感性分析)
- [ ] Θ* 相对稳定 (敏感性分析)
- [ ] 控制台输出显示 "✓ ALL PARAMETER SETS EXHIBIT VALID THRESHOLD STRUCTURE"

## 论文使用

这个实验的输出将作为论文的 **Figure 1**:

> **Figure 1: Empirical Validation of Two-Level Threshold Structure**
> 
> The optimal policy exhibits three distinct regions: (a) Retrieve for U < Θ_cont,
> (b) Reason for Θ_cont ≤ U < Θ*, and (c) Terminate for U ≥ Θ*. The value function
> V*(U) is monotonically increasing, and the advantage function A(U) demonstrates
> the single-crossing property predicted by optimal stopping theory.

## 故障排除

### 问题 1: "Config file not found"
**解决方案**: 代码会自动创建默认配置，这不是错误。

### 问题 2: 策略图没有显示三个清晰区域
**可能原因**:
- 参数设置不合理
- delta_r 或 delta_p 太小
- c_r 和 c_p 差异太小

**解决方案**: 调整 `create_default_config()` 中的参数

### 问题 3: V*(U) 不是单调的
**可能原因**:
- 数值精度问题 (通常可以忽略 < 1e-6 的违例)
- 收敛阈值太大

**解决方案**: 减小 `convergence_threshold` 或增加 `max_iterations`

### 问题 4: 多个零交叉点
**可能原因**:
- 参数组合导致复杂的策略结构
- 质量函数形状不合适

**解决方案**: 检查 quality function 的设置 (sigmoid, linear, sqrt, saturating)

## 理论背景

这个实验验证了以下理论结果:

### Theorem 1 (Two-Level Threshold Structure)
在合理的假设下，最优策略 π* 具有两级阈值结构。

**证明思路**:
1. 优势函数 A(U) = V_cont(U) - V_term(U) 是递减的
2. 由于 A(U) 递减且连续，它最多与零相交一次
3. 在交叉点之前，继续最优；之后，终止最优
4. 在继续区域，Retrieve 和 Reason 之间也有阈值

### Key Properties

1. **Value Function Properties**
   - V*(U) 是凸函数
   - V*(U) 关于 U 单调递增
   - V*(U_max) = σ(U_max) (质量函数)

2. **Optimal Stopping**
   - Θ* 是唯一的最优停止阈值
   - 满足 A(Θ*) = 0

3. **Action Selection**
   - Θ_cont 分离 Retrieve 和 Reason
   - 由 Q(U, Retrieve) = Q(U, Reason) 定义

## 下一步

验证通过后，继续后续实验:

1. **Experiment 1** (`run_exp1_full.py`): 检索成本影响
   - 验证 ARGO 如何响应 c_r 变化
   - 对比固定策略的性能

2. **Experiment 2** (`run_exp2_full.py`): 检索成功率影响
   - 验证 ARGO 如何响应 p_s 变化
   - 测试环境不确定性的鲁棒性

3. **Experiment 3** (`run_exp3_full.py`): Pareto 前沿分析
   - 多目标优化权衡
   - 成本-质量 Pareto 前沿

## 参考文献

- Puterman, M. L. (2014). Markov Decision Processes: Discrete Stochastic Dynamic Programming.
- Peskir, G., & Shiryaev, A. (2006). Optimal Stopping and Free-Boundary Problems.
- Ferguson, T. S. (2008). Optimal Stopping and Applications.

## 联系信息

如有问题，请查看:
- `ARGO_MDP/README.md` - MDP 求解器文档
- `ARCHITECTURE_EXPLANATION.md` - 系统架构说明
- `EXPERIMENTS_INDEX.md` - 所有实验概览
