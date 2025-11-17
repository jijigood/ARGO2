# Experiment 0 快速开始指南

## 🎯 目标
验证 ARGO MDP 解的理论基础：两级阈值结构 (Theorem 1)

## 📦 已创建的文件

```
ARGO2/ARGO/
├── Exp0_threshold_structure_validation.py  # 核心实验代码 (600+ 行)
├── run_exp0.py                             # 完整实验运行脚本
├── test_exp0_quick.py                      # 快速测试脚本 (~30秒)
├── check_exp0_setup.sh                     # 环境检查脚本
├── EXPERIMENT0_README.md                   # 详细文档
└── EXPERIMENT0_COMPLETION_SUMMARY.md       # 完成总结
```

## 🚀 快速开始

### 步骤 1: 验证环境
```bash
cd /data/user/huangxiaolin/ARGO2/ARGO
./check_exp0_setup.sh
```
应该看到: `✓ 所有必要文件都已就绪!`

### 步骤 2: 快速测试 (推荐)
```bash
python test_exp0_quick.py
```
- **运行时间**: ~30秒
- **输出**: 单个测试图 + 控制台报告
- **目的**: 验证代码是否正常工作

**预期输出**:
```
快速测试: 阈值结构验证
...
终止阈值 Θ* = 0.9XXX
继续阈值 Θ_cont = 0.XXX
价值函数单调性: ✓ PASS
单交叉性质: ✓ PASS
✓ 所有验证通过! 代码工作正常。
```

### 步骤 3: 完整实验
```bash
python run_exp0.py
```
- **运行时间**: ~2-3分钟
- **输出**: 5个图表 + 2个CSV文件
- **目的**: 完整的理论验证

## 📊 输出文件

### 图表 (figs/)
1. **exp0_threshold_structure_0_baseline.png** - 基线参数 ⭐ **论文 Figure 1(a)**
   - 4个子图展示策略结构
   - 清晰的 Retrieve → Reason → Terminate 区域

2. **exp0_threshold_structure_1_high_c_r.png** - 高检索成本
3. **exp0_threshold_structure_2_low_p_s.png** - 低成功率
4. **exp0_threshold_structure_3_high_p_s.png** - 高成功率

5. **exp0_threshold_sensitivity.png** - 敏感性分析 ⭐ **论文 Figure 1(b)**
   - 阈值如何响应参数变化

### 数据 (results/exp0_threshold_validation/)
- **threshold_validation_summary.csv** - 验证结果表
- **threshold_sensitivity_analysis.csv** - 敏感性数据

## 🔍 理解输出

### 每个策略结构图包含4个子图:

```
┌─────────────────────┬─────────────────────┐
│ 1. 最优策略 π*(U)    │ 2. Q 函数           │
│ - 蓝点: Retrieve    │ - 蓝线: Q(Retrieve) │
│ - 绿点: Reason      │ - 绿线: Q(Reason)   │
│ - 红点: Terminate   │ - 红线: Q(Terminate)│
├─────────────────────┼─────────────────────┤
│ 3. 价值函数 V*(U)   │ 4. 优势函数 A(U)    │
│ - 应该单调递增      │ - 应该递减          │
│ - ✓ Monotonic 标签  │ - 只有1个零点       │
└─────────────────────┴─────────────────────┘
```

### ✅ 正确的结果应该显示:

**左上 (最优策略)**:
- 低 U: 蓝色区域 (Retrieve)
- 中 U: 绿色区域 (Reason)
- 高 U: 红色区域 (Terminate)
- 两条清晰的阈值虚线

**右上 (Q函数)**:
- 三条线在不同区域交叉
- 红线 (Terminate) 在高 U 时最高

**左下 (价值函数)**:
- 平滑上升的黑色曲线
- 绿色 "✓ Monotonic" 标签

**右下 (优势函数)**:
- 紫色曲线从正到负
- 只有一个零点 (橙色虚线)
- 绿色 "✓ Single Crossing" 标签

## 🎓 理论背景

### Theorem 1: Two-Level Threshold Structure

**声明**: 最优策略有两个阈值 Θ_cont 和 Θ*，使得:
```
π*(U) = Retrieve,    if U < Θ_cont
π*(U) = Reason,      if Θ_cont ≤ U < Θ*
π*(U) = Terminate,   if U ≥ Θ*
```

**验证方法**:
1. **直接验证**: 检查策略是否遵循上述结构
2. **性质验证**: V*(U) 单调, A(U) 单交叉
3. **敏感性验证**: 阈值理性响应参数变化

### 为什么重要?

- ✅ **理论基础**: ARGO 不是启发式，有理论保证
- ✅ **可解释性**: 阈值结构易于理解
- ✅ **鲁棒性**: 单调性质保证稳定
- ✅ **适应性**: 阈值调整展示智能

## 🐛 故障排除

### 问题 1: ImportError
```python
ImportError: No module named 'mdp_solver'
```
**解决**: 检查路径
```bash
ls ../ARGO_MDP/src/mdp_solver.py
```

### 问题 2: 策略没有三个区域
**可能原因**: 参数不合理  
**解决**: 检查参数满足 delta_r > delta_p, c_r > c_p

### 问题 3: V*(U) 不单调
**可能原因**: 数值精度  
**解决**: 允许 < 1e-6 的误差，这是正常的

### 问题 4: 多个零交叉点
**可能原因**: 质量函数形状  
**解决**: 尝试不同的 quality mode (sigmoid/linear/sqrt)

## 📝 检查清单

运行实验后，检查:
- [ ] 快速测试通过 (test_exp0_quick.py)
- [ ] 完整实验生成5个图表
- [ ] 所有4个参数集显示清晰的三区域结构
- [ ] 价值函数单调 (✓ Monotonic)
- [ ] 优势函数单交叉 (✓ Single Crossing)
- [ ] 控制台显示 "ALL PARAMETER SETS EXHIBIT VALID THRESHOLD STRUCTURE"
- [ ] Θ_cont 随 c_r 增加而减少 (敏感性图)

## 📄 论文使用

### Figure 1: Threshold Structure Validation

**Panel (a)**: 使用 `exp0_threshold_structure_0_baseline.png`
> The optimal policy exhibits three distinct regions: Retrieve (blue),
> Reason (green), and Terminate (red), separated by two thresholds
> Θ_cont and Θ*.

**Panel (b)**: 使用 `exp0_threshold_sensitivity.png`
> The continuation threshold Θ_cont adapts rationally to retrieval
> cost c_r, decreasing as retrieval becomes more expensive.

## 🔗 相关文件

- **详细文档**: `EXPERIMENT0_README.md`
- **完成总结**: `EXPERIMENT0_COMPLETION_SUMMARY.md`
- **实验索引**: `EXPERIMENTS_INDEX.md`
- **MDP求解器**: `../ARGO_MDP/src/mdp_solver.py`

## ⏭️ 下一步

验证通过后，运行后续实验:

```bash
# Experiment 1: 检索成本影响 (使用真实LLM)
python run_exp1_full.py

# Experiment 2: 检索成功率影响 (使用真实LLM)
python run_exp2_full.py

# Experiment 3: Pareto前沿分析
python run_exp3_full.py
```

## 💡 技巧

1. **先运行快速测试**: 确保代码正常再运行完整实验
2. **查看图表**: 比文字更直观
3. **保存结果**: 图表和CSV会自动保存
4. **参考文档**: `EXPERIMENT0_README.md` 有详细说明

## 📧 支持

如有问题:
1. 查看 `EXPERIMENT0_README.md` 的故障排除部分
2. 检查控制台输出的错误信息
3. 验证环境: `./check_exp0_setup.sh`

---

**准备就绪! 现在可以运行实验了 🚀**

```bash
python test_exp0_quick.py   # 30秒快速测试
python run_exp0.py          # 2-3分钟完整验证
```
