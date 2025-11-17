# 实验1: 检索成本影响 - 统计学有效实验指南

## 📋 概述

本实验评估检索成本对ARGO策略性能的影响，使用**多个随机种子**确保统计学有效性。

### ✅ 修正内容

1. **多随机种子** (5个) - 确保统计显著性
2. **多难度级别** (Easy/Medium/Hard) - 展示泛化能力
3. **合理样本量** (100题/难度) - 平衡效率与精度
4. **统计分析** - 置信区间、t检验、效应量

### ⏱️ 运行时间估算

| 模式 | 配置 | 时间 (8 GPUs) |
|------|------|--------------|
| 快速验证 | 3种子 × Hard | ~3-4小时 |
| 推荐配置 | 5种子 × 3难度 | ~10-12小时 |
| 最小配置 | 3种子 × 3难度 | ~6-8小时 |

---

## 🚀 快速开始

### 方式1: 快速验证 (推荐新手)

先运行快速验证确保配置正确：

```bash
# 3种子 × Hard难度 × 100题 = 3小时
bash run_exp1_quick_validation.sh
```

成功后，查看结果：

```bash
python Exp1_aggregate_and_analyze.py
```

### 方式2: 完整实验 (发表级别)

直接运行完整实验：

```bash
# 5种子 × 3难度 × 100题 = 10-12小时
bash run_exp1_full_optimized.sh
```

---

## 📊 完整工作流

### 步骤1: 运行实验

```bash
# 选项A: 快速验证
bash run_exp1_quick_validation.sh

# 选项B: 完整实验 (推荐)
bash run_exp1_full_optimized.sh

# 选项C: 自定义配置
python Exp1_multi_seed_wrapper.py \
    --n-seeds 5 \
    --n-questions 100 \
    --difficulties easy,medium,hard \
    --gpus 0,1,2,3,4,5,6,7
```

### 步骤2: 聚合结果与统计分析

```bash
python Exp1_aggregate_and_analyze.py
```

**输出:**
- `draw_figs/data/exp1_aggregated_XXXXXX.csv` - 聚合统计数据
- `draw_figs/data/exp1_statistical_tests_XXXXXX.csv` - 显著性检验结果
- 终端输出包含论文摘要建议文本

### 步骤3: 生成图表

```bash
# 使用步骤2生成的聚合文件
python Exp1_plots.py draw_figs/data/exp1_aggregated_XXXXXX.csv
```

**输出:**
- `figs/exp1_graph1A_cost_vs_accuracy_with_ci.png` - 成本vs准确率 (带误差条)
- `figs/exp1_graph1B_cost_vs_retrievals_with_ci.png` - 成本vs检索次数 (带误差条)
- `figs/exp1_combined_all_difficulties.png` - 所有难度组合视图
- `figs/exp1_supplementary_reduction_percentage.png` - 检索减少百分比

---

## 📁 文件结构

```
ARGO/
├── Exp_real_cost_impact_v2.py          # 核心实验脚本 (支持custom模式)
├── Exp1_multi_seed_wrapper.py          # 多种子包装器
├── Exp1_aggregate_and_analyze.py       # 统计分析
├── Exp1_plots.py                       # 增强版可视化 (带误差条)
├── run_exp1_quick_validation.sh        # 快速验证脚本
├── run_exp1_full_optimized.sh          # 完整实验脚本
└── EXPERIMENT1_WORKFLOW.md             # 本文档

draw_figs/data/
├── exp1_real_cost_impact_custom_*.json # 原始结果 (每个种子一个)
├── exp1_aggregated_*.csv               # 聚合统计数据
└── exp1_statistical_tests_*.csv        # 统计检验结果

figs/
└── exp1_*.png                          # 生成的图表
```

---

## 🔧 高级用法

### 单独运行某个难度和种子

```bash
python Exp_real_cost_impact_v2.py \
    --mode custom \
    --n-questions 100 \
    --difficulty hard \
    --seed 42 \
    --gpus 0,1,2,3,4,5,6,7
```

### 只运行特定难度的多种子

```bash
python Exp1_multi_seed_wrapper.py \
    --n-seeds 5 \
    --n-questions 100 \
    --difficulties hard \
    --gpus 0,1,2,3,4,5,6,7
```

### 增加种子数量 (更强的统计能力)

```bash
python Exp1_multi_seed_wrapper.py \
    --n-seeds 10 \
    --n-questions 100 \
    --difficulties easy,medium,hard \
    --gpus 0,1,2,3,4,5,6,7
```

---

## 📈 结果解读

### 1. 聚合统计数据 (`exp1_aggregated_*.csv`)

包含每个成本点的均值、标准差、标准误、95%置信区间：

```
difficulty,c_r,c_r_multiplier,ARGO_accuracy_mean,ARGO_accuracy_ci95,...
hard,0.02,1.0,0.852,0.023,...
hard,0.04,2.0,0.863,0.019,...
```

### 2. 统计检验结果 (`exp1_statistical_tests_*.csv`)

包含配对t检验、p值、Cohen's d、效应量：

```
difficulty,comparison,p_value,cohens_d,effect_size,percent_reduction
hard,ARGO vs Always-Retrieve,0.0003,1.24,large,43.2
```

### 3. 论文中的表述

基于统计分析，你可以这样写：

> "在ORAN-Bench-13K基准测试中，ARGO在高成本场景下将检索调用次数
> 减少了43 ± 5% (t(4)=8.23, p<0.001, Cohen's d=1.24)，同时保持
> 与always-retrieve基线相当的答案质量 (87.4 ± 2.1% vs 86.2 ± 3.1%, 
> p=0.18)。"

---

## ⚠️ 注意事项

### 为什么不使用全部12K题？

1. **统计上不必要**: 100题 × 5种子 = 500个样本点，足够统计分析
2. **时间成本**: 12K题需要20+小时，但不会提高统计有效性
3. **资源浪费**: 更多数据不等于更好的统计结果

### 为什么需要多个种子？

单个种子的结果无法：
- 计算置信区间
- 进行显著性检验
- 声称统计显著性
- 通过同行评审

最少需要**3个种子**，推荐**5个种子**。

### 为什么需要多个难度？

1. 展示ARGO在不同复杂度下的泛化能力
2. 证明策略不是针对特定难度过拟合
3. 提供更全面的性能分析

---

## 🐛 故障排除

### 问题1: GPU内存不足

**解决方案:**
```bash
# 减少同时使用的GPU数
python Exp1_multi_seed_wrapper.py --gpus 0,1,2,3
```

### 问题2: 实验中断

**解决方案:**
```bash
# 查看已完成的种子
ls draw_figs/data/exp1_real_cost_impact_custom_*.json

# 手动运行缺失的种子
python Exp_real_cost_impact_v2.py \
    --mode custom \
    --n-questions 100 \
    --difficulty hard \
    --seed 45 \
    --gpus 0,1,2,3,4,5,6,7
```

### 问题3: 找不到结果文件

**检查:**
```bash
ls -lh draw_figs/data/exp1_real_cost_impact_custom_*.json
```

确保文件名包含 `custom` (不是 `small` 或 `full`)。

---

## 📚 相关文档

- `Exp_real_cost_impact_v2.py` - 查看代码注释了解实现细节
- `EXPERIMENT_DESIGN.md` - 查看完整实验设计
- `COMPARISON_v1_v2.md` - 查看v1与v2的差异

---

## ✅ 检查清单

实验完成后，确保你有：

- [ ] 至少3个不同种子的结果文件
- [ ] 至少1个难度级别 (推荐3个)
- [ ] 聚合统计CSV文件
- [ ] 统计检验CSV文件
- [ ] 4张高分辨率图表 (DPI 300)
- [ ] 终端输出的统计报告

如果全部完成，你的实验结果已达到**发表标准**！

---

**最后更新**: 2025-01-17
**维护者**: ARGO Team
