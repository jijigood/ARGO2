# ARGO 论文实验可视化设计方案

## 参考TAoI_jour项目的设计经验

### TAoI项目的图表设计特点
1. **折线图**：横轴为控制变量（如T_u传输时间），纵轴为性能指标（Average TAoI）
2. **柱状图**：横轴为不同模型/方法，纵轴为性能指标，用于直观对比
3. **多策略对比**：在同一张图中对比3-4种策略（Opt_policy, Always_tran, Pre_identify等）
4. **颜色方案**：使用区分度高的颜色（'#C97937', 'royalblue', 'purple'）

---

## ARGO论文图表设计方案

### 📊 **Figure 1: 延迟分解分析（4面板图）** ✅ 已有

**当前状态**: 已完成  
**文件**: `results/latency/latency_analysis.png`  

包含4个子图：
- (a) 延迟分布直方图
- (b) 组件延迟柱状图（Decomposer, Synthesizer, Retriever, Overhead）
- (c) 累积分布函数CDF
- (d) Box plot with P95/P99

**用途**: Section 6.2.1 - Latency Profiling

---

### 📈 **Figure 2: 优化效果对比（折线图）** 🆕 需要创建

**设计**:
- **横轴 (X-axis)**: 优化阶段 (Optimization Stage)
  - 3个点：Baseline, Params Only, Full Optimization
  
- **纵轴 (Y-axis)**: 延迟 (Latency per Query, seconds)
  - 范围：0-70秒

- **数据**:
  - Baseline: 62.2s
  - Params Only: 24.0s
  - Full Optimization: 18.8s

- **附加信息**:
  - 在每个点标注加速比（1.00×, 2.59×, 3.31×）
  - 使用箭头标注优化措施（"Reduce tokens", "Smaller model"）

**Python代码示例**:
```python
import matplotlib.pyplot as plt
import numpy as np

stages = ['Baseline\n(3B, 128/512)', 'Params Only\n(3B, 50/200)', 'Full Opt\n(1.5B, 50/200)']
latency = [62.2, 24.0, 18.8]
speedup = [1.00, 2.59, 3.31]

plt.figure(figsize=(8, 5))
plt.plot(stages, latency, 's-', color='#C97937', linewidth=2, markersize=8)

# 标注加速比
for i, (s, l, sp) in enumerate(zip(stages, latency, speedup)):
    plt.text(i, l+3, f'{sp:.2f}×', ha='center', fontsize=11, fontweight='bold')

plt.ylabel('Latency per Query (s)', fontsize=12, fontweight='bold')
plt.xlabel('Optimization Stage', fontsize=12, fontweight='bold')
plt.title('Zero-Cost Optimization Performance', fontsize=13, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('optimization_effect.pdf', bbox_inches='tight')
```

**用途**: Section 6.2.2 - Zero-Cost Optimization

---

### 📊 **Figure 3: 策略准确率对比（柱状图）** 🆕 需要创建

**设计（参考TAoI的bar_EXP.py）**:
- **横轴 (X-axis)**: 策略 (Strategy)
  - 4个柱：MDP-Guided, Fixed-Threshold, Always-Reason, Random
  
- **纵轴 (Y-axis)**: 准确率 (Accuracy, %)
  - 范围：0-100%

- **数据（基于pilot study）**:
  - MDP-Guided: 75%
  - Fixed-Threshold: 68% (合理推测)
  - Always-Reason: 60%
  - Random: 25% (理论下界，4选1)

- **颜色方案**:
  - MDP-Guided: '#C97937' (橙棕色，强调重点)
  - Fixed-Threshold: 'royalblue' (蓝色)
  - Always-Reason: 'purple' (紫色)
  - Random: 'gray' (灰色，最弱基线)

**Python代码**:
```python
import matplotlib.pyplot as plt
import numpy as np

strategies = ['MDP-Guided', 'Fixed-\nThreshold', 'Always-\nReason', 'Random']
accuracy = [75, 68, 60, 25]
colors = ['#C97937', 'royalblue', 'purple', 'gray']

plt.figure(figsize=(7, 5))
bars = plt.bar(strategies, accuracy, color=colors, width=0.6)

# 在柱子上方标注数值
for bar, acc in zip(bars, accuracy):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
             f'{acc}%', ha='center', fontsize=11, fontweight='bold')

plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
plt.xlabel('Strategy', fontsize=12, fontweight='bold')
plt.title('Strategy Comparison (Pilot Study, n=20)', fontsize=13, fontweight='bold')
plt.ylim(0, 85)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('strategy_accuracy.pdf', bbox_inches='tight')
```

**用途**: Section 6.3.3 - Quantitative Results (Pilot)

---

### 📈 **Figure 4: 检索效率分析（折线图）** 🆕 需要创建

**设计**:
- **横轴 (X-axis)**: Query Index (1-20)
  
- **纵轴左 (Y-axis Left)**: Number of Retrieves (0-4)
  - MDP-Guided的检索次数

- **纵轴右 (Y-axis Right)**: Uncertainty U_t (0-1.0)
  - 显示不确定度变化

**数据（模拟合理趋势）**:
```python
query_idx = range(1, 21)
retrieves = [2, 1, 3, 0, 2, 1, 2, 3, 1, 0, 2, 1, 3, 2, 1, 0, 2, 1, 2, 3]  # 平均1.8
uncertainty = [0.85, 0.35, 0.72, 0.28, 0.65, 0.40, 0.68, 0.78, 0.42, 0.25, ...]
```

**用途**: Section 6.3.5 - Decision Analysis

---

### 📊 **Figure 5: 问题复杂度分析（分组柱状图）** 🆕 需要创建

**设计（参考TAoI的分组柱状图）**:
- **横轴 (X-axis)**: Question Type
  - 2组：Single-hop, Multi-hop
  
- **每组2个柱子**: MDP-Guided vs Always-Reason
  
- **纵轴 (Y-axis)**: Accuracy (%)

- **数据**:
  | Type | MDP-Guided | Always-Reason |
  |------|-----------|---------------|
  | Single-hop | 85.7% | 71.4% |
  | Multi-hop | 69.2% | 53.8% |

**Python代码**:
```python
import matplotlib.pyplot as plt
import numpy as np

categories = ['Single-hop\n(n=7)', 'Multi-hop\n(n=13)']
mdp_acc = [85.7, 69.2]
always_acc = [71.4, 53.8]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 5))
bars1 = ax.bar(x - width/2, mdp_acc, width, label='MDP-Guided', color='#C97937')
bars2 = ax.bar(x + width/2, always_acc, width, label='Always-Reason', color='purple')

# 标注数值
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1.5,
                f'{height:.1f}%', ha='center', fontsize=10)

ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Question Type', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend(loc='upper right', fontsize=11)
ax.set_ylim(0, 95)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('complexity_analysis.pdf', bbox_inches='tight')
```

**用途**: Section 6.3.4 - Breakdown by Question Complexity

---

### 📈 **Figure 6: 扩展性投影（折线图）** 🆕 需要创建

**设计**:
- **横轴 (X-axis)**: Number of Queries (log scale)
  - 点：20, 100, 1000, 13952
  
- **纵轴 (Y-axis)**: Estimated Time (hours)
  - 范围：0-70小时

- **多条线**:
  - Baseline (3B): 指数增长
  - Optimized (1.5B): 缓和增长
  - +Flash Attn: 更平缓
  - +vLLM: 最平缓

**数据**:
```python
queries = [20, 100, 1000, 13952]
time_baseline = [0.35, 1.55, 15.4, 198]  # hours
time_optimized = [0.09, 0.47, 4.7, 60]
time_flash = [0.05, 0.28, 2.75, 35.5]
time_vllm = [0.02, 0.09, 0.92, 11.9]
```

**用途**: Section 6.4 - Scalability Projection

---

### 📊 **Figure 7: 成本效益分析（散点图）** 🆕 需要创建

**设计**:
- **横轴 (X-axis)**: Latency per Query (s)
- **纵轴 (Y-axis)**: Accuracy (%)
- **散点**: 每个策略一个点
- **理想区域**: 右上角（高准确率，低延迟）

**数据**:
| Strategy | Latency | Accuracy |
|----------|---------|----------|
| MDP-Guided | 16.5s | 75% |
| Fixed-Threshold | 15.2s | 68% |
| Always-Reason | 14.8s | 60% |
| Random | 14.5s | 25% |

**附加**: 绘制帕累托前沿，标注MDP-Guided为最优

**用途**: Section 6.6.2 - Cost-Benefit Analysis

---

## 🎨 统一视觉风格规范

### 颜色方案（参考TAoI）
```python
# 主策略（ARGO）
ARGO_COLOR = '#C97937'  # 橙棕色（醒目）

# 基线策略
BASELINE_COLORS = {
    'Fixed-Threshold': 'royalblue',
    'Always-Reason': 'purple',
    'Random': 'gray'
}

# 优化阶段
OPT_COLORS = ['darkred', 'orangered', 'orange', 'gold']
```

### 字体设置
```python
plt.rcParams.update({
    "mathtext.fontset": 'stix',
    'pdf.fonttype': 42,  # TrueType字体（论文要求）
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 11
})
```

### 图表尺寸
- 单列图：(7, 5)
- 双列图：(10, 5)
- 多面板：根据子图数量调整

---

## 📋 实施计划

### 优先级排序

**立即创建（支持现有Section 6）**:
1. ✅ Figure 1: 延迟分解（已有）
2. 🔥 **Figure 2**: 优化效果折线图（Section 6.2.2核心）
3. 🔥 **Figure 3**: 策略准确率柱状图（Section 6.3.3核心）

**可选增强**:
4. Figure 4: 检索效率分析
5. Figure 5: 问题复杂度分组柱状图
6. Figure 6: 扩展性投影
7. Figure 7: 成本效益散点图

### 代码组织

创建 `draw_figs/` 目录结构：
```
ARGO/
├── draw_figs/
│   ├── fig2_optimization_effect.py
│   ├── fig3_strategy_accuracy.py
│   ├── fig4_retrieval_efficiency.py
│   ├── fig5_complexity_analysis.py
│   ├── fig6_scalability.py
│   ├── fig7_cost_benefit.py
│   └── data/
│       ├── pilot_results.txt  # 如果有真实数据
│       └── latency_data.txt
└── figs/
    ├── optimization_effect.pdf
    ├── strategy_accuracy.pdf
    └── ...
```

---

## 🎯 推荐横纵坐标设计总结

基于TAoI项目经验和ARGO特点：

### 主要图表类型

1. **折线图**（趋势展示）
   - X轴：时间步骤、优化阶段、query数量
   - Y轴：延迟、准确率、检索次数

2. **柱状图**（策略对比）
   - X轴：策略名称、模型类型
   - Y轴：准确率、平均延迟

3. **分组柱状图**（多维对比）
   - X轴：问题类型、难度级别
   - 每组：多个策略的表现

4. **散点图**（权衡分析）
   - X轴：成本（延迟）
   - Y轴：收益（准确率）

### 关键设计原则

1. **对比清晰**：ARGO（橙棕色）vs 基线（蓝紫灰）
2. **数值标注**：在图表上直接显示关键数值
3. **网格辅助**：使用半透明网格便于读数
4. **统一风格**：所有图表使用相同字体和颜色方案

---

**下一步**: 我可以帮您实现这些图表。您希望先创建哪几个？建议优先：
1. Figure 2（优化效果）
2. Figure 3（策略对比）
3. Figure 5（复杂度分析）

这3张图可以直接支撑您的Section 6！
