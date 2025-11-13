"""
Figure 2: Optimization Effect Comparison
=========================================

参考TAoI项目的avg_AoI_T.py设计
展示ARGO零成本优化的3阶段性能提升
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['pdf.fonttype'] = 42

# 设置全局字体
plt.rcParams.update({
    "mathtext.fontset": 'stix',
    'font.size': 11
})

# 优化阶段数据
stages = ['Baseline\n(3B, 128/512)', 'Params Only\n(3B, 50/200)', 'Full Opt\n(1.5B, 50/200)']
latency = [62.2, 24.0, 18.8]
speedup = [1.00, 2.59, 3.31]
colors_stage = ['darkred', 'orangered', '#C97937']  # 渐变色显示优化进度

# 创建图表
fig, ax = plt.subplots(figsize=(8, 5))

# 绘制折线图
line = ax.plot(range(len(stages)), latency, 's-', 
               color='#C97937', linewidth=2.5, markersize=10,
               markerfacecolor='white', markeredgewidth=2,
               label='Latency per Query')

# 填充区域显示改进
for i in range(len(stages)-1):
    ax.fill_between([i, i+1], [latency[i], latency[i+1]], 
                      alpha=0.15, color=colors_stage[i])

# 标注加速比
for i, (l, sp) in enumerate(zip(latency, speedup)):
    # 数值标注
    ax.text(i, l+4, f'{l:.1f}s', ha='center', fontsize=11, fontweight='bold')
    # 加速比标注
    ax.text(i, l-5, f'{sp:.2f}×', ha='center', fontsize=10, 
            color='darkgreen', fontweight='bold')

# 优化措施标注（箭头）
ax.annotate('Reduce tokens\n(-57%)', 
            xy=(0.5, (latency[0]+latency[1])/2), 
            xytext=(0.5, 50),
            fontsize=9, ha='center',
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

ax.annotate('Smaller model\n(-22%)', 
            xy=(1.5, (latency[1]+latency[2])/2), 
            xytext=(1.5, 35),
            fontsize=9, ha='center',
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))

# 设置坐标轴
ax.set_xticks(range(len(stages)))
ax.set_xticklabels(stages, fontsize=11)
ax.set_ylabel('Latency per Query (s)', fontsize=12, fontweight='bold')
ax.set_xlabel('Optimization Stage', fontsize=12, fontweight='bold')
ax.set_ylim(0, 75)

# 网格
ax.grid(alpha=0.3, linewidth=0.5)

# 边框
for spine in ax.spines.values():
    spine.set_linewidth(1.0)

# 图例
ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

# 标题
ax.set_title('Zero-Cost Optimization: 3.31× Speedup', 
             fontsize=13, fontweight='bold', pad=15)

plt.tight_layout()

# 保存
plt.savefig('figs/fig2_optimization_effect.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figs/fig2_optimization_effect.png', bbox_inches='tight', dpi=300)

print("✅ Figure 2 saved: figs/fig2_optimization_effect.pdf")
print(f"   Baseline: {latency[0]:.1f}s → Optimized: {latency[2]:.1f}s ({speedup[2]:.2f}× faster)")
