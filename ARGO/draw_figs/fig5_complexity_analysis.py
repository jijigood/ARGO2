"""
Figure 5: Question Complexity Analysis
========================================

参考TAoI项目的分组柱状图设计
对比MDP vs Always-Reason在不同复杂度问题上的表现
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['pdf.fonttype'] = 42

# 设置全局字体
plt.rcParams.update({
    "mathtext.fontset": 'stix',
    'font.size': 11
})

# 数据
categories = ['Single-hop\n(n=7)', 'Multi-hop\n(n=13)']
mdp_acc = [85.7, 69.2]
always_acc = [71.4, 53.8]

x = np.arange(len(categories))
width = 0.35

# 创建图表
fig, ax = plt.subplots(figsize=(8, 6))

# 绘制分组柱状图
bars1 = ax.bar(x - width/2, mdp_acc, width, 
               label='MDP-Guided', color='#C97937', 
               edgecolor='black', linewidth=1.2, alpha=0.85)
bars2 = ax.bar(x + width/2, always_acc, width, 
               label='Always-Reason', color='purple', 
               edgecolor='black', linewidth=1.2, alpha=0.85)

# 标注数值
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1.5,
                f'{height:.1f}%', ha='center', fontsize=11, fontweight='bold')

# 标注改进幅度
improvements = [mdp_acc[i] - always_acc[i] for i in range(len(categories))]
for i, imp in enumerate(improvements):
    mid_y = (mdp_acc[i] + always_acc[i]) / 2
    ax.annotate(f'+{imp:.1f}%', 
                xy=(i, mid_y), 
                fontsize=10, ha='center',
                color='darkgreen', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

# 设置坐标轴
ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_xlabel('Question Type', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12)
ax.set_ylim(0, 95)

# 网格
ax.grid(axis='y', alpha=0.3, linewidth=0.5)

# 边框
for spine in ax.spines.values():
    spine.set_linewidth(1.0)

# 图例
ax.legend(loc='upper right', fontsize=12, framealpha=0.9, 
          ncol=1, columnspacing=1.0)

# 标题
ax.set_title('MDP-Guided Shows Consistent Advantage', 
             fontsize=14, fontweight='bold', pad=15)

# 添加观察说明
textstr = 'Observation:\n• Single-hop: +14.3%\n• Multi-hop: +15.4%'
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
ax.text(0.97, 0.40, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props, ha='right')

plt.tight_layout()

# 保存
plt.savefig('figs/fig5_complexity_analysis.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figs/fig5_complexity_analysis.png', bbox_inches='tight', dpi=300)

print("✅ Figure 5 saved: figs/fig5_complexity_analysis.pdf")
print(f"   Single-hop improvement: +{improvements[0]:.1f}%")
print(f"   Multi-hop improvement: +{improvements[1]:.1f}%")
