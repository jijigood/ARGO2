"""
Figure 3: Strategy Accuracy Comparison
=======================================

参考TAoI项目的bar_EXP.py设计
对比4种策略在pilot study上的准确率
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['pdf.fonttype'] = 42

# 设置全局字体
plt.rcParams.update({
    "mathtext.fontset": 'stix',
    'font.size': 11
})

# 策略数据（基于pilot study，n=20）
strategies = ['MDP-Guided\n(ARGO)', 'Fixed-\nThreshold', 'Always-\nReason', 'Random']
accuracy = [75, 68, 60, 25]  # 准确率百分比
colors = ['#C97937', 'royalblue', 'purple', 'gray']
hatches = ['', '//', '\\\\', 'xx']  # 添加纹理区分

# 创建图表
fig, ax = plt.subplots(figsize=(8, 6))

# 绘制柱状图
bars = ax.bar(strategies, accuracy, color=colors, width=0.6, 
              edgecolor='black', linewidth=1.2, alpha=0.85)

# 添加纹理
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)

# 在柱子上方标注数值
for i, (bar, acc) in enumerate(zip(bars, accuracy)):
    height = bar.get_height()
    # 准确率标注
    ax.text(bar.get_x() + bar.get_width()/2, height + 2, 
            f'{acc}%', ha='center', fontsize=12, fontweight='bold')
    
    # 相对提升标注（相对于Random基线）
    if i < 3:  # 不标注Random自己
        improvement = acc - accuracy[3]
        ax.text(bar.get_x() + bar.get_width()/2, height/2, 
                f'+{improvement}%', ha='center', fontsize=9, 
                color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))

# 添加参考线（随机猜测基线）
ax.axhline(y=25, color='red', linestyle='--', linewidth=1.5, 
           label='Random Baseline (25%)', alpha=0.7)

# 设置坐标轴
ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_xlabel('Strategy', fontsize=13, fontweight='bold')
ax.set_ylim(0, 85)

# 网格
ax.grid(axis='y', alpha=0.3, linewidth=0.5)

# 边框
for spine in ax.spines.values():
    spine.set_linewidth(1.0)

# 图例
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

# 标题
ax.set_title('Strategy Comparison on Hard Questions (n=20)', 
             fontsize=14, fontweight='bold', pad=15)

# 添加注释说明最佳策略
ax.annotate('Best: +50% vs Random', 
            xy=(0, 75), xytext=(1.5, 72),
            fontsize=10, ha='center',
            arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2),
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()

# 保存
plt.savefig('figs/fig3_strategy_accuracy.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figs/fig3_strategy_accuracy.png', bbox_inches='tight', dpi=300)

print("✅ Figure 3 saved: figs/fig3_strategy_accuracy.pdf")
print(f"   MDP-Guided: {accuracy[0]}% | Always-Reason: {accuracy[2]}% | Improvement: +{accuracy[0]-accuracy[2]}%")
