#!/usr/bin/env python
"""
实验3快速运行指南
==================

这是修复后的实验3的快速参考。

运行实验:
---------
cd /data/user/huangxiaolin/ARGO2/ARGO
python run_exp3_full.py

关键修复:
---------
1. ✅ 随机检索成功 - 现在检索以概率p_s成功
2. ✅ 分离质量度量 - 信息完整性 vs 答案正确性
3. ✅ 完整历史追踪 - (q_t, r_t) 对
4. ✅ 4个基线策略 - Always-Retrieve, Always-Reason, Fixed-Threshold, Random
5. ✅ 3个可视化图表 - Pareto边界, 阈值演化, 综合仪表板

期望输出:
---------
- draw_figs/data/exp3_real_pareto_frontier_<timestamp>.json
- figs/exp3_real_pareto_frontier.png
- figs/exp3_threshold_evolution.png
- figs/exp3_dashboard.png

验证结果:
---------
1. 检索成功率应该约为80% (= p_s)
2. 所有基线点应该在ARGO曲线下方
3. 阈值应该随μ单调变化
4. ARGO形成Pareto最优边界

详细文档:
---------
参见 EXP3_FIXES_SUMMARY.md
"""

if __name__ == "__main__":
    print(__doc__)
