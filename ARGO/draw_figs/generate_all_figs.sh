#!/bin/bash
# 批量生成所有论文图表

echo "================================"
echo "ARGO 论文图表生成脚本"
echo "================================"
echo ""

# 创建输出目录
mkdir -p figs

echo "生成 Figure 2: Optimization Effect..."
python draw_figs/fig2_optimization_effect.py

echo ""
echo "生成 Figure 3: Strategy Accuracy..."
python draw_figs/fig3_strategy_accuracy.py

echo ""
echo "生成 Figure 5: Complexity Analysis..."
python draw_figs/fig5_complexity_analysis.py

echo ""
echo "================================"
echo "✅ 所有图表生成完成！"
echo "================================"
echo ""
echo "图表位置: figs/"
ls -lh figs/*.pdf
echo ""
echo "可以在论文中使用："
echo "  \\includegraphics[width=0.8\\textwidth]{figs/fig2_optimization_effect.pdf}"
echo "  \\includegraphics[width=0.8\\textwidth]{figs/fig3_strategy_accuracy.pdf}"
echo "  \\includegraphics[width=0.8\\textwidth]{figs/fig5_complexity_analysis.pdf}"
