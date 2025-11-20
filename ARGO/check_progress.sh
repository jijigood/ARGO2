#!/bin/bash

###############################################################################
# 实验进度监控脚本
###############################################################################

echo "================================================================================"
echo "3B模型实验进度监控"
echo "================================================================================"
echo "当前时间: $(date)"
echo ""

# 检查实验进程
echo "1. 实验进程状态:"
if ps aux | grep -q "[E]xp_real_cost_impact_v2.py.*3B-Instruct"; then
    echo "   ✓ 实验正在运行"
    ps aux | grep "[E]xp_real_cost_impact_v2.py" | grep -v grep | awk '{print "   PID:", $2, "  CPU:", $3"%", "  内存:", $4"%", "  运行时间:", $10}'
else
    echo "   ✗ 实验未运行"
fi

echo ""

# 检查完成的文件
echo "2. 已完成的结果文件:"
NUM_FILES=$(ls draw_figs/data/exp1_real_cost_impact_custom_*.json 2>/dev/null | wc -l)
echo "   已完成: $NUM_FILES / 3 个种子"
if [ $NUM_FILES -gt 0 ]; then
    ls -lh draw_figs/data/exp1_real_cost_impact_custom_*.json | tail -5 | awk '{print "   -", $9, "("$5")", $6, $7, $8}'
fi

echo ""

# 显示实验日志最新进度
echo "3. 最新实验进度 (最后15行):"
tail -15 /data/user/huangxiaolin/ARGO2/ARGO/exp1_3B_quick_medium.log | grep -E "(进度:|c_r =|Accuracy=|运行|Seed)" | sed 's/^/   /'

echo ""

# 检查自动分析脚本
echo "4. 自动分析脚本状态:"
if ps aux | grep -q "[w]ait_and_analyze_3B.sh"; then
    echo "   ✓ 自动分析脚本正在等待"
    tail -3 /data/user/huangxiaolin/ARGO2/ARGO/auto_analysis.log | sed 's/^/   /'
else
    echo "   ✗ 自动分析脚本未运行"
fi

echo ""
echo "================================================================================"
echo "提示:"
echo "  - 查看完整日志: tail -f /data/user/huangxiaolin/ARGO2/ARGO/exp1_3B_quick_medium.log"
echo "  - 查看分析日志: tail -f /data/user/huangxiaolin/ARGO2/ARGO/auto_analysis.log"
echo "  - 重新运行监控: bash $0"
echo "================================================================================"
