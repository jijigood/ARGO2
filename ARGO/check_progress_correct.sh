#!/bin/bash

###############################################################################
# 实验进度监控脚本 (正确配置版本)
###############################################################################

echo "================================================================================"
echo "3B模型实验进度监控 (c_p=0.02 正确配置)"
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

# 检查配置是否正确
echo "2. 配置验证:"
if grep -q "c_p固定: 0.020" /data/user/huangxiaolin/ARGO2/ARGO/exp1_3B_correct_cp.log 2>/dev/null; then
    echo "   ✓ c_p = 0.02 (正确)"
else
    echo "   ✗ 配置可能不正确，请检查日志"
fi

if grep -q "c_r范围: 0.020 ~ 0.200" /data/user/huangxiaolin/ARGO2/ARGO/exp1_3B_correct_cp.log 2>/dev/null; then
    echo "   ✓ c_r = 0.02~0.2 (正确，与14B一致)"
else
    echo "   ✗ c_r范围可能不正确"
fi

echo ""

# 检查完成的文件
echo "3. 已完成的结果文件:"
NUM_FILES=$(ls draw_figs/data/exp1_real_cost_impact_custom_*.json 2>/dev/null | grep -v "backup" | wc -l)
echo "   已完成: $NUM_FILES / 3 个种子"

# 显示最新的文件（排除备份）
LATEST_FILES=$(ls -lht draw_figs/data/exp1_real_cost_impact_custom_*.json 2>/dev/null | grep -v "backup" | head -3)
if [ ! -z "$LATEST_FILES" ]; then
    echo "$LATEST_FILES" | awk '{print "   -", $9, "("$5")", $6, $7, $8}'
fi

echo ""

# 显示实验日志最新进度
echo "4. 最新实验进度 (最后15行):"
tail -15 /data/user/huangxiaolin/ARGO2/ARGO/exp1_3B_correct_cp.log 2>/dev/null | grep -E "(进度:|c_r =|Accuracy=|运行|Seed|完成)" | sed 's/^/   /' | tail -8

echo ""

# 检查自动分析脚本
echo "5. 自动分析脚本状态:"
if ps aux | grep -q "[w]ait_and_analyze_3B_correct.sh"; then
    echo "   ✓ 自动分析脚本正在等待"
    tail -3 /data/user/huangxiaolin/ARGO2/ARGO/auto_analysis_correct.log 2>/dev/null | sed 's/^/   /'
else
    echo "   ✗ 自动分析脚本未运行"
fi

echo ""
echo "================================================================================"
echo "提示:"
echo "  - 查看完整日志: tail -f /data/user/huangxiaolin/ARGO2/ARGO/exp1_3B_correct_cp.log"
echo "  - 查看分析日志: tail -f /data/user/huangxiaolin/ARGO2/ARGO/auto_analysis_correct.log"
echo "  - 重新运行监控: bash $0"
echo ""
echo "  预计完成时间: 约1.5-2小时"
echo "  横坐标将正确对齐: c_r = 0.02~0.2 (与14B实验一致)"
echo "================================================================================"
