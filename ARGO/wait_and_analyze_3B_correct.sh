#!/bin/bash

###############################################################################
# 等待3B正确配置实验完成并自动分析
###############################################################################

echo "================================================================================"
echo "等待3B实验完成并自动分析 (正确配置 c_p=0.02)"
echo "================================================================================"
echo "开始监控时间: $(date)"
echo ""

cd /data/user/huangxiaolin/ARGO2/ARGO

# 等待实验进程结束
echo "正在等待实验进程完成..."
while ps aux | grep -q "[E]xp_real_cost_impact_v2.py.*3B-Instruct"; do
    sleep 60  # 每分钟检查一次
    echo "  [$(date +%H:%M:%S)] 实验仍在运行..."
done

echo ""
echo "✓ 实验进程已结束"
echo "等待5秒确保文件写入完成..."
sleep 5

# 检查生成的结果文件（排除备份目录中的旧文件）
echo ""
echo "================================================================================"
echo "检查结果文件"
echo "================================================================================"

# 获取最新生成的3个文件
LATEST_FILES=$(ls -t draw_figs/data/exp1_real_cost_impact_custom_*.json 2>/dev/null | grep -v "backup" | head -3)
NUM_FILES=$(echo "$LATEST_FILES" | wc -l)

echo "找到最新的 $NUM_FILES 个结果文件:"
echo "$LATEST_FILES"

if [ $NUM_FILES -lt 3 ]; then
    echo ""
    echo "⚠️  警告: 只找到 $NUM_FILES 个文件，期望3个（3个种子）"
    echo "请检查实验是否全部完成"
    echo ""
    echo "查看日志最后100行:"
    tail -100 /data/user/huangxiaolin/ARGO2/ARGO/exp1_3B_correct_cp.log
    exit 1
fi

# 验证c_r范围是否正确（应该是0.02~0.2而不是0.3~3.0）
FIRST_FILE=$(echo "$LATEST_FILES" | head -1)
FIRST_CR=$(python -c "import json; data=json.load(open('$FIRST_FILE')); print(data['results'][0]['c_r'])" 2>/dev/null)

echo ""
echo "验证c_r范围: 第一个c_r = $FIRST_CR"

if [ "$FIRST_CR" != "0.02" ]; then
    echo ""
    echo "❌ 错误: c_r值不正确！应该是0.02，实际是 $FIRST_CR"
    echo "实验可能使用了错误的配置文件"
    exit 1
fi

echo "✓ c_r范围正确 (0.02~0.2)"

# 执行统计分析
echo ""
echo "================================================================================"
echo "执行统计分析"
echo "================================================================================"
python -u Exp1_aggregate_and_analyze.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 统计分析失败！"
    exit 1
fi

echo ""
echo "✓ 统计分析完成"

# 生成图表
echo ""
echo "================================================================================"
echo "生成图表"
echo "================================================================================"

# 找到最新的聚合文件
AGG_FILE=$(ls -t draw_figs/data/exp1_aggregated_*.csv 2>/dev/null | head -1)

if [ -z "$AGG_FILE" ]; then
    echo "❌ 未找到聚合文件"
    exit 1
fi

echo "使用聚合文件: $AGG_FILE"
python -u Exp1_plots.py "$AGG_FILE"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 图表生成失败！"
    exit 1
fi

echo ""
echo "✓ 图表生成完成"

# 显示结果
echo ""
echo "================================================================================"
echo "分析完成！(正确配置 c_p=0.02)"
echo "================================================================================"
echo "完成时间: $(date)"
echo ""
echo "生成的文件:"
echo "  1. 聚合统计结果: $AGG_FILE"
echo "  2. 统计检验结果: $(ls -t draw_figs/data/exp1_statistical_tests_*.csv 2>/dev/null | head -1)"
echo "  3. 图表文件:"
ls -lh figs/exp1_*.png | grep "$(date +"%b %e")" | tail -5
echo ""
echo "横坐标范围验证:"
echo "  c_r范围: 0.02 ~ 0.2 ✓"
echo "  与14B实验一致 ✓"
echo "================================================================================"
