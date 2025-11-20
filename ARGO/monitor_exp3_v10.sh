#!/bin/bash
# 实验3监控脚本

LOG_FILE="/data/user/huangxiaolin/ARGO2/ARGO/exp3_v10.log"
RESULTS_DIR="/data/user/huangxiaolin/ARGO2/ARGO/draw_figs/data"

echo "========================================"
echo "实验3 (v10) 进度监控"
echo "========================================"
echo "PID: 4128922"
echo "日志: exp3_v10.log"
echo "配置: Option B + v10参数"
echo "预计时间: 4-6小时"
echo "========================================"
echo ""

# 检查进程
if ps -p 4128922 > /dev/null 2>&1; then
    echo "✓ 实验进程运行中 (PID 4128922)"
else
    echo "✗ 实验进程已结束"
fi
echo ""

# 查看当前进度
echo "--- 当前进度 (最后20行) ---"
tail -20 "$LOG_FILE" | grep -E "\[|μ =|ARGO:|进度:|✓|评估" || tail -20 "$LOG_FILE"
echo ""

# 统计已完成的μ点
completed_mu=$(grep -c "^\[.*\] μ =" "$LOG_FILE" 2>/dev/null || echo "0")
echo "已完成μ点: $completed_mu / 20"

# 统计基线策略
completed_baseline=$(grep -c "评估 Always-\|评估 Fixed-\|评估 Random" "$LOG_FILE" 2>/dev/null || echo "0")
echo "已完成基线: $completed_baseline / 4"
echo ""

# 检查是否有结果文件生成
echo "--- 生成的文件 ---"
ls -lht "$RESULTS_DIR"/exp3_real_pareto_frontier_*.json 2>/dev/null | head -3 || echo "尚未生成结果文件"
echo ""

# 检查是否有错误
error_count=$(grep -c "Error\|Exception\|Traceback" "$LOG_FILE" 2>/dev/null || echo "0")
if [ "$error_count" -gt 0 ]; then
    echo "⚠️  检测到 $error_count 个错误"
    echo "最近的错误:"
    grep -A 3 "Error\|Exception" "$LOG_FILE" | tail -20
else
    echo "✓ 无错误"
fi
echo ""

# 预计剩余时间
if [ "$completed_mu" -gt 0 ]; then
    # 计算已运行时间
    start_time=$(stat -c %Y "$LOG_FILE" 2>/dev/null)
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    elapsed_min=$((elapsed / 60))
    
    # 估算总时间和剩余时间
    if [ "$completed_mu" -lt 20 ]; then
        avg_time_per_mu=$((elapsed / completed_mu))
        remaining_mu=$((20 - completed_mu))
        remaining_sec=$((avg_time_per_mu * remaining_mu))
        remaining_min=$((remaining_sec / 60))
        
        echo "已运行: ${elapsed_min}分钟"
        echo "预计剩余: ${remaining_min}分钟 (~$((remaining_min / 60))小时)"
    else
        echo "已运行: ${elapsed_min}分钟"
        echo "ARGO部分已完成，正在评估基线策略..."
    fi
fi

echo ""
echo "========================================"
echo "监控命令:"
echo "  实时日志: tail -f exp3_v10.log"
echo "  重新监控: bash monitor_exp3_v10.sh"
echo "  停止实验: kill 4128922"
echo "========================================"
