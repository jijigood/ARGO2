#!/bin/bash
# 实时监控实验3进度 (修复版)

LOG_FILE="/data/user/huangxiaolin/ARGO2/ARGO/exp3_fixed_log.txt"

echo "=========================================="
echo "实验3监控 (修复版) - $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# 检查进程
PID=$(ps aux | grep "run_exp3_full.py" | grep -v grep | awk '{print $2}')
if [ -z "$PID" ]; then
    echo "❌ 实验3进程未运行"
else
    echo "✓ 实验3进程运行中 (PID: $PID)"
    # 显示CPU和内存使用
    ps aux | grep $PID | grep -v grep | awk '{printf "  CPU: %s%%, MEM: %s%%\n", $3, $4}'
fi

echo ""
echo "=========================================="
echo "当前进度:"
echo "=========================================="

# 提取所有μ点的求解结果
grep -E "^\[|θ_cont|ARGO: Quality" "$LOG_FILE" | tail -30

echo ""
echo "=========================================="
echo "最新输出 (最后10行):"
echo "=========================================="
tail -10 "$LOG_FILE"

echo ""
echo "=========================================="
echo "统计:"
echo "=========================================="
echo "已完成的μ点: $(grep -c 'ARGO: Quality' $LOG_FILE) / 10"
echo "已完成的基线: $(grep -c 'Always-' $LOG_FILE | head -1)"
