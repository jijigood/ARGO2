#!/bin/bash
# 实验3进度监控脚本

echo "========================================"
echo "实验3 Pareto边界 - 进度监控"
echo "========================================"
echo ""

# 检查进程
PID=$(ps aux | grep "python.*run_exp3" | grep -v grep | awk '{print $2}')

if [ -z "$PID" ]; then
    echo "❌ 实验3进程未运行"
    exit 1
fi

echo "✅ 实验3正在运行 (PID: $PID)"
echo ""

# 显示进程信息
echo "进程信息:"
ps aux | grep "$PID" | grep -v grep | awk '{printf "  CPU: %s%%  内存: %sMB  运行时间: %s\n", $3, int($6/1024), $10}'
echo ""

# 显示最新日志
echo "========================================"
echo "最新日志 (最后30行):"
echo "========================================"
tail -30 /data/user/huangxiaolin/ARGO2/ARGO/exp3_full_log.txt
echo ""

# 统计进度
echo "========================================"
echo "进度统计:"
echo "========================================"
grep -c "\\[.*\\] μ =" /data/user/huangxiaolin/ARGO2/ARGO/exp3_full_log.txt 2>/dev/null | xargs -I {} echo "  已完成μ点: {} / 10"
grep -c "评估 Always-" /data/user/huangxiaolin/ARGO2/ARGO/exp3_full_log.txt 2>/dev/null | xargs -I {} echo "  基线策略: {} / 2"

echo ""
echo "提示: 使用 'bash monitor_exp3.sh' 再次检查进度"
