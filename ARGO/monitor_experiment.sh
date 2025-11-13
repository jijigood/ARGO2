#!/bin/bash

# 监控 MDP vs Fixed 对比实验进度

echo "========================================"
echo "MDP vs Fixed Experiment Monitor"
echo "========================================"
echo ""

# 检查进程是否运行
PID=$(pgrep -f "compare_mdp_vs_fixed.py")
if [ -z "$PID" ]; then
    echo "❌ Experiment not running"
    echo ""
    echo "Check results:"
    if [ -f "results/comparison/*.json" ]; then
        echo "✓ Results found in results/comparison/"
        ls -lh results/comparison/*.json
    else
        echo "⚠ No results found yet"
    fi
    exit 0
fi

echo "✓ Experiment running (PID: $PID)"
echo ""

# 显示日志
LOG_FILE="mdp_vs_fixed_100_medium.log"
if [ -f "$LOG_FILE" ]; then
    echo "Progress (last 30 lines):"
    echo "----------------------------------------"
    tail -30 "$LOG_FILE" | grep -E "\[MDP |\[Fixed |Accuracy|Results|✓|✗"
    echo "----------------------------------------"
    echo ""
    
    # 统计进度
    MDP_DONE=$(grep -c "\[MDP " "$LOG_FILE" 2>/dev/null || echo 0)
    FIXED_DONE=$(grep -c "\[Fixed " "$LOG_FILE" 2>/dev/null || echo 0)
    
    echo "MDP Strategy:   $MDP_DONE/100 questions"
    echo "Fixed Strategy: $FIXED_DONE/100 questions"
else
    echo "⚠ Log file not found: $LOG_FILE"
fi

echo ""
echo "Commands:"
echo "  Watch progress: watch -n 5 ./monitor_experiment.sh"
echo "  View full log:  tail -f mdp_vs_fixed_100_medium.log"
echo "  Kill process:   kill $PID"
