#!/bin/bash
# Phase 4.3 后台实验运行脚本

cd /data/user/huangxiaolin/ARGO2/ARGO

echo "================================"
echo "启动 Phase 4.3 Hard 实验"
echo "================================"
echo "时间: $(date)"
echo "PID: $$"
echo ""

# 运行实验，输出到日志文件
nohup python -u run_hard_experiment.py > results/phase4.3_hard/experiment_output.log 2>&1 &

PID=$!
echo "实验已启动，PID=$PID"
echo "日志文件: results/phase4.3_hard/experiment_output.log"
echo ""
echo "查看进度:"
echo "  tail -f results/phase4.3_hard/experiment_output.log"
echo ""
echo "检查进程:"
echo "  ps aux | grep $PID"
echo ""
echo "停止实验:"
echo "  kill $PID"
echo ""
