#!/bin/bash
#
# 运行优化后的实验 - 消除"悬崖"效应
#
# 配置:
#   - 成本范围: [0.005, 0.100] (0.25x-5.0x c_p)
#   - 采样点数: 20个
#   - 问题数量: 1000题 (Hard)
#   - 预计时间: ~38小时
#
# 预期效果:
#   - 准确率曲线呈平滑斜坡 (而非悬崖)
#   - 7个点在检索主导区 (旧版只有1个)
#   - 4个点在过渡区 (捕捉策略切换)
#   - 更好地展示ARGO自适应能力
#

# 设置
SCRIPT="Exp_3B_quick_validation.py"
MODE="full"
DIFFICULTY="hard"
GPU="0"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="exp1_3B_full_smooth_${TIMESTAMP}.log"

# 检查GPU是否空闲
echo "检查GPU ${GPU} 状态..."
GPU_UTIL=$(nvidia-smi --id=${GPU} --query-gpu=utilization.gpu --format=csv,noheader,nounits)
if [ ${GPU_UTIL} -gt 20 ]; then
    echo "⚠️  GPU ${GPU} 当前使用率: ${GPU_UTIL}% (建议<20%)"
    echo "继续运行可能影响性能，是否继续? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "已取消"
        exit 1
    fi
fi

echo ""
echo "="================================================================
echo "          运行优化后的实验 - 消除悬崖效应"
echo "================================================================="
echo ""
echo "配置信息:"
echo "  模式: ${MODE}"
echo "  难度: ${DIFFICULTY}"
echo "  GPU: ${GPU}"
echo "  成本范围: [0.005, 0.100]"
echo "  采样点数: 20个"
echo "  问题数量: 1000题"
echo "  日志文件: ${LOGFILE}"
echo ""
echo "预计时间: ~38小时"
echo ""
echo "================================================================="
echo ""

# 启动实验
echo "启动实验..."
nohup python -u ${SCRIPT} \
    --mode ${MODE} \
    --difficulty ${DIFFICULTY} \
    --gpus ${GPU} \
    > ${LOGFILE} 2>&1 &

PID=$!
echo "✓ 实验已启动!"
echo "  进程ID: ${PID}"
echo "  日志文件: ${LOGFILE}"
echo ""
echo "监控命令:"
echo "  查看进程: ps aux | grep ${PID}"
echo "  实时日志: tail -f ${LOGFILE}"
echo "  GPU监控: watch -n 1 nvidia-smi"
echo ""
echo "预计完成时间: $(date -d '+38 hours' '+%Y-%m-%d %H:%M:%S')"
echo ""
