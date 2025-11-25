#!/bin/bash
# 7B优化实验启动脚本
# 使用GPU 0-1运行Qwen2.5-7B模型

set -e  # 遇到错误立即退出

echo "========================================"
echo "7B优化实验启动"
echo "========================================"
echo ""

# 检查GPU可用性
if ! command -v nvidia-smi &> /dev/null; then
    echo "错误: 未找到nvidia-smi命令"
    exit 1
fi

echo "GPU状态:"
nvidia-smi --query-gpu=index,name,memory.free --format=csv
echo ""

# 默认参数
MODE="${1:-small}"      # small/medium/full
GPU="${2:-0,1}"         # 默认使用GPU 0-1
DIFFICULTY="${3:-hard}" # 难度

echo "运行参数:"
echo "  模式: $MODE"
echo "  GPU: $GPU"
echo "  难度: $DIFFICULTY"
echo ""

# 设置CUDA设备 (可选，device_map会自动处理)
export CUDA_VISIBLE_DEVICES=$GPU

# 运行实验
echo "启动实验..."
python Exp_7B_optimized_full.py \
    --mode $MODE \
    --difficulty $DIFFICULTY \
    --gpus $GPU \
    --seed 42

echo ""
echo "========================================"
echo "实验完成!"
echo "========================================"
