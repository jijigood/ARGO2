#!/bin/bash

###############################################################################
# 实验1: 3B模型 - 正确配置 (c_p=0.02)
# 
# 修正:
#   ✓ 使用 multi_gpu_data_calibrated.yaml 配置 (c_p=0.02)
#   ✓ c_r范围将是 0.02~0.2 (与14B实验一致)
#   ✓ 3个随机种子 (42-44)
#   ✓ Medium难度
#   ✓ 100题/种子
#
# 预计运行时间: 1.5-2小时
###############################################################################

echo "================================================================================"
echo "实验1: 检索成本影响 - 3B模型 (正确配置)"
echo "================================================================================"
echo "配置:"
echo "  - 模型: Qwen2.5-3B-Instruct"
echo "  - 配置文件: multi_gpu_data_calibrated.yaml (c_p=0.02)"
echo "  - 随机种子: 3个 (42-44)"
echo "  - 难度级别: Medium"
echo "  - 问题数: 100"
echo "  - c_r范围: 0.02~0.2 (1×c_p 到 10×c_p)"
echo "  - 总运行次数: 3"
echo "  - GPU: 8张 (0-7)"
echo "================================================================================"
echo "开始时间: $(date)"
echo ""

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1

cd /data/user/huangxiaolin/ARGO2/ARGO

# 使用3B模型
MODEL_PATH="/data/user/huangxiaolin/ARGO/RAG_Models/models/Qwen2.5-3B-Instruct"

# 使用正确的配置文件
CONFIG_PATH="configs/multi_gpu_data_calibrated.yaml"

echo "运行多种子实验 (3B模型, c_p=0.02)..."
echo ""

python -u Exp1_multi_seed_wrapper.py \
    --n-seeds 3 \
    --base-seed 42 \
    --n-questions 100 \
    --difficulties medium \
    --gpus 0,1,2,3,4,5,6,7 \
    --model-path "$MODEL_PATH" \
    --config-path "$CONFIG_PATH"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 实验失败!"
    echo "查看日志: tail -100 exp1_3B_correct_cp.log"
    exit 1
fi

echo ""
echo "================================================================================"
echo "✓ 3B实验完成 (正确配置)!"
echo "================================================================================"
echo "结束时间: $(date)"
echo ""
echo "下一步:"
echo "  1. 聚合结果:"
echo "     python Exp1_aggregate_and_analyze.py"
echo "  2. 生成图表:"
echo "     python Exp1_plots.py draw_figs/data/exp1_aggregated_*.csv"
echo "  3. 如果效果好，可以运行完整版 (5种子×3难度):"
echo "     bash run_exp1_full.sh"
echo "================================================================================"
