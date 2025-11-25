#!/bin/bash

###############################################################################
# 实验1: 3B模型快速验证 (3种子, Hard难度)
# 
# 优势:
#   ✓ 3B模型 - 速度快，更能突出ARGO优势
#   ✓ 3个随机种子 - 足够统计分析
#   ✓ Hard难度 - 最能体现检索价值
#   ✓ 100题 - 平衡效率与精度
#
# 预计运行时间: 6-8小时 (vs 14B的45+小时)
###############################################################################

echo "================================================================================"
echo "实验1: 检索成本影响 - 3B模型快速验证"
echo "================================================================================"
echo "配置:"
echo "  - 模型: Qwen2.5-3B-Instruct"
echo "  - 随机种子: 3个 (42-44)"
echo "  - 难度级别: Medium (Hard数据文件缺失)"
echo "  - 问题数: 100"
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

echo "运行多种子实验 (3B模型)..."
echo ""

python -u Exp1_multi_seed_wrapper.py \
    --n-seeds 3 \
    --base-seed 42 \
    --n-questions 100 \
    --difficulties medium \
    --gpus 0,1,2,3,4,5,6,7 \
    --model-path "$MODEL_PATH"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 实验失败!"
    exit 1
fi

echo ""
echo "================================================================================"
echo "✓ 3B快速验证完成!"
echo "================================================================================"
echo "结束时间: $(date)"
echo ""
echo "下一步:"
echo "  1. 统计分析:"
echo "     python Exp1_aggregate_and_analyze.py"
echo ""
echo "  2. 生成图表:"
echo "     python Exp1_plots.py draw_figs/data/exp1_aggregated_*.csv"
echo ""
echo "  3. 如果效果好，运行完整版 (5种子×3难度):"
echo "     bash run_exp1_3B_full.sh"
echo ""
echo "================================================================================"
