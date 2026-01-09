#!/bin/bash
# Exp3 V2 快速测试脚本
# 使用GPU 4,5,6,7 和 hard难度

cd /data/user/huangxiaolin/ARGO2/ARGO

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ARGO

echo "==============================================="
echo "Experiment 3 V2 - Quick Test"
echo "==============================================="
echo "Config: configs/pareto_optimized.yaml"
echo "GPUs: 4,5,6,7"
echo "Difficulty: hard"
echo "Questions: 10 (quick test)"
echo "==============================================="

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=4,5,6,7

python experiments/runners/Exp_real_pareto_frontier_v2.py \
    --n-questions 10 \
    --difficulty hard \
    --mu-min 0.0 \
    --mu-max 1.0 \
    --n-mu-steps 5 \
    --gpus 0,1,2,3 \
    --config-path configs/pareto_optimized.yaml \
    --seed 42

echo "==============================================="
echo "Test Complete!"
echo "==============================================="
