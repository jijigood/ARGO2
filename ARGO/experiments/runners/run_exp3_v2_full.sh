#!/bin/bash
# ===============================================
# Experiment 3 V2 - Pareto Frontier Analysis
# ===============================================
# 优化后规模:
#   - ARGO μ扫描: 12点 × 20题 = 240题
#   - Fixed-Threshold: 7点 × 20题 = 140题
#   - Baselines: 3 × 20题 = 60题
#   - 总计: 440题
#   - 预计时间: ~12-15小时
# ===============================================

cd /data/user/huangxiaolin/ARGO2/ARGO

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ARGO

echo "==============================================="
echo "Experiment 3 V2 - Pareto Frontier Analysis"
echo "==============================================="
echo "Config: configs/pareto_optimized.yaml"
echo "GPUs: 4,5,6,7 (CUDA_VISIBLE_DEVICES映射为0,1,2,3)"
echo "Difficulty: hard"
echo "Questions per point: 20"
echo "μ steps: 12"
echo "Fixed-Threshold points: 7"
echo "Total questions: ~440"
echo "Estimated time: 12-15 hours"
echo "==============================================="
echo "Start time: $(date)"
echo ""

# 设置CUDA设备 (使用GPU 4,5,6,7)
export CUDA_VISIBLE_DEVICES=4,5,6,7

python -u experiments/runners/Exp_real_pareto_frontier_v2.py \
    --n-questions 20 \
    --difficulty hard \
    --mu-min 0.0 \
    --mu-max 2.0 \
    --n-mu-steps 12 \
    --gpus 0,1,2,3 \
    --config-path configs/pareto_optimized.yaml \
    --seed 42

echo ""
echo "==============================================="
echo "Experiment Complete!"
echo "End time: $(date)"
echo "==============================================="
