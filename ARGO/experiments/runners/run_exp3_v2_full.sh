#!/bin/bash
# ===============================================
# Experiment 3 V3 - Pareto Frontier Analysis
# ===============================================
# 使用 ARGO_System 的完整版本
# 规模:
#   - ARGO μ扫描: 12点 × 20题 = 240题
#   - Baselines: 3 × 20题 = 60题
#   - 总计: 300题
#   - 预计时间: ~8-10小时
# ===============================================

cd /data/user/huangxiaolin/ARGO2/ARGO

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ARGO

echo "==============================================="
echo "Experiment 3 V3 - Pareto Frontier Analysis"
echo "==============================================="
echo "Config: configs/pareto_optimized_v2.yaml"
echo "Chroma DB: /data/user/huangxiaolin/ARGO2/Environments/chroma_store_v2"
echo "GPUs: 4,5,6,7 (CUDA_VISIBLE_DEVICES映射为0,1,2,3)"
echo "Difficulty: hard"
echo "Questions per point: 20"
echo "μ range: [0.0, 1.0] (12 steps)"
echo "Total questions: ~300"
echo "Estimated time: 8-10 hours"
echo "==============================================="
echo "Start time: $(date)"
echo ""

# 设置CUDA设备 (使用GPU 4,5,6,7)
export CUDA_VISIBLE_DEVICES=4,5,6,7

python -u experiments/runners/Exp_real_pareto_frontier_v3.py \
    --n-questions 20 \
    --difficulty hard \
    --mu-min 0.0 \
    --mu-max 1.0 \
    --n-mu-steps 12 \
    --gpus 0,1,2,3 \
    --config-path configs/pareto_optimized_v2.yaml \
    --chroma-db-path /data/user/huangxiaolin/ARGO2/Environments/chroma_store_v2 \
    --collection-name oran_specs_semantic \
    --seed 42

echo ""
echo "==============================================="
echo "Experiment Complete!"
echo "End time: $(date)"
echo "==============================================="
