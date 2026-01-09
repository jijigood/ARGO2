#!/bin/bash

###############################################################################
# 实验1-Mini (4 GPU版): 使用前4张GPU
###############################################################################

echo "================================================================================"
echo "实验1-Mini (4 GPU): 最小快速验证"
echo "================================================================================"
echo "配置:"
echo "  - 随机种子: 1个 (42)"
echo "  - 难度级别: Easy, Medium, Hard"  
echo "  - 每成本点问题数: 30"
echo "  - 成本采样点: 3个"
echo "  - 总问题数: 270"
echo "  - GPU: 4张 (0-3)"
echo "================================================================================"
echo "开始时间: $(date)"
echo ""

# 只使用前4张GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1

PYTHON_BIN=${PYTHON_BIN:-python}
FALLBACK_CONDA_ENV=${CONDA_FALLBACK_ENV:-ARGO}

if ! "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
    CONDA_BASE=$(conda info --base 2>/dev/null)
    if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        conda activate "$FALLBACK_CONDA_ENV"
        PYTHON_BIN=python
    fi
fi

cd /data/user/huangxiaolin/ARGO2/ARGO

if [ ! -f "configs/adaptive_policy.yaml" ]; then
    echo "❌ 错误: 找不到 configs/adaptive_policy.yaml"
    exit 1
fi
echo "✓ 配置检查通过"

SEEDS="42"
DIFFICULTIES="easy medium hard"
N_QUESTIONS=30
COST_VALUES="0.005,0.021,0.050"

echo "================================================================================"
echo "实验规模: 270 问题, 预计 ~10-12 小时 (4 GPU)"
echo "================================================================================"
echo ""

run_count=0
total_runs=3

for diff in $DIFFICULTIES; do
    for seed in $SEEDS; do
        run_count=$((run_count + 1))
        
        echo "========================================================================"
        echo "运行 $run_count/$total_runs: Difficulty=$diff, Seed=$seed"
        echo "========================================================================"
        
        "$PYTHON_BIN" -u experiments/runners/Exp_real_cost_impact_v2.py \
            --mode custom \
            --n-questions $N_QUESTIONS \
            --difficulty $diff \
            --gpus 0,1,2,3 \
            --seed $seed \
            --config-path configs/multi_gpu_data_calibrated.yaml \
            --policy-config-path configs/adaptive_policy.yaml \
            --c-r-values "$COST_VALUES" \
            --verbose
        
        echo "✓ 运行 $run_count 完成"
        echo ""
    done
done

echo "================================================================================"
echo "✓ 实验1-Mini 完成!"
echo "结束时间: $(date)"
echo "================================================================================"
