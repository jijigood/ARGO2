#!/bin/bash

###############################################################################
# 实验1-Mini: 最小验证版本
# 
# 极简配置:
#   - 种子数: 1个 (仅验证趋势)
#   - 难度: 3个 (Easy/Medium/Hard)
#   - 问题数: 30个/成本点
#   - 成本点: 3个 (关键对比)
#
# 预计时间: ~8-10小时
# 问题总数: 1×3×3×30 = 270 (原 Lite 版 1,200 的 22%)
###############################################################################

echo "================================================================================"
echo "实验1-Mini: 最小快速验证"
echo "================================================================================"
echo "配置 (极简版):"
echo "  - 随机种子: 1个 (42)"
echo "  - 难度级别: Easy, Medium, Hard"  
echo "  - 每成本点问题数: 30"
echo "  - 成本采样点: 3个"
echo "  - 总问题数: 270"
echo "  - GPU: 8张 (0-7)"
echo "================================================================================"
echo "成本采样策略 (3个关键点):"
echo "  0.005 (0.25×c_p) - 低成本，θ_cont≈0.97 (检索策略)"
echo "  0.021 (1.05×c_p) - 相变点，θ_cont≈0.03"
echo "  0.050 (2.50×c_p) - 高成本，θ_cont≈0.00 (推理策略)"
echo "================================================================================"
echo "开始时间: $(date)"
echo ""

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
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
echo ""

# 极简配置
SEEDS="42"
DIFFICULTIES="easy medium hard"
N_QUESTIONS=30
COST_VALUES="0.005,0.021,0.050"  # 只保留3个关键点

total_questions=$((1 * 3 * 3 * N_QUESTIONS))
echo "================================================================================"
echo "实验规模: 总问题数 = $total_questions"
echo "预计时间: ~8-10 小时"
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
            --gpus 0,1,2,3,4,5,6,7 \
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
echo "================================================================================"
echo "结束时间: $(date)"
echo ""
echo "预期观察:"
echo "  c_r=0.005: 高准确率 (~85%), 多检索"
echo "  c_r=0.021: 准确率下降, 相变点"
echo "  c_r=0.050: 最低准确率 (~70%), 少检索"
echo "================================================================================"
