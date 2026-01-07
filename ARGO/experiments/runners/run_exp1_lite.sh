#!/bin/bash

###############################################################################
# 实验1-Lite: 快速验证版本
# 
# 简化配置:
#   - 种子数: 2个 (足够验证趋势一致性)
#   - 难度: 3个 (Easy/Medium/Hard)
#   - 问题数: 50个/成本点 (统计上足够)
#   - 成本点: 4个 (关键对比点)
#
# 预计时间: ~50小时 (原版490小时)
# 问题总数: 2×3×4×50 = 1,200 (原版10,500)
###############################################################################

echo "================================================================================"
echo "实验1-Lite: 快速验证成本敏感性"
echo "================================================================================"
echo "配置 (简化版):"
echo "  - 随机种子: 2个 (42, 43)"
echo "  - 难度级别: Easy, Medium, Hard"  
echo "  - 每成本点问题数: 50"
echo "  - 成本采样点: 4个"
echo "  - 总问题数: 1,200 (原版10,500的11%)"
echo "  - GPU: 8张 (0-7)"
echo "================================================================================"
echo "成本采样策略 (4个关键点):"
echo "  0.005 (0.25×c_p) - 最优点，预期θ_cont≈0.97"
echo "  0.015 (0.75×c_p) - 转折前，预期θ_cont≈0.96"
echo "  0.021 (1.05×c_p) - 转折后，预期θ_cont≈0.03"
echo "  0.050 (2.50×c_p) - 高成本，预期θ_cont≈0.00"
echo "================================================================================"
echo "开始时间: $(date)"
echo ""

# GPU配置
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1

# Python环境检测
PYTHON_BIN=${PYTHON_BIN:-python}
FALLBACK_CONDA_ENV=${CONDA_FALLBACK_ENV:-ARGO}

if ! "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
    echo "⚠ 激活 conda 环境: ${FALLBACK_CONDA_ENV}"
    CONDA_BASE=$(conda info --base 2>/dev/null)
    if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        conda activate "$FALLBACK_CONDA_ENV"
        PYTHON_BIN=python
        if ! "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
            echo "❌ 无法导入 torch"
            exit 1
        fi
        echo "✓ 已激活: ${FALLBACK_CONDA_ENV}"
    else
        echo "❌ 找不到 conda"
        exit 1
    fi
fi

run_python() {
    "$PYTHON_BIN" -u "$@"
}

# 工作目录
cd /data/user/huangxiaolin/ARGO2/ARGO

# 检查配置
if [ ! -f "configs/adaptive_policy.yaml" ]; then
    echo "❌ 错误: 找不到 configs/adaptive_policy.yaml"
    exit 1
fi
echo "✓ 配置检查通过"
echo ""

# ============================================================================
# 简化配置
# ============================================================================
SEEDS="42 43"                                    # 2个种子 (原5个)
DIFFICULTIES="easy medium hard"                   # 3个难度
N_QUESTIONS=50                                    # 50个问题/成本点 (原100)
COST_VALUES="0.005,0.015,0.021,0.050"            # 4个关键点 (原7个)

# 计算总工作量
n_seeds=2
n_diffs=3
n_costs=4
total_questions=$((n_seeds * n_diffs * n_costs * N_QUESTIONS))

echo "================================================================================"
echo "实验规模:"
echo "  - 种子: $n_seeds 个"
echo "  - 难度: $n_diffs 个"
echo "  - 成本点: $n_costs 个"
echo "  - 每点问题: $N_QUESTIONS"
echo "  - 总问题数: $total_questions"
echo "  - 预计时间: 约 50 小时"
echo "================================================================================"
echo ""

# 运行计数器
run_count=0
total_runs=$((n_seeds * n_diffs))

# 主循环
for diff in $DIFFICULTIES; do
    for seed in $SEEDS; do
        run_count=$((run_count + 1))
        
        echo "========================================================================"
        echo "运行 $run_count/$total_runs: Difficulty=$diff, Seed=$seed"
        echo "========================================================================"
        echo ""
        
        # 运行实验
        run_python experiments/runners/Exp_real_cost_impact_v2.py \
            --mode custom \
            --n-questions $N_QUESTIONS \
            --difficulty $diff \
            --gpus 0,1,2,3,4,5,6,7 \
            --seed $seed \
            --config-path configs/multi_gpu_data_calibrated.yaml \
            --policy-config-path configs/adaptive_policy.yaml \
            --c-r-values "$COST_VALUES" \
            --verbose
        
        if [ $? -ne 0 ]; then
            echo "❌ 运行 $run_count 失败 (diff=$diff, seed=$seed)"
        else
            echo "✓ 运行 $run_count 完成"
        fi
        
        echo ""
    done
done

echo "================================================================================"
echo "✓ 实验1-Lite 完成!"
echo "================================================================================"
echo "结束时间: $(date)"
echo ""
echo "预期观察:"
echo "  c_r=0.005: 高准确率 (~86%), θ_cont≈0.97"
echo "  c_r=0.015: 高准确率 (~85%), θ_cont≈0.96"  
echo "  c_r=0.021: 准确率下降, θ_cont≈0.03 (相变!)"
echo "  c_r=0.050: 最低准确率 (~67%), θ_cont≈0.00"
echo "================================================================================"
