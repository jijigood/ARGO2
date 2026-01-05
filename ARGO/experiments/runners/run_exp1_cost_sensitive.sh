#!/bin/bash
# ============================================================================
# 实验1: 成本敏感性验证 - 扩展成本范围
# ============================================================================
# 
# 问题诊断:
#   MDP策略在 c_r ≈ 0.02 (1x c_p) 处发生转换
#   原实验成本范围 (0.04-0.20) 全在 θ_cont=0 区域
#   
# 解决方案:
#   扩展成本扫描范围到 0.01-0.05，覆盖策略转换点
#
# 预期结果:
#   c_r=0.01 (0.5x): 高检索次数 (~6-8)
#   c_r=0.02 (1.0x): 策略转换点
#   c_r=0.03 (1.5x): 低检索次数 (~2-3)
#   c_r=0.05 (2.5x): 最低检索次数 (~1-2)
#
# ============================================================================

set -e

echo "=============================================================================="
echo "实验1: 成本敏感性验证 - 扩展成本范围"
echo "=============================================================================="
echo "开始时间: $(date)"
echo ""

# 配置
SEEDS=(42 43 44)
DIFFICULTIES=("easy" "medium" "hard")
N_QUESTIONS=50  # 使用较少问题快速验证

# 关键: 扩展成本范围覆盖策略转换点
# c_p = 0.02, 所以:
#   0.01 = 0.5x c_p
#   0.015 = 0.75x c_p  
#   0.02 = 1.0x c_p (转换点)
#   0.025 = 1.25x c_p
#   0.03 = 1.5x c_p
#   0.04 = 2.0x c_p
#   0.05 = 2.5x c_p
COST_VALUES="0.01,0.015,0.02,0.025,0.03,0.04,0.05"

echo "配置:"
echo "  - 随机种子: ${SEEDS[@]}"
echo "  - 难度级别: ${DIFFICULTIES[@]}"
echo "  - 每难度问题数: $N_QUESTIONS"
echo "  - 成本点: $COST_VALUES"
echo "=============================================================================="

cd /data/user/huangxiaolin/ARGO2/ARGO

for difficulty in "${DIFFICULTIES[@]}"; do
    echo ""
    echo "================ 难度: ${difficulty^^} ================"
    
    for seed in "${SEEDS[@]}"; do
        echo ""
        echo "---- Seed $seed, Difficulty $difficulty ----"
        
        python experiments/runners/Exp_real_cost_impact_v2.py \
            --mode custom \
            --n-questions $N_QUESTIONS \
            --difficulty $difficulty \
            --gpus 0,1,2,3,4,5,6,7 \
            --seed $seed \
            --config-path configs/multi_gpu_data_calibrated.yaml \
            --policy-config-path configs/adaptive_policy.yaml \
            --c-r-values "$COST_VALUES" \
            --verbose
        
        echo "✓ 完成: $difficulty / Seed $seed"
    done
done

echo ""
echo "=============================================================================="
echo "所有实验完成!"
echo "结束时间: $(date)"
echo "=============================================================================="
echo ""
echo "下一步:"
echo "  1. 运行聚合脚本: python Exp1_aggregate_and_analyze.py"
echo "  2. 生成图表查看成本敏感性"
