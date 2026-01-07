#!/bin/bash

###############################################################################
# 实验1-并行版: 使用两组GPU同时运行
#
# 策略:
#   - Group A (GPU 0-3): 运行 easy + hard 难度
#   - Group B (GPU 4-7): 运行 medium 难度 + 其他种子
#
# 预期加速: 原版 50h → 约 25-30h (接近2倍)
###############################################################################

echo "================================================================================"
echo "实验1-并行版: 双GPU组并行运行"
echo "================================================================================"
echo "配置:"
echo "  - Group A: GPU 0-3, 处理 easy/hard"
echo "  - Group B: GPU 4-7, 处理 medium"
echo "  - 每组独立加载模型,并行执行"
echo "================================================================================"
echo "开始时间: $(date)"
echo ""

cd /data/user/huangxiaolin/ARGO2/ARGO

# 激活conda环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ARGO

# 共享配置
N_QUESTIONS=50
COST_VALUES="0.005,0.015,0.021,0.050"

# 创建日志目录
mkdir -p /tmp/argo_parallel

# ============================================================================
# Group A: GPU 0-3, 处理 easy 和 hard (seed 42, 43)
# ============================================================================
run_group_a() {
    echo "[Group A] 启动 GPU 0-3..."
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    
    for diff in easy hard; do
        for seed in 42 43; do
            echo "[Group A] Running: $diff, seed=$seed"
            python experiments/runners/Exp_real_cost_impact_v2.py \
                --mode custom \
                --n-questions $N_QUESTIONS \
                --difficulty $diff \
                --gpus 0,1,2,3 \
                --seed $seed \
                --config-path configs/multi_gpu_data_calibrated.yaml \
                --policy-config-path configs/adaptive_policy.yaml \
                --c-r-values "$COST_VALUES" \
                --verbose 2>&1 | tee -a /tmp/argo_parallel/group_a_${diff}_${seed}.log
            echo "[Group A] 完成: $diff, seed=$seed"
        done
    done
    echo "[Group A] 全部完成!"
}

# ============================================================================
# Group B: GPU 4-7, 处理 medium (seed 42, 43)
# ============================================================================
run_group_b() {
    echo "[Group B] 启动 GPU 4-7..."
    export CUDA_VISIBLE_DEVICES=4,5,6,7
    
    for seed in 42 43; do
        echo "[Group B] Running: medium, seed=$seed"
        python experiments/runners/Exp_real_cost_impact_v2.py \
            --mode custom \
            --n-questions $N_QUESTIONS \
            --difficulty medium \
            --gpus 0,1,2,3 \
            --seed $seed \
            --config-path configs/multi_gpu_data_calibrated.yaml \
            --policy-config-path configs/adaptive_policy.yaml \
            --c-r-values "$COST_VALUES" \
            --verbose 2>&1 | tee -a /tmp/argo_parallel/group_b_medium_${seed}.log
        echo "[Group B] 完成: medium, seed=$seed"
    done
    echo "[Group B] 全部完成!"
}

# ============================================================================
# 并行启动两个组
# ============================================================================
echo "启动并行实验..."
echo ""

# 后台启动 Group A
run_group_a &
PID_A=$!
echo "Group A 启动, PID: $PID_A"

# 等待一小段时间让模型加载错开,避免内存峰值
sleep 30

# 后台启动 Group B
run_group_b &
PID_B=$!
echo "Group B 启动, PID: $PID_B"

echo ""
echo "================================================================================"
echo "两组实验已并行启动!"
echo "  - Group A (GPU 0-3): PID $PID_A, 日志: /tmp/argo_parallel/group_a_*.log"
echo "  - Group B (GPU 4-7): PID $PID_B, 日志: /tmp/argo_parallel/group_b_*.log"
echo ""
echo "监控命令:"
echo "  tail -f /tmp/argo_parallel/group_a_*.log  # 查看 Group A"
echo "  tail -f /tmp/argo_parallel/group_b_*.log  # 查看 Group B"
echo "  nvidia-smi -l 5                            # 监控 GPU 使用"
echo "================================================================================"

# 等待两组都完成
wait $PID_A
STATUS_A=$?
echo "[Group A] 退出状态: $STATUS_A"

wait $PID_B
STATUS_B=$?
echo "[Group B] 退出状态: $STATUS_B"

echo ""
echo "================================================================================"
echo "✓ 并行实验全部完成!"
echo "结束时间: $(date)"
echo "================================================================================"
