#!/bin/bash
# 实验1测试脚本
# 快速验证实验是否能跑通

echo "========================================"
echo "实验1: 检索成本影响 - 测试脚本"
echo "========================================"
echo ""

# 检查Python环境
echo "[1/3] 检查Python环境..."
python --version
echo ""

# 检查GPU
echo "[2/3] 检查GPU..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# 运行小规模测试
echo "[3/3] 运行小规模测试 (50题, 5个c_r点)..."
echo "预计运行时间: 10-30分钟 (取决于GPU速度)"
echo ""

cd /data/user/huangxiaolin/ARGO2/ARGO

python Exp_real_cost_impact_v2.py \
    --mode small \
    --difficulty hard \
    --gpus 0,1,2,3,4,5,6,7 \
    --seed 42

echo ""
echo "========================================"
echo "测试完成!"
echo "========================================"
echo ""
echo "如果测试成功，请运行完整实验:"
echo "  bash run_exp1_full.sh"
echo ""
