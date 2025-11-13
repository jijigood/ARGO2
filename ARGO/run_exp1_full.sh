#!/bin/bash
# 实验1完整实验脚本
# 运行全部~12K题的完整评估

echo "========================================"
echo "实验1: 检索成本影响 - 完整实验"
echo "========================================"
echo ""
echo "⚠️  警告: 这是完整实验模式!"
echo "   - 问题数量: ~12K题"
echo "   - c_r采样点: 10个"
echo "   - 预计运行时间: 数小时到1天"
echo ""

read -p "确认要运行完整实验吗? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "开始运行..."
echo ""

cd /data/user/huangxiaolin/ARGO2/ARGO

# 记录开始时间
start_time=$(date +%s)

python Exp_real_cost_impact_v2.py \
    --mode full \
    --difficulty hard \
    --gpus 0,1,2,3,4,5,6,7 \
    --seed 42

# 计算运行时间
end_time=$(date +%s)
elapsed=$((end_time - start_time))
hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))

echo ""
echo "========================================"
echo "完整实验完成!"
echo "========================================"
echo "运行时间: ${hours}小时 ${minutes}分钟"
echo ""
