#!/bin/bash
# 实验1完整实验脚本 (自动运行版本，无需确认)
# 运行全部~12K题的完整评估

echo "========================================"
echo "实验1: 检索成本影响 - 完整实验"
echo "========================================"
echo ""
echo "📊 实验配置:"
echo "   - 问题数量: ~12K题"
echo "   - c_r采样点: 10个"
echo "   - 难度: Hard"
echo "   - GPU: 0,1,2,3,4,5,6,7 (8张 - 全部)"
echo "   - 预计运行时间: 8-24小时"
echo ""
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""

cd /data/user/huangxiaolin/ARGO2/ARGO

# 记录开始时间
start_time=$(date +%s)

# 运行实验 (关闭Python缓冲，实时输出日志)
PYTHONUNBUFFERED=1 python -u Exp_real_cost_impact_v2.py \
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
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "运行时间: ${hours}小时 ${minutes}分钟"
echo ""
echo "结果文件:"
echo "  - 数据: draw_figs/data/exp1_real_cost_impact_full_*.json"
echo "  - 图表: figs/exp1_graph1A_cost_vs_accuracy_full.png"
echo "  - 图表: figs/exp1_graph1B_cost_vs_retrievals_full.png"
echo ""
