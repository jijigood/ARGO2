#!/bin/bash

###############################################################################
# 实验1: 多种子统计学有效实验 (Question-Adaptive Version)
# 
# 新特性:
#   ✓ Question-adaptive U_max estimation - 简单问题快速终止
#   ✓ Complexity-based threshold scaling - 不同难度不同阈值
#   ✓ Dynamic progress tracking - 智能进度估计
#   ✓ 多个随机种子 (5个) - 确保统计显著性
#   ✓ 多个难度级别 (Easy/Medium/Hard) - 展示泛化能力
#
# 预期效果:
#   - Easy questions: ~2-3步平均 (vs 之前 8-10步)
#   - Medium questions: ~4-5步平均 (vs 之前 8-10步)
#   - Hard questions: ~7-9步平均 (vs 之前 8-10步)
#   - 总体效率提升: ~40-50%
#
# 预计运行时间: 6-8小时 (8 GPUs) - 比之前快 30-40%!
###############################################################################

echo "================================================================================"
echo "实验1: 检索成本影响 - Question-Adaptive版本"
echo "================================================================================"
echo "配置:"
echo "  - 随机种子: 5个 (42-46)"
echo "  - 难度级别: Easy, Medium, Hard"
echo "  - 每难度问题数: 100"
echo "  - 总运行次数: 15 (5种子 × 3难度)"
echo "  - GPU: 8张 (0-7)"
echo "  - 策略: Question-Adaptive (自适应终止)"
echo "================================================================================"
echo "新特性:"
echo "  ✓ 简单问题自动早停 (预期 2-3步 vs 之前 8-10步)"
echo "  ✓ 复杂度感知阈值 (每个问题不同的U_max)"
echo "  ✓ 动态进度追踪 (coverage + novelty + confidence)"
echo "================================================================================"
echo "开始时间: $(date)"
echo ""

# 使用所有8张GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1

# 切换到工作目录
cd /data/user/huangxiaolin/ARGO2/ARGO

# 确保配置文件存在
if [ ! -f "configs/adaptive_policy.yaml" ]; then
    echo "❌ 错误: 找不到配置文件 configs/adaptive_policy.yaml"
    echo "   请先创建配置文件 (参考前面提供的模板)"
    exit 1
fi

echo "✓ 配置文件检查通过"
echo ""

# 运行多种子实验
echo "运行多种子实验 (Question-Adaptive)..."
echo ""

python -u experiments/runners/Exp1_multi_seed_wrapper.py \
    --n-seeds 5 \
    --base-seed 42 \
    --n-questions 100 \
    --difficulties easy,medium,hard \
    --gpus 0,1,2,3,4,5,6,7 \
    --policy-config-path configs/adaptive_policy.yaml

# 检查是否成功
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 实验失败!"
    exit 1
fi

echo ""
echo "================================================================================"
echo "✓ 实验完成!"
echo "================================================================================"
echo "结束时间: $(date)"
echo ""
echo "下一步:"
echo "  1. 聚合结果并生成统计分析:"
echo "     python Exp1_aggregate_and_analyze.py"
echo ""
echo "  2. 生成图表:"
echo "     python Exp1_plots.py draw_figs/data/exp1_aggregated_XXXXXX.csv"
echo ""
echo "  3. 检查question-adaptive效果:"
echo "     - 查看CSV中的 'question_umax' 列"
echo "     - 对比不同complexity的平均steps"
echo "     - 检查 'terminated_early' 比率"
echo ""
echo "================================================================================"
