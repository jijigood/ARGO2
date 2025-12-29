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

# 选择可用的 Python 解释器 (默认: 当前shell中的python)，若缺少 torch 自动回退到指定conda环境
PYTHON_BIN=${PYTHON_BIN:-python}
PYTHON_ARGS=(-u)
FALLBACK_CONDA_ENV=${CONDA_FALLBACK_ENV:-ARGO}

if ! "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
    echo "⚠ 当前 Python (${PYTHON_BIN}) 未检测到 torch, 尝试激活 conda 环境: ${FALLBACK_CONDA_ENV}"
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE=$(conda info --base 2>/dev/null)
        if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
            # shellcheck source=/dev/null
            source "$CONDA_BASE/etc/profile.d/conda.sh"
            if conda env list | grep -Eq "^[[:space:]]*${FALLBACK_CONDA_ENV}[[:space:]]"; then
                conda activate "$FALLBACK_CONDA_ENV"
                PYTHON_BIN=python
                PYTHON_ARGS=(-u)
                if ! "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
                    echo "❌ 已激活 ${FALLBACK_CONDA_ENV} 但仍无法导入 torch, 请检查环境。"
                    exit 1
                fi
                echo "✓ 已自动激活 conda 环境: ${FALLBACK_CONDA_ENV}"
            else
                echo "❌ 未找到名为 ${FALLBACK_CONDA_ENV} 的 conda 环境, 请设置 CONDA_FALLBACK_ENV 或手动激活环境。"
                exit 1
            fi
        else
            echo "❌ 找不到 conda 初始化脚本, 请先运行 'conda init bash' 或手动激活环境。"
            exit 1
        fi
    else
        echo "❌ 当前环境缺少 torch, 且未检测到 conda 命令。"
        echo "   请手动激活包含 torch 的环境，或通过环境变量 PYTHON_BIN 指定解释器后重新运行。"
        exit 1
    fi
fi

run_python() {
    "$PYTHON_BIN" "${PYTHON_ARGS[@]}" "$@"
}

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

run_python experiments/runners/Exp1_multi_seed_wrapper.py \
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

# 运行聚合脚本
echo ""
echo "================================================================================"
echo "运行聚合脚本: Exp1_aggregate_and_analyze.py"
echo "================================================================================"
run_python Exp1_aggregate_and_analyze.py
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 聚合脚本执行失败!"
    exit 1
fi

# 自动识别最新的聚合CSV
latest_csv=$(ls -t draw_figs/data/exp1_aggregated_*.csv 2>/dev/null | head -n 1)
if [ -z "$latest_csv" ]; then
    echo ""
    echo "❌ 未找到聚合结果 CSV (draw_figs/data/exp1_aggregated_*.csv)"
    exit 1
fi

echo ""
echo "✓ 最新聚合结果: $latest_csv"

# 使用最新CSV生成图表
echo "================================================================================"
echo "生成图表: python Exp1_plots.py $latest_csv"
echo "================================================================================"
run_python Exp1_plots.py "$latest_csv"
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 生成图表失败!"
    exit 1
fi

echo ""
echo "================================================================================"
echo "✓ 实验完成!"
echo "================================================================================"
echo "结束时间: $(date)"
echo ""
echo "下一步:"
echo "  1. 聚合结果: 已自动运行 (如需重新聚合: python Exp1_aggregate_and_analyze.py)"
echo ""
echo "  2. 生成图表: 已使用最新CSV ($latest_csv) 运行"
echo "     若需再次生成: python Exp1_plots.py $latest_csv"
echo ""
echo "  3. 检查question-adaptive效果:"
echo "     - 查看CSV中的 'question_umax' 列"
echo "     - 对比不同complexity的平均steps"
echo "     - 检查 'terminated_early' 比率"
echo ""
echo "================================================================================"
