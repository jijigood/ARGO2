#!/bin/bash

###############################################################################
# 实验1: 多种子统计学有效实验 (Question-Adaptive + Cost-Optimized Version)
# 
# 新特性:
#   ✓ Question-adaptive U_max estimation - 简单问题快速终止
#   ✓ Complexity-based threshold scaling - 不同难度不同阈值
#   ✓ Dynamic progress tracking - 智能进度估计
#   ✓ 多个随机种子 (5个) - 确保统计显著性
#   ✓ 多个难度级别 (Easy/Medium/Hard) - 展示泛化能力
#   ✓ [NEW] 成本优化 - 基于2026-01-05实验结果
#
# ============================================================================
# MDP策略的"相变"特性 (2026-01-05实验发现):
# ============================================================================
#
#   c_r ≤ 0.020 (≤ c_p):  θ_cont ≈ 0.96  → 纯检索策略
#   c_r = 0.0205:          θ_cont = 0.03  → 突然转折!
#   c_r > 0.021:           θ_cont ≈ 0.00  → 纯推理策略
#
# 这是MDP阈值策略的数学特性：
#   - 转折发生在 c_r ≈ c_p = 0.02 (理论预测的临界点)
#   - 转折窗口极窄 (~0.0005)
#   - 无法通过改变采样消除，但可以精确定位
#
# 采样策略:
#   - 低成本区 (c_r < c_p): 展示θ_cont稳定性
#   - 转折点区 (c_r ≈ c_p): 精确定位相变
#   - 高成本区 (c_r > c_p): 展示推理策略效果
# ============================================================================
#
# 运行模式:
#   默认:      ./run_exp1_full_optimized.sh        # 精简对比 (7点)
#   详细扫描:  COST_SWEEP=1 ./run_exp1_full_optimized.sh  # 完整扫描 (15点)
#   仅最优:    OPTIMAL_ONLY=1 ./run_exp1_full_optimized.sh # 仅c_r=0.005
#
# 预计运行时间: 6-10小时 (8 GPUs, 取决于采样点数)
###############################################################################

echo "================================================================================"
echo "实验1: 检索成本影响 - Question-Adaptive + Cost-Optimized 版本"
echo "================================================================================"
echo "配置:"
echo "  - 随机种子: 5个 (42-46)"
echo "  - 难度级别: Easy, Medium, Hard"
echo "  - 每难度问题数: 100"
echo "  - 总运行次数: 15 (5种子 × 3难度) × 成本点数"
echo "  - GPU: 8张 (0-7)"
echo "  - 策略: Question-Adaptive (自适应终止)"
echo "================================================================================"
echo "MDP相变特性:"
echo "  - 转折点: c_r ≈ c_p = 0.02"
echo "  - 转折窗口: ~0.0005 (极窄)"
echo "  - c_r < c_p: 纯检索 (θ_cont ≈ 0.96)"
echo "  - c_r > c_p: 纯推理 (θ_cont ≈ 0)"
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

# ============================================================================
# 成本采样策略 (基于MDP相变分析)
# ============================================================================
# 
# 相变特性:
#   c_r = 0.020 → θ_cont = 0.96 (检索)
#   c_r = 0.0205 → θ_cont = 0.03 (转折!)
#   c_r = 0.021 → θ_cont = 0.03 (推理)
#
# 采样原则:
#   1. 低成本区: 间距 0.005, 展示稳定性
#   2. 转折点区: 间距 0.0005-0.001, 精确定位
#   3. 高成本区: 间距 0.01-0.02, 减少冗余
# ============================================================================

if [ "${COST_SWEEP:-0}" = "1" ]; then
    echo "================================================================================"
    echo "模式: 详细扫描 (15点)"
    echo "================================================================================"
    # 完整扫描: 精确定位相变
    COST_VALUES="0.002,0.005,0.008,0.010,0.012,0.015,0.018,0.019,0.020,0.0205,0.021,0.022,0.025,0.030,0.050"
    echo "采样点: $COST_VALUES"
    echo ""
    echo "采样分布:"
    echo "  低成本区 (c_r < 0.02):  8点 - 展示θ_cont稳定在~0.96"
    echo "  转折点区 (c_r ≈ 0.02):  4点 - 精确定位0.020→0.0205的突变"
    echo "  高成本区 (c_r > 0.02):  3点 - 展示推理策略效果"
    echo ""
    
elif [ "${OPTIMAL_ONLY:-0}" = "1" ]; then
    echo "================================================================================"
    echo "模式: 仅最优成本 (c_r = 0.005)"
    echo "================================================================================"
    COST_VALUES="0.005"
    echo "使用最优成本: $COST_VALUES"
    echo "说明: c_r=0.005时准确率最高(86.7%)"
    echo ""
    
else
    echo "================================================================================"
    echo "模式: 精简对比 (7点) [默认]"
    echo "================================================================================"
    # 默认: 关键节点对比
    COST_VALUES="0.005,0.010,0.015,0.020,0.021,0.025,0.050"
    echo "采样点: $COST_VALUES"
    echo ""
    echo "采样设计:"
    echo "  0.005 (0.25×c_p) - 最优点,准确率86.7%"
    echo "  0.010 (0.50×c_p) - 低成本区中点"
    echo "  0.015 (0.75×c_p) - 接近转折点"
    echo "  0.020 (1.00×c_p) - 转折点前,θ_cont≈0.96"
    echo "  0.021 (1.05×c_p) - 转折点后,θ_cont≈0.03"
    echo "  0.025 (1.25×c_p) - 确认推理策略"
    echo "  0.050 (2.50×c_p) - 高成本参考"
    echo ""
fi

echo "可选模式:"
echo "  OPTIMAL_ONLY=1 ./run_exp1_full_optimized.sh  # 仅最优成本"
echo "  COST_SWEEP=1 ./run_exp1_full_optimized.sh    # 详细扫描"
echo ""

# 运行多种子实验
echo "运行多种子实验 (Question-Adaptive + Cost-Optimized)..."
echo ""

run_python experiments/runners/Exp1_multi_seed_wrapper.py \
    --n-seeds 5 \
    --base-seed 42 \
    --n-questions 100 \
    --difficulties easy,medium,hard \
    --gpus 0,1,2,3,4,5,6,7 \
    --policy-config-path configs/adaptive_policy.yaml \
    --c-r-values "$COST_VALUES"

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
echo "实验结果汇总:"
echo "  - 聚合结果: $latest_csv"
echo ""
echo "MDP相变特性验证:"
echo "  - 转折点: c_r = 0.020 → 0.021"
echo "  - θ_cont: 0.96 → 0.03 (突变)"
echo "  - 窗口宽度: ~0.001"
echo ""
echo "论文图表建议:"
echo "  1. θ_cont vs c_r 曲线: 展示阶跃函数特性"
echo "  2. 准确率 vs c_r: 展示最优点在 c_r=0.005"
echo "  3. 使用分段坐标轴突出转折区域"
echo ""
echo "================================================================================"
