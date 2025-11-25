#!/bin/bash
# 运行ARGO所有实验的便捷脚本
# 使用方法: ./run_all_experiments.sh

set -e  # 遇到错误立即退出

echo "========================================"
echo "ARGO实验批量运行脚本"
echo "========================================"
echo "日期: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 确保在ARGO环境中
if [[ "$CONDA_DEFAULT_ENV" != "ARGO" ]]; then
    echo "错误: 请先激活ARGO环境"
    echo "运行: source activate ARGO"
    exit 1
fi

# 进入项目目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "当前目录: $(pwd)"
echo "Python版本: $(python --version)"
echo ""

# 创建结果目录
mkdir -p figs
mkdir -p draw_figs/data

# 运行实验1
echo "========================================"
echo "实验1: 检索成本(c_r)的影响"
echo "========================================"
echo "开始时间: $(date '+%H:%M:%S')"
python Exp_retrieval_cost_impact.py
echo "完成时间: $(date '+%H:%M:%S')"
echo ""

# 运行实验2
echo "========================================"
echo "实验2: 检索成功率(p_s)的影响"
echo "========================================"
echo "开始时间: $(date '+%H:%M:%S')"
python Exp_retrieval_success_impact.py
echo "完成时间: $(date '+%H:%M:%S')"
echo ""

# 总结
echo "========================================"
echo "所有实验完成!"
echo "========================================"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "生成的文件:"
echo "----------------------------------------"
echo "图表 (figs/):"
ls -1 figs/exp*.png | while read file; do
    size=$(du -h "$file" | cut -f1)
    echo "  - $file ($size)"
done
echo ""
echo "数据 (draw_figs/data/):"
ls -1 draw_figs/data/exp*.json | tail -2 | while read file; do
    size=$(du -h "$file" | cut -f1)
    echo "  - $file ($size)"
done
echo ""
echo "报告文档:"
echo "  - EXPERIMENT1_REPORT.md"
echo "  - EXPERIMENT2_REPORT.md"
echo "  - EXPERIMENTS_INDEX.md"
echo ""
echo "========================================"
echo "查看图表: eog figs/exp*.png"
echo "查看报告: cat EXPERIMENTS_INDEX.md"
echo "========================================"
