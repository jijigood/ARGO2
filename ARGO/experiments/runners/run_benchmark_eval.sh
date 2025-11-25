#!/bin/bash
# ORAN-Bench-13K RAG 评估系统 - 快速启动脚本

# 设置环境
export ARGO_ENV=/root/miniconda/envs/ARGO/bin/python
export WORK_DIR=/home/data2/huangxiaolin2/ARGO

echo "========================================="
echo "ORAN-Bench-13K RAG 评估系统"
echo "========================================="
echo

# 检查基准数据
echo "[1] 检查基准数据..."
if [ -d "$WORK_DIR/data/benchmark/ORAN-Bench-13K/Benchmark" ]; then
    echo "✓ 基准数据已就绪"
    echo "  - Easy: $(wc -l < $WORK_DIR/data/benchmark/ORAN-Bench-13K/Benchmark/fin_E.json) 问题"
    echo "  - Medium: $(wc -l < $WORK_DIR/data/benchmark/ORAN-Bench-13K/Benchmark/fin_M.json) 问题"
    echo "  - Hard: $(wc -l < $WORK_DIR/data/benchmark/ORAN-Bench-13K/Benchmark/fin_H.json) 问题"
else
    echo "✗ 基准数据未找到"
    exit 1
fi
echo

# 测试加载器
echo "[2] 测试基准加载器..."
cd $WORK_DIR
$ARGO_ENV oran_benchmark_loader.py 2>&1 | head -30
echo

# 运行评估实验
echo "[3] 运行 RAG 评估实验..."
echo "  测试配置: 100 问题 (混合难度)"
$ARGO_ENV -c "
import sys
sys.path.insert(0, '.')
from Exp_RAG_benchmark import run_benchmark_experiment, save_results

results, questions = run_benchmark_experiment(n_questions=100, seed=42)
save_results(results, questions, 'oran_benchmark_mixed.json')
" 2>&1 | grep -E "(Evaluating|Accuracy|Question distribution)" | tail -15
echo

# 生成可视化
echo "[4] 生成可视化图表..."
$ARGO_ENV plot_benchmark_results.py 2>&1 | grep -E "(Saved|completed)"
echo

# 显示结果摘要
echo "[5] 结果摘要"
echo "========================================="
cat $WORK_DIR/draw_figs/benchmark_summary.txt
echo

# 列出生成的文件
echo "[6] 生成的文件"
echo "========================================="
echo "数据文件:"
ls -lh $WORK_DIR/draw_figs/data/oran_benchmark*.json 2>/dev/null | awk '{print "  "$9" ("$5")"}'
echo
echo "可视化图表:"
ls -lh $WORK_DIR/draw_figs/benchmark_*.png 2>/dev/null | awk '{print "  "$9" ("$5")"}'
echo

echo "========================================="
echo "✓ 所有步骤完成！"
echo "========================================="
echo
echo "查看结果:"
echo "  - 图表: $WORK_DIR/draw_figs/benchmark_*.png"
echo "  - 数据: $WORK_DIR/draw_figs/data/oran_benchmark_mixed.json"
echo "  - 摘要: $WORK_DIR/draw_figs/benchmark_summary.txt"
echo
echo "详细文档: $WORK_DIR/ORAN_BENCHMARK_README.md"
