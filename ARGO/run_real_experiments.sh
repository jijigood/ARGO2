#!/bin/bash
# 运行真实LLM实验 - 多GPU并行
# 
# 硬件要求:
# - 4张GPU (RTX 3060或更好)
# - CUDA 12.x
# - 60GB+ 总GPU内存
#
# 模型:
# - Qwen2.5-14B-Instruct (本地)
# - all-MiniLM-L6-v2 (本地)
#
# 预计时间:
# - 实验1: ~2-3小时 (5个c_r点 × 3策略 × 50题)
# - 实验2: ~2-3小时 (4个p_s点 × 3策略 × 50题)

set -e

echo "=========================================="
echo "真实LLM实验 - 多GPU并行"
echo "=========================================="
echo "日期: $(date)"
echo "工作目录: $(pwd)"
echo ""

# 检查GPU
echo "检查GPU..."
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo ""

# 检查模型
echo "检查模型文件..."
LLM_MODEL="/data/user/huangxiaolin/ARGO/RAG_Models/models/Qwen2.5-14B-Instruct"
EMB_MODEL="/data/user/huangxiaolin/ARGO/models/all-MiniLM-L6-v2"

if [ ! -d "$LLM_MODEL" ]; then
    echo "❌ 错误: LLM模型不存在 $LLM_MODEL"
    exit 1
fi

if [ ! -d "$EMB_MODEL" ]; then
    echo "❌ 错误: 嵌入模型不存在 $EMB_MODEL"
    exit 1
fi

echo "✓ LLM模型: $LLM_MODEL"
echo "✓ 嵌入模型: $EMB_MODEL"
echo ""

# 选择实验
echo "请选择要运行的实验:"
echo "  1) 实验1: 检索成本影响"
echo "  2) 实验2: 检索成功率影响"
echo "  3) 运行全部实验"
echo ""
read -p "输入选项 [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "=========================================="
        echo "运行实验1: 检索成本影响"
        echo "=========================================="
        echo "参数: 50道Hard题, 5个c_r点, 4张GPU"
        echo ""
        python Exp_real_cost_impact.py
        ;;
    2)
        echo ""
        echo "=========================================="
        echo "运行实验2: 检索成功率影响"
        echo "=========================================="
        echo "参数: 50道Hard题, 4个p_s点, 4张GPU"
        echo ""
        python Exp_real_success_impact.py
        ;;
    3)
        echo ""
        echo "=========================================="
        echo "运行全部实验"
        echo "=========================================="
        echo ""
        
        echo ">>> 实验1: 检索成本影响"
        python Exp_real_cost_impact.py
        
        echo ""
        echo ">>> 实验2: 检索成功率影响"
        python Exp_real_success_impact.py
        ;;
    *)
        echo "无效选项: $choice"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "实验完成!"
echo "=========================================="
echo "结果保存在:"
echo "  - draw_figs/data/exp*_real_*.json"
echo "  - figs/exp*_real_*.png"
echo ""
echo "查看结果:"
echo "  python view_results.py"
echo ""
