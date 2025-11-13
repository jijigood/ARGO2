#!/bin/bash

# ORAN QA提取脚本
# 使用8张GPU并行推理

echo "========================================="
echo "ORAN QA Extraction from TeleQnA Dataset"
echo "========================================="
echo ""
echo "配置信息:"
echo "  - 模型: Qwen2.5-14B-Instruct"
echo "  - GPU数量: 8"
echo "  - 批处理大小: 32"
echo ""

# 设置可见GPU (使用0-7号GPU)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 切换到脚本目录
cd "$(dirname "$0")"

# 运行提取脚本
python extract_oran_qa.py

echo ""
echo "========================================="
echo "提取完成!"
echo "========================================="
