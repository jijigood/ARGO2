#!/bin/bash

# 快速测试ORAN提取功能

echo "========================================="
echo "Quick Test: ORAN QA Extraction"
echo "========================================="
echo ""
echo "仅测试前10个问题以验证功能"
echo ""

# 设置可见GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 切换到脚本目录
cd "$(dirname "$0")"

# 运行测试
python test_extraction.py

echo ""
echo "========================================="
echo "测试完成!"
echo "========================================="
