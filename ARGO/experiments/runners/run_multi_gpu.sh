#!/bin/bash
# Multi-GPU MDP-RAG 运行脚本
# 硬件: 8x RTX 3060 (12GB each)

set -e

echo "=========================================="
echo "Multi-GPU MDP-RAG Evaluation Suite"
echo "=========================================="
echo ""

# 激活环境
source activate ARGO

# 检查GPU
echo "Checking GPU availability..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# ========================================
# 实验1: 快速测试 (7B模型, 单GPU)
# ========================================
echo "=========================================="
echo "Test 1: Quick Test (7B model, 1 GPU)"
echo "=========================================="
python mdp_rag_multi_gpu.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --n_questions 10 \
  --difficulty easy \
  --gpu_mode single \
  --gpu_ids 0 \
  --seed 42

echo ""
echo "✓ Test 1 completed!"
echo ""

# ========================================
# 实验2: 中等规模 (7B模型, 数据并行, 4 GPU)
# ========================================
echo "=========================================="
echo "Test 2: Medium Scale (7B, 4 GPUs)"
echo "=========================================="
python mdp_rag_multi_gpu.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --n_questions 100 \
  --difficulty medium \
  --gpu_mode data_parallel \
  --gpu_ids 0 1 2 3 \
  --seed 42

echo ""
echo "✓ Test 2 completed!"
echo ""

# ========================================
# 实验3: 大模型 (14B, Accelerate自动分配)
# ========================================
echo "=========================================="
echo "Test 3: Large Model (14B, Auto)"
echo "=========================================="
python mdp_rag_multi_gpu.py \
  --model Qwen/Qwen2.5-14B-Instruct \
  --n_questions 50 \
  --difficulty medium \
  --gpu_mode accelerate \
  --seed 42

echo ""
echo "✓ Test 3 completed!"
echo ""

# ========================================
# 实验4: 全量评估 (7B, 8 GPU)
# ========================================
# echo "=========================================="
# echo "Test 4: Full Evaluation (7B, 8 GPUs, 1000q)"
# echo "=========================================="
# python mdp_rag_multi_gpu.py \
#   --model Qwen/Qwen2.5-7B-Instruct \
#   --n_questions 1000 \
#   --difficulty mixed \
#   --gpu_mode data_parallel \
#   --seed 42

# echo ""
# echo "✓ Test 4 completed!"
# echo ""

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Results saved in: results/multi_gpu/"
ls -lh results/multi_gpu/
