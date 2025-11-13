#!/bin/bash
# 快速测试多GPU配置
# 使用小规模数据验证系统工作正常

set -e

echo "=========================================="
echo "Multi-GPU Quick Test"
echo "=========================================="
echo ""

# 激活环境
source activate ARGO

# 检查GPU
echo "Step 1: Checking GPU availability..."
echo ""
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# 检查PyTorch CUDA
echo "Step 2: Checking PyTorch CUDA support..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'Number of GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"
echo ""

# 检查必要的库
echo "Step 3: Checking required libraries..."
python -c "
import transformers
import accelerate
import torch
print(f'✓ transformers: {transformers.__version__}')
print(f'✓ accelerate: {accelerate.__version__}')
print(f'✓ torch: {torch.__version__}')
"
echo ""

# 测试1: 单GPU快速测试 (5题)
echo "=========================================="
echo "Test 1: Single GPU (5 questions)"
echo "=========================================="
python mdp_rag_multi_gpu.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --n_questions 5 \
  --difficulty easy \
  --gpu_mode single \
  --gpu_ids 0 \
  --seed 42

echo ""
echo "✓ Test 1 passed!"
echo ""

# 测试2: 双GPU测试 (5题)
echo "=========================================="
echo "Test 2: Dual GPU (5 questions)"
echo "=========================================="
python mdp_rag_multi_gpu.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --n_questions 5 \
  --difficulty easy \
  --gpu_mode data_parallel \
  --gpu_ids 0 1 \
  --seed 42

echo ""
echo "✓ Test 2 passed!"
echo ""

# 测试3: 对比实验测试 (5题)
echo "=========================================="
echo "Test 3: MDP vs Fixed Comparison (5 questions)"
echo "=========================================="
python compare_mdp_vs_fixed_multigpu.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --n_questions 5 \
  --difficulty easy \
  --fixed_k 3 \
  --gpu_mode single \
  --gpu_ids 0 \
  --seed 42

echo ""
echo "✓ Test 3 passed!"
echo ""

echo "=========================================="
echo "All tests passed! ✓"
echo "=========================================="
echo ""
echo "Your multi-GPU setup is working correctly!"
echo ""
echo "Next steps:"
echo "  1. Run medium-scale test: ./run_multi_gpu.sh"
echo "  2. Check results: ls -lh results/multi_gpu/"
echo "  3. Read guide: cat MULTI_GPU_GUIDE.md"
echo ""
