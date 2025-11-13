#!/bin/bash

# 快速测试脚本 - 使用小模型验证 MDP-RAG

echo "========================================"
echo "MDP-RAG Small Model Quick Test"
echo "========================================"

# 检查 Python 环境
echo ""
echo "[1/4] Checking Python environment..."
python --version
if [ $? -ne 0 ]; then
    echo "❌ Python not found!"
    exit 1
fi
echo "✓ Python OK"

# 检查依赖
echo ""
echo "[2/4] Checking dependencies..."

# transformers
python -c "import transformers; print(f'transformers: {transformers.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠ transformers not found, installing..."
    pip install transformers>=4.37.0 -q
fi

# torch
python -c "import torch; print(f'torch: {torch.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ torch not found!"
    exit 1
fi

echo "✓ Dependencies OK"

# 检查 GPU
echo ""
echo "[3/4] Checking device..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'✓ GPU available: {torch.cuda.get_device_name(0)}')
else:
    print('✓ Using CPU (this is fine for small models)')
"

# 运行测试
echo ""
echo "[4/4] Running quick test (5 easy questions)..."
echo "========================================"
echo ""

python mdp_rag_small_llm.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  -n 5 \
  -d easy \
  --seed 42

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ Test completed successfully!"
    echo "========================================"
    echo ""
    echo "Next steps:"
    echo "  1. Run 100 medium questions:"
    echo "     python mdp_rag_small_llm.py --model Qwen/Qwen2.5-3B-Instruct -n 100 -d medium"
    echo ""
    echo "  2. Check results in results/small_llm/"
    echo ""
    echo "  3. See SMALL_MODEL_GUIDE.md for more options"
else
    echo ""
    echo "========================================"
    echo "❌ Test failed!"
    echo "========================================"
    echo "Check error messages above"
fi
