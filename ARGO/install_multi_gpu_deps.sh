#!/bin/bash
# 一键安装多GPU支持所需的依赖（如果缺失）

echo "Checking and installing Multi-GPU dependencies..."

# 检查并安装 accelerate
python -c "import accelerate" 2>/dev/null || {
    echo "Installing accelerate..."
    pip install accelerate
}

# 检查并安装 bitsandbytes (用于量化，可选)
python -c "import bitsandbytes" 2>/dev/null || {
    echo "Installing bitsandbytes (optional, for quantization)..."
    pip install bitsandbytes
}

echo ""
echo "✓ All dependencies checked/installed!"
echo ""
echo "Verification:"
python -c "
import torch
import transformers
import accelerate
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ Transformers: {transformers.__version__}')
print(f'✓ Accelerate: {accelerate.__version__}')
print(f'✓ CUDA: {torch.cuda.is_available()} ({torch.cuda.device_count()} GPUs)')
"
