#!/bin/bash
# ARGO Enhanced Prompts V2.0 - 快速开始脚本
# =============================================

set -e  # 遇到错误立即退出

echo "========================================"
echo "ARGO Enhanced Prompts V2.0 快速开始"
echo "========================================"
echo ""

# 检查Python环境
echo "1️⃣ 检查Python环境..."
if ! command -v python &> /dev/null; then
    echo "❌ 未找到Python，请先安装Python 3.8+"
    exit 1
fi
python --version
echo "✅ Python环境正常"
echo ""

# 检查必要的包
echo "2️⃣ 检查依赖包..."
python -c "import torch; import transformers; print('✅ PyTorch和Transformers已安装')" || {
    echo "❌ 缺少必要的包，请运行: pip install torch transformers"
    exit 1
}
echo ""

# 检查模型
echo "3️⃣ 检查LLM模型..."
MODEL_PATH="/data/user/huangxiaolin/ARGO/RAG_Models/models/Qwen2.5-1.5B-Instruct"
if [ -d "$MODEL_PATH" ]; then
    echo "✅ 找到模型: $MODEL_PATH"
else
    echo "⚠️  未找到默认模型"
    echo "   请修改 test_enhanced_prompts.py 中的 --model 参数"
    MODEL_PATH=""
fi
echo ""

# 运行快速测试
echo "4️⃣ 运行快速测试（Mock模式）..."
echo "   这将测试所有增强的prompts组件"
echo ""

if [ -n "$MODEL_PATH" ]; then
    python test_enhanced_prompts.py --mode quick --model "$MODEL_PATH"
else
    echo "跳过测试（未找到模型）"
    echo ""
    echo "手动运行测试："
    echo "  python test_enhanced_prompts.py --mode quick --model /path/to/your/model"
fi

echo ""
echo "========================================"
echo "✅ 设置完成！"
echo "========================================"
echo ""
echo "下一步："
echo ""
echo "1. 快速测试（Mock检索）："
echo "   python test_enhanced_prompts.py --mode quick"
echo ""
echo "2. 完整测试（真实Chroma检索）："
echo "   python test_enhanced_prompts.py --mode full"
echo ""
echo "3. 集成到实验中："
echo "   from src.argo_system import ARGO_System"
echo "   argo = ARGO_System(model, tokenizer)"
echo "   answer, _, _ = argo.run_episode(question)"
echo ""
echo "4. 查看README了解详细信息："
echo "   cat PROMPTS_V2_README.md"
echo ""
