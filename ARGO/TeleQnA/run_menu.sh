#!/bin/bash

# ========================================
# ORAN QA提取工具 - 一键运行脚本
# ========================================

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║     ORAN QA Extraction Tool - TeleQnA Dataset             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 检查当前目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "📁 当前目录: $SCRIPT_DIR"
echo ""

# 主菜单
echo "请选择操作:"
echo ""
echo "  1) 快速测试 (前10个问题)"
echo "  2) 完整提取 - 基础版"
echo "  3) 完整提取 - 增强版 (推荐, 支持断点续传)"
echo "  4) 查看提取进度"
echo "  5) 检查结果统计"
echo "  6) 安装依赖"
echo "  0) 退出"
echo ""
read -p "请输入选项 [0-6]: " choice

case $choice in
    1)
        echo ""
        echo "🚀 运行快速测试..."
        echo ""
        ./run_test.sh
        ;;
    2)
        echo ""
        echo "🚀 运行完整提取 (基础版)..."
        echo "⚠️  注意: 基础版不支持断点续传"
        echo ""
        read -p "确认继续? [y/N]: " confirm
        if [[ $confirm == [yY] ]]; then
            export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
            python extract_oran_qa.py
        else
            echo "已取消"
        fi
        ;;
    3)
        echo ""
        echo "🚀 运行完整提取 (增强版)..."
        echo "✅ 支持断点续传, 错误恢复"
        echo ""
        read -p "确认继续? [y/N]: " confirm
        if [[ $confirm == [yY] ]]; then
            export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
            python extract_oran_qa_enhanced.py
        else
            echo "已取消"
        fi
        ;;
    4)
        echo ""
        echo "📊 查看提取进度..."
        echo ""
        if [ -f "progress.json" ]; then
            cat progress.json | jq '.'
        else
            echo "⚠️  未找到进度文件 (可能还未开始提取或已完成)"
        fi
        ;;
    5)
        echo ""
        echo "📊 结果统计..."
        echo ""
        if [ -f "TeleQnA_ORAN_only.json" ]; then
            oran_count=$(cat TeleQnA_ORAN_only.json | jq 'length')
            total_count=$(cat TeleQnA.txt | grep -c '"question"')
            percentage=$(echo "scale=2; $oran_count * 100 / $total_count" | bc)
            
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "总问题数:      $total_count"
            echo "ORAN问题数:    $oran_count"
            echo "ORAN占比:      $percentage%"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo ""
            echo "随机抽样 (5个ORAN问题):"
            echo ""
            cat TeleQnA_ORAN_only.json | jq -r '.[] | .question' | shuf | head -5 | nl
        else
            echo "⚠️  未找到输出文件 (可能还未完成提取)"
        fi
        ;;
    6)
        echo ""
        echo "📦 安装Python依赖..."
        echo ""
        pip install -r requirements.txt
        echo ""
        echo "✅ 依赖安装完成"
        ;;
    0)
        echo ""
        echo "👋 再见!"
        exit 0
        ;;
    *)
        echo ""
        echo "❌ 无效选项"
        exit 1
        ;;
esac

echo ""
echo "✅ 完成!"
echo ""
