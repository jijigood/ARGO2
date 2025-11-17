#!/bin/bash
# Experiment 0 完整性检查脚本
# 验证所有必要的文件都已创建

echo "========================================"
echo "Experiment 0 完整性检查"
echo "========================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1"
        return 0
    else
        echo -e "${RED}✗${NC} $1 (缺失)"
        return 1
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} $1/"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} $1/ (将自动创建)"
        return 0
    fi
}

total=0
passed=0

echo "1. 核心文件检查"
echo "----------------------------------------"
if check_file "Exp0_threshold_structure_validation.py"; then ((passed++)); fi; ((total++))
if check_file "run_exp0.py"; then ((passed++)); fi; ((total++))
if check_file "test_exp0_quick.py"; then ((passed++)); fi; ((total++))
echo ""

echo "2. 文档文件检查"
echo "----------------------------------------"
if check_file "EXPERIMENT0_README.md"; then ((passed++)); fi; ((total++))
if check_file "EXPERIMENT0_COMPLETION_SUMMARY.md"; then ((passed++)); fi; ((total++))
if check_file "EXPERIMENTS_INDEX.md"; then ((passed++)); fi; ((total++))
echo ""

echo "3. 目录检查"
echo "----------------------------------------"
if check_dir "figs"; then ((passed++)); fi; ((total++))
if check_dir "results"; then ((passed++)); fi; ((total++))
if check_dir "../ARGO_MDP"; then ((passed++)); fi; ((total++))
if check_dir "../ARGO_MDP/src"; then ((passed++)); fi; ((total++))
echo ""

echo "4. 依赖检查"
echo "----------------------------------------"
if check_file "../ARGO_MDP/src/mdp_solver.py"; then ((passed++)); fi; ((total++))
echo ""

echo "========================================"
echo "检查结果: $passed/$total 通过"
echo "========================================"
echo ""

if [ $passed -eq $total ]; then
    echo -e "${GREEN}✓ 所有必要文件都已就绪!${NC}"
    echo ""
    echo "可以运行实验:"
    echo "  1. 快速测试: python test_exp0_quick.py"
    echo "  2. 完整实验: python run_exp0.py"
    echo ""
    exit 0
else
    echo -e "${RED}✗ 某些文件缺失，请检查上面的列表${NC}"
    echo ""
    exit 1
fi
