#!/bin/bash
# 快速查看3B实验进度 (单次查看)

LOG_FILE="exp1_3B_quick_validation_20251101_015834.log"
PID=2512580

echo ""
echo "========================================================================"
echo "3B快速验证实验 - 进度快照"
echo "========================================================================"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"

# 检查进程
if ps -p $PID > /dev/null 2>&1; then
    echo "状态: ✓ 运行中 (PID $PID)"
    
    # GPU状态
    echo ""
    echo "GPU使用:"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -1 | awk -F', ' '{printf "  GPU %s: %s%% 利用率, %s/%s MB 显存\n", $1, $2, $3, $4}'
    
    # 运行时间
    ELAPSED=$(ps -p $PID -o etime --no-headers | tr -d ' ')
    echo "  运行时间: $ELAPSED"
else
    echo "状态: ✗ 已停止"
fi

echo ""
echo "实验进度:"
echo "------------------------------------------------------------------------"

# 当前c_r
CURRENT_CR=$(grep -oP '\[\d+/10\] c_r = \K[0-9.]+' $LOG_FILE | tail -1)
CR_INDEX=$(grep -oP '\[\K\d+(?=/10\] c_r)' $LOG_FILE | tail -1)

if [ -n "$CURRENT_CR" ]; then
    echo "  当前 c_r: $CURRENT_CR ($CR_INDEX/10)"
fi

# 题目进度
LATEST_PROGRESS=$(grep -oP '进度: \K\d+/\d+' $LOG_FILE | tail -1)

if [ -n "$LATEST_PROGRESS" ]; then
    CURRENT=$(echo $LATEST_PROGRESS | cut -d'/' -f1)
    TOTAL=$(echo $LATEST_PROGRESS | cut -d'/' -f2)
    PERCENT=$(awk "BEGIN {printf \"%.1f\", ($CURRENT/$TOTAL)*100}")
    
    echo "  本轮进度: $CURRENT/$TOTAL ($PERCENT%)"
    
    # 总体进度
    if [ -n "$CR_INDEX" ]; then
        COMPLETED=$((($CR_INDEX - 1) * $TOTAL + $CURRENT))
        TOTAL_Q=$(($TOTAL * 10))
        OVERALL=$(awk "BEGIN {printf \"%.2f\", ($COMPLETED/$TOTAL_Q)*100}")
        echo "  总体进度: $COMPLETED/$TOTAL_Q ($OVERALL%)"
        
        # 速度估算
        if ps -p $PID > /dev/null 2>&1 && [ $COMPLETED -gt 0 ]; then
            ELAPSED_SEC=$(ps -p $PID -o etimes --no-headers | tr -d ' ')
            SPEED=$(awk "BEGIN {printf \"%.1f\", $ELAPSED_SEC/$COMPLETED}")
            REMAINING=$((10000 - $COMPLETED))
            REMAINING_SEC=$(awk "BEGIN {printf \"%.0f\", $REMAINING*$SPEED}")
            REMAINING_HR=$(($REMAINING_SEC / 3600))
            ETA=$(date -d "+$REMAINING_SEC seconds" '+%m-%d %H:%M')
            
            echo ""
            echo "  平均速度: ${SPEED}秒/题"
            echo "  剩余时间: 约${REMAINING_HR}小时"
            echo "  预计完成: $ETA"
        fi
    fi
fi

echo ""
echo "========================================================================"
echo ""
