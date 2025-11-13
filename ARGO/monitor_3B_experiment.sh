#!/bin/bash
# 3B快速验证实验实时监控脚本
# 每30秒更新一次进度

LOG_FILE="exp1_3B_quick_validation_20251101_015834.log"
PID=2512580

echo "========================================================================"
echo "3B快速验证实验 - 实时监控"
echo "========================================================================"
echo "进程ID: $PID"
echo "日志文件: $LOG_FILE"
echo "开始时间: $(date)"
echo "========================================================================"
echo ""

while true; do
    # 检查进程是否还在运行
    if ! ps -p $PID > /dev/null 2>&1; then
        echo ""
        echo "========================================================================"
        echo "⚠️  实验进程已停止 (PID $PID 不存在)"
        echo "时间: $(date)"
        echo "========================================================================"
        
        # 显示最后的日志
        echo ""
        echo "最后50行日志:"
        echo "------------------------------------------------------------------------"
        tail -50 $LOG_FILE
        break
    fi
    
    # 清屏并显示最新状态
    clear
    
    echo "========================================================================"
    echo "3B快速验证实验 - 实时监控"
    echo "========================================================================"
    echo "当前时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "进程状态: ✓ 运行中 (PID $PID)"
    echo "========================================================================"
    echo ""
    
    # 显示GPU使用情况
    echo "GPU状态:"
    echo "------------------------------------------------------------------------"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader | head -1
    echo ""
    
    # 显示进程CPU和内存使用
    echo "进程资源:"
    echo "------------------------------------------------------------------------"
    ps -p $PID -o pid,pcpu,pmem,etime,cmd --no-headers
    echo ""
    
    # 提取最新进度信息
    echo "实验进度:"
    echo "------------------------------------------------------------------------"
    
    # 获取当前c_r值
    CURRENT_CR=$(grep -oP '\[\d+/10\] c_r = \K[0-9.]+' $LOG_FILE | tail -1)
    CR_INDEX=$(grep -oP '\[\K\d+(?=/10\] c_r)' $LOG_FILE | tail -1)
    
    if [ -n "$CURRENT_CR" ]; then
        echo "当前 c_r: $CURRENT_CR ($CR_INDEX/10)"
    fi
    
    # 获取最新进度
    LATEST_PROGRESS=$(grep -oP '进度: \K\d+/\d+' $LOG_FILE | tail -1)
    
    if [ -n "$LATEST_PROGRESS" ]; then
        CURRENT=$(echo $LATEST_PROGRESS | cut -d'/' -f1)
        TOTAL=$(echo $LATEST_PROGRESS | cut -d'/' -f2)
        PERCENT=$(awk "BEGIN {printf \"%.1f\", ($CURRENT/$TOTAL)*100}")
        echo "题目进度: $CURRENT/$TOTAL ($PERCENT%)"
        
        # 计算总体进度 (考虑10个c_r值)
        if [ -n "$CR_INDEX" ]; then
            COMPLETED_QUESTIONS=$((($CR_INDEX - 1) * $TOTAL + $CURRENT))
            TOTAL_QUESTIONS=$(($TOTAL * 10))
            OVERALL_PERCENT=$(awk "BEGIN {printf \"%.2f\", ($COMPLETED_QUESTIONS/$TOTAL_QUESTIONS)*100}")
            echo "总体进度: $COMPLETED_QUESTIONS/$TOTAL_QUESTIONS ($OVERALL_PERCENT%)"
        fi
    fi
    
    # 计算速度和预估完成时间
    if [ -n "$LATEST_PROGRESS" ] && [ -n "$CR_INDEX" ]; then
        COMPLETED_QUESTIONS=$((($CR_INDEX - 1) * $TOTAL + $CURRENT))
        
        # 获取运行时间
        ELAPSED=$(ps -p $PID -o etime --no-headers | tr -d ' ')
        
        # 转换运行时间为秒
        ELAPSED_SECONDS=$(echo $ELAPSED | awk -F: '{
            if (NF==3) {
                split($1, hm, "-");
                if (length(hm) == 2) {
                    print hm[1]*86400 + hm[2]*3600 + $2*60 + $3
                } else {
                    print $1*3600 + $2*60 + $3
                }
            } else if (NF==2) {
                print $1*60 + $2
            } else {
                print $1
            }
        }')
        
        if [ $COMPLETED_QUESTIONS -gt 0 ] && [ $ELAPSED_SECONDS -gt 0 ]; then
            # 计算速度 (秒/题)
            SPEED=$(awk "BEGIN {printf \"%.1f\", $ELAPSED_SECONDS/$COMPLETED_QUESTIONS}")
            
            # 计算剩余时间
            REMAINING_QUESTIONS=$((10000 - $COMPLETED_QUESTIONS))
            REMAINING_SECONDS=$(awk "BEGIN {printf \"%.0f\", $REMAINING_QUESTIONS*$SPEED}")
            
            # 转换为小时和分钟
            REMAINING_HOURS=$(($REMAINING_SECONDS / 3600))
            REMAINING_MINS=$((($REMAINING_SECONDS % 3600) / 60))
            
            # 计算预计完成时间
            ETA=$(date -d "+$REMAINING_SECONDS seconds" '+%Y-%m-%d %H:%M')
            
            echo ""
            echo "运行时间: $ELAPSED"
            echo "平均速度: ${SPEED}秒/题"
            echo "剩余时间: ${REMAINING_HOURS}小时${REMAINING_MINS}分钟"
            echo "预计完成: $ETA"
        fi
    fi
    
    echo ""
    echo "最新日志 (最后10行):"
    echo "------------------------------------------------------------------------"
    tail -10 $LOG_FILE | grep -v "^$"
    
    echo ""
    echo "========================================================================"
    echo "按 Ctrl+C 停止监控 (不影响实验运行)"
    echo "下次更新: 30秒后"
    echo "========================================================================"
    
    # 等待30秒
    sleep 30
done
