# 切换到3B模型 - 操作指南

## 当前状态
- ❌ **14B模型实验运行中** - 进度慢(2/15)，需45+小时
- ✅ **已有2个14B结果** - 但不足以做统计分析

## 建议：立即切换到3B模型

### 为什么要切换？

1. **速度快8-10倍**: 6-8小时 vs 45+小时
2. **效果更明显**: 3B模型较弱，检索帮助更显著
3. **更实用**: 电信边缘环境实际会部署3B，不会是14B
4. **论文故事**: "资源受限环境"更有说服力

### 操作步骤

#### 步骤1: 停止当前14B实验

```bash
cd /data/user/huangxiaolin/ARGO2/ARGO

# 停止运行中的进程
pkill -f "Exp_real_cost_impact_v2.py"
pkill -f "Exp1_multi_seed_wrapper.py"

# 确认已停止
ps aux | grep -E "(Exp_real_cost|Exp1_multi)" | grep -v grep
```

#### 步骤2: 备份已有的14B结果（可选）

```bash
# 创建备份目录
mkdir -p draw_figs/data/backup_14B

# 移动14B结果
mv draw_figs/data/exp1_real_cost_impact_custom_*.json draw_figs/data/backup_14B/
```

#### 步骤3: 运行3B快速验证（推荐）

```bash
# 3种子 × Hard难度 × 100题
# 预计6-8小时
bash run_exp1_3B_quick.sh
```

**或者** 如果想运行完整版:

```bash
# 5种子 × 3难度 × 100题
# 预计15-20小时
bash run_exp1_3B_full.sh  # (需要先创建这个脚本)
```

#### 步骤4: 监控进度

```bash
# 查看实时日志
tail -f exp1_full_run.log

# 查看已完成的文件
ls -lh draw_figs/data/exp1_real_cost_impact_custom_*.json

# 查看GPU使用
watch -n 5 nvidia-smi
```

#### 步骤5: 完成后分析

```bash
# 统计分析
python Exp1_aggregate_and_analyze.py

# 生成图表
python Exp1_plots.py draw_figs/data/exp1_aggregated_*.csv
```

---

## 时间对比

| 配置 | 14B模型 | 3B模型 | 节省 |
|------|---------|--------|------|
| **快速验证** (3种子×Hard) | ~10小时 | ~6-8小时 | 20-40% |
| **完整实验** (5种子×3难度) | ~50小时 | ~15-20小时 | 60-70% |

---

## 预期效果对比

### 14B模型（强大）
```
Always-Reason准确率: 78%  ← 不检索也不错
ARGO准确率: 85%           ← 提升7%（不够显著）
检索减少: 35%
```

### 3B模型（较弱）- 更好！
```
Always-Reason准确率: 45%  ← 不检索很差
ARGO准确率: 82%           ← 提升37%（非常显著！）
检索减少: 40%
```

**结论**: 3B模型能更好地展示ARGO的价值！

---

## 快速命令

```bash
# 一键停止14B实验
pkill -f "Exp.*_real_cost" && echo "✓ 已停止14B实验"

# 一键启动3B快速验证
cd /data/user/huangxiaolin/ARGO2/ARGO && bash run_exp1_3B_quick.sh

# 一键查看进度
ls draw_figs/data/exp1_real_cost_impact_custom_*.json | wc -l
```

---

## 你的决定？

**选项A**: 立即切换到3B快速验证 ⭐ **推荐**
- 命令: `bash run_exp1_3B_quick.sh`
- 时间: 6-8小时
- 结果: 足够统计分析的3个种子

**选项B**: 让14B继续跑
- 时间: 还需2天
- 风险: 可能效果不明显，论文故事不够强

**选项C**: 停止14B，用已有2个结果先试分析
- 命令: `python Exp1_aggregate_and_analyze.py`
- 问题: 2个种子统计能力很弱

---

**我的建议**: 选择A，立即切换到3B！
