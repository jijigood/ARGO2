# 实验1: 检索成本影响 (真实LLM版本) - 修正版

## 📋 修正内容

相比原始版本 `Exp_real_cost_impact.py`，本版本 (`Exp_real_cost_impact_v2.py`) 做了以下修正：

### ✅ 1. 添加Random策略
- 新增 `simulate_random_policy()` 方法
- Random策略在每一步随机选择Retrieve或Reason (50%概率)
- 现在有4个策略: **ARGO, Always-Retrieve, Always-Reason, Random**

### ✅ 2. 基线策略使用动态θ*
- **修正前**: `Always-Retrieve`和`Always-Reason`都硬编码 `theta_star = 0.9`
- **修正后**: 所有基线策略都接受MDP求解出的`theta_star`作为参数
- 确保所有策略使用相同的终止条件

### ✅ 3. 支持小规模测试和大规模实验切换
- **小规模模式** (`--mode small`):
  - 50道Hard题
  - 5个c_r采样点
  - 预计运行时间: 10-30分钟
  - 用于快速验证实验是否能跑通

- **完整实验模式** (`--mode full`):
  - 全部~12K道题
  - 10个c_r采样点
  - 预计运行时间: 数小时到1天
  - 用于正式的实验结果

### ✅ 4. 图表命名符合实验设计文档
- **Graph 1.A**: Cost vs. Accuracy
- **Graph 1.B**: Cost vs. Retrieval Calls
- **Supplementary**: Cost vs. Total Cost (额外补充分析)

### ✅ 5. 命令行参数支持
```bash
python Exp_real_cost_impact_v2.py --mode small --difficulty hard --gpus 0,1,2,3
```

---

## 🚀 使用方法

### 方法1: 使用Shell脚本 (推荐)

#### Step 1: 小规模测试 (验证能否跑通)
```bash
chmod +x test_exp1.sh
bash test_exp1.sh
```

#### Step 2: 如果测试成功，运行完整实验
```bash
chmod +x run_exp1_full.sh
bash run_exp1_full.sh
```

### 方法2: 直接使用Python

#### 小规模测试
```bash
cd /data/user/huangxiaolin/ARGO2/ARGO

python Exp_real_cost_impact_v2.py \
    --mode small \
    --difficulty hard \
    --gpus 0,1,2,3 \
    --seed 42
```

#### 完整实验
```bash
python Exp_real_cost_impact_v2.py \
    --mode full \
    --difficulty hard \
    --gpus 0,1,2,3 \
    --seed 42
```

---

## 📊 输出结果

### 1. 数据文件
保存在 `draw_figs/data/` 目录:
- `exp1_real_cost_impact_small_YYYYMMDD_HHMMSS.json` (小规模测试结果)
- `exp1_real_cost_impact_full_YYYYMMDD_HHMMSS.json` (完整实验结果)

JSON格式:
```json
{
  "metadata": {
    "test_mode": "small",
    "n_questions": 50,
    "difficulty": "hard",
    "n_cost_steps": 5,
    "timestamp": "20251029_143022"
  },
  "results": [
    {
      "c_r": 0.05,
      "theta_cont": 0.45,
      "theta_star": 0.90,
      "ARGO_accuracy": 0.82,
      "ARGO_quality": 0.91,
      "ARGO_cost": 0.35,
      "ARGO_retrievals": 3.2,
      "Always-Retrieve_accuracy": 0.84,
      ...
    }
  ]
}
```

### 2. 图表文件
保存在 `figs/` 目录:
- `exp1_graph1A_cost_vs_accuracy_small.png` / `_full.png`
- `exp1_graph1B_cost_vs_retrievals_small.png` / `_full.png`
- `exp1_supplementary_cost_vs_total_small.png` / `_full.png`

---

## 🔧 参数说明

```bash
python Exp_real_cost_impact_v2.py [OPTIONS]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `small` | 测试模式: `small` (50题) 或 `full` (~12K题) |
| `--difficulty` | `hard` | 问题难度: `easy`, `medium`, `hard` |
| `--gpus` | `0,1,2,3` | 使用的GPU ID列表，逗号分隔 |
| `--seed` | `42` | 随机种子 |

---

## 📈 预期结果 (按实验设计文档)

### Graph 1.A: Cost vs. Accuracy
- **X轴**: Retrieval Cost ($c_r$)
- **Y轴**: Average Accuracy
- **预期趋势**:
  - **ARGO**: 高准确率且稳定 (自适应调整策略)
  - **Always-Retrieve**: 平坦 (静态策略)
  - **Always-Reason**: 平坦且较低 (缺少检索)
  - **Random**: 平坦 (随机策略，无优化)

### Graph 1.B: Cost vs. Retrieval Calls
- **X轴**: Retrieval Cost ($c_r$)
- **Y轴**: Average Retrieval Calls ($E[R_T]$)
- **预期趋势**:
  - **ARGO**: **随c_r增加而下降** (证明自适应性)
  - **Always-Retrieve**: 平坦且高 (始终检索)
  - **Random**: 平坦 (随机行为)

---

## 🐛 故障排查

### 问题1: ModuleNotFoundError: No module named 'oran_benchmark_loader'
**解决方案**: 确保 `oran_benchmark_loader.py` 在同目录下
```bash
ls /data/user/huangxiaolin/ARGO2/ARGO/oran_benchmark_loader.py
```

### 问题2: Chroma集合不存在
**解决方案**: 脚本会自动降级到模拟检索模式，不影响运行

### 问题3: GPU内存不足
**解决方案**: 减少使用的GPU数量或使用更小的模型
```bash
python Exp_real_cost_impact_v2.py --mode small --gpus 0,1
```

### 问题4: MDP求解失败
**解决方案**: 检查 `configs/multi_gpu.yaml` 配置文件是否存在
```bash
ls /data/user/huangxiaolin/ARGO2/ARGO/configs/multi_gpu.yaml
```

---

## 📝 与原始版本的对比

| 特性 | 原始版本 | 修正版本 v2 |
|------|----------|-------------|
| 策略数量 | 3个 | **4个** (添加Random) |
| θ* 使用 | 硬编码0.9 | **动态传入** |
| 数据集规模 | 固定50题 | **可切换** (50题/12K题) |
| c_r采样点 | 固定5个 | **可切换** (5个/10个) |
| 命令行参数 | 无 | **支持** |
| 测试模式 | 无 | **支持** |
| 图表数量 | 3张 | 3张 (符合文档要求的2张+1张补充) |
| 图表命名 | 通用 | **符合文档规范** |

---

## ⏱️ 预计运行时间

基于 4×RTX 3060 (12GB) 的估算:

| 模式 | 问题数 | c_r点 | GPU利用率 | 预计时间 |
|------|--------|-------|-----------|----------|
| Small | 50 | 5 | ~60% | 10-30分钟 |
| Full | 12K | 10 | ~80% | 8-24小时 |

*实际时间取决于GPU型号、LLM推理速度和Chroma检索速度*

---

## 📧 联系方式

如有问题，请检查:
1. 是否按照 `test_exp1.sh` 的顺序运行
2. 是否有足够的GPU内存
3. 是否所有依赖文件都存在

Happy Experimenting! 🎉
