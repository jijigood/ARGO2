# 实验1脚本修正对比表

## 📊 核心修正汇总

| # | 问题 | 原始版本 | 修正版本 v2 | 影响 |
|---|------|----------|-------------|------|
| 1 | **Random策略缺失** | ❌ 只有3个策略 | ✅ 4个策略 (ARGO, Always-Retrieve, Always-Reason, **Random**) | 缺少重要基线对比 |
| 2 | **θ* 硬编码** | ❌ `theta_star = 0.9` (硬编码) | ✅ 动态传入MDP求解的θ* | 基线策略不公平 |
| 3 | **数据集规模** | ⚠️ 固定50题 | ✅ 可切换 (50题测试 / 12K题完整实验) | 缺少大规模验证 |
| 4 | **图表命名** | ⚠️ 通用命名 | ✅ 符合文档规范 (Graph 1.A/1.B) | 与文档不一致 |
| 5 | **测试模式** | ❌ 无 | ✅ 支持小规模快速测试 | 难以验证 |
| 6 | **命令行参数** | ❌ 无 | ✅ 完整参数支持 | 灵活性差 |

---

## 🔍 详细对比

### 1. 策略对比

#### 原始版本 (3个策略)
```python
def evaluate_all_policies(...):
    results = {
        'ARGO': [],
        'Always-Retrieve': [],
        'Always-Reason': []
    }
    # ❌ 缺少 Random 策略
```

#### 修正版本 (4个策略)
```python
def evaluate_all_policies(...):
    results = {
        'ARGO': [],
        'Always-Retrieve': [],
        'Always-Reason': [],
        'Random': []  # ✅ 新增
    }
    
    # ✅ 新增 Random 策略实现
    result = self.simulate_random_policy(question, c_r, theta_star)
    results['Random'].append(result)
```

---

### 2. θ* 使用对比

#### 原始版本 (硬编码)
```python
def simulate_always_retrieve_policy(self, question: Dict, c_r: float) -> Dict:
    theta_star = 0.9  # ❌ 硬编码，不随c_r变化
    # ...
```

#### 修正版本 (动态传入)
```python
def simulate_always_retrieve_policy(self, question: Dict, c_r: float, theta_star: float) -> Dict:
    # ✅ 使用MDP求解的theta_star
    # theta_star会随着c_r的变化而动态调整
    # ...
```

**为什么重要？**
- MDP求解会根据c_r调整最优θ*
- 硬编码0.9会导致基线策略使用错误的终止条件
- 修正后所有策略使用相同的θ*，对比更公平

---

### 3. 数据集规模对比

#### 原始版本
```python
def __init__(self, ..., n_test_questions: int = 50, ...):
    # ❌ 固定50题，无法切换到大规模实验
    self.test_questions = self.benchmark.sample_questions(
        n=50,  # 固定值
        difficulty=difficulty,
        seed=seed
    )
```

#### 修正版本
```python
def __init__(self, ..., test_mode: str = "small", ...):
    # ✅ 支持两种模式
    if test_mode == "small":
        self.n_test_questions = 50
        self.n_cost_steps = 5
    elif test_mode == "full":
        self.n_test_questions = None  # 全部~12K题
        self.n_cost_steps = 10
    
    if self.n_test_questions:
        self.test_questions = self.benchmark.sample_questions(...)
    else:
        self.test_questions = self.benchmark.get_test_set(...)  # 全部
```

**使用示例：**
```bash
# 小规模测试 (50题, 5个c_r点, 10-30分钟)
python Exp_real_cost_impact_v2.py --mode small

# 完整实验 (12K题, 10个c_r点, 8-24小时)
python Exp_real_cost_impact_v2.py --mode full
```

---

### 4. 图表对比

#### 原始版本
```python
# 图1: exp1_real_cost_vs_quality.png
plt.ylabel('Average Quality')  # ⚠️ 与文档要求的"Accuracy"不一致

# 图2: exp1_real_cost_vs_retrievals.png
# ✅ 基本符合

# 图3: exp1_real_cost_vs_accuracy.png
# ⚠️ 文档只要求2张图，但实现了3张
```

#### 修正版本
```python
# 图1.A: exp1_graph1A_cost_vs_accuracy_small.png
plt.ylabel('Average Accuracy')  # ✅ 符合文档
plt.title('Graph 1.A: Cost vs. Accuracy')  # ✅ 符合文档命名

# 图1.B: exp1_graph1B_cost_vs_retrievals_small.png
plt.ylabel('Average Retrieval Calls ($E[R_T]$)')  # ✅ 符合文档
plt.title('Graph 1.B: Cost vs. Retrieval Calls')  # ✅ 符合文档命名

# 补充图: exp1_supplementary_cost_vs_total_small.png
plt.title('Supplementary: Cost vs. Total Cost')  # ✅ 标注为补充分析
```

---

### 5. 运行方式对比

#### 原始版本
```python
# ❌ 只能修改代码中的参数
if __name__ == "__main__":
    experiment = RealCostImpactExperiment(
        n_test_questions=50,  # 硬编码，需要改代码
        difficulty="hard",    # 硬编码，需要改代码
        gpu_ids=[0, 1, 2, 3]  # 硬编码，需要改代码
    )
    main()
```

#### 修正版本
```bash
# ✅ 支持命令行参数
python Exp_real_cost_impact_v2.py \
    --mode small \           # 灵活切换
    --difficulty hard \      # 灵活切换
    --gpus 0,1,2,3 \        # 灵活切换
    --seed 42               # 灵活切换

# ✅ 更简单的shell脚本
bash test_exp1.sh           # 一键测试
bash run_exp1_full.sh       # 一键运行完整实验
```

---

## 📋 实验设计文档要求检查表

| 要求 | 原始版本 | 修正版本 v2 |
|------|----------|-------------|
| 4个策略 (ARGO, Always-Retrieve, Always-Reason, Random) | ❌ 只有3个 | ✅ 4个 |
| 所有策略使用相同的θ* | ❌ 硬编码0.9 | ✅ 动态传入 |
| 全部~12K测试集 | ❌ 只有50题 | ✅ 支持切换 |
| Graph 1.A: Cost vs. Accuracy | ⚠️ 命名不符 | ✅ 符合 |
| Graph 1.B: Cost vs. Retrieval Calls | ⚠️ 命名不符 | ✅ 符合 |
| ARGO应随c_r增加减少检索 | ✅ 逻辑正确 | ✅ 逻辑正确 |
| 基线策略应保持平坦 | ✅ 逻辑正确 | ✅ 逻辑正确 |

---

## 🚀 使用建议

### 测试流程

1. **先运行小规模测试** (验证代码逻辑)
   ```bash
   bash test_exp1.sh
   ```
   - 50题, 5个c_r点
   - 预计10-30分钟
   - 检查输出图表趋势是否符合预期

2. **如果测试通过，运行完整实验**
   ```bash
   bash run_exp1_full.sh
   ```
   - 12K题, 10个c_r点
   - 预计8-24小时
   - 获得正式的实验结果

### 预期图表趋势验证

#### Graph 1.A (Accuracy vs. c_r)
- ✅ ARGO: 高且稳定 (~80-85%)
- ✅ Always-Retrieve: 平坦 (~85%)
- ✅ Always-Reason: 平坦但低 (~60-70%)
- ✅ Random: 平坦 (~70-75%)

#### Graph 1.B (Retrievals vs. c_r)
- ✅ **ARGO: 下降趋势** (这是关键！证明自适应性)
- ✅ Always-Retrieve: 平坦且高 (~15-20次)
- ✅ Random: 平坦 (~7-10次)

---

## 📁 文件清单

修正版本新增文件:
```
ARGO2/ARGO/
├── Exp_real_cost_impact.py       # 原始版本 (保留)
├── Exp_real_cost_impact_v2.py    # ✅ 修正版本 (新增)
├── test_exp1.sh                  # ✅ 测试脚本 (新增)
├── run_exp1_full.sh              # ✅ 完整实验脚本 (新增)
├── README_Exp1_v2.md             # ✅ 使用文档 (新增)
└── COMPARISON_v1_v2.md           # ✅ 对比文档 (本文件)
```

---

## 💡 关键改进总结

1. **完整性**: 添加Random策略，符合实验设计文档的4个策略要求
2. **公平性**: 所有策略使用相同的动态θ*，对比更公平
3. **灵活性**: 支持小规模测试和大规模实验切换
4. **规范性**: 图表命名符合实验设计文档
5. **易用性**: 命令行参数和shell脚本，更方便使用

---

## 📌 快速启动

**立即测试:**
```bash
cd /data/user/huangxiaolin/ARGO2/ARGO
bash test_exp1.sh
```

**查看详细文档:**
```bash
cat README_Exp1_v2.md
```

**运行完整实验:**
```bash
bash run_exp1_full.sh  # 确认测试通过后再运行
```

---

生成时间: 2025-10-29
版本: v2.0
状态: ✅ Ready for Testing
