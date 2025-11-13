# 实验脚本详细分析

**分析时间**: 2025-10-29  
**分析对象**: `Exp_retrieval_cost_impact.py` 和 `Exp_retrieval_success_impact.py`

---

## 📊 实验概览对比

| 维度 | 实验1 (成本影响) | 实验2 (成功率影响) |
|-----|----------------|------------------|
| **脚本文件** | `Exp_retrieval_cost_impact.py` | `Exp_retrieval_success_impact.py` |
| **代码行数** | 632行 | 648行 |
| **文件大小** | 22KB | 21KB |
| **核心类** | `CostImpactExperiment` | `RetrievalSuccessExperiment` |

---

## 🔬 实验1: 检索成本影响 - 详细分析

### 基本配置

```python
# 文件: Exp_retrieval_cost_impact.py
class CostImpactExperiment:
    def __init__(
        self,
        config_path: str = "configs/multi_gpu.yaml",
        n_test_questions: int = 100,        # 测试问题数量
        difficulty: str = "medium",         # 问题难度
        seed: int = 42                      # 随机种子
    )
```

#### 数据集配置
- **数据源**: ORAN-Bench-13K
- **数据路径**: `ORAN-Bench-13K/Benchmark/`
- **问题数量**: 100道 (从9570道Medium问题中抽样)
- **难度级别**: Medium (中等难度)
- **随机种子**: 42 (保证可重现)
- **数据格式**: JSONL (每行一个JSON数组)

#### 测试问题分布
```
总问题池:
  - Easy: 1,139题
  - Medium: 9,570题  ← 从这里抽样
  - Hard: 3,243题
  - Total: 13,952题

实际使用: 100题 (Medium, seed=42)
```

### 模型配置

**注意**: 这个实验使用**仿真模型**,不是真实LLM!

#### 仿真模型组件

1. **质量函数 (Quality Function)**
```python
def simulate_quality_function(self, U: float) -> float:
    """
    模拟质量函数 σ(U)
    
    模式: Linear (线性)
    公式: σ(U) = U / U_max
    """
    mode = "linear"  # 从config读取
    U_max = 1.0
    return U / U_max  # 简单线性映射
```

2. **检索模拟**
```python
def simulate_argo_policy(self, question, theta_cont, theta_star):
    """
    模拟ARGO策略执行
    
    参数:
      - delta_r = 0.25  (检索成功时U增量)
      - p_s = 0.8       (检索成功概率)
      - max_steps = 20  (最大步数)
    """
    if U < theta_cont:
        # Retrieve action
        if random() < 0.8:  # p_s = 0.8
            U += 0.25       # delta_r
    else:
        # Reason action
        U += 0.08           # delta_p
```

3. **基线策略模拟**
```python
# Always-Retrieve: 固定检索
def simulate_always_retrieve_policy(self, question):
    while U < 0.9:  # 固定theta_star
        retrieval_count += 1
        if random() < p_s:
            U += delta_r

# Always-Reason: 固定推理
def simulate_always_reason_policy(self, question):
    while U < 0.9:
        reason_count += 1
        U += delta_p

# Random: 随机50-50
def simulate_random_policy(self, question):
    while U < 0.9:
        if random() < 0.5:
            retrieval_count += 1
            if random() < p_s:
                U += delta_r
        else:
            reason_count += 1
            U += delta_p
```

### MDP求解器配置

```python
# 从 configs/multi_gpu.yaml 加载
config = {
    'mdp': {
        'U_max': 1.0,              # 信息进度上限
        'delta_r': 0.25,           # 检索增量 (固定)
        'delta_p': 0.08,           # 推理增量 (固定)
        'p_s': 0.8,                # 检索成功率 (固定)
        'c_r': [0.02 ~ 0.20],      # 检索成本 (变量!扫描10个值)
        'c_p': 0.02,               # 推理成本 (固定)
        'mu': 0.6,                 # 质量权重
        'gamma': 0.98,             # 折扣因子
        'U_grid_size': 101         # 状态空间离散化粒度
    },
    'quality': {
        'mode': 'linear',          # 质量函数类型
        'k': 5.0                   # 参数k (linear模式下未使用)
    },
    'solver': {
        'max_iterations': 1000,    # Value Iteration最大迭代次数
        'convergence_threshold': 1e-6,  # 收敛阈值
        'verbose': False           # 不打印详细日志
    }
}
```

### 实验参数扫描

```python
def run_experiment(
    self,
    c_r_min_multiplier: float = 1.0,   # c_r最小值 = 1.0 * c_p
    c_r_max_multiplier: float = 10.0,  # c_r最大值 = 10.0 * c_p
    n_steps: int = 10                  # 扫描步数
):
    """
    扫描c_r从c_p到10*c_p, 10个均匀分布的点
    """
    c_r_values = np.linspace(
        1.0 * 0.02,   # 0.020
        10.0 * 0.02,  # 0.200
        10
    )
    # 结果: [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
```

**实际测试的c_r值:**
```
c_r = 0.020 (1.0x c_p)
c_r = 0.040 (2.0x c_p)
c_r = 0.060 (3.0x c_p)
c_r = 0.080 (4.0x c_p)
c_r = 0.100 (5.0x c_p)
c_r = 0.120 (6.0x c_p)
c_r = 0.140 (7.0x c_p)
c_r = 0.160 (8.0x c_p)
c_r = 0.180 (9.0x c_p)
c_r = 0.200 (10.0x c_p)
```

### 评估策略

对每个c_r值,评估4种策略:

```python
policies = {
    'ARGO': lambda q: self.simulate_argo_policy(q, theta_cont, theta_star),
    'Always-Retrieve': self.simulate_always_retrieve_policy,
    'Always-Reason': self.simulate_always_reason_policy,
    'Random': self.simulate_random_policy
}

# 每个策略在100道问题上运行
for question in self.test_questions:  # 100题
    result = policy_fn(question)
    # 记录: quality, retrieval_count, reason_count, steps
```

### 计算复杂度

**总运行次数**:
```
10 (c_r值) × 4 (策略) × 100 (问题) = 4,000 次策略执行
10 (c_r值) × 1 (MDP求解) = 10 次 Value Iteration
```

**单次Value Iteration复杂度**:
```
状态数: 101 (U_grid_size)
动作数: 3 (Retrieve, Reason, Terminate)
迭代次数: ~100-200次 (通常快速收敛)

复杂度: O(101 × 3 × 200) ≈ 60,600 次状态更新
```

**实际运行时间**: ~2分钟 (无GPU,纯CPU仿真)

---

## 🔬 实验2: 检索成功率影响 - 详细分析

### 基本配置

```python
# 文件: Exp_retrieval_success_impact.py
class RetrievalSuccessExperiment:
    def __init__(
        self,
        config_path: str = "configs/multi_gpu.yaml",
        n_test_questions: int = 100,        # 测试问题数量
        difficulty: str = "medium",         # 问题难度
        seed: int = 42                      # 随机种子
    )
```

#### 数据集配置
- **数据源**: ORAN-Bench-13K (与实验1相同)
- **问题数量**: 100道 (相同抽样,seed=42)
- **难度级别**: Medium
- **使用相同的100道题**: 保证实验可比性

### 模型配置

同样使用**仿真模型**,但有关键差异:

#### 关键参数变化

```python
# 实验1固定p_s, 变化c_r
p_s = 0.8         (固定)
c_r = [0.02~0.20] (变量)

# 实验2固定c_r, 变化p_s  
p_s = [0.3~1.0]   (变量)
c_r = 0.05        (固定)
```

#### 仿真模型调整

```python
def simulate_argo_policy(self, question, theta_cont, theta_star, p_s):
    """
    与实验1的区别: p_s是参数!
    """
    max_steps = 30  # 增加到30 (因为低p_s可能需要更多步)
    
    while U < theta_star and step < max_steps:
        if U < theta_cont:
            retrieval_count += 1
            if random() < p_s:  # 使用变化的p_s!
                U += delta_r
        else:
            reason_count += 1
            U += delta_p
```

**为什么max_steps=30?**
- 低p_s时(如0.3),检索成功率低
- Always-Retrieve可能需要很多次重试
- 避免无限循环

### MDP求解器配置

```python
config = {
    'mdp': {
        'U_max': 1.0,
        'delta_r': 0.25,           # 固定
        'delta_p': 0.08,           # 固定
        'p_s': [0.3 ~ 1.0],        # 变量!扫描8个值
        'c_r': 0.05,               # 固定
        'c_p': 0.02,               # 固定
        'mu': 0.6,
        'gamma': 0.98,
        'U_grid_size': 101
    },
    'quality': {
        'mode': 'linear',
        'k': 5.0
    },
    'solver': {
        'max_iterations': 1000,
        'convergence_threshold': 1e-6,
        'verbose': False
    }
}
```

### 实验参数扫描

```python
def run_experiment(
    self,
    p_s_min: float = 0.3,    # 最小成功率30%
    p_s_max: float = 1.0,    # 最大成功率100%
    n_steps: int = 8         # 扫描8个点
):
    """
    扫描p_s从0.3到1.0, 8个均匀分布的点
    """
    p_s_values = np.linspace(0.3, 1.0, 8)
    # 结果: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
```

**实际测试的p_s值:**
```
p_s = 0.30 (30%成功率)
p_s = 0.40 (40%成功率)
p_s = 0.50 (50%成功率)
p_s = 0.60 (60%成功率)
p_s = 0.70 (70%成功率)
p_s = 0.80 (80%成功率)
p_s = 0.90 (90%成功率)
p_s = 1.00 (100%成功率)
```

### 计算复杂度

**总运行次数**:
```
8 (p_s值) × 4 (策略) × 100 (问题) = 3,200 次策略执行
8 (p_s值) × 1 (MDP求解) = 8 次 Value Iteration
```

**实际运行时间**: ~2分钟

---

## 📊 两个实验的对比总结

### 数据层面

| 维度 | 实验1 | 实验2 | 说明 |
|-----|-------|-------|------|
| 数据集 | ORAN-Bench-13K | ORAN-Bench-13K | 相同 |
| 问题数量 | 100题 | 100题 | 相同 |
| 难度 | Medium | Medium | 相同 |
| 随机种子 | 42 | 42 | **相同100题!** |
| 问题格式 | JSONL | JSONL | 相同 |

### 参数层面

| 参数 | 实验1 | 实验2 |
|-----|-------|-------|
| **自变量** | c_r (检索成本) | p_s (检索成功率) |
| 扫描范围 | 0.02 ~ 0.20 | 0.3 ~ 1.0 |
| 扫描点数 | 10个 | 8个 |
| delta_r | 0.25 (固定) | 0.25 (固定) |
| delta_p | 0.08 (固定) | 0.08 (固定) |
| p_s | 0.8 (固定) | 变量 |
| c_r | 变量 | 0.05 (固定) |
| c_p | 0.02 (固定) | 0.02 (固定) |
| gamma | 0.98 (固定) | 0.98 (固定) |

### 模型层面

| 组件 | 实验1 | 实验2 | 说明 |
|-----|-------|-------|------|
| **LLM模型** | ❌ 无 | ❌ 无 | 使用仿真 |
| **嵌入模型** | ❌ 无 | ❌ 无 | 使用仿真 |
| **检索器** | ❌ 无 | ❌ 无 | 使用仿真 |
| 质量函数 | Linear | Linear | 相同 |
| MDP求解器 | Value Iteration | Value Iteration | 相同 |
| 状态空间 | 101维 | 101维 | 相同 |
| 动作空间 | 3个 | 3个 | 相同 |

**重要**: 这两个实验都是**纯仿真实验**,不需要加载任何LLM或嵌入模型!

### 计算资源

| 资源 | 实验1 | 实验2 |
|-----|-------|-------|
| **GPU需求** | ❌ 不需要 | ❌ 不需要 |
| **CPU** | ✅ 单核足够 | ✅ 单核足够 |
| **内存** | ~500MB | ~500MB |
| **运行时间** | ~2分钟 | ~2分钟 |
| **磁盘** | ~3KB (JSON) | ~3KB (JSON) |

### 输出层面

| 输出 | 实验1 | 实验2 |
|-----|-------|-------|
| 图表数量 | 3张 | 3张 |
| 数据文件 | 1个JSON | 1个JSON |
| 报告文档 | 1个MD | 1个MD |
| 核心图 | cost_vs_retrievals | ps_vs_retrievals |

---

## 🎯 关键设计决策

### 为什么使用仿真而非真实LLM?

#### 优点:
1. **速度快**: 2分钟 vs 数小时(真实LLM)
2. **可控**: 参数确定,结果可重现
3. **成本低**: 无需GPU,无API费用
4. **专注MDP**: 验证MDP求解器,而非LLM性能

#### 缺点:
1. **真实性**: 无法反映真实RAG性能
2. **质量简化**: Linear函数过于简单
3. **适用性**: 需要后续真实LLM验证

### 为什么选择100题?

1. **平衡**: 足够统计意义,不会太慢
2. **可重现**: seed=42固定抽样
3. **可扩展**: 可轻松改为1000题

### 为什么固定其他参数?

**单变量控制法**:
- 实验1: 只变c_r,固定p_s
- 实验2: 只变p_s,固定c_r
- 目的: 清晰展示单一参数的影响

---

## 💡 如何改为真实LLM实验?

如果要使用真实模型,需要修改:

### 1. 加载LLM和嵌入模型

```python
# 在__init__中添加
from transformers import AutoModelForCausalLM, AutoTokenizer

self.model_name = "Qwen/Qwen2.5-7B-Instruct"
self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
self.model = AutoModelForCausalLM.from_pretrained(
    self.model_name,
    device_map="auto",
    torch_dtype=torch.float16
)
```

### 2. 集成真实检索

```python
from chromadb import Client
from sentence_transformers import SentenceTransformer

self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
self.chroma_client = Client()
self.collection = self.chroma_client.get_collection("oran_specs")
```

### 3. 替换仿真函数

```python
def real_argo_policy(self, question, theta_cont, theta_star):
    """使用真实RAG系统"""
    U = 0.0
    
    while U < theta_star:
        if U < theta_cont:
            # 真实检索
            docs = self.collection.query(
                query_texts=[question],
                n_results=5
            )
            context = docs['documents']
            
            # 真实LLM推理
            prompt = f"Context: {context}\n\nQuestion: {question}"
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_new_tokens=100)
            answer = self.tokenizer.decode(outputs[0])
            
            # 评估质量(需要ground truth或评判模型)
            quality = evaluate_answer(answer, question['correct_answer'])
            U += quality
        else:
            # 真实推理
            # ...
```

### 4. 调整参数

```python
# 真实实验需要更少问题(因为很慢)
n_test_questions = 20  # 而非100
n_steps = 5            # 而非10

# 需要GPU
device = "cuda:0"
```

### 5. 预计资源

**真实LLM实验**:
- GPU: 至少1张A100 (40GB)
- 时间: 每题~30秒,总计~10分钟(20题)
- 内存: ~20GB GPU VRAM
- 成本: 如果用API,~$1-5

---

## 📈 实验数据流

### 实验1数据流

```
输入:
  ├─ ORAN-Bench-13K/Benchmark/fin_M.json (9570题)
  ├─ configs/multi_gpu.yaml (MDP参数)
  └─ seed=42
         ↓
  [抽样100题] (ORANBenchmark.sample_questions)
         ↓
  [循环10次,c_r从0.02到0.20]
    ├─ [MDP求解] (MDPSolver.solve)
    │    └─ 输出: θ_cont, θ*
    │
    ├─ [评估ARGO策略] (100题 × simulate_argo_policy)
    ├─ [评估Always-Retrieve] (100题)
    ├─ [评估Always-Reason] (100题)
    └─ [评估Random] (100题)
         ↓
  [聚合结果]
    ├─ 平均质量: 1.000
    ├─ 平均检索次数: 5.1 → 0.0
    └─ 平均推理次数: ...
         ↓
  [保存]
    ├─ JSON: draw_figs/data/exp1_*.json (3.2KB)
    └─ PNG: figs/exp1_*.png (3张,482KB)
```

### 实验2数据流

```
输入:
  ├─ 相同的100题 (seed=42)
  ├─ configs/multi_gpu.yaml
  └─ p_s范围: 0.3~1.0
         ↓
  [循环8次,p_s从0.3到1.0]
    ├─ [MDP求解] (p_s变化)
    │    └─ 输出: θ_cont, θ*
    │
    ├─ [评估ARGO] (p_s作为参数)
    ├─ [评估Always-Retrieve] (p_s影响结果)
    ├─ [评估Always-Reason] (p_s无影响)
    └─ [评估Random] (p_s影响结果)
         ↓
  [聚合结果]
    ├─ p_s=0.3: ARGO 0次检索, Always-R 12.7次
    ├─ p_s=1.0: ARGO 1次检索, Always-R 4.0次
    └─ ...
         ↓
  [保存]
    ├─ JSON: draw_figs/data/exp2_*.json (3.4KB)
    └─ PNG: figs/exp2_*.png (3张,707KB)
```

---

## 🔍 代码质量分析

### 代码结构

```python
# 两个脚本都采用相同的类结构
class Experiment:
    __init__()              # 初始化,加载数据
    create_mdp_config()     # 创建MDP配置
    solve_mdp()             # 求解MDP
    simulate_quality_function()  # 质量函数
    simulate_argo_policy()       # ARGO策略仿真
    simulate_always_retrieve()   # Always-Retrieve仿真
    simulate_always_reason()     # Always-Reason仿真
    simulate_random()            # Random仿真
    evaluate_all_policies()      # 评估所有策略
    run_experiment()             # 主实验循环
    save_results()               # 保存JSON
    plot_results()               # 绘图
```

### 代码复用

**共享逻辑** (~70%代码相同):
- 数据加载
- MDP配置生成
- 仿真函数结构
- 结果保存
- 绘图逻辑

**差异点** (~30%):
- 参数扫描 (c_r vs p_s)
- 仿真函数参数传递
- 图表标题和标签

### 改进建议

1. **提取基类**: 创建`BaseExperiment`,减少代码重复
2. **配置驱动**: 用YAML配置实验参数
3. **并行化**: 使用多进程加速策略评估
4. **日志**: 添加logging而非print

---

## 📝 总结

### 实验1核心要素
- **目标**: 验证成本自适应性
- **数据**: 100题Medium难度
- **模型**: 仿真(无LLM)
- **参数**: 扫描c_r (10个值)
- **输出**: 3张图,1个JSON
- **时间**: 2分钟
- **结论**: ARGO检索次数从5.1降至0

### 实验2核心要素
- **目标**: 验证不确定性管理
- **数据**: 相同100题
- **模型**: 仿真(无LLM)
- **参数**: 扫描p_s (8个值)
- **输出**: 3张图,1个JSON
- **时间**: 2分钟
- **结论**: 低p_s时ARGO避免检索

### 关键特点
✅ 快速: 2分钟完成  
✅ 可控: 仿真保证可重现  
✅ 轻量: 无需GPU  
✅ 专注: 验证MDP理论  
⚠️ 限制: 需要真实LLM验证  

---

**文档生成时间**: 2025-10-29 01:10  
**分析者**: GitHub Copilot
