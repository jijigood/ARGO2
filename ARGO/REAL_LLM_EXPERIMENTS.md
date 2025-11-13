# 真实LLM实验指南

## 📋 概述

这是升级版的ARGO实验，使用**真实的Qwen模型**和**嵌入模型**，支持**多GPU并行**。

### 主要改进

| 特性 | 仿真版 | 真实LLM版 |
|-----|--------|----------|
| **LLM模型** | ❌ 无 (数学仿真) | ✅ Qwen2.5-14B-Instruct |
| **嵌入模型** | ❌ 无 | ✅ all-MiniLM-L6-v2 |
| **检索系统** | ❌ 无 | ✅ Chroma (ORAN规范库) |
| **问题难度** | Medium | **Hard** |
| **GPU支持** | 不需要 | ✅ 多GPU并行 |
| **运行时间** | 2分钟 | 2-3小时 |
| **答案准确性** | 模拟 | ✅ 真实LLM推理 |

---

## 🖥️ 硬件要求

### GPU配置
- **推荐**: 4-8张 RTX 3060 (12GB each) 或更好
- **最小**: 2张 GPU (总共20GB+ VRAM)
- **CUDA**: 12.x 或 11.x

### 内存要求
- **GPU内存**: 40GB+ (推荐 48GB+)
- **系统内存**: 32GB+
- **磁盘空间**: 50GB (模型文件)

---

## 📦 模型准备

### 1. LLM模型 (Qwen2.5-14B-Instruct)

**位置**: `/data/user/huangxiaolin/ARGO/RAG_Models/models/Qwen2.5-14B-Instruct`

如果还没下载:
```bash
# 从HuggingFace下载
huggingface-cli download Qwen/Qwen2.5-14B-Instruct \
    --local-dir /data/user/huangxiaolin/ARGO/RAG_Models/models/Qwen2.5-14B-Instruct

# 或使用git-lfs
cd /data/user/huangxiaolin/ARGO/RAG_Models/models
git clone https://huggingface.co/Qwen/Qwen2.5-14B-Instruct
```

### 2. 嵌入模型 (all-MiniLM-L6-v2)

**位置**: `/data/user/huangxiaolin/ARGO/models/all-MiniLM-L6-v2`

如果还没下载:
```bash
# 从HuggingFace下载
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
    --local-dir /data/user/huangxiaolin/ARGO/models/all-MiniLM-L6-v2
```

### 3. 检索库 (Chroma)

**位置**: `/data/user/huangxiaolin/ARGO2/ARGO/Environments/chroma_store`

如果还没创建,运行:
```bash
cd /data/user/huangxiaolin/ARGO2/ARGO
python run_chroma_pipeline.py
```

---

## 🚀 快速开始

### 方法1: 交互式脚本 (推荐)

```bash
cd /data/user/huangxiaolin/ARGO2/ARGO
chmod +x run_real_experiments.sh
./run_real_experiments.sh
```

会提示选择:
1. 实验1: 检索成本影响
2. 实验2: 检索成功率影响
3. 运行全部实验

### 方法2: 直接运行Python

**实验1 (检索成本影响):**
```bash
python Exp_real_cost_impact.py
```

**实验2 (检索成功率影响):**
```bash
python Exp_real_success_impact.py
```

---

## 📊 实验详情

### 实验1: 检索成本影响

**目标**: 验证ARGO在高成本时避免检索

**参数设置**:
- 问题难度: **Hard** (3,243题池)
- 问题数量: 50题
- c_r扫描: 0.02 → 0.20 (5个点)
- p_s固定: 0.8
- GPU: 4张

**预计时间**: 2-3小时

**输出**:
- `draw_figs/data/exp1_real_cost_impact_*.json`
- `figs/exp1_real_cost_vs_quality.png`
- `figs/exp1_real_cost_vs_retrievals.png`
- `figs/exp1_real_cost_vs_accuracy.png`

### 实验2: 检索成功率影响

**目标**: 验证ARGO在低成功率时避免检索

**参数设置**:
- 问题难度: **Hard**
- 问题数量: 50题
- p_s扫描: 0.3 → 1.0 (4个点)
- c_r固定: 0.05
- GPU: 4张

**预计时间**: 2-3小时

**输出**:
- `draw_figs/data/exp2_real_success_impact_*.json`
- `figs/exp2_real_ps_vs_quality.png`
- `figs/exp2_real_ps_vs_retrievals.png`
- `figs/exp2_real_ps_vs_accuracy.png`

---

## ⚙️ 自定义配置

### 修改问题数量

编辑 `Exp_real_cost_impact.py`:
```python
experiment = RealCostImpactExperiment(
    n_test_questions=100,  # 改为100题 (更长时间)
    ...
)
```

### 修改问题难度

```python
experiment = RealCostImpactExperiment(
    difficulty="medium",  # 改为Medium
    ...
)
```

### 使用不同GPU

```python
experiment = RealCostImpactExperiment(
    gpu_ids=[0, 1],  # 只用前2张GPU
    ...
)
```

### 更换LLM模型

如果要用7B模型 (更快):
```python
experiment = RealCostImpactExperiment(
    llm_model_path="/data/user/huangxiaolin/ARGO/RAG_Models/models/qwen2.5-7b-instruct",
    gpu_ids=[0, 1],  # 7B模型2张GPU足够
    ...
)
```

### 调整参数扫描

**实验1 (c_r扫描)**:
```python
results = experiment.run_experiment(
    c_r_min_multiplier=0.5,   # c_r最小 = 0.5 × c_p
    c_r_max_multiplier=20.0,  # c_r最大 = 20 × c_p
    n_steps=10                # 扫描10个点 (更细)
)
```

**实验2 (p_s扫描)**:
```python
results = experiment.run_experiment(
    p_s_min=0.2,   # 从20%开始
    p_s_max=1.0,   # 到100%
    n_steps=8      # 8个点
)
```

---

## 📈 查看结果

### 快速查看

```bash
python view_results.py
```

### 手动查看JSON

```bash
cd draw_figs/data
ls -lh exp*_real_*.json
cat exp1_real_cost_impact_*.json | jq
```

### 查看图表

```bash
cd figs
ls -lh exp*_real_*.png
```

在VS Code中打开PNG文件查看。

---

## 🔍 监控GPU使用

### 实时监控

另开一个终端:
```bash
watch -n 1 nvidia-smi
```

### 查看详细信息

```bash
nvidia-smi dmon -i 0,1,2,3
```

---

## ⚠️ 故障排除

### 1. GPU内存不足 (OOM)

**方案A**: 减少问题数量
```python
n_test_questions=20  # 从50减到20
```

**方案B**: 使用更小的模型
```python
llm_model_path="/path/to/qwen2.5-7b-instruct"
```

**方案C**: 增加CPU卸载
编辑代码中的 `max_memory`:
```python
max_memory = {i: "8GB" for i in self.gpu_ids}  # 减少到8GB
max_memory["cpu"] = "50GB"  # 增加CPU内存
```

### 2. Chroma集合不存在

运行:
```bash
python run_chroma_pipeline.py
```

如果失败,实验会自动切换到**模拟检索模式** (仍能运行,但检索是假的)。

### 3. 模型加载失败

检查路径:
```bash
ls -lh /data/user/huangxiaolin/ARGO/RAG_Models/models/Qwen2.5-14B-Instruct
```

确保包含:
- `config.json`
- `model*.safetensors`
- `tokenizer*`

### 4. CUDA版本不匹配

检查CUDA:
```bash
nvcc --version
python -c "import torch; print(torch.version.cuda)"
```

如果不匹配,重新安装PyTorch:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 5. 实验太慢

**选项1**: 减少问题数量
```python
n_test_questions=20
```

**选项2**: 减少扫描点数
```python
n_steps=3  # 从5减到3
```

**选项3**: 使用7B模型
```python
llm_model_path="/path/to/qwen2.5-7b-instruct"
```

---

## 📊 预期结果

### 实验1: 检索成本影响

**假设**:
- ✅ c_r增加 → ARGO检索次数减少
- ✅ Always-Retrieve保持不变
- ✅ ARGO准确率保持较高

### 实验2: 检索成功率影响

**假设**:
- ✅ p_s降低 → ARGO避免检索
- ✅ Always-Retrieve在低p_s时大量重试
- ✅ ARGO切换到Reason策略

---

## 🆚 对比仿真版

| 维度 | 仿真版 | 真实LLM版 |
|-----|--------|----------|
| 运行时间 | 2分钟 | 2-3小时 |
| GPU需求 | 无 | 4张GPU |
| 问题数量 | 100题 | 50题 |
| 问题难度 | Medium | **Hard** |
| 答案质量 | 模拟 | 真实推理 |
| 检索质量 | 模拟 | 真实检索 |
| 可重现性 | 完美 | 较高 (随机性) |
| 成本 | 免费 | GPU时间 |

---

## 📝 引用

如果使用这些实验,请引用:

```bibtex
@article{argo2025,
  title={ARGO: Adaptive Retrieval-Augmented Generation with MDP},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

---

## 🔗 相关文件

- **实验脚本**:
  - `Exp_real_cost_impact.py` (实验1)
  - `Exp_real_success_impact.py` (实验2)
  - `run_real_experiments.sh` (启动脚本)

- **仿真版** (对比):
  - `Exp_retrieval_cost_impact.py`
  - `Exp_retrieval_success_impact.py`

- **配置**:
  - `configs/multi_gpu.yaml` (MDP参数)

- **文档**:
  - `EXPERIMENT_ANALYSIS.md` (仿真版分析)
  - `EXPERIMENTS_INDEX.md` (实验索引)

---

## 💡 提示

1. **第一次运行**: 先用20题测试,确保一切正常
2. **监控GPU**: 用 `nvidia-smi` 监控显存
3. **保存结果**: 结果自动保存,不会丢失
4. **对比仿真**: 可以和仿真版对比验证MDP理论
5. **调整参数**: 根据硬件调整问题数量和GPU数量

---

**创建时间**: 2025-10-29  
**作者**: GitHub Copilot  
**版本**: 1.0
