# 多GPU MDP-RAG 使用指南

## 硬件配置
- **GPU**: 8x NVIDIA GeForce RTX 3060 (12GB each)
- **CUDA**: 12.4
- **PyTorch**: 2.6.0+cu124

## 快速开始

### 1. 环境准备
```bash
cd /data/user/huangxiaolin/ARGO2/ARGO
source activate ARGO

# 检查GPU
nvidia-smi
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

### 2. 快速测试（单GPU，10题）
```bash
python mdp_rag_multi_gpu.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --n_questions 10 \
  --difficulty easy \
  --gpu_mode single \
  --gpu_ids 0 \
  --seed 42
```

**预计时间**: ~2-3分钟

### 3. 中等规模评估（4 GPU，100题）
```bash
python mdp_rag_multi_gpu.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --n_questions 100 \
  --difficulty medium \
  --gpu_mode data_parallel \
  --gpu_ids 0 1 2 3 \
  --seed 42
```

**预计时间**: ~10-15分钟

### 4. MDP vs Fixed 对比（4 GPU，100题）
```bash
python compare_mdp_vs_fixed_multigpu.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --n_questions 100 \
  --difficulty medium \
  --fixed_k 3 \
  --gpu_mode data_parallel \
  --gpu_ids 0 1 2 3 \
  --seed 42
```

**预计时间**: ~20-30分钟

### 5. 大模型评估（14B，自动分配）
```bash
python mdp_rag_multi_gpu.py \
  --model Qwen/Qwen2.5-14B-Instruct \
  --n_questions 50 \
  --difficulty medium \
  --gpu_mode accelerate \
  --seed 42
```

**预计时间**: ~15-20分钟

### 6. 一键运行所有测试
```bash
./run_multi_gpu.sh
```

## GPU模式说明

### 1. **single** - 单GPU模式
- 适用: 小模型 (1.5B, 3B)
- 特点: 简单直接
- 示例:
  ```bash
  --gpu_mode single --gpu_ids 0
  ```

### 2. **data_parallel** - 数据并行模式 ⭐ 推荐
- 适用: 中等模型 (7B)，批量评估
- 特点: 多个问题并行处理，提升吞吐量
- 示例:
  ```bash
  --gpu_mode data_parallel --gpu_ids 0 1 2 3
  ```

### 3. **accelerate** - Accelerate自动分配 ⭐ 推荐大模型
- 适用: 大模型 (14B, 32B)
- 特点: 自动将模型分配到多个GPU
- 示例:
  ```bash
  --gpu_mode accelerate
  ```

### 4. **auto** - 自动选择
- 根据模型大小自动选择最佳模式
- 推荐初学者使用

## 模型选择指南

| 模型 | 大小 | GPU需求 | 推荐模式 | 速度 | 准确率 |
|-----|------|---------|---------|------|--------|
| Qwen2.5-1.5B | ~3GB | 1 GPU | single | 快 | 中 |
| Qwen2.5-3B | ~6GB | 1 GPU | single | 中 | 中+ |
| Qwen2.5-7B | ~14GB | 2 GPU | data_parallel | 中 | 高 |
| Qwen2.5-14B | ~28GB | 3 GPU | accelerate | 慢 | 高+ |

## 性能优化建议

### 1. 7B模型优化（推荐配置）
```bash
# 使用4个GPU，数据并行
python mdp_rag_multi_gpu.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --gpu_mode data_parallel \
  --gpu_ids 0 1 2 3 \
  --n_questions 100
```

**优势**:
- 速度提升: 4x（理论）
- 吞吐量高
- GPU利用率高

### 2. 14B模型优化
```bash
# 使用Accelerate自动分配，限制每个GPU 10GB
python mdp_rag_multi_gpu.py \
  --model Qwen/Qwen2.5-14B-Instruct \
  --gpu_mode accelerate \
  --n_questions 100
```

**特点**:
- 自动将模型分层到3-4个GPU
- 显存使用均衡
- 推理速度适中

### 3. 批量评估优化
如果需要评估大量问题（1000+），建议:
```bash
# 使用全部8个GPU
python mdp_rag_multi_gpu.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --gpu_mode data_parallel \
  --n_questions 1000 \
  --difficulty mixed
```

## 常见问题

### Q1: CUDA Out of Memory?
**解决方案**:
1. 减少使用的GPU数量
2. 使用更小的模型
3. 切换到`accelerate`模式
4. 添加内存限制:
   ```python
   max_memory_per_gpu="8GB"
   ```

### Q2: GPU利用率不高?
**原因**: 可能是模型太小或batch_size=1
**解决方案**:
- 使用更大的模型
- 增加问题数量
- 使用`data_parallel`模式

### Q3: 多GPU速度没有提升?
**原因**: 
- 单GPU可能是瓶颈在LLM推理，而非数据加载
- DataParallel有通信开销

**解决方案**:
- 对于7B模型，使用2-4个GPU即可
- 对于14B+模型，使用`accelerate`模式

## 结果文件

所有结果保存在:
```
results/
├── multi_gpu/                      # 单策略评估结果
│   └── Qwen2.5-7B-Instruct_medium_100q.json
└── multi_gpu_comparison/           # MDP vs Fixed对比
    └── Qwen2.5-7B-Instruct_medium_100q_mdp_vs_fixed_k3.json
```

## 监控GPU使用

在另一个终端运行:
```bash
# 实时监控
watch -n 1 nvidia-smi

# 或者
nvidia-smi dmon -i 0,1,2,3
```

## 预期结果

### 7B模型 (100题, medium)
- **MDP准确率**: ~75-78%
- **Fixed准确率**: ~60-65%
- **提升**: +13-15%
- **评估时间**: ~10-15分钟 (4 GPU)

### 14B模型 (100题, medium)
- **MDP准确率**: ~82-85%
- **Fixed准确率**: ~70-73%
- **提升**: +12-15%
- **评估时间**: ~15-20分钟 (accelerate)

## 下一步

1. **快速验证**: 运行`./run_multi_gpu.sh`
2. **对比实验**: 运行MDP vs Fixed
3. **全量评估**: 增加到1000题
4. **大模型测试**: 尝试14B模型

## 技术支持

如遇问题，检查:
1. GPU内存: `nvidia-smi`
2. PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. 日志文件: 查看错误信息
