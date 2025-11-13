# ORAN QA提取工具

## 概述

从TeleQnA数据集中提取仅涉及O-RAN (Open Radio Access Network)知识的问答对。使用Qwen2.5-14B-Instruct模型和vLLM框架进行8卡GPU并行推理。

## 文件说明

- `extract_oran_qa.py` - 主要提取脚本
- `run_extraction.sh` - 运行脚本(bash)
- `TeleQnA.txt` - 输入数据集
- `TeleQnA_ORAN_only.json` - 输出的ORAN问题集
- `extraction_log.txt` - 提取日志(包含每个问题的判断理由)

## 运行方法

### 方法1: 使用shell脚本

```bash
cd /data/user/huangxiaolin/ARGO2/ARGO/TeleQnA
chmod +x run_extraction.sh
./run_extraction.sh
```

### 方法2: 直接运行Python脚本

```bash
cd /data/user/huangxiaolin/ARGO2/ARGO/TeleQnA
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python extract_oran_qa.py
```

## 配置说明

可以在`extract_oran_qa.py`中修改以下配置:

```python
# 模型配置
MODEL_PATH = "/data/user/huangxiaolin/ARGO/RAG_Models/models/Qwen2.5-14B-Instruct"
TENSOR_PARALLEL_SIZE = 8  # GPU数量

# 文件路径
INPUT_FILE = "/data/user/huangxiaolin/ARGO2/ARGO/TeleQnA/TeleQnA.txt"
OUTPUT_FILE = "/data/user/huangxiaolin/ARGO2/ARGO/TeleQnA/TeleQnA_ORAN_only.json"

# 推理参数
BATCH_SIZE = 32  # 批处理大小
MAX_MODEL_LEN = 4096  # 最大序列长度
```

## ORAN知识判定标准

### 包含的内容(YES):
- O-RAN Alliance规范和架构
- O-RAN组件: O-CU, O-DU, O-RU
- O-RAN接口: E2, A1, O1, O2, F1, fronthaul等
- RAN智能控制器: Near-RT RIC, Non-RT RIC
- xApps和rApps
- O-RAN特定的协议、流程和实现
- O-RAN网络切片、QoS和资源管理
- O-RAN特定的用例和部署场景

### 排除的内容(NO):
- 通用3GPP规范(除非明确涉及O-RAN实现)
- 通用电信概念(VPN、加密、MIMO等,非O-RAN特定)
- IEEE标准(802.11, 802.15.4等)
- 数学概念、通用无线理论
- 非O-RAN网络架构
- 云/边缘计算概念(除非明确与O-RAN相关)
- 通用术语或缩写

## Prompt设计

脚本使用精心设计的prompt模板,要求LLM:

1. **明确O-RAN的定义范围** - 详细列出O-RAN相关的技术和组件
2. **明确排除标准** - 清晰说明哪些内容不属于O-RAN
3. **提供上下文** - 包含问题、选项、答案、解释和分类
4. **简洁输出** - 只需返回YES/NO和一行理由

## 输出格式

### TeleQnA_ORAN_only.json

```json
{
  "question 0": {
    "question": "Which components are responsible for embedding intelligence in the O-RAN architecture?",
    "option 1": "O-CU and O-DU",
    "option 2": "Non-RT RIC and Near-RT RIC",
    "option 3": "O-RU and O-DU",
    "option 4": "O-CU and O-RU",
    "answer": "option 2: Non-RT RIC and Near-RT RIC",
    "explanation": "Both Non-RT RIC and Near-RT RIC are responsible for embedding intelligence in the O-RAN architecture.",
    "category": "Standards specifications"
  }
}
```

### extraction_log.txt

```
================================================================================
Question ID: question 0
Question: Which components are responsible for embedding intelligence in the O-RAN architecture?
Is ORAN: True
Reason: This question is about O-RAN architecture components (RIC).
LLM Response: YES - This question is about O-RAN architecture components (RIC).
================================================================================
```

## 性能预估

- **数据集规模**: ~106,324个问题
- **GPU配置**: 8x GPU (Tesla/A100等)
- **批处理大小**: 32
- **预计时间**: 
  - 每批次推理: ~2-5秒
  - 总批次数: ~3,323批
  - 总耗时: ~3-5小时

## 依赖安装

```bash
pip install vllm torch transformers tqdm
```

或使用requirements.txt:

```bash
pip install -r requirements.txt
```

## 注意事项

1. **GPU内存**: 确保每张GPU有足够内存(建议至少16GB)
2. **模型路径**: 确认模型路径正确且模型已下载
3. **输入文件**: 确认TeleQnA.txt文件存在且格式正确
4. **GPU可见性**: 脚本默认使用GPU 0-7,可通过CUDA_VISIBLE_DEVICES调整

## 故障排除

### 问题1: CUDA out of memory

**解决方案**:
- 减小`BATCH_SIZE`
- 减小`MAX_MODEL_LEN`
- 降低`gpu_memory_utilization`(默认0.9)

### 问题2: vLLM导入错误

**解决方案**:
```bash
pip install vllm --upgrade
```

### 问题3: 模型加载失败

**解决方案**:
- 检查模型路径是否正确
- 确认模型文件完整性
- 尝试添加`trust_remote_code=True`

## 后续处理建议

1. **质量检查**: 随机抽样检查提取结果的准确性
2. **二次过滤**: 如需更严格的标准,可调整prompt或添加后处理规则
3. **数据增强**: 可基于提取的ORAN问题进行数据增强
4. **统计分析**: 分析ORAN问题的分布、类别等

## 联系信息

如有问题或建议,请联系项目维护者。
