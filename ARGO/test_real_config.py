#!/usr/bin/env python
"""
测试脚本 - 验证真实LLM实验配置
快速检查:
1. GPU可用性
2. 模型文件
3. 数据集
4. Chroma数据库
5. 依赖库
"""

import sys
import os

print("="*80)
print("真实LLM实验 - 配置检查")
print("="*80)
print()

# 1. 检查GPU
print("[1/6] 检查GPU...")
try:
    import torch
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"  ✓ 检测到 {n_gpus} 张GPU:")
        for i in range(n_gpus):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"    GPU {i}: {name} ({mem:.1f}GB)")
    else:
        print(f"  ❌ 未检测到GPU!")
        sys.exit(1)
except Exception as e:
    print(f"  ❌ 错误: {e}")
    sys.exit(1)

print()

# 2. 检查LLM模型
print("[2/6] 检查LLM模型...")
llm_paths = [
    "/data/user/huangxiaolin/ARGO/RAG_Models/models/Qwen2.5-14B-Instruct",
    "/data/user/huangxiaolin/ARGO/RAG_Models/models/qwen2.5-7b-instruct"
]

llm_found = None
for path in llm_paths:
    if os.path.exists(path):
        # 检查关键文件
        required_files = ['config.json', 'tokenizer.json']
        all_exist = all(os.path.exists(os.path.join(path, f)) for f in required_files)
        
        if all_exist:
            print(f"  ✓ 找到LLM模型: {path}")
            llm_found = path
            break
        else:
            print(f"  ⚠ 模型路径存在但文件不完整: {path}")

if not llm_found:
    print(f"  ❌ 未找到可用的LLM模型!")
    print(f"     请下载 Qwen2.5-14B-Instruct 或 Qwen2.5-7B-Instruct")
    sys.exit(1)

print()

# 3. 检查嵌入模型
print("[3/6] 检查嵌入模型...")
emb_path = "/data/user/huangxiaolin/ARGO/models/all-MiniLM-L6-v2"

if os.path.exists(emb_path):
    print(f"  ✓ 找到嵌入模型: {emb_path}")
else:
    print(f"  ❌ 未找到嵌入模型: {emb_path}")
    sys.exit(1)

print()

# 4. 检查数据集
print("[4/6] 检查ORAN-Bench-13K数据集...")
dataset_path = "ORAN-Bench-13K/Benchmark"

if os.path.exists(dataset_path):
    files = ['fin_E.json', 'fin_M.json', 'fin_H.json']
    found_files = [f for f in files if os.path.exists(os.path.join(dataset_path, f))]
    
    if len(found_files) == 3:
        print(f"  ✓ 找到数据集: {dataset_path}")
        
        # 统计问题数量
        import json
        for f in files:
            with open(os.path.join(dataset_path, f), 'r') as fp:
                lines = fp.readlines()
                print(f"    {f}: {len(lines)} 道题")
    else:
        print(f"  ❌ 数据集不完整，缺失: {set(files) - set(found_files)}")
        sys.exit(1)
else:
    print(f"  ❌ 未找到数据集: {dataset_path}")
    sys.exit(1)

print()

# 5. 检查Chroma数据库
print("[5/6] 检查Chroma数据库...")
chroma_path = "Environments/chroma_store"

if os.path.exists(chroma_path):
    print(f"  ✓ 找到Chroma数据库: {chroma_path}")
    
    try:
        import chromadb
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_collection("oran_specs")
        count = collection.count()
        print(f"    集合 'oran_specs': {count} 个文档")
    except Exception as e:
        print(f"  ⚠ Chroma数据库存在但无法连接: {e}")
        print(f"    实验将使用模拟检索模式")
else:
    print(f"  ⚠ 未找到Chroma数据库: {chroma_path}")
    print(f"    实验将使用模拟检索模式")

print()

# 6. 检查依赖库
print("[6/6] 检查Python依赖...")
required_packages = {
    'torch': 'PyTorch',
    'transformers': 'Transformers',
    'sentence_transformers': 'Sentence Transformers',
    'chromadb': 'ChromaDB',
    'numpy': 'NumPy',
    'matplotlib': 'Matplotlib',
    'yaml': 'PyYAML'
}

missing = []
for package, name in required_packages.items():
    try:
        __import__(package)
        print(f"  ✓ {name}")
    except ImportError:
        print(f"  ❌ {name}")
        missing.append(package)

if missing:
    print(f"\n  安装缺失的包:")
    print(f"    pip install {' '.join(missing)}")
    sys.exit(1)

print()

# 总结
print("="*80)
print("✅ 所有检查通过!")
print("="*80)
print()
print("推荐配置:")
print(f"  LLM模型: {llm_found}")
print(f"  嵌入模型: {emb_path}")
print(f"  使用GPU: 0,1,2,3 (前4张)")
print(f"  问题难度: Hard")
print(f"  问题数量: 50题 (首次测试建议20题)")
print()
print("运行实验:")
print("  方法1: ./run_real_experiments.sh")
print("  方法2: python Exp_real_cost_impact.py")
print()
print("快速测试 (20题):")
print("  编辑 Exp_real_cost_impact.py")
print("  修改: n_test_questions=20")
print()
