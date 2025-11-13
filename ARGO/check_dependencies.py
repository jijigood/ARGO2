#!/usr/bin/env python
"""
实验1前置检查脚本
检查所有依赖是否就绪
"""

import os
import sys
from pathlib import Path

def print_section(title):
    """打印分节标题"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def check_file_exists(filepath, description):
    """检查文件是否存在"""
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    print(f"{status} {description}")
    if exists:
        size = os.path.getsize(filepath)
        print(f"   路径: {filepath}")
        if size > 1024*1024:
            print(f"   大小: {size / (1024*1024):.1f} MB")
        elif size > 1024:
            print(f"   大小: {size / 1024:.1f} KB")
        else:
            print(f"   大小: {size} bytes")
    else:
        print(f"   路径: {filepath} (不存在)")
    return exists

def check_import(module_name, package_name=None):
    """检查Python模块是否可导入"""
    try:
        __import__(module_name)
        print(f"✅ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"❌ {package_name or module_name}")
        print(f"   错误: {e}")
        return False

def check_gpu():
    """检查GPU"""
    try:
        import torch
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            print(f"✅ CUDA可用 ({n_gpus}张GPU)")
            for i in range(n_gpus):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"   GPU {i}: {name} ({mem:.1f} GB)")
            return True
        else:
            print(f"❌ CUDA不可用")
            return False
    except ImportError:
        print(f"❌ PyTorch未安装")
        return False

def main():
    print_section("实验1前置检查")
    
    base_dir = "/data/user/huangxiaolin/ARGO2/ARGO"
    
    # 1. 检查脚本文件
    print_section("1. 检查脚本文件")
    files_ok = True
    files_ok &= check_file_exists(
        f"{base_dir}/Exp_real_cost_impact_v2.py",
        "实验脚本 (v2)"
    )
    files_ok &= check_file_exists(
        f"{base_dir}/oran_benchmark_loader.py",
        "数据集加载器"
    )
    files_ok &= check_file_exists(
        f"{base_dir}/../ARGO_MDP/src/mdp_solver.py",
        "MDP求解器"
    )
    files_ok &= check_file_exists(
        f"{base_dir}/configs/multi_gpu.yaml",
        "配置文件"
    )
    
    # 2. 检查模型文件
    print_section("2. 检查模型文件")
    models_ok = True
    models_ok &= check_file_exists(
        "/data/user/huangxiaolin/ARGO/RAG_Models/models/Qwen2.5-14B-Instruct",
        "LLM模型目录"
    )
    models_ok &= check_file_exists(
        "/data/user/huangxiaolin/ARGO/models/all-MiniLM-L6-v2",
        "嵌入模型目录"
    )
    
    # 3. 检查数据文件
    print_section("3. 检查数据文件")
    data_ok = True
    benchmark_dir = f"{base_dir}/ORAN-Bench-13K"
    data_ok &= check_file_exists(
        benchmark_dir,
        "ORAN-Bench-13K数据集目录"
    )
    
    # 4. 检查Python依赖
    print_section("4. 检查Python依赖")
    deps_ok = True
    deps_ok &= check_import("torch", "PyTorch")
    deps_ok &= check_import("transformers", "Transformers")
    deps_ok &= check_import("sentence_transformers", "SentenceTransformers")
    deps_ok &= check_import("numpy", "NumPy")
    deps_ok &= check_import("yaml", "PyYAML")
    deps_ok &= check_import("matplotlib", "Matplotlib")
    
    try:
        import chromadb
        print(f"✅ ChromaDB")
        chroma_ok = True
    except ImportError:
        print(f"⚠️  ChromaDB (可选，会使用模拟检索模式)")
        chroma_ok = False
    
    # 5. 检查GPU
    print_section("5. 检查GPU")
    gpu_ok = check_gpu()
    
    # 6. 检查Chroma数据库
    print_section("6. 检查Chroma数据库 (可选)")
    chroma_db_ok = check_file_exists(
        f"{base_dir}/Environments/chroma_store",
        "Chroma数据库目录"
    )
    if not chroma_db_ok:
        print("   ⚠️  Chroma数据库不存在，将使用模拟检索模式")
    
    # 总结
    print_section("检查总结")
    
    all_critical_ok = files_ok and models_ok and data_ok and deps_ok and gpu_ok
    
    if all_critical_ok:
        print("✅ 所有关键依赖都已就绪!")
        print("\n可以开始运行实验:")
        print("  bash test_exp1.sh          # 小规模测试 (推荐先运行)")
        print("  bash run_exp1_full.sh      # 完整实验")
        return 0
    else:
        print("❌ 存在缺失的依赖，请先解决以下问题:")
        if not files_ok:
            print("  - 脚本文件缺失")
        if not models_ok:
            print("  - 模型文件缺失")
        if not data_ok:
            print("  - 数据文件缺失")
        if not deps_ok:
            print("  - Python依赖缺失")
        if not gpu_ok:
            print("  - GPU不可用")
        return 1
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    sys.exit(main())
