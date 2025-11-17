#!/usr/bin/env python
"""
实验1: 依赖检查脚本
确保所有必要的包都已安装
"""

import sys

def check_dependencies():
    """检查依赖包"""
    
    print("="*80)
    print("检查实验1所需依赖包...")
    print("="*80)
    print()
    
    missing = []
    installed = []
    
    # 必需的包
    required_packages = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('scipy', 'SciPy'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('sentence_transformers', 'Sentence-Transformers'),
    ]
    
    # 可选的包
    optional_packages = [
        ('chromadb', 'ChromaDB'),
    ]
    
    print("必需依赖:")
    print("-"*80)
    
    for module, name in required_packages:
        try:
            __import__(module)
            version = __import__(module).__version__ if hasattr(__import__(module), '__version__') else 'unknown'
            print(f"  ✓ {name:25s} - {version}")
            installed.append(name)
        except ImportError:
            print(f"  ✗ {name:25s} - 缺失")
            missing.append(name)
    
    print()
    print("可选依赖:")
    print("-"*80)
    
    for module, name in optional_packages:
        try:
            __import__(module)
            version = __import__(module).__version__ if hasattr(__import__(module), '__version__') else 'unknown'
            print(f"  ✓ {name:25s} - {version}")
        except ImportError:
            print(f"  ⚠ {name:25s} - 缺失 (将使用模拟模式)")
    
    print()
    print("="*80)
    
    if missing:
        print(f"❌ 缺少 {len(missing)} 个必需依赖包:")
        for pkg in missing:
            print(f"  - {pkg}")
        print()
        print("请安装缺失的包:")
        print("  pip install " + " ".join([p.lower().replace('-', '_') for p in missing]))
        print()
        return False
    else:
        print(f"✓ 所有必需依赖已安装 ({len(installed)}/{len(required_packages)})")
        print()
        return True


def check_cuda():
    """检查CUDA可用性"""
    
    print("="*80)
    print("检查CUDA环境...")
    print("="*80)
    print()
    
    try:
        import torch
        
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            print(f"✓ CUDA可用")
            print(f"  GPU数量: {n_gpus}")
            print()
            
            for i in range(n_gpus):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  GPU {i}: {name}")
                print(f"    内存: {mem:.1f} GB")
            
            print()
            return True
        else:
            print("❌ CUDA不可用")
            print("  实验需要GPU才能运行")
            print()
            return False
    
    except ImportError:
        print("❌ PyTorch未安装")
        return False


def check_files():
    """检查必要的文件"""
    
    print("="*80)
    print("检查必要文件...")
    print("="*80)
    print()
    
    from pathlib import Path
    
    required_files = [
        'Exp_real_cost_impact_v2.py',
        'Exp1_multi_seed_wrapper.py',
        'Exp1_aggregate_and_analyze.py',
        'Exp1_plots.py',
        'run_exp1_full_optimized.sh',
        'run_exp1_quick_validation.sh',
        'oran_benchmark_loader.py',
        'configs/multi_gpu.yaml',
    ]
    
    missing_files = []
    
    for file_path in required_files:
        p = Path(file_path)
        if p.exists():
            size = p.stat().st_size
            print(f"  ✓ {file_path:45s} ({size:,} bytes)")
        else:
            print(f"  ✗ {file_path:45s} - 缺失")
            missing_files.append(file_path)
    
    print()
    
    if missing_files:
        print(f"❌ 缺少 {len(missing_files)} 个文件")
        return False
    else:
        print(f"✓ 所有文件完整")
        return True


def check_directories():
    """检查必要的目录"""
    
    print("="*80)
    print("检查必要目录...")
    print("="*80)
    print()
    
    from pathlib import Path
    
    required_dirs = [
        'configs',
        'draw_figs/data',
        'figs',
        '../ARGO_MDP/src',
    ]
    
    optional_dirs = [
        'Environments/chroma_store',
        '/data/user/huangxiaolin/ARGO/RAG_Models/models/Qwen2.5-14B-Instruct',
        '/data/user/huangxiaolin/ARGO/models/all-MiniLM-L6-v2',
    ]
    
    print("必需目录:")
    print("-"*80)
    
    missing_dirs = []
    
    for dir_path in required_dirs:
        p = Path(dir_path)
        if p.exists() and p.is_dir():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} - 缺失")
            missing_dirs.append(dir_path)
    
    print()
    print("可选目录 (模型/数据):")
    print("-"*80)
    
    for dir_path in optional_dirs:
        p = Path(dir_path)
        if p.exists() and p.is_dir():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ⚠ {dir_path} - 缺失")
    
    print()
    
    if missing_dirs:
        print(f"❌ 缺少 {len(missing_dirs)} 个必需目录")
        return False
    else:
        print(f"✓ 所有必需目录存在")
        return True


def main():
    """主函数"""
    
    print("\n" + "="*80)
    print("实验1: 环境检查")
    print("="*80)
    print()
    
    checks = []
    
    # 检查依赖
    checks.append(("依赖包", check_dependencies()))
    
    # 检查CUDA
    checks.append(("CUDA环境", check_cuda()))
    
    # 检查文件
    checks.append(("必要文件", check_files()))
    
    # 检查目录
    checks.append(("必要目录", check_directories()))
    
    # 汇总
    print("="*80)
    print("检查汇总:")
    print("="*80)
    print()
    
    all_passed = all(result for _, result in checks)
    
    for name, result in checks:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {name:20s}: {status}")
    
    print()
    
    if all_passed:
        print("="*80)
        print("✓ 所有检查通过! 可以开始实验")
        print("="*80)
        print()
        print("推荐的下一步:")
        print("  1. 快速验证:")
        print("     bash run_exp1_quick_validation.sh")
        print()
        print("  2. 完整实验:")
        print("     bash run_exp1_full_optimized.sh")
        print()
        return 0
    else:
        print("="*80)
        print("❌ 部分检查未通过，请先解决问题")
        print("="*80)
        print()
        print("常见问题:")
        print("  1. 缺少依赖包:")
        print("     pip install numpy pandas scipy matplotlib seaborn torch transformers sentence-transformers")
        print()
        print("  2. CUDA不可用:")
        print("     确保已安装CUDA驱动和PyTorch GPU版本")
        print()
        print("  3. 缺少文件:")
        print("     确保所有脚本文件已创建")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
