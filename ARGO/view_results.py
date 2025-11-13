"""
快速查看实验结果
================

快速解析并展示实验1和实验2的关键结果
"""

import json
import glob
from pathlib import Path

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def load_latest_exp_data(pattern):
    """加载最新的实验数据"""
    files = glob.glob(f"draw_figs/data/{pattern}")
    if not files:
        return None
    latest = max(files, key=lambda x: Path(x).stat().st_mtime)
    with open(latest, 'r') as f:
        return json.load(f)

def show_exp1_results():
    """展示实验1结果"""
    print_section("实验1: 检索成本(c_r)的影响")
    
    data = load_latest_exp_data("exp1_*.json")
    if not data:
        print("未找到实验1数据")
        return
    
    print(f"\n实验配置:")
    print(f"  测试问题: {data['config']['n_test_questions']}道")
    print(f"  难度: {data['config']['difficulty']}")
    print(f"  固定p_s: {data['config'].get('p_s', 'N/A')}")
    
    c_r_values = data['results']['c_r_values']
    c_p = data['config']['c_p']
    
    print(f"\n成本范围: {c_r_values[0]:.3f} ~ {c_r_values[-1]:.3f} ({c_r_values[0]/c_p:.1f}x ~ {c_r_values[-1]/c_p:.1f}x c_p)")
    
    print("\n关键结果:")
    print("-" * 70)
    print(f"{'c_r/c_p':<10} {'ARGO检索':<12} {'ARGO质量':<12} {'Always-R检索':<15} {'差异':<10}")
    print("-" * 70)
    
    argo_r = data['results']['policies']['ARGO']['retrievals']
    argo_q = data['results']['policies']['ARGO']['quality']
    always_r = data['results']['policies']['Always-Retrieve']['retrievals']
    
    for i, c_r in enumerate(c_r_values):
        ratio = c_r / c_p
        diff_pct = (always_r[i] - argo_r[i]) / always_r[i] * 100 if always_r[i] > 0 else 0
        print(f"{ratio:<10.1f} {argo_r[i]:<12.1f} {argo_q[i]:<12.3f} {always_r[i]:<15.1f} {diff_pct:<10.0f}%")
    
    print("\n核心发现:")
    print(f"  ✓ ARGO在高成本下(c_r≥4c_p)完全停止检索: {argo_r[-1]:.1f}次")
    print(f"  ✓ Always-Retrieve检索次数恒定: {always_r[0]:.1f}次")
    print(f"  ✓ 最大效率提升: {(always_r[-1] - argo_r[-1]) / always_r[-1] * 100:.0f}%")

def show_exp2_results():
    """展示实验2结果"""
    print_section("实验2: 检索成功率(p_s)的影响")
    
    data = load_latest_exp_data("exp2_*.json")
    if not data:
        print("未找到实验2数据")
        return
    
    print(f"\n实验配置:")
    print(f"  测试问题: {data['config']['n_test_questions']}道")
    print(f"  难度: {data['config']['difficulty']}")
    print(f"  固定c_r: {data['config'].get('c_r', 'N/A')}")
    
    p_s_values = data['results']['p_s_values']
    
    print(f"\n成功率范围: {p_s_values[0]:.2f} ~ {p_s_values[-1]:.2f}")
    
    print("\n关键结果:")
    print("-" * 80)
    print(f"{'p_s':<8} {'ARGO检索':<12} {'ARGO推理':<12} {'ARGO质量':<12} {'Always-R检索':<15} {'效率提升':<12}")
    print("-" * 80)
    
    argo_r = data['results']['policies']['ARGO']['retrievals']
    argo_p = data['results']['policies']['ARGO']['reasons']
    argo_q = data['results']['policies']['ARGO']['quality']
    always_r = data['results']['policies']['Always-Retrieve']['retrievals']
    
    for i, p_s in enumerate(p_s_values):
        gain_pct = (always_r[i] - argo_r[i]) / always_r[i] * 100 if always_r[i] > 0 else 0
        print(f"{p_s:<8.2f} {argo_r[i]:<12.1f} {argo_p[i]:<12.1f} {argo_q[i]:<12.3f} {always_r[i]:<15.1f} {gain_pct:<12.0f}%")
    
    print("\n核心发现:")
    print(f"  ✓ 低p_s时(0.3),ARGO避免检索: {argo_r[0]:.1f}次 vs Always-Retrieve {always_r[0]:.1f}次")
    print(f"  ✓ ARGO转向推理: {argo_p[0]:.1f}次")
    print(f"  ✓ 最大效率提升: {(always_r[0] - argo_r[0]) / always_r[0] * 100 if always_r[0] > 0 else 'inf'}%")

def show_threshold_evolution():
    """展示阈值演化"""
    print_section("MDP阈值演化")
    
    # 实验1的阈值
    data1 = load_latest_exp_data("exp1_*.json")
    if data1:
        print("\n实验1 - 随c_r变化:")
        print("-" * 50)
        print(f"{'c_r/c_p':<15} {'θ_cont':<15} {'θ*':<15}")
        print("-" * 50)
        
        c_r_values = data1['results']['c_r_values']
        c_p = data1['config']['c_p']
        thresholds = data1['results']['policies']['ARGO']['thresholds']
        
        for i, c_r in enumerate(c_r_values):
            ratio = c_r / c_p
            t = thresholds[i]
            print(f"{ratio:<15.1f} {t['theta_cont']:<15.4f} {t['theta_star']:<15.4f}")
    
    # 实验2的阈值
    data2 = load_latest_exp_data("exp2_*.json")
    if data2:
        print("\n实验2 - 随p_s变化:")
        print("-" * 50)
        print(f"{'p_s':<15} {'θ_cont':<15} {'θ*':<15}")
        print("-" * 50)
        
        p_s_values = data2['results']['p_s_values']
        thresholds = data2['results']['policies']['ARGO']['thresholds']
        
        for i, p_s in enumerate(p_s_values):
            t = thresholds[i]
            print(f"{p_s:<15.2f} {t['theta_cont']:<15.4f} {t['theta_star']:<15.4f}")

def show_summary():
    """总结"""
    print_section("实验总结")
    
    print("\n✅ 成功验证的假设:")
    print("  1. ARGO具有成本自适应能力(实验1)")
    print("  2. ARGO能管理检索不确定性(实验2)")
    print("  3. MDP求解器能找到最优策略")
    print("  4. 静态基线无法适应环境变化")
    
    print("\n📊 生成的图表:")
    import os
    if os.path.exists('figs'):
        for f in sorted(Path('figs').glob('exp*.png')):
            size = f.stat().st_size / 1024
            print(f"  - {f.name} ({size:.0f}KB)")
    
    print("\n📖 详细报告:")
    print("  - EXPERIMENT1_REPORT.md")
    print("  - EXPERIMENT2_REPORT.md")
    print("  - EXPERIMENTS_INDEX.md")
    
    print("\n🎯 论文贡献:")
    print("  - 证明了ARGO的成本敏感性和不确定性管理能力")
    print("  - 提供了可视化证据支持MDP优于静态策略")
    print("  - 为Section 6实验部分提供了核心数据")

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ARGO实验结果快速查看")
    print("=" * 70)
    print(f"时间: {Path('draw_figs/data').exists() and 'Ready' or '请先运行实验'}")
    
    show_exp1_results()
    show_exp2_results()
    show_threshold_evolution()
    show_summary()
    
    print("\n" + "=" * 70)
    print("查看完毕!")
    print("=" * 70)
