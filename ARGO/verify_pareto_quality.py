#!/usr/bin/env python3
"""
快速验证Pareto曲线质量
检查是否满足：
1. 至少10个不同的操作点
2. 质量范围: [0.55, 1.0] (平滑分布，不是跳跃)
3. 成本范围: [1.5, 4.5] (平滑过渡)
4. 检索次数: [0, 1, 2, 3, 4, 5] 分布在不同μ值
"""

import sys
import os

import yaml
import numpy as np

# 直接导入mdp_solver模块（避免触发__init__.py）
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mdp_solver_path = os.path.join(parent_dir, 'ARGO_MDP', 'src', 'mdp_solver.py')

# 动态加载模块
import importlib.util
spec = importlib.util.spec_from_file_location("mdp_solver", mdp_solver_path)
mdp_solver_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mdp_solver_module)
MDPSolver = mdp_solver_module.MDPSolver

def load_config():
    """加载配置"""
    with open('configs/multi_gpu.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def diagnose_pareto_curve():
    """诊断Pareto曲线质量"""
    config = load_config()
    mdp_config = config['mdp']
    
    print("=" * 80)
    print("Pareto曲线质量诊断")
    print("=" * 80)
    print(f"当前参数:")
    print(f"  c_r = {mdp_config['c_r']}")
    print(f"  c_p = {mdp_config['c_p']}")
    print(f"  delta_r = {mdp_config['delta_r']}")
    print(f"  delta_p = {mdp_config['delta_p']}")
    print(f"  quality_k = {mdp_config['quality_k']}")
    print(f"  p_s = {mdp_config['p_s']}")
    print()
    
    # 测试更宽的μ范围
    mu_values = np.linspace(0, 5, 51)  # 51个点，密集采样
    
    results = []
    
    print(f"测试 {len(mu_values)} 个μ值...")
    print("-" * 80)
    
    for mu in mu_values:
        # 创建完整的MDP求解器配置
        mdp_config_copy = mdp_config.copy()
        mdp_config_copy['mu'] = mu
        if 'U_grid_size' not in mdp_config_copy:
            mdp_config_copy['U_grid_size'] = mdp_config.get('grid_size', 101)
        
        # 从MDP配置构建quality字典 (FIX: 使用配置的quality_function)
        quality_config = {
            'mode': mdp_config.get('quality_function', 'linear'),
            'k': mdp_config.get('quality_k', 1.0)
        }
        
        solver_config = {
            'mdp': mdp_config_copy,
            'quality': quality_config,
            'reward_shaping': mdp_config.get('reward_shaping', {'enabled': False, 'k': 1.0}),
            'solver': {
                'max_iterations': 1000,
                'convergence_threshold': 1e-6,
                'verbose': False
            }
        }
        
        solver = MDPSolver(config=solver_config)
        
        # 求解MDP (μ已在配置中)
        result = solver.solve()
        theta_cont = result['theta_cont']
        theta_star = result['theta_star']
        
        # 模拟典型场景的期望值
        U_max = mdp_config['U_max']
        delta_r = mdp_config['delta_r']
        delta_p = mdp_config['delta_p']
        c_r = mdp_config['c_r']
        c_p = mdp_config['c_p']
        p_s = mdp_config['p_s']
        
        # 估算期望检索次数和成本（简化模型）
        if theta_star == 0:
            # 立即终止
            expected_retrievals = 0
            expected_cost = 0
            expected_quality = 0
        elif theta_cont >= U_max:
            # 只推理
            steps_to_threshold = int(np.ceil(theta_star / delta_p))
            expected_retrievals = 0
            expected_cost = steps_to_threshold * c_p
            expected_quality = min(theta_star, U_max) / U_max
        else:
            # 混合策略：先检索到theta_cont，再推理到theta_star
            # 期望检索次数（考虑失败）
            retrievals_needed = int(np.ceil(theta_cont / delta_r))
            expected_successful = retrievals_needed * p_s
            expected_retrievals = retrievals_needed / p_s  # 考虑重试
            
            # 达到theta_cont后的推理步数
            U_after_retrieve = expected_successful * delta_r
            remaining = max(0, theta_star - U_after_retrieve)
            reason_steps = int(np.ceil(remaining / delta_p))
            
            expected_cost = expected_retrievals * c_r + reason_steps * c_p
            expected_quality = min(theta_star, U_max) / U_max
        
        results.append({
            'mu': mu,
            'theta_cont': theta_cont,
            'theta_star': theta_star,
            'expected_retrievals': expected_retrievals,
            'expected_cost': expected_cost,
            'expected_quality': expected_quality
        })
    
    # 分析结果
    print("\n" + "=" * 80)
    print("Pareto曲线分析")
    print("=" * 80)
    
    # 提取有意义的操作点（θ* > 0）
    valid_points = [r for r in results if r['theta_star'] > 0]
    
    print(f"\n1. 操作点数量:")
    print(f"   总点数: {len(results)}")
    print(f"   有效点 (θ*>0): {len(valid_points)}")
    print(f"   无效点 (θ*=0): {len(results) - len(valid_points)}")
    
    if len(valid_points) < 10:
        print(f"   ❌ 警告: 有效点少于10个，Pareto曲线太稀疏！")
    else:
        print(f"   ✅ 操作点充足")
    
    # 找到不同θ*值的数量
    unique_thetas = len(set(r['theta_star'] for r in valid_points))
    print(f"   不同的θ*值: {unique_thetas}")
    
    if len(valid_points) > 0:
        qualities = [r['expected_quality'] for r in valid_points]
        costs = [r['expected_cost'] for r in valid_points]
        retrievals = [r['expected_retrievals'] for r in valid_points]
        
        print(f"\n2. 质量范围:")
        print(f"   最小: {min(qualities):.3f}")
        print(f"   最大: {max(qualities):.3f}")
        print(f"   范围: {max(qualities) - min(qualities):.3f}")
        
        # 检查是否有足够的质量层次
        unique_qualities = len(set(np.round(qualities, 2)))
        print(f"   不同质量值 (精度0.01): {unique_qualities}")
        
        if min(qualities) < 0.55:
            print(f"   ❌ 警告: 最小质量 {min(qualities):.3f} < 0.55")
        elif max(qualities) - min(qualities) < 0.3:
            print(f"   ❌ 警告: 质量范围太窄 ({max(qualities) - min(qualities):.3f} < 0.3)")
        elif unique_qualities < 10:
            print(f"   ⚠️  警告: 质量层次较少 ({unique_qualities} < 10)")
        else:
            print(f"   ✅ 质量范围良好")
        
        print(f"\n3. 成本范围:")
        print(f"   最小: {min(costs):.3f}")
        print(f"   最大: {max(costs):.3f}")
        print(f"   范围: {max(costs) - min(costs):.3f}")
        
        if min(costs) < 1.5:
            print(f"   ⚠️  最小成本 {min(costs):.3f} < 1.5 (可能太便宜)")
        if max(costs) > 4.5:
            print(f"   ⚠️  最大成本 {max(costs):.3f} > 4.5 (可能太贵)")
        
        if max(costs) - min(costs) < 2.0:
            print(f"   ❌ 警告: 成本范围太窄 ({max(costs) - min(costs):.3f} < 2.0)")
        else:
            print(f"   ✅ 成本范围充足")
        
        print(f"\n4. 检索次数分布:")
        print(f"   最小: {min(retrievals):.1f}")
        print(f"   最大: {max(retrievals):.1f}")
        
        # 统计检索次数直方图
        retrieval_bins = np.arange(0, 7)
        hist, _ = np.histogram(retrievals, bins=retrieval_bins)
        print(f"   分布:")
        for i, count in enumerate(hist):
            if count > 0:
                print(f"     {i}次: {count}个点")
        
        covered_bins = np.sum(hist > 0)
        if covered_bins < 4:
            print(f"   ❌ 警告: 检索次数分布太集中 (只有{covered_bins}个不同值)")
        else:
            print(f"   ✅ 检索次数分布良好")
        
        # 显示详细的采样点
        print(f"\n5. 详细采样点 (前20个有效点):")
        print(f"{'μ':>6} {'θ_cont':>7} {'θ*':>7} {'Quality':>8} {'Cost':>7} {'Retrievals':>10}")
        print("-" * 60)
        
        for r in valid_points[:20]:
            print(f"{r['mu']:6.2f} {r['theta_cont']:7.3f} {r['theta_star']:7.3f} "
                  f"{r['expected_quality']:8.3f} {r['expected_cost']:7.3f} "
                  f"{r['expected_retrievals']:10.2f}")
        
        if len(valid_points) > 20:
            print(f"... 还有 {len(valid_points) - 20} 个点未显示")
        
        # 找到θ*发生变化的μ值
        print(f"\n6. θ*变化点:")
        last_theta = None
        change_count = 0
        for r in results:
            if last_theta is not None and r['theta_star'] != last_theta:
                change_count += 1
                if change_count <= 10:  # 只显示前10个变化点
                    print(f"   μ={r['mu']:.3f}: θ*从 {last_theta:.3f} 变为 {r['theta_star']:.3f}")
            last_theta = r['theta_star']
        
        print(f"   总共 {change_count} 个变化点")
        
        if change_count < 10:
            print(f"   ❌ 警告: θ*变化太少，Pareto曲线不够平滑")
        else:
            print(f"   ✅ θ*变化充足")
    
    else:
        print("\n❌ 严重错误: 没有任何有效操作点！")
        print("   所有μ值都导致θ*=0 (立即终止)")
        print("\n原因分析:")
        print("   当 μ·c_r > quality_k·delta_r 时，检索的成本超过收益")
        print(f"   当前: μ·{c_r} > {quality_k}·{delta_r}")
        print(f"   即: μ > {quality_k * delta_r / c_r:.3f}")
        print("\n建议:")
        print("   1. 进一步降低 c_r (当前 {:.3f})".format(c_r))
        print("   2. 或提高 quality_k (当前 {:.3f})".format(quality_k))
        print("   3. 或提高 delta_r (当前 {:.3f})".format(delta_r))
    
    print("\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80)
    
    # 给出总体评估
    if len(valid_points) >= 10:
        qualities = [r['expected_quality'] for r in valid_points]
        costs = [r['expected_cost'] for r in valid_points]
        
        quality_ok = (min(qualities) >= 0.5 and 
                     max(qualities) - min(qualities) >= 0.3 and
                     len(set(np.round(qualities, 2))) >= 8)
        
        cost_ok = max(costs) - min(costs) >= 2.0
        
        if quality_ok and cost_ok:
            print("\n✅ 总体评估: 参数配置良好，可以运行完整实验")
        else:
            print("\n⚠️  总体评估: 参数配置需要调整")
            if not quality_ok:
                print("   - 质量范围或分布需要改善")
            if not cost_ok:
                print("   - 成本范围需要扩大")
    else:
        print("\n❌ 总体评估: 参数配置有严重问题，需要重新调整")

if __name__ == "__main__":
    diagnose_pareto_curve()
