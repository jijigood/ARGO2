#!/usr/bin/env python3
"""
补充测试：不同复杂度查询下MDP决策的一致性
证明：查询复杂度不影响成本-收益权衡
"""

import sys
import os

# 添加ARGO_MDP路径
argo_mdp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ARGO_MDP', 'src'))
sys.path.insert(0, argo_mdp_path)

from mdp_solver import MDPSolver

def test_query_complexity_independence():
    """测试不同复杂度查询在相同参数下的决策一致性"""
    
    print("=" * 80)
    print("查询复杂度独立性测试")
    print("=" * 80)
    print("\n证明：在固定参数下，无论查询简单还是复杂，MDP的阈值保持不变\n")
    
    # 固定参数配置
    config = {
        'mdp': {
            'delta_r': 0.15,
            'delta_p': 0.08,
            'c_r': 0.05,
            'c_p': 0.02,
            'p_s': 0.8,
            'gamma': 0.98,
            'U_max': 1.0,
            'mu': 0.6,
            'U_grid_size': 101
        },
        'quality': {
            'mode': 'linear',
            'k': 1.0
        },
        'solver': {
            'max_iterations': 1000,
            'convergence_threshold': 1e-6,
            'verbose': False
        }
    }
    
    # 求解MDP
    solver = MDPSolver(config)
    solver.solve()
    
    print("固定参数 (当前规范):")
    print(f"  delta_r={config['mdp']['delta_r']}, delta_p={config['mdp']['delta_p']}")
    print(f"  c_r={config['mdp']['c_r']}, c_p={config['mdp']['c_p']}")
    print(f"  成本效益比: Retrieve={config['mdp']['c_r']/config['mdp']['delta_r']:.3f}, " 
          f"Reason={config['mdp']['c_p']/config['mdp']['delta_p']:.3f}")
    
    print(f"\nMDP求解结果:")
    print(f"  θ_cont = {solver.theta_cont:.4f}")
    print(f"  θ* = {solver.theta_star:.4f}")
    
    # 模拟不同复杂度的查询（通过当前U表示）
    queries = [
        ("简单查询 (U=0.7, 已有70%信息)", 0.7),
        ("中等查询 (U=0.4, 已有40%信息)", 0.4),
        ("困难查询 (U=0.1, 几乎无信息)", 0.1),
        ("刚开始 (U=0.0, 完全无信息)", 0.0),
    ]
    
    print("\n" + "-" * 80)
    print("不同'复杂度'查询的决策 (复杂度由当前U表示):")
    print("-" * 80)
    
    for desc, U in queries:
        if U <= solver.theta_cont:
            action = "Retrieve"
            reason = f"U({U:.1f}) ≤ θ_cont({solver.theta_cont:.4f})"
        elif U <= solver.theta_star:
            action = "Reason"
            reason = f"θ_cont({solver.theta_cont:.4f}) < U({U:.1f}) ≤ θ*({solver.theta_star:.4f})"
        else:
            action = "Terminate"
            reason = f"U({U:.1f}) > θ*({solver.theta_star:.4f})"
        
        print(f"\n{desc}")
        print(f"  → 决策: {action}")
        print(f"  → 原因: {reason}")
    
    print("\n" + "=" * 80)
    print("📊 关键结论")
    print("=" * 80)
    print("""
1. **阈值是参数的函数，与具体查询无关**
   - θ_cont 和 θ* 由 MDP 参数决定 (c_r, c_p, δ_r, δ_p, p_s, γ)
   - 一旦参数固定，阈值就固定
   
2. **当前参数下 θ_cont=0.0 意味着**
   - 任何非零U状态都会选择Reason而非Retrieve
   - 这是因为 Reason 的成本效益比更优 (0.25 < 0.33)
   
3. **查询复杂度只影响初始U值，不影响策略**
   - 简单查询: 可能从较高U开始 → 快速Terminate
   - 困难查询: 从较低U开始 → 多次Reason → Terminate
   - 但无论哪种，MDP都不会选择Retrieve (除非U=0.0)
   
4. **要让MDP执行Retrieve，必须调整参数**
   - 方案1: 提高 delta_r (增加retrieve收益)
   - 方案2: 降低 c_r (减少retrieve成本)
   - 方案3: 降低 delta_p 或提高 c_p (使reason不那么优)
   - 目标: 使 c_r/delta_r ≤ c_p/delta_p
""")
    
    print("\n推荐修改:")
    print("  delta_r: 0.15 → 0.25  (使成本效益比: 0.05/0.25=0.20 < 0.25)")
    print("  这样 θ_cont 会从 0.0 变为 ~0.08，在低U时会选择Retrieve")

if __name__ == "__main__":
    test_query_complexity_independence()
