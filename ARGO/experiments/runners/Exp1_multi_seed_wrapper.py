#!/usr/bin/env python
"""
多种子实验包装器 - 实验1 (Question-Adaptive Version)
=========================================================
自动运行多个随机种子、多个难度级别的实验
使用问题自适应终止策略确保高效性

修改内容:
    ✓ 启用 question-adaptive U_max estimation
    ✓ 复杂度自适应的终止阈值
    ✓ 动态进度追踪
    ✓ 增强的结果记录

用法:
        # 快速验证 (3个种子, Hard难度)
        python Exp1_multi_seed_wrapper.py --n-seeds 3 --difficulties hard
    
        # 完整实验 (5个种子, 所有难度)
        python Exp1_multi_seed_wrapper.py --n-seeds 5 --n-questions 100 --difficulties easy,medium,hard
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import argparse
import json


def run_multi_seed_experiment(
    n_seeds: int = 5,
    base_seed: int = 42,
    n_questions: int = 100,
    difficulties: list = ['easy', 'medium', 'hard'],
    gpus: str = '0,1,2,3,4,5,6,7',
    model_path: str = None,
    config_path: str = 'configs/multi_gpu_data_calibrated.yaml',
    policy_config_path: str = 'configs/adaptive_policy.yaml'
):
    """运行多种子实验 (Question-Adaptive版本)"""
    
    print("="*80)
    print("实验1: 多种子实验执行 (Question-Adaptive版本)")
    print("="*80)
    print(f"随机种子数: {n_seeds} (从 {base_seed} 到 {base_seed + n_seeds - 1})")
    print(f"每难度问题数: {n_questions}")
    print(f"难度级别: {difficulties}")
    print(f"使用GPU: {gpus}")
    if model_path:
        model_name = Path(model_path).name
        print(f"LLM模型: {model_name}")
    print(f"策略配置: {policy_config_path}")
    print(f"总运行次数: {n_seeds * len(difficulties)}")
    print("="*80)
    print("\n特性:")
    print("  ✓ Question-adaptive U_max estimation")
    print("  ✓ Complexity-based threshold scaling")
    print("  ✓ Dynamic progress tracking")
    print("  ✓ Early termination for simple questions")
    print("="*80)
    print()
    
    start_time = datetime.now()
    completed_runs = []
    failed_runs = []
    run_statistics = []
    
    total_runs = n_seeds * len(difficulties)
    current_run = 0
    
    for difficulty in difficulties:
        print(f"\n{'='*80}")
        print(f"难度级别: {difficulty.upper()}")
        print(f"{'='*80}\n")
        
        for seed_idx in range(n_seeds):
            seed = base_seed + seed_idx
            current_run += 1
            
            print(f"\n{'─'*80}")
            print(f"运行 {current_run}/{total_runs}: Difficulty={difficulty}, Seed={seed} ({seed_idx+1}/{n_seeds})")
            print(f"{'─'*80}\n")
            
            # 构建命令
            cmd = [
                sys.executable,
                'experiments/runners/Exp_real_cost_impact_v2.py',
                '--mode', 'custom',
                '--n-questions', str(n_questions),
                '--difficulty', difficulty,
                '--gpus', gpus,
                '--seed', str(seed)
            ]
            
            # 如果指定了模型路径，添加到命令中
            if model_path:
                cmd.extend(['--model-path', model_path])
            
            # 添加配置文件路径
            cmd.extend(['--config-path', config_path])
            
            # 添加策略配置文件路径
            if policy_config_path:
                cmd.extend(['--policy-config-path', policy_config_path])
            
            # 添加question-adaptive观察的verbose标志
            cmd.append('--verbose')
            
            print(f"执行命令: {' '.join(cmd)}\n")
            
            # 运行实验
            run_start = datetime.now()
            try:
                subprocess.run(cmd, capture_output=False, check=True)
                run_end = datetime.now()
                run_duration = (run_end - run_start).total_seconds()
                
                print(f"\n✓ 运行成功: {difficulty} / Seed {seed}")
                print(f"  运行时间: {run_duration:.1f}秒 ({run_duration/60:.1f}分钟)")
                completed_runs.append({
                    'difficulty': difficulty,
                    'seed': seed,
                    'status': 'success',
                    'duration': run_duration
                })
                run_statistics.append({
                    'difficulty': difficulty,
                    'seed': seed,
                    'duration_seconds': run_duration,
                    'status': 'success'
                })
                
            except subprocess.CalledProcessError as e:
                run_end = datetime.now()
                run_duration = (run_end - run_start).total_seconds()
                print(f"\n❌ 运行失败: {difficulty} / Seed {seed}")
                print(f"   错误码: {e.returncode}")
                print(f"   运行时间: {run_duration:.1f}秒")
                failed_runs.append({
                    'difficulty': difficulty,
                    'seed': seed,
                    'status': 'failed',
                    'error': str(e),
                    'duration': run_duration
                })
                run_statistics.append({
                    'difficulty': difficulty,
                    'seed': seed,
                    'duration_seconds': run_duration,
                    'status': 'failed',
                    'error': str(e)
                })
                
                # 询问是否继续
                print("\n发生错误! 选项:")
                print("  [c] 继续下一个运行")
                print("  [s] 停止并退出")
                choice = input("请选择 (c/s): ").strip().lower()
                
                if choice == 's':
                    print("\n用户选择停止实验")
                    break
            
            print(f"\n{'─'*80}")
            print(f"进度: {current_run}/{total_runs} 完成 ({current_run/total_runs*100:.1f}%)")
            if completed_runs:
                avg_duration = sum(r['duration'] for r in completed_runs) / len(completed_runs)
                remaining = total_runs - current_run
                eta_seconds = remaining * avg_duration
                print(f"平均运行时间: {avg_duration:.1f}秒/次")
                print(f"预计剩余时间: {eta_seconds/60:.1f}分钟")
            print(f"{'─'*80}")
        
        # 如果在内层循环中选择停止，也退出外层循环
        if failed_runs and failed_runs[-1]['status'] == 'failed':
            choice = input("\n是否继续下一个难度级别? (y/n): ").strip().lower()
            if choice != 'y':
                break
    
    end_time = datetime.now()
    elapsed = end_time - start_time
    stats_dir = Path('draw_figs/data')
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_file = stats_dir / f"exp1_run_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(stats_file, 'w') as f:
        json.dump({
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'total_duration_seconds': elapsed.total_seconds(),
            'config': {
                'n_seeds': n_seeds,
                'base_seed': base_seed,
                'n_questions': n_questions,
                'difficulties': difficulties,
                'policy_config': policy_config_path,
                'gpus': gpus
            },
            'runs': run_statistics,
            'summary': {
                'total_runs': total_runs,
                'completed': len(completed_runs),
                'failed': len(failed_runs)
            }
        }, f, indent=2)
    print(f"\n✓ 运行统计已保存到: {stats_file}")
    
    # 汇总报告
    print("\n" + "="*80)
    print("多种子实验完成!")
    print("="*80)
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {elapsed}")
    print(f"\n成功运行: {len(completed_runs)}/{total_runs}")
    print(f"失败运行: {len(failed_runs)}/{total_runs}")
    
    if completed_runs:
        print("\n✓ 成功的运行:")
        total_time = 0.0
        for run in completed_runs:
            total_time += run['duration']
            print(f"  - {run['difficulty']:6s} / Seed {run['seed']:2d} ({run['duration']/60:.1f}分钟)")
        avg_time = total_time / len(completed_runs) / 60
        print(f"\n  平均运行时间: {avg_time:.1f}分钟/次")
    
    if failed_runs:
        print("\n❌ 失败的运行:")
        for run in failed_runs:
            print(f"  - {run['difficulty']:6s} / Seed {run['seed']}")
    
    print("\n" + "="*80)
    print("下一步操作:")
    print("="*80)
    
    if len(completed_runs) >= 3:  # 至少3个种子才有统计意义
        print("✓ 已完成足够的运行，可以进行统计分析:")
        print("  1. 聚合结果:")
        print("     python Exp1_aggregate_and_analyze.py")
        print("\n  2. 生成图表:")
        print("     python Exp1_plots.py draw_figs/data/exp1_aggregated_*.csv")
        print("\n  3. 检查question-adaptive效果:")
        print("     查看结果中的 'question_umax', 'progress_efficiency', 'terminated_early' 字段")
    else:
        print("⚠ 完成的运行不足3个，统计分析可能不可靠")
        print("  建议重新运行或增加种子数")
    
    return len(failed_runs) == 0


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='实验1多种子包装器 (Question-Adaptive版本)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 快速验证 (3种子, Hard)
  python Exp1_multi_seed_wrapper.py --n-seeds 3 --difficulties hard
  
  # 标准实验 (5种子, 所有难度)
  python Exp1_multi_seed_wrapper.py --n-seeds 5 --n-questions 100
  
  # 完整实验 (更多种子)
  python Exp1_multi_seed_wrapper.py --n-seeds 10 --n-questions 100
  
  # 使用自定义策略配置
  python Exp1_multi_seed_wrapper.py --policy-config-path configs/my_adaptive_policy.yaml
        """
    )
    
    parser.add_argument('--n-seeds', type=int, default=5,
                       help='随机种子数量 (默认: 5)')
    parser.add_argument('--base-seed', type=int, default=42,
                       help='起始种子值 (默认: 42)')
    parser.add_argument('--n-questions', type=int, default=100,
                       help='每个难度的问题数量 (默认: 100)')
    parser.add_argument('--difficulties', type=str, default='easy,medium,hard',
                       help='难度级别，逗号分隔 (默认: easy,medium,hard)')
    parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7',
                       help='使用的GPU ID (默认: 0,1,2,3,4,5,6,7)')
    parser.add_argument('--model-path', type=str, default=None,
                       help='LLM模型路径 (可选，用于覆盖默认模型)')
    parser.add_argument('--config-path', type=str, default='configs/multi_gpu_data_calibrated.yaml',
                       help='MDP配置文件路径 (默认使用data_calibrated版本，c_p=0.02)')
    parser.add_argument('--policy-config-path', type=str, default='configs/adaptive_policy.yaml',
                       help='自适应策略配置文件路径 (启用question-adaptive features)')
    
    args = parser.parse_args()
    
    # 解析难度列表
    difficulties = [d.strip() for d in args.difficulties.split(',')]
    
    # 验证难度
    valid_difficulties = ['easy', 'medium', 'hard']
    for diff in difficulties:
        if diff not in valid_difficulties:
            parser.error(f"无效的难度: {diff}. 必须是: {valid_difficulties}")
    
    # 验证策略配置存在
    policy_config_file = Path(args.policy_config_path)
    if not policy_config_file.exists():
        print(f"⚠ 警告: 策略配置文件不存在: {policy_config_file}")
        print("   将创建默认配置文件...")
        policy_config_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"   请手动创建配置文件: {policy_config_file}")
        print("   参考上面提供的 adaptive_policy.yaml 模板")
        sys.exit(1)
    
    # 运行实验
    success = run_multi_seed_experiment(
        n_seeds=args.n_seeds,
        base_seed=args.base_seed,
        n_questions=args.n_questions,
        difficulties=difficulties,
        gpus=args.gpus,
        model_path=args.model_path,
        config_path=args.config_path,
        policy_config_path=args.policy_config_path
    )
    
    # 返回退出码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
