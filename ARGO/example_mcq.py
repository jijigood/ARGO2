#!/usr/bin/env python3
"""
ARGO选择题使用示例
==================

演示如何使用ARGO系统回答O-RAN选择题。
"""

import json
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.argo_system import ARGO_System


def example_single_question():
    """示例1: 回答单个选择题"""
    print("="*80)
    print("示例1: 回答单个选择题")
    print("="*80)
    
    # 初始化ARGO系统
    argo = ARGO_System(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        retriever_mode="chroma",  # 使用Chroma检索
        chroma_dir="chroma_db",
        use_mdp=True,  # 使用MDP策略
        verbose=True   # 显示详细过程
    )
    
    # 准备问题和选项
    question = "What is a key function of the O-RAN Fronthaul CUS Plane specification?"
    options = [
        "Support for slice differentiation to meet specific SLAs.",
        "Optimizing power consumption for the gNB DU system.",
        "Managing network security protocols.",
        "Determining the optimal frequency band for transmission."
    ]
    
    # 回答问题
    answer, choice, history, metadata = argo.answer_question(
        question=question,
        options=options,
        return_history=True
    )
    
    # 显示结果
    print("\n" + "="*80)
    print("结果")
    print("="*80)
    print(f"详细答案:\n{answer}\n")
    print(f"选择的选项: {choice}")
    print(f"\n元数据:")
    print(f"  - 推理步数: {metadata['total_steps']}")
    print(f"  - 检索次数: {metadata['retrieve_count']}")
    print(f"  - 推理次数: {metadata['reason_count']}")
    print(f"  - 耗时: {metadata['elapsed_time']:.2f}秒")
    
    if metadata.get('sources'):
        print(f"  - 引用来源: {', '.join(metadata['sources'])}")


def example_batch_evaluation():
    """示例2: 批量评估数据集"""
    print("\n\n" + "="*80)
    print("示例2: 批量评估数据集")
    print("="*80)
    
    # 加载数据集
    dataset_path = "ORAN-Bench-13K/Benchmark/fin_H_clean.json"
    
    if not os.path.exists(dataset_path):
        print(f"⚠️  数据集未找到: {dataset_path}")
        print("请确保数据集文件存在")
        return
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"加载数据集: {len(dataset)} 题")
    
    # 初始化系统（关闭详细输出以加快速度）
    argo = ARGO_System(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        retriever_mode="chroma",
        chroma_dir="chroma_db",
        use_mdp=True,
        verbose=False
    )
    
    # 评估前N题
    num_samples = 10
    print(f"\n评估前 {num_samples} 题...\n")
    
    correct_count = 0
    results = []
    
    for i, item in enumerate(dataset[:num_samples]):
        question_text = item[0]
        raw_options = item[1]
        correct_answer = item[2]
        
        # 清理选项（移除编号前缀）
        options = [opt.split('. ', 1)[1] if '. ' in opt else opt 
                  for opt in raw_options]
        
        # 推理
        print(f"[{i+1}/{num_samples}] 处理中...", end=" ")
        
        try:
            _, choice, _, metadata = argo.answer_question(
                question=question_text,
                options=options,
                return_history=False
            )
            
            is_correct = (choice == correct_answer)
            if is_correct:
                correct_count += 1
            
            status = "✅" if is_correct else "❌"
            print(f"{status} 预测={choice}, 正确={correct_answer}, "
                  f"步数={metadata['total_steps']}, "
                  f"耗时={metadata['elapsed_time']:.1f}s")
            
            results.append({
                'question_id': i,
                'predicted': choice,
                'correct': correct_answer,
                'is_correct': is_correct,
                'steps': metadata['total_steps'],
                'time': metadata['elapsed_time']
            })
            
        except Exception as e:
            print(f"❌ 错误: {e}")
            results.append({
                'question_id': i,
                'predicted': None,
                'correct': correct_answer,
                'is_correct': False,
                'error': str(e)
            })
    
    # 统计
    print("\n" + "="*80)
    print("评估结果")
    print("="*80)
    
    accuracy = correct_count / num_samples
    avg_steps = sum(r.get('steps', 0) for r in results) / num_samples
    avg_time = sum(r.get('time', 0) for r in results) / num_samples
    
    print(f"准确率: {accuracy*100:.2f}% ({correct_count}/{num_samples})")
    print(f"平均步数: {avg_steps:.1f}")
    print(f"平均耗时: {avg_time:.2f}秒")
    
    # 保存结果
    output_file = "evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total': num_samples,
                'correct': correct_count,
                'accuracy': accuracy,
                'avg_steps': avg_steps,
                'avg_time': avg_time
            },
            'details': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细结果已保存至: {output_file}")


def example_custom_options():
    """示例3: 自定义选项格式"""
    print("\n\n" + "="*80)
    print("示例3: 自定义选项格式")
    print("="*80)
    
    argo = ARGO_System(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        retriever_mode="mock",
        verbose=False
    )
    
    # 自定义问题和选项（不带编号）
    question = "Which component is responsible for E2 Service Model management in O-RAN?"
    options = [
        "O-DU",
        "Near-RT RIC",
        "SMO",
        "O-CU"
    ]
    
    print(f"问题: {question}")
    print("\n选项:")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    
    answer, choice, _, _ = argo.answer_question(
        question=question,
        options=options
    )
    
    print(f"\n选择: {choice}")
    print(f"解释: {answer[:200]}...")


def main():
    """主函数"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*18 + "ARGO 选择题使用示例" + " "*18 + "║")
    print("╚" + "="*78 + "╝")
    print()
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num == "1":
            example_single_question()
        elif example_num == "2":
            example_batch_evaluation()
        elif example_num == "3":
            example_custom_options()
        else:
            print(f"未知示例编号: {example_num}")
            print("用法: python example_mcq.py [1|2|3]")
    else:
        # 运行所有示例
        try:
            example_single_question()
        except Exception as e:
            print(f"示例1错误: {e}")
        
        try:
            example_batch_evaluation()
        except Exception as e:
            print(f"示例2错误: {e}")
        
        try:
            example_custom_options()
        except Exception as e:
            print(f"示例3错误: {e}")
    
    print("\n✅ 示例运行完成！")


if __name__ == "__main__":
    main()
