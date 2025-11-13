"""
Zero-Cost Optimization Test - Phase 4.2.1
=========================================

测试零成本优化方案的效果：
1. 使用Qwen2.5-1.5B-Instruct (替代3B)
2. 减少max_new_tokens (Decomposer: 128→50, Synthesizer: 512→200)

预期加速: 3倍 (55秒 → 18秒)
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import pandas as pd
import logging

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import ARGO_System
from measure_latency import LatencyProfiler


def main():
    print("="*80)
    print("Zero-Cost Optimization Test")
    print("="*80)
    print("\n优化措施:")
    print("  1. 模型: Qwen2.5-3B → Qwen2.5-1.5B (2倍加速)")
    print("  2. Decomposer max_tokens: 128 → 50 (约1.5倍加速)")
    print("  3. Synthesizer max_tokens: 512 → 200 (约1.5倍加速)")
    print("  预期总加速: 2 × 1.5 = 3倍")
    print("  预期延迟: 55秒 → 18秒")
    print("="*80)
    
    # 使用1.5B模型
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    print(f"\nLoading model: {model_name}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print(f"✅ Model loaded")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        print(f"\n⚠️  Model loading failed: {e}")
        print("\n备选方案：模型可能未下载，将使用3B模型测试参数优化")
        
        # 回退到3B模型
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        print(f"\nFalling back to: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print(f"✅ Model loaded (using 3B as fallback)")
    
    # 创建系统（使用优化参数）
    print("\nInitializing ARGO System with optimized parameters...")
    
    # 注意：我们需要修改ARGO_System来接受这些参数
    # 这里先用默认参数创建，然后手动修改组件参数
    system = ARGO_System(
        model,
        tokenizer,
        use_mdp=True,
        retriever_mode='mock',
        max_steps=6,
        verbose=False
    )
    
    # 手动优化参数
    system.decomposer.max_subquery_length = 50  # 从128减少到50
    system.synthesizer.max_answer_length = 200  # 从512减少到200
    
    # 降低temperature以加速
    system.decomposer.temperature = 0.5  # 从0.7降低
    system.synthesizer.temperature = 0.2  # 从0.3降低
    
    print("✅ System ready with optimized parameters:")
    print(f"  - Decomposer max_tokens: 50 (原128)")
    print(f"  - Synthesizer max_tokens: 200 (原512)")
    print(f"  - Decomposer temperature: 0.5 (原0.7)")
    print(f"  - Synthesizer temperature: 0.2 (原0.3)")
    
    # 创建性能分析器
    profiler = LatencyProfiler(system)
    
    # 测试问题（与原始测试相同）
    test_questions = [
        "What are the latency requirements for O-RAN fronthaul?",
        "How does O-RAN handle network slicing?",
        "What is the role of RIC in O-RAN architecture?",
    ]
    
    print("\n" + "="*80)
    print("Running optimized latency measurement...")
    print("="*80)
    
    try:
        # 测量延迟
        df = profiler.measure_batch(test_questions, verbose=True)
        
        # 分析结果
        print("\n" + "="*80)
        print("Optimization Results")
        print("="*80)
        
        # 与原始结果对比
        original_avg = 55573.2  # ms (从之前的测试)
        optimized_avg = df['total_latency_ms'].mean()
        speedup = original_avg / optimized_avg
        
        print(f"\n对比原始版本:")
        print(f"  Original (3B, max_tokens=128/512): {original_avg:.1f}ms")
        print(f"  Optimized ({model_name.split('/')[-1]}, max_tokens=50/200): {optimized_avg:.1f}ms")
        print(f"  加速比: {speedup:.2f}x")
        print(f"  延迟降低: {(1 - optimized_avg/original_avg)*100:.1f}%")
        
        if speedup >= 2.5:
            print(f"\n✅ 优化成功！达到预期加速 ({speedup:.2f}x >= 2.5x)")
        elif speedup >= 2.0:
            print(f"\n⚠️  优化有效但低于预期 ({speedup:.2f}x, 预期3x)")
        else:
            print(f"\n❌ 优化效果不理想 ({speedup:.2f}x < 2x)")
        
        # 检查是否更接近O-RAN要求
        pass_rate = (df['total_latency_ms'] <= 1000).sum() / len(df) * 100
        original_pass_rate = 0.0
        
        print(f"\nO-RAN要求 (≤1000ms):")
        print(f"  Original: {original_pass_rate:.1f}% passed")
        print(f"  Optimized: {pass_rate:.1f}% passed")
        
        if pass_rate > original_pass_rate:
            print(f"  ✅ 改进 (+{pass_rate - original_pass_rate:.1f}%)")
        
        # 保存结果
        output_dir = "results/latency_optimized"
        os.makedirs(output_dir, exist_ok=True)
        
        df.to_csv(os.path.join(output_dir, 'latency_optimized.csv'), index=False)
        print(f"\n✅ Saved results to {output_dir}/latency_optimized.csv")
        
        # 组件分析
        profiler.analyze(df, output_dir=output_dir)
        profiler.visualize(df, output_dir=output_dir)
        
        # 保存对比报告
        comparison = {
            'metric': ['Model', 'Decomposer max_tokens', 'Synthesizer max_tokens', 
                      'Avg Latency (ms)', 'Speedup', 'Pass Rate (%)'],
            'original': ['Qwen2.5-3B', '128', '512', f'{original_avg:.1f}', '1.00x', f'{original_pass_rate:.1f}'],
            'optimized': [model_name.split('/')[-1], '50', '200', f'{optimized_avg:.1f}', 
                         f'{speedup:.2f}x', f'{pass_rate:.1f}']
        }
        
        comparison_df = pd.DataFrame(comparison)
        comparison_df.to_csv(os.path.join(output_dir, 'optimization_comparison.csv'), index=False)
        
        print("\n" + "="*80)
        print("ZERO-COST OPTIMIZATION COMPLETED! ✅")
        print("="*80)
        print(f"\n下一步建议:")
        
        if optimized_avg > 15000:  # 仍然 > 15秒
            print("  1. 安装Flash Attention 2 (预期1.5-2倍加速)")
            print("     pip install flash-attn --no-build-isolation")
            print("  2. 安装vLLM (预期2-5倍加速)")
            print("     pip install vllm")
        elif optimized_avg > 5000:  # 5-15秒
            print("  1. 考虑安装vLLM以进一步加速到<5秒")
            print("  2. 或直接进行Phase 4.3实验（当前速度可接受）")
        else:  # < 5秒
            print("  ✅ 性能已足够优秀，可以直接进行Phase 4.3完整实验！")
        
    except Exception as e:
        logger.error(f"Measurement failed: {e}", exc_info=True)
        print(f"\n❌ Measurement failed: {e}")


if __name__ == "__main__":
    main()
