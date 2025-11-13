"""
Ultra-Small Experiment - 10 Queries
===================================

快速验证实验流程（10个问题）
"""

import os
import sys

# 修改run_small_scale_experiment.py的配置
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入并修改配置
from run_small_scale_experiment import *

# 覆盖主函数
def main_ultra_small():
    """主函数 - 超小规模版本"""
    
    print("="*80)
    print("ARGO Phase 4.3 - Ultra-Small Experiment (10 queries)")
    print("="*80)
    
    # 配置
    N_QUESTIONS = 10  # 超小规模
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    OUTPUT_DIR = "results/phase4.3_ultra_small"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 加载数据
    print("\n1. Loading ORAN-Bench-13K...")
    loader = ORANBenchLoader()
    questions = loader.sample(n=N_QUESTIONS, difficulties=['E', 'M', 'H'], random_seed=42)
    
    # 2. 加载模型
    print(f"\n2. Loading model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print("✅ Model loaded")
    
    # 3. 只测试2种策略（加快速度）
    print("\n3. Initializing strategies (2 strategies only)...")
    
    strategies = {}
    
    # MDP-Guided
    argo_system = ARGO_System(
        model, tokenizer,
        use_mdp=True,
        retriever_mode='mock',
        max_steps=4,  # 减少步数
        verbose=False
    )
    argo_system.decomposer.max_subquery_length = 50
    argo_system.synthesizer.max_answer_length = 150  # 进一步减少
    argo_system.decomposer.temperature = 0.3  # 降低温度加速
    argo_system.synthesizer.temperature = 0.1
    
    strategies['MDP_Guided'] = argo_system
    
    # Always Reason (对比基线)
    always_system = AlwaysReasonStrategy(
        model, tokenizer,
        retriever_mode='mock',
        max_steps=4,
        verbose=False
    )
    always_system.decomposer.max_subquery_length = 50
    always_system.synthesizer.max_answer_length = 150
    always_system.decomposer.temperature = 0.3
    always_system.synthesizer.temperature = 0.1
    
    strategies['Always_Reason'] = always_system
    
    print(f"✅ Created {len(strategies)} strategies")
    
    # 4. 运行实验
    print("\n4. Running ultra-small experiments...")
    all_results = {}
    
    for strategy_name, system in strategies.items():
        try:
            df = run_experiment(strategy_name, system, questions, OUTPUT_DIR)
            all_results[strategy_name] = df
        except Exception as e:
            logger.error(f"Failed to run {strategy_name}: {e}", exc_info=True)
            print(f"❌ {strategy_name} failed: {e}")
    
    # 5. 分析结果
    if len(all_results) > 0:
        print("\n5. Analyzing results...")
        analyze_results(all_results, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("ULTRA-SMALL EXPERIMENT COMPLETED! ✅")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main_ultra_small()
