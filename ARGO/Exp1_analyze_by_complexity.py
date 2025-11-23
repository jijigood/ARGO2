import json
import os
import glob
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_complexity_stratification(data_dir="draw_figs/data"):
    # 查找所有结果文件
    files = glob.glob(os.path.join(data_dir, "exp1_real_cost_impact_*.json"))
    if not files:
        print("未找到结果文件!")
        return

    print(f"找到 {len(files)} 个结果文件")
    
    all_data = []
    
    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
                
            raw_results = data.get('raw_results', [])
            if not raw_results:
                continue
                
            # 我们只取第一个c_r的结果，因为自适应策略不依赖c_r (或者我们假设它是一样的)
            # 如果自适应策略在所有c_r下都运行了，我们只需要一份
            # 但是要注意，如果policy_config被使用，c_r循环可能还在跑，但结果是一样的
            # 只要取第一个即可
            first_cr_results = raw_results[0]['details']['ARGO']
            
            for res in first_cr_results:
                # 确保有complexity字段 (旧的结果可能没有)
                if 'complexity' in res:
                    all_data.append({
                        'complexity': res['complexity'],
                        'steps': res['steps'],
                        'correct': res['correct'],
                        'retrievals': res['retrieval_count'],
                        'reasons': res['reason_count']
                    })
        except Exception as e:
            print(f"读取文件 {fpath} 失败: {e}")
    
    if not all_data:
        print("没有包含复杂度信息的数据!")
        return

    df = pd.DataFrame(all_data)
    
    # 排序: Simple, Medium, Complex
    order = ['simple', 'medium', 'complex']
    df['complexity'] = pd.Categorical(df['complexity'], categories=order, ordered=True)
    
    print("\n" + "="*60)
    print("按复杂度分层分析 (Stratification Analysis)")
    print("="*60)
    
    # 1. 描述性统计
    # 重命名列以便显示
    df_display = df.copy()
    df_display.columns = ['Complexity', 'Steps', 'Accuracy', 'Retrievals', 'Reasons']
    
    stats_df = df_display.groupby('Complexity').agg({
        'Steps': ['mean', 'std', 'count'],
        'Accuracy': 'mean',
        'Retrievals': 'mean',
        'Reasons': 'mean'
    })
    
    print("\n各复杂度级别的平均表现:")
    print(stats_df)
    
    # 2. 统计检验 (ANOVA)
    groups = [df[df['complexity'] == c]['steps'].values for c in order if len(df[df['complexity'] == c]) > 0]
    
    if len(groups) > 1:
        f_val, p_val = stats.f_oneway(*groups)
        print(f"\nANOVA (Steps ~ Complexity): F={f_val:.2f}, p={p_val:.4e}")
        if p_val < 0.05:
            print("✓ 步数在不同复杂度之间存在显著差异 (验证了分层假设)")
        else:
            print("✗ 未发现显著差异")
            
    # 3. 可视化
    os.makedirs('figs', exist_ok=True)
    
    # Boxplot for Steps
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='complexity', y='steps', data=df, order=order, palette="Set2")
    plt.title('Reasoning Steps by Question Complexity')
    plt.ylabel('Steps')
    plt.xlabel('Complexity')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('figs/complexity_stratification_steps.png', dpi=300)
    print("\n图表已保存至 figs/complexity_stratification_steps.png")
    
    # Barplot for Accuracy
    plt.figure(figsize=(10, 6))
    sns.barplot(x='complexity', y='correct', data=df, order=order, palette="Set2", errorbar=None)
    plt.title('Accuracy by Question Complexity')
    plt.ylabel('Accuracy')
    plt.xlabel('Complexity')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('figs/complexity_stratification_accuracy.png', dpi=300)
    print("图表已保存至 figs/complexity_stratification_accuracy.png")

if __name__ == "__main__":
    analyze_complexity_stratification()