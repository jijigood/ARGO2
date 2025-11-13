"""
Visualize ORAN-Bench-13K RAG Evaluation Results
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_results(filename):
    """Load results from JSON file"""
    filepath = os.path.join('draw_figs/data', filename)
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_strategy_comparison(results_data):
    """Plot accuracy comparison across strategies"""
    results = results_data['results']
    
    strategies = list(results.keys())
    accuracies = [results[s]['accuracy'] for s in strategies]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars
    x_pos = np.arange(len(strategies))
    bars = ax.bar(x_pos, accuracies, alpha=0.8, 
                   color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12'])
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Formatting
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_xlabel('Retrieval Strategy', fontsize=12, fontweight='bold')
    ax.set_title('RAG Performance on ORAN-Bench-13K\n(100 Mixed Difficulty Questions)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([s.replace('_', ' ').title() for s in strategies], 
                        rotation=15, ha='right')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('draw_figs/benchmark_strategy_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: draw_figs/benchmark_strategy_comparison.png")
    plt.close()


def plot_difficulty_breakdown(results_data):
    """Plot accuracy by difficulty level for each strategy"""
    results = results_data['results']
    
    # Collect data
    strategies = list(results.keys())
    difficulties = ['easy', 'medium', 'hard']
    
    data = {diff: [] for diff in difficulties}
    
    for strategy in strategies:
        details = results[strategy]['details']
        
        # Calculate accuracy per difficulty
        for diff in difficulties:
            diff_questions = [d for d in details if d['difficulty'] == diff]
            if diff_questions:
                correct = sum(1 for d in diff_questions if d['is_correct'])
                acc = correct / len(diff_questions)
            else:
                acc = 0
            data[diff].append(acc)
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(strategies))
    width = 0.25
    
    colors = {'easy': '#2ecc71', 'medium': '#f39c12', 'hard': '#e74c3c'}
    
    for i, diff in enumerate(difficulties):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, data[diff], width, label=diff.capitalize(), 
                      color=colors[diff], alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_xlabel('Retrieval Strategy', fontsize=12, fontweight='bold')
    ax.set_title('RAG Performance by Question Difficulty', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', ' ').title() for s in strategies], 
                        rotation=15, ha='right')
    ax.set_ylim([0, 1.1])
    ax.legend(title='Difficulty', loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('draw_figs/benchmark_difficulty_breakdown.png', dpi=300, bbox_inches='tight')
    print("Saved: draw_figs/benchmark_difficulty_breakdown.png")
    plt.close()


def plot_confusion_heatmap(results_data, strategy='fixed_k5'):
    """Plot confusion analysis for a specific strategy"""
    results = results_data['results'][strategy]
    details = results['details']
    
    # Create confusion matrix (predicted vs correct)
    confusion = np.zeros((4, 4), dtype=int)
    
    for d in details:
        pred = d['predicted'] - 1  # Convert 1-4 to 0-3
        correct = d['correct'] - 1
        confusion[correct, pred] += 1
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    
    im = ax.imshow(confusion, cmap='Blues', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', rotation=270, labelpad=20, fontweight='bold')
    
    # Set ticks
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels(['Option 1', 'Option 2', 'Option 3', 'Option 4'])
    ax.set_yticklabels(['Option 1', 'Option 2', 'Option 3', 'Option 4'])
    
    # Add text annotations
    for i in range(4):
        for j in range(4):
            text = ax.text(j, i, confusion[i, j],
                          ha="center", va="center", color="black" if confusion[i, j] < confusion.max()/2 else "white",
                          fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Predicted Answer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Correct Answer', fontsize=12, fontweight='bold')
    ax.set_title(f'Answer Distribution - {strategy.replace("_", " ").title()} Strategy\n'
                 f'(Accuracy: {results["accuracy"]:.3f})', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'draw_figs/benchmark_confusion_{strategy}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: draw_figs/benchmark_confusion_{strategy}.png")
    plt.close()


def plot_retrieval_config_impact(results_data):
    """Plot how retrieval configuration affects accuracy"""
    results = results_data['results']
    
    # Extract retrieval configs and accuracies
    configs = []
    accuracies = []
    
    for strategy in ['fixed_k3', 'fixed_k5', 'fixed_k7']:
        if strategy in results:
            # Get typical config from first question
            config = results[strategy]['details'][0]['retrieval_config']
            top_k = config['top_k']
            configs.append(f"k={top_k}")
            accuracies.append(results[strategy]['accuracy'])
    
    # Create line plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x_pos = np.arange(len(configs))
    ax.plot(x_pos, accuracies, marker='o', linewidth=2, markersize=10, 
            color='#3498db', label='Accuracy')
    ax.fill_between(x_pos, accuracies, alpha=0.3, color='#3498db')
    
    # Add value labels
    for i, (x, y) in enumerate(zip(x_pos, accuracies)):
        ax.text(x, y + 0.02, f'{y:.3f}', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_xlabel('Retrieval Configuration (Top-K)', fontsize=12, fontweight='bold')
    ax.set_title('Impact of Retrieval Depth on RAG Performance', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(configs)
    ax.set_ylim([0.6, 0.9])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('draw_figs/benchmark_retrieval_impact.png', dpi=300, bbox_inches='tight')
    print("Saved: draw_figs/benchmark_retrieval_impact.png")
    plt.close()


def generate_summary_report(results_data):
    """Generate text summary report"""
    results = results_data['results']
    
    report = []
    report.append("=" * 80)
    report.append("ORAN-Bench-13K RAG Evaluation Summary")
    report.append("=" * 80)
    report.append(f"Timestamp: {results_data['timestamp']}")
    report.append(f"Questions: {results_data['num_questions']}")
    report.append("")
    
    # Overall rankings
    report.append("Strategy Rankings (by Overall Accuracy):")
    report.append("-" * 80)
    
    sorted_strategies = sorted(results.items(), 
                              key=lambda x: x[1]['accuracy'], 
                              reverse=True)
    
    for rank, (strategy, data) in enumerate(sorted_strategies, 1):
        report.append(f"{rank}. {strategy.upper():20s} - "
                     f"Accuracy: {data['accuracy']:.3f} "
                     f"({data['correct']}/{data['total']})")
    
    report.append("")
    report.append("=" * 80)
    
    # Save report
    report_text = "\n".join(report)
    with open('draw_figs/benchmark_summary.txt', 'w') as f:
        f.write(report_text)
    
    print("\nSaved: draw_figs/benchmark_summary.txt")
    print(report_text)


if __name__ == "__main__":
    print("Loading ORAN-Bench-13K evaluation results...")
    
    # Load results
    results_data = load_results('oran_benchmark_mixed.json')
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_strategy_comparison(results_data)
    plot_difficulty_breakdown(results_data)
    plot_confusion_heatmap(results_data, strategy='fixed_k5')
    plot_retrieval_config_impact(results_data)
    
    # Generate summary
    generate_summary_report(results_data)
    
    print("\n" + "=" * 80)
    print("All visualizations completed!")
    print("=" * 80)
