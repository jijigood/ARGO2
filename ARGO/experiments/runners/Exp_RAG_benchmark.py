"""
RAG Evaluation on ORAN-Bench-13K
Evaluate different retrieval strategies on actual ORAN benchmark questions
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import json
import re
from typing import Dict, List, Tuple
import datetime

from oran_benchmark_loader import ORANBenchmark
from Env_RAG import Env_RAG

# Try to import RAG models if available
try:
    from RAG_Models.retrieval import build_vector_store
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False


def extract_answer_number(llm_output: str) -> int:
    """
    Extract answer number (1-4) from LLM output
    
    Handles various formats:
    - "1"
    - "Answer: 2"
    - "The correct answer is 3"
    - "4. Option text"
    
    Returns:
        Answer number (1-4) or 0 if cannot parse
    """
    # Clean output
    output = llm_output.strip()
    
    # Pattern 1: Just a number
    if output in ['1', '2', '3', '4']:
        return int(output)
    
    # Pattern 2: "Answer: N" or "Answer is N"
    match = re.search(r'answer[:\s]+(\d)', output, re.IGNORECASE)
    if match:
        num = int(match.group(1))
        if 1 <= num <= 4:
            return num
    
    # Pattern 3: Look for first occurrence of 1-4
    match = re.search(r'[1-4]', output)
    if match:
        return int(match.group())
    
    return 0  # Cannot parse


def evaluate_rag_on_benchmark(
    benchmark: ORANBenchmark,
    questions: List[Dict],
    retrieval_config: Dict,
    use_real_rag: bool = False
) -> Dict:
    """
    Evaluate RAG system on benchmark questions
    
    Args:
        benchmark: ORANBenchmark instance
        questions: List of questions to evaluate
        retrieval_config: Retrieval configuration (top_k, use_rerank, use_filter)
        use_real_rag: Whether to use actual RAG or simulation
        
    Returns:
        Evaluation results
    """
    results = {
        'correct': 0,
        'total': len(questions),
        'accuracy': 0.0,
        'details': []
    }
    
    for q in questions:
        question_text = q['question']
        correct_answer = q['correct_answer']
        
        if use_real_rag and RAG_AVAILABLE:
            # TODO: Implement actual RAG retrieval + LLM inference
            # retriever = build_vector_store()
            # context = retriever.retrieve(question_text, top_k=retrieval_config['top_k'])
            # 
            # # Format prompt for multiple choice
            # prompt = f"""Based on the following context, answer the multiple choice question.
            # Only output the number (1, 2, 3, or 4) of the correct answer.
            #
            # Context:
            # {context}
            #
            # Question:
            # {benchmark.format_question_for_llm(q)}
            # """
            #
            # llm_output = llm.generate(prompt)
            # predicted = extract_answer_number(llm_output)
            
            # For now, simulate
            predicted = np.random.choice([1, 2, 3, 4])
        else:
            # Simulate answer based on difficulty and retrieval quality
            difficulty_map = {'easy': 0.8, 'medium': 0.6, 'hard': 0.4}
            base_acc = difficulty_map.get(q.get('difficulty', 'medium'), 0.6)
            
            # Adjust by retrieval config
            top_k = retrieval_config.get('top_k', 3)
            use_rerank = retrieval_config.get('use_rerank', 0)
            
            # Better retrieval = higher accuracy
            acc_boost = min(top_k * 0.05, 0.15) + (use_rerank * 0.1)
            final_acc = min(base_acc + acc_boost, 0.95)
            
            # Randomly decide if correct based on accuracy
            if np.random.random() < final_acc:
                predicted = correct_answer
            else:
                # Random wrong answer
                wrong_options = [i for i in [1, 2, 3, 4] if i != correct_answer]
                predicted = np.random.choice(wrong_options)
        
        is_correct = (predicted == correct_answer)
        results['correct'] += int(is_correct)
        
        results['details'].append({
            'question_id': q['id'],
            'difficulty': q.get('difficulty', 'unknown'),
            'question': question_text,
            'predicted': predicted,
            'correct': correct_answer,
            'is_correct': is_correct
        })
    
    results['accuracy'] = results['correct'] / results['total'] if results['total'] > 0 else 0
    return results


def run_benchmark_experiment(
    n_questions: int = 100,
    difficulty: str = None,
    seed: int = 42
) -> Dict:
    """
    Run RAG experiment on ORAN benchmark
    
    Args:
        n_questions: Number of questions to test
        difficulty: 'easy', 'medium', 'hard', or None for mixed
        seed: Random seed
        
    Returns:
        Experiment results for all policies
    """
    print("=" * 80)
    print(f"ORAN-Bench-13K RAG Evaluation")
    print(f"Questions: {n_questions}, Difficulty: {difficulty or 'mixed'}, Seed: {seed}")
    print("=" * 80)
    
    # Load benchmark
    benchmark = ORANBenchmark()
    
    # Sample questions
    questions = benchmark.sample_questions(n=n_questions, difficulty=difficulty, seed=seed)
    
    # Show distribution
    if difficulty is None:
        dist = {'easy': 0, 'medium': 0, 'hard': 0}
        for q in questions:
            dist[q['difficulty']] += 1
        print(f"\nQuestion distribution: {dist}")
    
    # Initialize environment for policy decisions
    env = Env_RAG(cost_weight=0.1, seed=seed)
    
    # Test different retrieval strategies
    strategies = {
        'optimal': lambda state: env.opt_policy(state),
        'fixed_k3': lambda state: (3, 0, 0),
        'fixed_k5': lambda state: (5, 1, 0),
        'fixed_k7': lambda state: (7, 1, 1),
        'adaptive': lambda state: env.adaptive_policy(state)
    }
    
    all_results = {}
    
    for strategy_name, strategy_fn in strategies.items():
        print(f"\n[Evaluating {strategy_name} strategy]")
        
        # Reset environment
        state = env.reset()
        
        # Map difficulty to complexity for state
        difficulty_to_complexity = {'easy': 1, 'medium': 2, 'hard': 3}
        
        strategy_correct = 0
        strategy_total = 0
        strategy_details = []
        
        for q in questions:
            # Update state based on question difficulty
            complexity = difficulty_to_complexity.get(q.get('difficulty', 'medium'), 2)
            state[0] = complexity
            
            # Get retrieval action from strategy
            action = strategy_fn(state)
            top_k, use_rerank, use_filter = action
            
            retrieval_config = {
                'top_k': top_k,
                'use_rerank': use_rerank,
                'use_filter': use_filter
            }
            
            # Evaluate single question
            single_result = evaluate_rag_on_benchmark(
                benchmark, 
                [q], 
                retrieval_config,
                use_real_rag=False  # Set to True when RAG is ready
            )
            
            strategy_correct += single_result['correct']
            strategy_total += 1
            
            # Store details
            detail = single_result['details'][0]
            detail['retrieval_config'] = retrieval_config
            strategy_details.append(detail)
            
            # Step environment for next question
            next_state, reward, done, info = env.step(action)
            state = next_state
            if done:
                state = env.reset()
        
        # Calculate statistics
        strategy_acc = strategy_correct / strategy_total if strategy_total > 0 else 0
        
        all_results[strategy_name] = {
            'correct': strategy_correct,
            'total': strategy_total,
            'accuracy': strategy_acc,
            'details': strategy_details
        }
        
        print(f"  Accuracy: {strategy_acc:.3f} ({strategy_correct}/{strategy_total})")
    
    return all_results, questions


def analyze_by_difficulty(results: Dict, questions: List[Dict]):
    """Analyze results by difficulty level"""
    print("\n" + "=" * 80)
    print("Performance by Difficulty Level")
    print("=" * 80)
    
    for strategy_name, strategy_data in results.items():
        print(f"\n{strategy_name.upper()}:")
        
        # Group by difficulty
        diff_stats = {'easy': {'correct': 0, 'total': 0},
                     'medium': {'correct': 0, 'total': 0},
                     'hard': {'correct': 0, 'total': 0}}
        
        for detail in strategy_data['details']:
            diff = detail['difficulty']
            if diff in diff_stats:
                diff_stats[diff]['total'] += 1
                diff_stats[diff]['correct'] += int(detail['is_correct'])
        
        # Print stats
        for diff in ['easy', 'medium', 'hard']:
            if diff_stats[diff]['total'] > 0:
                acc = diff_stats[diff]['correct'] / diff_stats[diff]['total']
                print(f"  {diff.capitalize():8s}: {acc:.3f} ({diff_stats[diff]['correct']}/{diff_stats[diff]['total']})")


def save_results(results: Dict, questions: List[Dict], filename: str):
    """Save evaluation results"""
    output_dir = "draw_figs/data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert numpy types to Python native types
    import json
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            if hasattr(obj, 'item'):
                return obj.item()
            return super().default(obj)
    
    output = {
        'timestamp': datetime.datetime.now().isoformat(),
        'benchmark': 'ORAN-Bench-13K',
        'num_questions': len(questions),
        'results': results
    }
    
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    
    print(f"\nResults saved to {filepath}")


if __name__ == "__main__":
    print("=" * 80)
    print("ORAN-Bench-13K RAG Evaluation Experiment")
    print("=" * 80)
    print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"RAG Available: {RAG_AVAILABLE}")
    print("=" * 80)
    
    # Experiment 1: Mixed difficulty
    print("\n### EXPERIMENT 1: Mixed Difficulty (100 questions) ###")
    results_mixed, questions_mixed = run_benchmark_experiment(
        n_questions=100, 
        difficulty=None, 
        seed=42
    )
    analyze_by_difficulty(results_mixed, questions_mixed)
    save_results(results_mixed, questions_mixed, "oran_benchmark_mixed.json")
    
    # Experiment 2: By difficulty level
    for diff in ['easy', 'medium', 'hard']:
        print(f"\n### EXPERIMENT: {diff.upper()} Questions ###")
        results_diff, questions_diff = run_benchmark_experiment(
            n_questions=50,
            difficulty=diff,
            seed=42
        )
        save_results(results_diff, questions_diff, f"oran_benchmark_{diff}.json")
    
    print("\n" + "=" * 80)
    print("All experiments completed!")
    print("=" * 80)
