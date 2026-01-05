#!/usr/bin/env python
"""Complexity-stratified adaptive depth experiment for ARGO.

This script samples O-RAN benchmark questions by predicted complexity,
configures ARGO with the new adaptive termination policy, and reports
step counts, retrieval/ reasoning breakdowns, and statistical tests
showing whether reasoning depth tracks question complexity.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from oran_benchmark_loader import ORANBenchmark
from src import ARGO_System
from src.complexity_v2 import ORANComplexityClassifier, ComplexityProfile
# Legacy import for backward compatibility
# from src.complexity import QuestionComplexityClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_question(item: Dict) -> str:
    lines = [item['question'].strip(), ""]
    for idx, option in enumerate(item['options'], 1):
        lines.append(f"{idx}. {option}")
    lines.append("\nAnswer (1-4):")
    return "\n".join(lines)


def extract_choice(text: str) -> str:
    import re

    patterns = [
        r"answer is\s*['\"]?(\d)['\"]?",
        r"correct answer is\s*['\"]?(\d)['\"]?",
        r"选择\s*(\d)",
        r"答案是\s*(\d)",
        r"^\s*(\d)\s*[\).]",
        r"\((\d)\)"
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)

    for char in text:
        if char in {'1', '2', '3', '4'}:
            return char
    return "N/A"


def stratified_questions(
    benchmark: ORANBenchmark,
    classifier: ORANComplexityClassifier,
    counts: Dict[str, int],
    seed: int
) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    rng = random.Random(seed)
    grouped = {'simple': [], 'medium': [], 'complex': []}

    for diff, entries in benchmark.questions.items():
        for item in entries:
            enriched = item.copy()
            enriched['difficulty'] = diff
            profile = classifier.classify(enriched['question'], metadata={'difficulty': diff})
            label = profile.label if hasattr(profile, 'label') else profile
            enriched['argo_complexity'] = label
            enriched['argo_umax'] = profile.umax if hasattr(profile, 'umax') else None
            grouped[label].append(enriched)

    selected: List[Dict] = []
    for label, target in counts.items():
        pool = grouped.get(label, [])
        if len(pool) < target:
            raise ValueError(f"Need {target} {label} questions but only have {len(pool)}")
        selected.extend(rng.sample(pool, target))

    rng.shuffle(selected)
    return selected, grouped


def build_policy_config(max_steps: int, theory_aligned: bool = True) -> Dict:
    """
    Build policy configuration.
    
    Args:
        max_steps: Base maximum steps
        theory_aligned: If True, use theory-aligned settings (Problem 1, 2, 3 fixes)
    """
    config = {
        'theta_star': 0.75,  # Fallback (ThresholdTable preferred)
        'theta_cont': 0.40,  # Fallback (ThresholdTable preferred)
        'max_steps': max_steps,
        'hard_cap_steps': max_steps + 2,
        'theta_star_by_complexity': {
            'simple': 0.75,
            'medium': 0.80,
            'complex': 0.85
        },
        'theta_cont_by_complexity': {
            'simple': 0.35,
            'medium': 0.45,
            'complex': 0.50
        },
        'max_steps_by_complexity': {
            'simple': max(3, max_steps - 4),
            'medium': max_steps,
            'complex': max_steps + 2
        },
        # Use V2 classifier (ORANComplexityClassifier)
        'classifier_version': 'v2',
        # U_max buckets for ThresholdTable alignment
        'umax_buckets': [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
    }
    
    if theory_aligned:
        # Theory-aligned progress tracking (Problem 2 fix)
        config['progress'] = {
            'enabled': True,
            'mode': 'fixed',  # Use FixedProgressTracker (Eq.2 exact)
            'delta_r': 0.25,  # Fixed retrieval gain
            'delta_p': 0.08,  # Fixed reasoning gain
            # Legacy fields (ignored in 'fixed' mode)
            'base_retrieval_gain': 0.25,
            'base_reason_gain': 0.08,
            'coverage_weight': 0.0,   # Disabled - violates Assumption 1
            'novelty_weight': 0.0,    # Disabled - violates Assumption 1
            'confidence_weight': 0.0, # Disabled - violates Assumption 1
            'min_gain': 0.08,
            'max_gain': 0.25,
        }
    else:
        # Legacy dynamic progress (may violate Assumption 1)
        config['progress'] = {
            'enabled': True,
            'mode': 'dynamic',
            'base_retrieval_gain': 0.30,
            'base_reason_gain': 0.10,
            'coverage_weight': 0.55,
            'novelty_weight': 0.30,
            'confidence_weight': 0.45,
            'min_gain': 0.02,
            'max_gain': 0.35,
            'gain_multipliers': {
                'simple': 1.2,
                'medium': 1.0,
                'complex': 0.85
            }
        }
    
    return config


def analyze_results(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for label in ['simple', 'medium', 'complex']:
        subset = df[df['complexity'] == label]
        if subset.empty:
            continue
        summary[label] = {
            'n': float(len(subset)),
            'steps_mean': float(subset['total_steps'].mean()),
            'steps_std': float(subset['total_steps'].std(ddof=1) if len(subset) > 1 else 0.0),
            'retrievals_mean': float(subset['retrieve_count'].mean()),
            'reasons_mean': float(subset['reason_count'].mean()),
            'accuracy_mean': float(subset['accuracy'].mean()),
            'final_U_mean': float(subset['final_uncertainty'].mean())
        }

    groups = [df[df['complexity'] == label]['total_steps'] for label in ['simple', 'medium', 'complex']]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) >= 2:
        f_stat, p_val = f_oneway(*groups)
        summary['anova'] = {'F': float(f_stat), 'p': float(p_val)}
    else:
        summary['anova'] = {'F': None, 'p': None}

    return summary


def maybe_save_plot(df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    order = ['simple', 'medium', 'complex']
    aggregates = df.groupby('complexity')['total_steps'].mean().reindex(order)
    fig, ax = plt.subplots(figsize=(6, 4))
    aggregates.plot(kind='bar', ax=ax, color=['#6baed6', '#fd8d3c', '#31a354'])
    ax.set_ylabel('Average Steps')
    ax.set_xlabel('Complexity')
    ax.set_title('ARGO Steps vs. Question Complexity')
    ax.set_ylim(0, max(aggregates.max() + 1, 5))
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_experiment(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    benchmark = ORANBenchmark(benchmark_dir=args.benchmark_dir, use_cleaned=not args.use_raw_benchmark)
    classifier = ORANComplexityClassifier()
    counts = {
        'simple': args.num_simple,
        'medium': args.num_medium,
        'complex': args.num_complex
    }
    selected_questions, grouped = stratified_questions(benchmark, classifier, counts, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map='auto' if args.device == 'auto' else None
    )
    if args.device != 'auto':
        model.to(args.device)
    model.eval()

    policy_config = build_policy_config(args.max_steps, theory_aligned=not args.legacy_progress)

    argo = ARGO_System(
        model=model,
        tokenizer=tokenizer,
        use_mdp=not args.disable_mdp,
        mdp_config=None,
        policy_config=policy_config,
        retriever_mode=args.retriever_mode,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection_name,
        max_steps=args.max_steps,
        verbose=args.verbose
    )

    records: List[Dict] = []
    failures = 0

    progress_bar = tqdm(selected_questions, desc="Adaptive Depth", disable=args.verbose)
    for item in progress_bar:
        formatted_question = format_question(item)
        try:
            answer, choice, history, metadata = argo.answer_question(
                formatted_question,
                return_history=True,
                options=item['options']
            )
            predicted = choice or extract_choice(answer)
            is_correct = str(predicted) == str(item['correct_answer'])
            record = {
                'question_id': item['id'],
                'difficulty': item.get('difficulty'),
                'complexity': metadata.get('complexity'),
                'total_steps': metadata['total_steps'],
                'retrieve_count': metadata['retrieve_count'],
                'reason_count': metadata['reason_count'],
                'accuracy': 1.0 if is_correct else 0.0,
                'prediction': str(predicted),
                'ground_truth': str(item['correct_answer']),
                'final_uncertainty': metadata['final_uncertainty'],
                'theta_star': metadata.get('theta_star'),
                'theta_cont': metadata.get('theta_cont'),
                'max_steps_cap': metadata.get('max_steps_cap'),
                'terminated_early': metadata.get('terminated_early'),
                'step_cap_hit': metadata.get('step_cap_hit'),
                'progress_mode': metadata.get('progress_mode'),
                'question_text': item['question'],
                'answer_text': answer[:200],
                'error': None
            }
            records.append(record)
        except Exception as exc:
            failures += 1
            record = {
                'question_id': item['id'],
                'difficulty': item.get('difficulty'),
                'complexity': item.get('argo_complexity'),
                'error': str(exc)
            }
            records.append(record)
            if args.verbose:
                print(f"Failed question {item['id']}: {exc}")

    df = pd.DataFrame(records)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / 'adaptive_depth_results.csv'
    df.to_csv(results_path, index=False)

    json_path = output_dir / 'adaptive_depth_results.json'
    df.to_json(json_path, orient='records', force_ascii=False, indent=2)

    summary = analyze_results(df[df['error'].isna()]) if 'error' in df.columns else analyze_results(df)
    summary_path = output_dir / 'adaptive_depth_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("Complexity-Stratified Results")
    print("=" * 80)
    for label in ['simple', 'medium', 'complex']:
        stats = summary.get(label)
        if not stats:
            continue
        print(
            f"\n{label.upper()} (n={int(stats['n'])}):\n"
            f"  Steps: {stats['steps_mean']:.2f} ± {stats['steps_std']:.2f}\n"
            f"  Retrievals: {stats['retrievals_mean']:.2f}\n"
            f"  Reasons: {stats['reasons_mean']:.2f}\n"
            f"  Accuracy: {stats['accuracy_mean']:.3f}\n"
            f"  Final U_t: {stats['final_U_mean']:.3f}"
        )

    if summary.get('anova', {}).get('F') is not None:
        print(
            f"\nANOVA: F={summary['anova']['F']:.2f}, p={summary['anova']['p']:.4f}"
        )
        if summary['anova']['p'] < 0.05:
            print("✓ Step count significantly varies with complexity")
        else:
            print("✗ No significant difference detected")

    if args.plot_file:
        plot_path = output_dir / args.plot_file
        maybe_save_plot(df[df['error'].isna()], plot_path)
        print(f"\nSaved plot to {plot_path}")

    if failures:
        print(f"\nWarnings: {failures} questions failed during inference")

    print(f"\nDetailed results: {results_path}")
    print(f"Summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ARGO adaptive depth experiment")
    parser.add_argument('--model-name', required=True, help='HF model name or local path')
    parser.add_argument('--benchmark-dir', default='data/benchmark/ORAN-Bench-13K/Benchmark', help='Benchmark directory')
    parser.add_argument('--num-simple', type=int, default=30)
    parser.add_argument('--num-medium', type=int, default=40)
    parser.add_argument('--num-complex', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-steps', type=int, default=10)
    parser.add_argument('--retriever-mode', choices=['mock', 'chroma'], default='mock')
    parser.add_argument('--chroma-dir', default=None)
    parser.add_argument('--collection-name', default='oran_specs')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--disable-mdp', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--output-dir', default='results/complexity_adaptive')
    parser.add_argument('--plot-file', default='steps_by_complexity.png')
    parser.add_argument('--use-raw-benchmark', action='store_true', help='Use raw fin_H.json instead of cleaned set')
    parser.add_argument('--legacy-progress', action='store_true', 
                       help='Use legacy dynamic progress (may violate Assumption 1)')
    return parser.parse_args()


if __name__ == '__main__':
    run_experiment(parse_args())
