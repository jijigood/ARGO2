#!/usr/bin/env python
"""
Experiment: Real Cost Impact Analysis (v2 - Question-Adaptive)
===============================================================

Evaluates ARGO system performance with question-adaptive termination.

Features:
  ✓ Question-adaptive U_max estimation
  ✓ Complexity-based threshold scaling
  ✓ Dynamic progress tracking
  ✓ Multi-seed support for statistical validity
  ✓ Multi-GPU support for parallel processing

Usage:
    # Single run (test)
    python Exp_real_cost_impact_v2.py --mode custom --n-questions 10 --difficulty easy --seed 42 --verbose
    
    # Full run (one seed, one difficulty)
    python Exp_real_cost_impact_v2.py --mode custom --n-questions 100 --difficulty hard --seed 42
    
    # With custom policy config
    python Exp_real_cost_impact_v2.py --policy-config-path configs/my_adaptive_policy.yaml

Author: ARGO Team
Version: 2.1 (Question-Adaptive)
"""

import sys
import os
from pathlib import Path
import argparse
import yaml
import json
import random
import math
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from copy import deepcopy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ARGO.src.argo_system import ARGO_System
except ModuleNotFoundError:
    from src.argo_system import ARGO_System

from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"✓ Random seed set to: {seed}")


def setup_gpu(gpus: str = "0,1,2,3,4,5,6,7") -> int:
    """Configure GPU visibility"""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"✓ Using {n_gpus} GPUs: {gpus}")
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        return n_gpus
    print("⚠ No GPU available, using CPU")
    return 0


def load_model_and_tokenizer(
    model_path: str = "Qwen/Qwen2.5-7B-Instruct",
    device_map: str = "auto",
    cache_dir: Optional[str] = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load LLM model and tokenizer"""
    print(f"\nLoading model: {model_path}")
    print(f"Device map: {device_map}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        trust_remote_code=True,
        use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        cache_dir=cache_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    print("✓ Model loaded successfully")
    print(f"  Model dtype: {model.dtype}")
    print(f"  Model device: {next(model.parameters()).device}")
    return model, tokenizer


def _load_json_benchmark(file_path: Path, difficulty_label: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(file_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON line in {file_path}: {line[:80]}") from exc
            if not isinstance(item, list) or len(item) < 3:
                raise ValueError(f"Unexpected record format in {file_path}: {item}")
            question = item[0]
            options = item[1] if isinstance(item[1], list) else None
            answer = item[2]
            rows.append({
                'question': question,
                'options': options,
                'answer': str(answer).strip(),
                'difficulty': difficulty_label
            })
    return rows


def load_dataset(
    dataset_path: Path,
    mode: str = "custom",
    n_questions: int = 100,
    difficulty: str = "all",
    seed: int = 42
) -> pd.DataFrame:
    """Load and filter ORAN-Bench dataset"""
    print(f"\nLoading dataset from: {dataset_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    def normalize_diff(value: str) -> str:
        mapping = {
            'easy': 'easy', 'e': 'easy',
            'medium': 'medium', 'm': 'medium',
            'hard': 'hard', 'h': 'hard'
        }
        return mapping.get(value.lower(), value.lower())

    normalized_diff = normalize_diff(difficulty)
    rows: List[Dict] = []

    if dataset_path.is_dir():
        suffix_map = {
            'easy': 'fin_E.json',
            'medium': 'fin_M.json',
            'hard': 'fin_H.json'
        }
        target_difficulties = [normalized_diff] if normalized_diff in suffix_map else list(suffix_map.keys())
        for diff in target_difficulties:
            file_name = suffix_map[diff]
            file_path = dataset_path / file_name
            if not file_path.exists():
                raise FileNotFoundError(f"Missing dataset file: {file_path}")
            rows.extend(_load_json_benchmark(file_path, diff))
    elif dataset_path.suffix.lower() == '.json':
        rows.extend(_load_json_benchmark(dataset_path, normalized_diff if normalized_diff in {'easy','medium','hard'} else 'mixed'))
    else:
        df = pd.read_csv(dataset_path)
        for _, row in df.iterrows():
            rows.append({
                'question': row.get('question'),
                'options': row.get('options'),
                'answer': str(row.get('answer')).strip() if row.get('answer') is not None else None,
                'difficulty': row.get('difficulty', normalized_diff)
            })

    dataset = pd.DataFrame(rows)
    print(f"✓ Loaded {len(dataset)} questions")

    if normalized_diff in {'easy', 'medium', 'hard'}:
        dataset = dataset[dataset['difficulty'] == normalized_diff].reset_index(drop=True)
        print(f"  Filtered to {normalized_diff}: {len(dataset)} questions")
    else:
        print("  Using all difficulty levels")

    if mode == "custom" and n_questions < len(dataset):
        dataset = dataset.sample(n=n_questions, random_state=seed).reset_index(drop=True)
        print(f"  Sampled {n_questions} questions (mode: {mode})")

    required_cols = ['question', 'answer']
    missing = [col for col in required_cols if col not in dataset.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    def parse_options(opts):
        if isinstance(opts, list):
            return opts
        if pd.isna(opts):
            return None
        if isinstance(opts, str):
            try:
                value = json.loads(opts)
                if isinstance(value, list):
                    return value
            except json.JSONDecodeError:
                if '|' in opts:
                    return [seg.strip() for seg in opts.split('|') if seg.strip()]
        return None

    dataset['options'] = dataset['options'].apply(parse_options)
    dataset['dataset_difficulty'] = dataset['difficulty']
    print(f"✓ Dataset prepared: {len(dataset)} questions")
    return dataset


def load_configs(
    config_path: Path,
    policy_config_path: Optional[Path]
) -> Tuple[Dict, Optional[Dict]]:
    """Load MDP and policy configurations"""
    print("\nLoading configurations:")
    if not config_path.exists():
        raise FileNotFoundError(f"MDP config not found: {config_path}")
    print(f"  MDP config: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as fh:
        mdp_config = yaml.safe_load(fh)
    print("  ✓ MDP config loaded")

    policy_config = None
    if policy_config_path:
        print(f"  Policy config: {policy_config_path}")
        if policy_config_path.exists():
            with open(policy_config_path, 'r', encoding='utf-8') as fh:
                policy_config = yaml.safe_load(fh)
            print("  ✓ Policy config loaded (question-adaptive enabled)")
        else:
            print(f"  ⚠ Warning: Policy config not found: {policy_config_path}")
            print("    Using default settings (no question-adaptive features)")
    else:
        print("  No policy config specified (using defaults)")
    return mdp_config, policy_config


def _parse_float_list(raw_value: Optional[str]) -> Optional[List[float]]:
    """Parse comma separated floats, returning None when empty"""
    if not raw_value:
        return None
    values: List[float] = []
    for chunk in raw_value.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            values.append(float(chunk))
        except ValueError:
            raise ValueError(f"Invalid float value in list: '{chunk}'") from None
    return values or None


def _build_cost_schedule(
    mdp_config: Dict,
    explicit_values: Optional[List[float]],
    multipliers: Optional[List[float]],
    base_cost_override: Optional[float]
) -> List[float]:
    """Determine which c_r values to evaluate"""
    if explicit_values:
        return [float(v) for v in explicit_values]

    base_cost = base_cost_override
    if base_cost is None:
        base_cost = mdp_config['mdp'].get('c_p')
    if base_cost is None:
        base_cost = mdp_config['mdp'].get('c_r')
    if base_cost is None:
        base_cost = 0.02  # final fallback

    if multipliers:
        return [base_cost * float(m) for m in multipliers]

    return [mdp_config['mdp'].get('c_r', base_cost)]


def run_experiment(
    argo_system: ARGO_System,
    dataset: pd.DataFrame,
    verbose: bool = False,
    cost_params: Optional[Dict[str, float]] = None
) -> List[Dict]:
    """Run ARGO on all questions and collect results"""
    print(f"\n{'='*80}")
    print("Running Experiment")
    print(f"{'='*80}")
    print(f"Total questions: {len(dataset)}")
    print(f"Verbose mode: {verbose}")
    print(f"{'='*80}\n")

    results: List[Dict] = []
    pbar = tqdm(dataset.iterrows(), total=len(dataset), desc="Processing questions")

    for idx, row in pbar:
        question = row['question']
        options = row.get('options', None)
        ground_truth = row.get('answer')
        metadata_difficulty = row.get('dataset_difficulty', None)
        preview = question[:50] + "..." if isinstance(question, str) and len(question) > 50 else question
        pbar.set_description(f"Q{idx+1}: {preview}")

        try:
            # Build question metadata with difficulty hint
            q_metadata = {'difficulty': metadata_difficulty} if metadata_difficulty else None
            
            answer, choice, history, metadata = argo_system.answer_question(
                question,
                return_history=True,
                options=options,
                question_metadata=q_metadata
            )
            metadata = metadata or {}
            if choice is not None and ground_truth is not None:
                correct = str(choice).strip() == str(ground_truth).strip()
            else:
                correct = None

            retrieve_count = metadata.get('retrieve_count')
            reason_count = metadata.get('reason_count')
            estimated_cost = None
            if cost_params and retrieve_count is not None and reason_count is not None:
                c_r_value = cost_params.get('c_r')
                c_p_value = cost_params.get('c_p')
                if c_r_value is not None and c_p_value is not None:
                    estimated_cost = retrieve_count * c_r_value + reason_count * c_p_value

            result = {
                'question_id': idx,
                'question': question,
                'ground_truth': ground_truth,
                'dataset_difficulty': metadata_difficulty,
                'predicted_choice': choice,
                'predicted_answer': answer,
                'correct': correct,
                'total_steps': metadata.get('total_steps'),
                'retrieve_count': retrieve_count,
                'reason_count': reason_count,
                'successful_retrievals': metadata.get('successful_retrievals'),
                'elapsed_time': metadata.get('elapsed_time'),
                'final_progress': metadata.get('final_progress', metadata.get('final_uncertainty')),
                'complexity': metadata.get('complexity', 'unknown'),
                'question_umax': metadata.get('question_umax', 1.0),
                'progress_efficiency': metadata.get('progress_efficiency'),
                'terminated_early': metadata.get('terminated_early', False),
                'step_cap_hit': metadata.get('step_cap_hit', False),
                'theta_star_base': metadata.get('theta_star_base'),
                'theta_star_scaled': metadata.get('theta_star'),
                'theta_cont_base': metadata.get('theta_cont_base'),
                'theta_cont_scaled': metadata.get('theta_cont'),
                'max_steps_cap': metadata.get('max_steps_cap'),
                'num_sources': len(metadata.get('sources', [])) if metadata.get('sources') else 0,
                'c_r': cost_params.get('c_r') if cost_params else None,
                'c_r_multiplier': (
                    cost_params.get('c_r') / cost_params.get('c_p')
                    if cost_params and cost_params.get('c_p')
                    else None
                ),
                'estimated_cost': estimated_cost
            }
            results.append(result)

            valid_results = [r for r in results if r.get('correct') is not None]
            if valid_results:
                avg_steps = np.nanmean([r.get('total_steps', np.nan) for r in valid_results])
                accuracy = np.nanmean([1.0 if r['correct'] else 0.0 for r in valid_results])
                pbar.set_postfix({
                    'Acc': f"{accuracy:.2%}",
                    'AvgSteps': f"{avg_steps:.1f}"
                })
        except Exception as exc:
            print(f"\n❌ Error processing question {idx}: {exc}")
            print(f"   Question: {question[:100]}...")
            result = {
                'question_id': idx,
                'question': question,
                'ground_truth': ground_truth,
                'error': str(exc),
                'correct': None,
            }
            results.append(result)
            continue

    print(f"\n✓ Experiment completed: {len(results)} questions processed")
    return results


def save_results(
    results: List[Dict],
    output_dir: Path,
    difficulty: str = "all",
    seed: int = 42,
    adaptive: bool = True,
    c_r: Optional[float] = None
) -> Path:
    """Save results to CSV file"""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    difficulty_str = difficulty if difficulty != 'all' else 'mixed'
    adaptive_tag = '_adaptive' if adaptive else '_baseline'
    cost_suffix = f"_cr{c_r:.3f}" if c_r is not None else ""
    n_questions = len(results)
    filename = (
        f"exp1_results_{difficulty_str}_seed{seed}_n{n_questions}{cost_suffix}{adaptive_tag}_{timestamp}.csv"
    )
    output_file = output_dir / filename
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    return output_file


def save_summary_json(
    summary_records: List[Dict],
    args: argparse.Namespace,
    output_dir: Path,
    timestamp: Optional[str] = None
) -> Optional[Path]:
    """Persist aggregated summaries for downstream analysis"""
    if not summary_records:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        'metadata': {
            'test_mode': args.mode,
            'n_questions': args.n_questions,
            'difficulty': args.difficulty,
            'seed': args.seed,
            'n_cost_steps': len(summary_records),
            'timestamp': ts
        },
        'results': []
    }

    def _clean(value):
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
        return value

    for record in summary_records:
        entry = {
            'c_r': record.get('c_r'),
            'c_r_multiplier': record.get('c_r_multiplier'),
            'theta_cont': record.get('theta_cont'),
            'theta_star': record.get('theta_star'),
            'ARGO_retrievals': record.get('ARGO_retrievals'),
            'ARGO_reasons': record.get('ARGO_reasons'),
            'ARGO_steps': record.get('ARGO_steps'),
            'ARGO_time': record.get('ARGO_time'),
            'ARGO_quality': record.get('ARGO_quality'),
            'ARGO_accuracy': record.get('ARGO_accuracy'),
            'ARGO_cost': record.get('ARGO_cost')
        }
        entry = {key: _clean(value) for key, value in entry.items()}
        payload['results'].append(entry)

    summary_path = output_dir / f"exp1_real_cost_impact_custom_{ts}.json"
    with open(summary_path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2)
    print(f"✓ Cost summary saved to: {summary_path}")
    return summary_path


def _safe_mean(series: pd.Series) -> float:
    clean = series.dropna()
    return float(clean.mean()) if not clean.empty else float('nan')


def summarize_run(
    results: List[Dict],
    c_r: Optional[float],
    c_p: Optional[float]
) -> Dict:
    """Compute aggregate metrics for a single cost point"""
    if not results:
        return {
            'c_r': c_r,
            'c_p': c_p,
            'n_questions': 0
        }

    df = pd.DataFrame(results)

    def _maybe_mean(column: str) -> float:
        if column not in df:
            return float('nan')
        return _safe_mean(df[column])

    accuracy = float('nan')
    if 'correct' in df and df['correct'].notna().any():
        accuracy = float(df['correct'].dropna().mean())

    summary = {
        'c_r': c_r,
        'c_p': c_p,
        'c_r_multiplier': (c_r / c_p) if (c_r is not None and c_p) else None,
        'n_questions': len(df),
        'ARGO_retrievals': _maybe_mean('retrieve_count'),
        'ARGO_reasons': _maybe_mean('reason_count'),
        'ARGO_steps': _maybe_mean('total_steps'),
        'ARGO_time': _maybe_mean('elapsed_time'),
        'ARGO_quality': _maybe_mean('final_progress'),
        'ARGO_accuracy': accuracy,
        'ARGO_cost': _maybe_mean('estimated_cost') if 'estimated_cost' in df else float('nan')
    }
    return summary


def print_summary(results: List[Dict], policy_config: Optional[Dict] = None):
    """Print comprehensive experiment summary"""
    df = pd.DataFrame(results)
    if df.empty:
        print("\n⚠ No results to summarize")
        return

    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    print(f"\nBasic Statistics:")
    print(f"  Total Questions: {len(df)}")

    if 'correct' in df:
        valid_correct = df['correct'].dropna()
        if not valid_correct.empty:
            accuracy = valid_correct.mean()
            print(f"  Accuracy: {accuracy:.1%} ({int(valid_correct.sum())}/{len(valid_correct)})")
        else:
            print("  Accuracy: N/A (no ground truth available)")

    if 'total_steps' in df:
        print("\nPerformance Metrics:")
        print(f"  Avg Steps: {_safe_mean(df['total_steps']):.2f}")
        print(f"  Avg Retrieval Calls: {_safe_mean(df['retrieve_count']):.2f}")
        print(f"  Avg Reason Calls: {_safe_mean(df['reason_count']):.2f}")
        print(f"  Avg Time: {_safe_mean(df['elapsed_time']):.2f}s")
        total_time = df['elapsed_time'].dropna().sum()
        print(f"  Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")

    if policy_config and 'complexity' in df:
        print("\nQuestion-Adaptive Metrics:")
        if 'question_umax' in df:
            print(f"  Avg U_max: {_safe_mean(df['question_umax']):.3f}")
        if 'progress_efficiency' in df:
            print(f"  Progress Efficiency: {_safe_mean(df['progress_efficiency']):.3f} progress/step")
        if 'terminated_early' in df:
            print(f"  Early Termination Rate: {df['terminated_early'].mean():.1%}")
        if 'step_cap_hit' in df:
            print(f"  Step Cap Hit Rate: {df['step_cap_hit'].mean():.1%}")

        for comp in ['simple', 'medium', 'complex']:
            subset = df[df['complexity'] == comp]
            if subset.empty:
                continue
            subset_correct = subset['correct'].dropna()
            acc = subset_correct.mean() if not subset_correct.empty else None
            print(f"\n  {comp.upper()}:")
            print(f"    Count: {len(subset)} ({len(subset)/len(df):.1%})")
            if 'question_umax' in subset:
                print(f"    Avg U_max: {_safe_mean(subset['question_umax']):.3f}")
            if 'theta_star_scaled' in subset:
                print(f"    Avg θ* (scaled): {_safe_mean(subset['theta_star_scaled']):.3f}")
            if 'total_steps' in subset:
                print(f"    Avg Steps: {_safe_mean(subset['total_steps']):.2f}")
            if 'elapsed_time' in subset:
                print(f"    Avg Time: {_safe_mean(subset['elapsed_time']):.2f}s")
            if acc is not None:
                print(f"    Accuracy: {acc:.1%}")
            if 'terminated_early' in subset:
                print(f"    Early Term: {subset['terminated_early'].mean():.1%}")
            if 'retrieve_count' in subset:
                print(f"    Retrieve: {_safe_mean(subset['retrieve_count']):.2f} calls")
            if 'reason_count' in subset:
                print(f"    Reason: {_safe_mean(subset['reason_count']):.2f} calls")

        simple_subset = df[df['complexity'] == 'simple']
        complex_subset = df[df['complexity'] == 'complex']
        if not simple_subset.empty and not complex_subset.empty and 'total_steps' in df:
            simple_steps = _safe_mean(simple_subset['total_steps'])
            complex_steps = _safe_mean(complex_subset['total_steps'])
            if simple_steps > 0 and complex_steps > 0:
                step_ratio = complex_steps / simple_steps
                print(f"\nEfficiency Analysis:")
                print(f"  Simple vs Complex step ratio: {step_ratio:.2f}x")
                if simple_steps < 4.0:
                    print(f"  ✓ Simple questions efficiently handled ({simple_steps:.1f} steps avg)")
                else:
                    print(f"  ⚠ Simple questions may need tuning ({simple_steps:.1f} steps avg)")

    if 'dataset_difficulty' in df and df['dataset_difficulty'].notna().any():
        print("\nDataset Difficulty Distribution:")
        for diff in df['dataset_difficulty'].dropna().unique():
            subset = df[df['dataset_difficulty'] == diff]
            print(f"  {diff}:")
            if 'total_steps' in subset:
                print(f"    Avg Steps: {_safe_mean(subset['total_steps']):.2f}")
            subset_correct = subset['correct'].dropna()
            if not subset_correct.empty:
                print(f"    Accuracy: {subset_correct.mean():.1%}")

    print("\n" + "="*80)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='ARGO Experiment: Real Cost Impact Analysis (Question-Adaptive)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test run (10 questions, verbose)
  python Exp_real_cost_impact_v2.py --mode custom --n-questions 10 --difficulty easy --seed 42 --verbose
  
  # Standard run (100 questions, one difficulty)
  python Exp_real_cost_impact_v2.py --mode custom --n-questions 100 --difficulty hard --seed 42
  
  # Full dataset (all difficulties)
  python Exp_real_cost_impact_v2.py --mode all --difficulty all --seed 42
  
  # With custom configs
  python Exp_real_cost_impact_v2.py --config-path configs/my_mdp.yaml --policy-config-path configs/my_policy.yaml
        """
    )

    parser.add_argument('--dataset-path', type=str,
                        default='data/benchmark/ORAN-Bench-13K/Benchmark',
                        help='Path to ORAN-Bench dataset directory or file')
    parser.add_argument('--mode', type=str,
                        choices=['custom', 'all'],
                        default='custom',
                        help='Dataset mode: custom (sample) or all (full)')
    parser.add_argument('--n-questions', type=int,
                        default=100,
                        help='Number of questions to sample (custom mode)')
    parser.add_argument('--difficulty', type=str,
                        choices=['easy', 'medium', 'hard', 'all', 'e', 'm', 'h'],
                        default='all',
                        help='Question difficulty filter')

    parser.add_argument('--model-path', type=str,
                        default='Qwen/Qwen2.5-7B-Instruct',
                        help='Path or name of LLM model')
    parser.add_argument('--cache-dir', type=str,
                        default=None,
                        help='Cache directory for model downloads')

    parser.add_argument('--config-path', type=str,
                        default='configs/multi_gpu_data_calibrated.yaml',
                        help='Path to MDP configuration file')
    parser.add_argument('--policy-config-path', type=str,
                        default='configs/adaptive_policy.yaml',
                        help='Path to policy configuration (enables question-adaptive features)')
    parser.add_argument('--c-r-values', type=str,
                        default=None,
                        help='Comma-separated retrieval cost values to evaluate (overrides config c_r)')
    parser.add_argument('--c-r-multipliers', type=str,
                        default=None,
                        help='Comma-separated multipliers applied to base retrieval cost (defaults to c_p)')
    parser.add_argument('--base-c-r', type=float,
                        default=None,
                        help='Base retrieval cost used when applying multipliers (defaults to c_p, then c_r)')

    parser.add_argument('--chroma-dir', type=str,
                        default='/data/user/huangxiaolin/ARGO2/Environments/chroma_store_v2',
                        help='Path to Chroma vector database')
    parser.add_argument('--collection-name', type=str,
                        default='oran_specs_semantic',
                        help='Chroma collection name')

    parser.add_argument('--gpus', type=str,
                        default='0,1,2,3,4,5,6,7',
                        help='Comma-separated GPU IDs')
    parser.add_argument('--device-map', type=str,
                        default='auto',
                        help='Device map for model loading')

    parser.add_argument('--seed', type=int,
                        default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output-dir', type=str,
                        default='draw_figs/data',
                        help='Output directory for results')
    parser.add_argument('--summary-dir', type=str,
                        default='analysis/figures/draw_figs/data',
                        help='Directory for aggregated cost summaries (JSON)')
    parser.add_argument('--skip-summary', action='store_true',
                        help='Do not persist aggregated per-cost summaries')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output to see question-adaptive behavior')
    return parser.parse_args()


def main() -> int:
    """Main experiment function"""
    args = parse_args()

    print("="*80)
    print("ARGO EXPERIMENT: Real Cost Impact Analysis")
    print("Version: 2.1 (Question-Adaptive)")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    set_seed(args.seed)
    setup_gpu(args.gpus)

    config_path = Path(args.config_path)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    policy_config_path = Path(args.policy_config_path) if args.policy_config_path else None
    if policy_config_path and not policy_config_path.is_absolute():
        policy_config_path = (PROJECT_ROOT / policy_config_path).resolve()
    mdp_config, policy_config = load_configs(config_path, policy_config_path)

    model, tokenizer = load_model_and_tokenizer(
        model_path=args.model_path,
        device_map=args.device_map,
        cache_dir=args.cache_dir
    )

    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = (PROJECT_ROOT / dataset_path).resolve()
    dataset = load_dataset(
        dataset_path=dataset_path,
        mode=args.mode,
        n_questions=args.n_questions,
        difficulty=args.difficulty,
        seed=args.seed
    )

    chroma_dir = Path(args.chroma_dir)
    if not chroma_dir.is_absolute():
        chroma_dir = (PROJECT_ROOT / chroma_dir).resolve()
    retriever_mode = "chroma"
    if not chroma_dir.exists():
        print(f"⚠ Chroma directory not found: {chroma_dir}. Falling back to mock retriever.")
        retriever_mode = "mock"

    cost_values = _build_cost_schedule(
        mdp_config=mdp_config,
        explicit_values=_parse_float_list(args.c_r_values),
        multipliers=_parse_float_list(args.c_r_multipliers),
        base_cost_override=args.base_c_r
    )
    c_p_value = mdp_config['mdp'].get('c_p')

    if len(cost_values) == 1:
        print(f"\n✓ Evaluating single retrieval cost: c_r = {cost_values[0]:.4f}")
    else:
        print(f"\n✓ Evaluating {len(cost_values)} retrieval cost points")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    summary_dir = Path(args.summary_dir)
    if not summary_dir.is_absolute():
        summary_dir = (PROJECT_ROOT / summary_dir).resolve()

    summary_records: List[Dict] = []
    csv_files: List[Path] = []

    for idx, cost_value in enumerate(cost_values, start=1):
        cost_value = float(cost_value)
        multiplier = None
        if c_p_value:
            multiplier = cost_value / c_p_value
        print("\n" + "-"*80)
        if multiplier:
            print(f"Cost point {idx}/{len(cost_values)}: c_r = {cost_value:.4f} ({multiplier:.2f}× c_p)")
        else:
            print(f"Cost point {idx}/{len(cost_values)}: c_r = {cost_value:.4f}")
        print("-"*80)

        run_mdp_config = deepcopy(mdp_config)
        run_mdp_config.setdefault('mdp', {})['c_r'] = cost_value

        argo_system = ARGO_System(
            model=model,
            tokenizer=tokenizer,
            use_mdp=True,
            mdp_config=run_mdp_config,
            policy_config=policy_config,
            retriever_mode=retriever_mode,
            chroma_dir=str(chroma_dir),
            collection_name=args.collection_name,
            max_steps=10,
            verbose=args.verbose
        )

        theta_cont = getattr(argo_system.mdp_solver, 'theta_cont', None) if argo_system.mdp_solver else None
        theta_star = getattr(argo_system.mdp_solver, 'theta_star', None) if argo_system.mdp_solver else None
        if theta_cont is not None and theta_star is not None:
            print(f"  θ_cont = {theta_cont:.4f}")
            print(f"  θ*      = {theta_star:.4f}")

        if policy_config:
            print("  ✓ Question-adaptive features enabled")
        else:
            print("  ⚠ Using baseline settings (no question-adaptive features)")

        cost_context = {'c_r': cost_value, 'c_p': run_mdp_config['mdp'].get('c_p')}
        results = run_experiment(
            argo_system=argo_system,
            dataset=dataset,
            verbose=args.verbose,
            cost_params=cost_context
        )

        results_file = save_results(
            results=results,
            output_dir=output_dir,
            difficulty=args.difficulty,
            seed=args.seed,
            adaptive=policy_config is not None,
            c_r=cost_value
        )
        csv_files.append(results_file)

        summary = summarize_run(results, cost_value, cost_context.get('c_p'))
        summary['theta_cont'] = theta_cont
        summary['theta_star'] = theta_star
        summary['results_file'] = str(results_file)
        summary_records.append(summary)

        print(f"  Avg retrievals/question: {summary.get('ARGO_retrievals', float('nan')):.2f}")
        print(f"  Avg reasons/question:    {summary.get('ARGO_reasons', float('nan')):.2f}")
        if not np.isnan(summary.get('ARGO_accuracy', float('nan'))):
            print(f"  Accuracy:                {summary['ARGO_accuracy']:.2%}")
        avg_cost = summary.get('ARGO_cost')
        if avg_cost == avg_cost:
            print(f"  Avg cost/question:       {avg_cost:.3f}")

        print(f"\n>>> Detailed metrics for c_r = {cost_value:.4f}")
        print_summary(results, policy_config)

        del argo_system

    if not args.skip_summary:
        save_summary_json(summary_records, args, summary_dir)

    if len(summary_records) > 1:
        print("\nRetrieval cost sweep summary:")
        for record in summary_records:
            c_r_value = record.get('c_r')
            if c_r_value is None:
                continue
            multiplier = record.get('c_r_multiplier')
            multiplier_text = f" ({multiplier:.2f}× c_p)" if multiplier else ""
            accuracy_value = record.get('ARGO_accuracy', float('nan'))
            if accuracy_value == accuracy_value:
                accuracy_text = f"{accuracy_value:.2%}"
            else:
                accuracy_text = "N/A"
            retrieval_value = record.get('ARGO_retrievals', float('nan'))
            print(
                f"  c_r={c_r_value:.4f}{multiplier_text} -> "
                f"retrievals={retrieval_value:.2f}, accuracy={accuracy_text}"
            )

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Result files:")
    for path in csv_files:
        print(f"  - {path}")
    print("="*80)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠ Experiment interrupted by user")
        sys.exit(1)
    except Exception as exc:
        print(f"\n\n❌ Experiment failed with error: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
