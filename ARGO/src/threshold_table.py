"""
Pre-computed MDP Threshold Lookup Table for Question-Adaptive ARGO

Solves Problem 1: Threshold Scaling Breaks Optimality

Instead of linear scaling (theta* = base_theta* * question_umax), we pre-compute
optimal thresholds for discrete U_max values offline, ensuring:
1. O(1) runtime lookup as claimed in paper
2. Preservation of MDP optimality guarantees (Theorem 1)
3. Correct parameter ratio relationships

Theory:
    The optimal threshold theta* depends on the ratio relationships:
    theta* = f(delta_r/U_max, delta_p/U_max, c_r, c_p, p_s, mu, gamma)
    
    Linear scaling violates these ratios. This module pre-computes
    correct thresholds for each U_max configuration.

Author: ARGO Team
Version: 1.0
"""

from __future__ import annotations

import copy
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Import MDP Solver
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../ARGO_MDP/src'))
try:
    from mdp_solver import MDPSolver
    MDP_SOLVER_AVAILABLE = True
except ImportError:
    MDP_SOLVER_AVAILABLE = False
    logger.warning("MDPSolver not available for threshold computation")


class ThresholdTable:
    """
    Pre-computed optimal thresholds for different U_max configurations.
    
    Theory Alignment:
    - Each U_max value has independently computed optimal thresholds
    - Preserves Theorem 1 guarantees for each configuration
    - O(1) runtime lookup as claimed in paper
    
    Usage:
        table = ThresholdTable(mdp_config, cache_path="configs/threshold_cache.json")
        theta_cont, theta_star, actual_umax = table.lookup(question_umax=0.65)
    """
    
    # Default U_max buckets aligned with complexity levels
    DEFAULT_UMAX_BUCKETS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    
    def __init__(
        self,
        mdp_base_config: Dict,
        umax_buckets: Optional[List[float]] = None,
        cache_path: Optional[str] = None,
        force_recompute: bool = False
    ):
        """
        Initialize threshold table.
        
        Args:
            mdp_base_config: Base MDP configuration dictionary
            umax_buckets: List of U_max values to pre-compute (default: 8 values)
            cache_path: Path to cache computed thresholds (JSON)
            force_recompute: If True, recompute even if cache exists
        """
        self.base_config = mdp_base_config
        self.umax_buckets = sorted(umax_buckets or self.DEFAULT_UMAX_BUCKETS)
        self.cache_path = Path(cache_path) if cache_path else None
        
        # Threshold table: {U_max: {theta_cont, theta_star, ...}}
        self.table: Dict[float, Dict] = {}
        
        # Cost-aware cache for experiments with varying c_r
        self._cost_cache: Dict[str, Dict] = {}
        
        # Load from cache or compute
        if not force_recompute and self.cache_path and self.cache_path.exists():
            self._load_cache()
            if not self._validate_cache():
                logger.warning("Cache mismatch, recomputing thresholds...")
                self._compute_all_thresholds()
                if self.cache_path:
                    self._save_cache()
        else:
            self._compute_all_thresholds()
            if self.cache_path:
                self._save_cache()
    
    def _compute_all_thresholds(self) -> None:
        """Compute optimal thresholds for all U_max buckets via value iteration."""
        if not MDP_SOLVER_AVAILABLE:
            logger.error("MDPSolver not available, using fallback thresholds")
            self._use_fallback_thresholds()
            return
        
        logger.info(f"Computing thresholds for {len(self.umax_buckets)} U_max values...")
        
        for umax in self.umax_buckets:
            config = self._make_config_for_umax(umax)
            
            try:
                solver = MDPSolver(config)
                result = solver.solve()
                
                self.table[umax] = {
                    'theta_star': float(result['theta_star']),
                    'theta_cont': float(result['theta_cont']),
                    'expected_value_at_0': float(solver.V[0]) if hasattr(solver, 'V') else None,
                    'converged': True
                }
                
                logger.info(
                    f"  U_max={umax:.2f}: theta_cont={result['theta_cont']:.4f}, "
                    f"theta*={result['theta_star']:.4f}"
                )
            except Exception as e:
                logger.error(f"Failed to compute for U_max={umax}: {e}")
                self.table[umax] = {
                    'theta_star': 0.75 * umax,
                    'theta_cont': 0.35 * umax,
                    'expected_value_at_0': None,
                    'converged': False
                }
        
        logger.info("Threshold table computation complete.")
    
    def _use_fallback_thresholds(self) -> None:
        """Use heuristic thresholds when MDP solver unavailable."""
        for umax in self.umax_buckets:
            self.table[umax] = {
                'theta_star': 0.75 * umax,
                'theta_cont': 0.35 * umax,
                'expected_value_at_0': None,
                'converged': False
            }
        logger.warning("Using fallback (heuristic) thresholds")
    
    def _make_config_for_umax(self, umax: float) -> Dict:
        """Create MDP config with specific U_max."""
        config = copy.deepcopy(self.base_config)
        config['mdp']['U_max'] = umax
        return config
    
    def lookup(
        self, 
        question_umax: float,
        strategy: str = 'ceiling'
    ) -> Tuple[float, float, float]:
        """
        O(1) threshold lookup with bucket mapping.
        
        Args:
            question_umax: Estimated U_max for the question
            strategy: Bucket selection strategy
                - 'nearest': Use closest bucket
                - 'ceiling': Use next higher bucket (conservative)
                - 'floor': Use next lower bucket (aggressive)
            
        Returns:
            (theta_cont, theta_star, actual_umax)
        """
        question_umax = max(min(self.umax_buckets), min(max(self.umax_buckets), question_umax))
        
        if strategy == 'nearest':
            actual_umax = min(self.umax_buckets, key=lambda x: abs(x - question_umax))
        elif strategy == 'ceiling':
            candidates = [u for u in self.umax_buckets if u >= question_umax]
            actual_umax = min(candidates) if candidates else max(self.umax_buckets)
        elif strategy == 'floor':
            candidates = [u for u in self.umax_buckets if u <= question_umax]
            actual_umax = max(candidates) if candidates else min(self.umax_buckets)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        entry = self.table[actual_umax]
        return entry['theta_cont'], entry['theta_star'], actual_umax
    
    def lookup_interpolated(self, question_umax: float) -> Tuple[float, float, float]:
        """Interpolated threshold lookup for smoother transitions."""
        umax_clamped = max(min(self.umax_buckets), min(max(self.umax_buckets), question_umax))
        
        if umax_clamped in self.umax_buckets:
            entry = self.table[umax_clamped]
            return entry['theta_cont'], entry['theta_star'], umax_clamped
        
        lower_buckets = [u for u in self.umax_buckets if u < umax_clamped]
        upper_buckets = [u for u in self.umax_buckets if u > umax_clamped]
        
        if not lower_buckets:
            entry = self.table[min(self.umax_buckets)]
            return entry['theta_cont'], entry['theta_star'], min(self.umax_buckets)
        if not upper_buckets:
            entry = self.table[max(self.umax_buckets)]
            return entry['theta_cont'], entry['theta_star'], max(self.umax_buckets)
        
        lower = max(lower_buckets)
        upper = min(upper_buckets)
        
        alpha = (umax_clamped - lower) / (upper - lower)
        
        theta_cont = (1 - alpha) * self.table[lower]['theta_cont'] + alpha * self.table[upper]['theta_cont']
        theta_star = (1 - alpha) * self.table[lower]['theta_star'] + alpha * self.table[upper]['theta_star']
        
        return theta_cont, theta_star, umax_clamped
    
    def lookup_with_cost(
        self, 
        question_umax: float, 
        c_r: float,
        recompute_if_missing: bool = True
    ) -> Tuple[float, float, float]:
        """Lookup with cost-aware threshold adjustment for experiments."""
        actual_umax = min(self.umax_buckets, key=lambda x: abs(x - question_umax))
        cache_key = f"{actual_umax:.2f}_{c_r:.4f}"
        
        if cache_key not in self._cost_cache:
            if not recompute_if_missing or not MDP_SOLVER_AVAILABLE:
                entry = self.table[actual_umax]
                return entry['theta_cont'], entry['theta_star'], actual_umax
            
            config = self._make_config_for_umax(actual_umax)
            config['mdp']['c_r'] = c_r
            
            solver = MDPSolver(config)
            result = solver.solve()
            
            self._cost_cache[cache_key] = {
                'theta_cont': float(result['theta_cont']),
                'theta_star': float(result['theta_star']),
                'actual_umax': actual_umax
            }
            
            logger.debug(f"Computed cost-aware thresholds for {cache_key}")
        
        entry = self._cost_cache[cache_key]
        return entry['theta_cont'], entry['theta_star'], entry['actual_umax']
    
    def _validate_cache(self) -> bool:
        """Check if cached table matches current configuration including c_r.
        
        CRITICAL: Must validate both buckets AND config hash to prevent
        using stale thresholds when c_r changes during cost sweep experiments.
        """
        if not self.table:
            return False
        
        # Check bucket match
        cached_buckets = set(self.table.keys())
        expected_buckets = set(self.umax_buckets)
        if cached_buckets != expected_buckets:
            return False
        
        # Check config hash (includes c_r, c_p, delta_r, delta_p, etc.)
        if not self.cache_path or not self.cache_path.exists():
            return False
            
        try:
            with open(self.cache_path, 'r') as f:
                data = json.load(f)
            cached_hash = data.get('base_config_hash')
            if cached_hash is None:
                return False  # Old cache format without hash
            current_hash = hash(json.dumps(self.base_config, sort_keys=True, default=str))
            if cached_hash != current_hash:
                logger.info(f"Cache config hash mismatch (c_r or other param changed)")
                return False
            return True
        except Exception as e:
            logger.warning(f"Cache validation error: {e}")
            return False
    
    def _load_cache(self) -> None:
        """Load threshold table from JSON cache."""
        try:
            with open(self.cache_path, 'r') as f:
                data = json.load(f)
            self.table = {float(k): v for k, v in data.get('table', {}).items()}
            if 'umax_buckets' in data:
                self.umax_buckets = sorted(data['umax_buckets'])
            logger.info(f"Loaded threshold table from {self.cache_path}")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            self.table = {}
    
    def _save_cache(self) -> None:
        """Save threshold table to JSON cache."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'table': {str(k): v for k, v in self.table.items()},
                'umax_buckets': self.umax_buckets,
                'base_config_hash': hash(json.dumps(self.base_config, sort_keys=True, default=str))
            }
            with open(self.cache_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved threshold table to {self.cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def get_statistics(self) -> Dict:
        """Return table statistics for logging/debugging."""
        if not self.table:
            return {'error': 'Table is empty'}
        
        theta_stars = [v['theta_star'] for v in self.table.values()]
        theta_conts = [v['theta_cont'] for v in self.table.values()]
        converged = [v.get('converged', True) for v in self.table.values()]
        
        return {
            'num_buckets': len(self.umax_buckets),
            'umax_range': (min(self.umax_buckets), max(self.umax_buckets)),
            'theta_star_range': (min(theta_stars), max(theta_stars)),
            'theta_cont_range': (min(theta_conts), max(theta_conts)),
            'all_converged': all(converged),
            'cost_cache_size': len(self._cost_cache)
        }
    

    def clear_cost_cache(self) -> None:
        """Clear the cost-aware cache. Call this when c_r changes in experiments."""
        old_size = len(self._cost_cache)
        self._cost_cache.clear()
        if old_size > 0:
            logger.info(f"Cleared cost cache ({old_size} entries)")
    
    def invalidate_for_new_cost(self, new_c_r: float) -> None:
        """Invalidate cached thresholds and recompute for new c_r value.
        
        IMPORTANT: Call this at the start of each cost sweep iteration
        to ensure thresholds are computed with the correct c_r.
        """
        # Update base config with new c_r
        self.base_config['mdp']['c_r'] = new_c_r
        
        # Clear all caches
        self._cost_cache.clear()
        self.table.clear()
        
        # Recompute thresholds with new c_r
        self._compute_all_thresholds()
        
        logger.info(f"Recomputed thresholds for c_r={new_c_r:.4f}")

    def print_table(self) -> None:
        """Pretty-print the threshold table."""
        print("\n" + "="*60)
        print("Pre-computed Threshold Table")
        print("="*60)
        print(f"{'U_max':>8} | {'theta_cont':>10} | {'theta*':>10} | {'theta*/U_max':>12} | {'Conv':>5}")
        print("-"*60)
        for umax in sorted(self.table.keys()):
            entry = self.table[umax]
            ratio = entry['theta_star'] / umax if umax > 0 else 0
            conv = "Y" if entry.get('converged', True) else "N"
            print(f"{umax:>8.2f} | {entry['theta_cont']:>10.4f} | {entry['theta_star']:>10.4f} | {ratio:>12.4f} | {conv:>5}")
        print("="*60 + "\n")


def precompute_threshold_grid(
    base_config: Dict,
    umax_values: List[float],
    cr_values: List[float],
    output_path: str
) -> Dict:
    """
    Pre-compute thresholds for (U_max, c_r) parameter grid.
    
    Run this ONCE before experiments to create complete lookup table.
    """
    if not MDP_SOLVER_AVAILABLE:
        raise RuntimeError("MDPSolver required for grid computation")
    
    results = {}
    total = len(umax_values) * len(cr_values)
    count = 0
    
    print(f"Pre-computing {total} threshold configurations...")
    
    for umax in umax_values:
        for c_r in cr_values:
            count += 1
            
            config = copy.deepcopy(base_config)
            config['mdp']['U_max'] = umax
            config['mdp']['c_r'] = c_r
            
            solver = MDPSolver(config)
            result = solver.solve()
            
            key = f"{umax:.2f}_{c_r:.4f}"
            results[key] = {
                'theta_cont': float(result['theta_cont']),
                'theta_star': float(result['theta_star']),
                'umax': umax,
                'c_r': c_r
            }
            
            if count % 10 == 0 or count == total:
                print(f"  Progress: {count}/{total} ({100*count/total:.1f}%)")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'grid': results,
            'umax_values': umax_values,
            'cr_values': cr_values,
            'base_config': base_config
        }, f, indent=2)
    
    print(f"\nSaved {len(results)} configurations to {output_path}")
    return results


# Convenience function
_default_table: Optional[ThresholdTable] = None


def get_default_table(mdp_config: Optional[Dict] = None) -> ThresholdTable:
    """Get or create the default threshold table (singleton)."""
    global _default_table
    
    if _default_table is None:
        if mdp_config is None:
            mdp_config = {
                'mdp': {
                    'U_max': 1.0,
                    'delta_r': 0.25,
                    'delta_p': 0.08,
                    'c_r': 0.05,
                    'c_p': 0.02,
                    'p_s': 0.8,
                    'mu': 0.0,
                    'gamma': 0.98,
                    'U_grid_size': 1000
                },
                'quality': {'mode': 'linear', 'k': 1.0},
                'reward_shaping': {'enabled': False, 'k': 0.0},
                'solver': {
                    'max_iterations': 1000,
                    'convergence_threshold': 1e-6,
                    'verbose': False
                }
            }
        
        cache_dir = Path(__file__).parent.parent / 'configs'
        _default_table = ThresholdTable(
            mdp_base_config=mdp_config,
            cache_path=str(cache_dir / 'threshold_cache.json')
        )
    
    return _default_table
