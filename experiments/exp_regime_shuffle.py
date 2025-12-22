#!/usr/bin/env python3
"""
Regime Shuffle Test - Negative Control.

This experiment validates that our regime detection is meaningful,
not just random noise. If shuffling regime labels causes SI to drop
to near-random levels (0.25), it proves:
1. Detected regimes are meaningful
2. Agents genuinely learn to specialize in specific regimes

Expected Results:
- Original SI: ~0.65-0.80
- Shuffled SI: ~0.25-0.35 (near random)
- p-value: < 0.001
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from lambda_zero experiment
from exp_lambda_zero_real import (
    PredictionPopulation,
    get_domain_predictor,
    load_domain_data,
    bootstrap_ci,
    cohens_d
)


def run_shuffle_experiment(domain: str, shuffle: bool, n_trials: int = 30,
                           n_iterations: int = 400, n_agents: int = 8,
                           seed: int = 42) -> Dict:
    """
    Run experiment with optional regime shuffling.

    Args:
        domain: Domain name
        shuffle: If True, randomly shuffle regime labels
        n_trials: Number of trials
        n_iterations: Iterations per trial
        n_agents: Agents in population
        seed: Random seed
    """
    values, regimes, methods = load_domain_data(domain)

    si_values = []

    for trial in tqdm(range(n_trials), desc=f"{domain} ({'shuffled' if shuffle else 'original'})", leave=False):
        trial_seed = seed + trial * 100 + (1000 if shuffle else 0)
        np.random.seed(trial_seed)

        # Shuffle regimes if requested
        if shuffle:
            regimes_to_use = np.random.permutation(regimes).tolist()
        else:
            regimes_to_use = regimes

        # Create population
        unique_regimes = list(set(regimes))
        pop = PredictionPopulation(
            n_agents=n_agents,
            methods=methods,
            regimes=unique_regimes,
            niche_bonus_lambda=0.5,  # Standard setting
            seed=trial_seed + 500
        )

        # Run iterations
        for it in range(min(n_iterations, len(values) - 20)):
            idx = it + 20
            regime = regimes_to_use[idx]  # Use potentially shuffled regime
            true_val = values[idx]
            history = values[:idx]

            def predict_fn(method):
                return get_domain_predictor(domain, method, history, idx)

            pop.run_iteration(regime, true_val, predict_fn)

        si_values.append(pop.get_population_si())

    si_arr = np.array(si_values)
    si_ci = bootstrap_ci(si_arr)

    return {
        "domain": domain,
        "shuffled": shuffle,
        "n_trials": n_trials,
        "si_mean": float(si_arr.mean()),
        "si_std": float(si_arr.std()),
        "si_ci_lower": float(si_ci[0]),
        "si_ci_upper": float(si_ci[1]),
        "si_all": [float(x) for x in si_values]
    }


def run_all_shuffle_tests(n_trials: int = 30):
    """Run shuffle test on all domains."""

    domains = ["energy", "weather", "finance"]

    results = {
        "experiment": "regime_shuffle_test",
        "date": datetime.now().isoformat(),
        "config": {
            "n_trials": n_trials,
            "n_shuffle_seeds": 10
        },
        "results": {}
    }

    for domain in domains:
        print(f"\n{'='*60}")
        print(f"Testing {domain.upper()}")
        print(f"{'='*60}")

        # Original (no shuffle)
        original = run_shuffle_experiment(domain, shuffle=False, n_trials=n_trials)

        # Shuffled (10 different random shuffles, average)
        shuffled_results = []
        for shuffle_seed in range(10):
            shuffled = run_shuffle_experiment(
                domain, shuffle=True, n_trials=n_trials // 3,  # Fewer trials per shuffle
                seed=42 + shuffle_seed * 10000
            )
            shuffled_results.extend(shuffled["si_all"])

        shuffled_arr = np.array(shuffled_results)
        shuffled_ci = bootstrap_ci(shuffled_arr)

        # Statistical test: original SI > shuffled SI
        t_stat, p_val = stats.ttest_ind(
            original["si_all"], shuffled_results,
            alternative='greater'
        )

        effect = cohens_d(np.array(original["si_all"]), shuffled_arr)

        results["results"][domain] = {
            "original": {
                "si_mean": original["si_mean"],
                "si_std": original["si_std"],
                "si_ci_lower": original["si_ci_lower"],
                "si_ci_upper": original["si_ci_upper"]
            },
            "shuffled": {
                "si_mean": float(shuffled_arr.mean()),
                "si_std": float(shuffled_arr.std()),
                "si_ci_lower": float(shuffled_ci[0]),
                "si_ci_upper": float(shuffled_ci[1]),
                "n_samples": len(shuffled_results)
            },
            "comparison": {
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
                "cohens_d": float(effect),
                "significant_001": bool(p_val < 0.001),
                "si_drop_pct": float((original["si_mean"] - shuffled_arr.mean()) / original["si_mean"] * 100)
            }
        }

        print(f"  Original SI:  {original['si_mean']:.3f} ± {original['si_std']:.3f}")
        print(f"  Shuffled SI:  {shuffled_arr.mean():.3f} ± {shuffled_arr.std():.3f}")
        print(f"  SI Drop:      {results['results'][domain]['comparison']['si_drop_pct']:.1f}%")
        print(f"  t-statistic:  {t_stat:.2f}")
        print(f"  p-value:      {p_val:.2e}")
        print(f"  Cohen's d:    {effect:.2f}")
        print(f"  Significant:  {'✓' if p_val < 0.001 else '✗'}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY: Regime Shuffle Test (Negative Control)")
    print("="*80)
    print(f"{'Domain':<12} {'Original SI':>12} {'Shuffled SI':>12} {'SI Drop':>10} {'p-value':>12} {'Sig?':>6}")
    print("-"*80)

    all_significant = True
    for domain in domains:
        r = results["results"][domain]
        orig = r["original"]["si_mean"]
        shuf = r["shuffled"]["si_mean"]
        drop = r["comparison"]["si_drop_pct"]
        pval = r["comparison"]["p_value"]
        sig = r["comparison"]["significant_001"]
        all_significant = all_significant and sig

        print(f"{domain:<12} {orig:>12.3f} {shuf:>12.3f} {drop:>9.1f}% {pval:>12.2e} {'✓' if sig else '✗':>6}")

    print("="*80)

    if all_significant:
        print("\n✓ All domains show significant SI drop when regimes are shuffled.")
        print("  This proves regime detection is MEANINGFUL, not random noise.")
    else:
        print("\n⚠ Some domains did not show significant SI drop.")

    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "regime_shuffle"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")

    return results


if __name__ == "__main__":
    print("="*80)
    print("Regime Shuffle Test - Negative Control")
    print("Validating that regime detection is meaningful")
    print("="*80)

    results = run_all_shuffle_tests(n_trials=30)
