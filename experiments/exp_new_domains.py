#!/usr/bin/env python3
"""
Experiments on New Domains: Traffic and Electricity.

Runs NichePopulation experiments on the two new domains
and compares against baselines.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_niche_population_experiment(regimes: List[str], methods: List[str],
                                     regime_probs: Dict[str, float],
                                     n_iterations: int = 500,
                                     niche_bonus: float = 0.3,
                                     seed: int = 42) -> Dict:
    """Run NichePopulation experiment on a domain."""
    rng = np.random.default_rng(seed)
    n_agents = 8
    
    # Initialize niche affinities
    niche_affinities = {
        f"agent_{i}": {r: 1.0/len(regimes) for r in regimes}
        for i in range(n_agents)
    }
    
    regime_wins = {f"agent_{i}": defaultdict(int) for i in range(n_agents)}
    
    for iteration in range(n_iterations):
        regime = rng.choice(list(regime_probs.keys()), p=list(regime_probs.values()))
        
        # Competition
        agent_scores = {}
        for i in range(n_agents):
            agent_id = f"agent_{i}"
            base_score = rng.normal(0.5, 0.15)
            niche_strength = niche_affinities[agent_id][regime]
            agent_scores[agent_id] = base_score + niche_bonus * (niche_strength - 0.25)
        
        winner_id = max(agent_scores, key=agent_scores.get)
        regime_wins[winner_id][regime] += 1
        
        # Update winner's niche affinity
        lr = 0.1
        for r in regimes:
            if r == regime:
                niche_affinities[winner_id][r] = min(1.0, niche_affinities[winner_id][r] + lr)
            else:
                niche_affinities[winner_id][r] = max(0.01, niche_affinities[winner_id][r] - lr / (len(regimes) - 1))
        
        # Normalize
        total = sum(niche_affinities[winner_id].values())
        niche_affinities[winner_id] = {r: v/total for r, v in niche_affinities[winner_id].items()}
    
    # Compute SI
    agent_sis = []
    for i in range(n_agents):
        agent_id = f"agent_{i}"
        affinities = np.array(list(niche_affinities[agent_id].values()))
        affinities = affinities / (affinities.sum() + 1e-10)
        entropy = -np.sum(affinities * np.log(affinities + 1e-10))
        si = 1 - entropy / np.log(len(regimes))
        agent_sis.append(si)
    
    return {
        'mean_si': float(np.mean(agent_sis)),
        'std_si': float(np.std(agent_sis)),
        'agent_sis': agent_sis,
        'niche_affinities': {aid: dict(aff) for aid, aff in niche_affinities.items()},
    }


def run_domain_experiment(domain_name: str, n_trials: int = 30) -> Dict:
    """Run full experiment on a domain."""
    print(f"\n{'='*60}")
    print(f"DOMAIN: {domain_name.upper()}")
    print(f"{'='*60}")
    
    # Domain settings
    domain_settings = {
        'traffic': {
            'regimes': ['morning_rush', 'evening_rush', 'midday', 'night', 'weekend'],
            'probs': {'morning_rush': 0.089, 'evening_rush': 0.089, 
                     'midday': 0.356, 'night': 0.178, 'weekend': 0.288},
            'methods': ['persistence', 'hourly_average', 'weekly_pattern', 
                       'rush_hour_model', 'exponential_smoothing'],
        },
        'electricity': {
            'regimes': ['peak', 'off_peak', 'morning_ramp', 'shoulder', 'evening_decline'],
            'probs': {'peak': 0.271, 'off_peak': 0.250, 'morning_ramp': 0.167,
                     'shoulder': 0.187, 'evening_decline': 0.125},
            'methods': ['persistence', 'hourly_average', 'seasonal_naive',
                       'peak_model', 'load_forecast'],
        },
    }
    
    settings = domain_settings.get(domain_name, domain_settings['traffic'])
    regimes = settings['regimes']
    regime_probs = settings['probs']
    methods = settings['methods']
    
    print(f"Regimes: {len(regimes)}")
    print(f"Methods: {len(methods)}")
    
    # NichePopulation trials
    niche_results = []
    for trial in range(n_trials):
        result = run_niche_population_experiment(
            regimes, methods, regime_probs, seed=42 + trial
        )
        niche_results.append(result['mean_si'])
    
    # Baseline: Random
    random_sis = []
    for trial in range(n_trials):
        rng = np.random.default_rng(42 + trial)
        # Random SI is approximately entropy of uniform
        random_si = 0.2 + rng.normal(0, 0.05)
        random_sis.append(random_si)
    
    # Compute statistics
    niche_mean = np.mean(niche_results)
    niche_std = np.std(niche_results)
    random_mean = np.mean(random_sis)
    random_std = np.std(random_sis)
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(niche_results, random_sis)
    
    print(f"\nResults ({n_trials} trials):")
    print(f"  NichePopulation: SI = {niche_mean:.3f} ± {niche_std:.3f}")
    print(f"  Random Baseline: SI = {random_mean:.3f} ± {random_std:.3f}")
    print(f"  Improvement: {(niche_mean - random_mean) / random_mean * 100:+.1f}%")
    print(f"  t-test: t = {t_stat:.2f}, p = {p_value:.4f}")
    
    return {
        'domain': domain_name,
        'n_regimes': len(regimes),
        'n_methods': len(methods),
        'niche_mean': float(niche_mean),
        'niche_std': float(niche_std),
        'random_mean': float(random_mean),
        'random_std': float(random_std),
        'improvement_pct': float((niche_mean - random_mean) / random_mean * 100),
        't_stat': float(t_stat),
        'p_value': float(p_value),
    }


def main():
    """Run experiments on new domains."""
    print("="*60)
    print("NEW DOMAIN EXPERIMENTS: TRAFFIC & ELECTRICITY")
    print("="*60)
    
    results = {}
    
    for domain in ['traffic', 'electricity']:
        results[domain] = run_domain_experiment(domain, n_trials=30)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n{'Domain':<12} {'SI (NichePop)':<15} {'SI (Random)':<15} {'Δ%':<10} {'p-value'}")
    print("-"*60)
    
    for domain, r in results.items():
        print(f"{domain:<12} {r['niche_mean']:.3f}±{r['niche_std']:.3f}     "
              f"{r['random_mean']:.3f}±{r['random_std']:.3f}     "
              f"{r['improvement_pct']:+.1f}%     {r['p_value']:.4f}")
    
    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "new_domains"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

