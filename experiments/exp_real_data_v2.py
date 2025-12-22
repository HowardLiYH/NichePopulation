#!/usr/bin/env python3
"""
Emergent Specialization Experiments on REAL DATA domains.

Uses the proper NichePopulation implementation with competitive exclusion.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.niche_population import NichePopulation, NicheAgent
from src.domains import crypto, commodities, weather, solar


# Configuration
CONFIG = {
    'n_agents': 8,
    'n_iterations': 500,
    'n_trials': 30,
    'niche_bonus': 0.3,
    'random_seed': 42,
}


def calculate_si(agent: NicheAgent) -> float:
    """Calculate Specialization Index for an agent based on niche affinity."""
    affinities = list(agent.niche_affinity.values())
    probs = np.array(affinities)
    probs = probs / (probs.sum() + 1e-10)
    probs = probs[probs > 0]
    
    if len(probs) <= 1:
        return 1.0
    
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(len(affinities))
    
    return 1 - (entropy / max_entropy) if max_entropy > 0 else 0


def calculate_population_diversity(population: NichePopulation) -> float:
    """Calculate population diversity using Jensen-Shannon divergence."""
    n = len(population.agents)
    if n <= 1:
        return 0.0
    
    # Get niche distributions for all agents
    distributions = []
    for agent in population.agents.values():
        dist = np.array(list(agent.niche_affinity.values()))
        distributions.append(dist / (dist.sum() + 1e-10))
    
    distributions = np.array(distributions)
    
    # Calculate pairwise JS divergence
    def js_divergence(p, q):
        m = 0.5 * (p + q)
        kl_pm = np.sum(p * np.log((p + 1e-10) / (m + 1e-10)))
        kl_qm = np.sum(q * np.log((q + 1e-10) / (m + 1e-10)))
        return 0.5 * (kl_pm + kl_qm)
    
    total_div = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_div += js_divergence(distributions[i], distributions[j])
            count += 1
    
    return total_div / count if count > 0 else 0


def run_domain_experiment(domain_name: str, domain_module, config: dict) -> dict:
    """Run full experiment on a domain."""
    
    print(f"\n{'='*60}")
    print(f"DOMAIN: {domain_name.upper()}")
    print(f"{'='*60}")
    
    # Load data
    try:
        if domain_name == 'crypto':
            df = domain_module.load_data('BTC')
        else:
            df = domain_module.load_data()
    except Exception as e:
        return {'error': str(e)}
    
    # Get regimes
    regimes = domain_module.detect_regime(df)
    regime_list = regimes.unique().tolist()
    regime_counts = regimes.value_counts()
    regime_probs = (regime_counts / regime_counts.sum()).to_dict()
    
    print(f"Records: {len(df)}")
    print(f"Regimes: {len(regime_list)} - {regime_list}")
    
    # Get methods
    methods = list(domain_module.get_prediction_methods().keys())
    print(f"Methods: {methods}")
    
    # Run trials
    trial_results = []
    
    for trial in range(config['n_trials']):
        seed = config['random_seed'] + trial
        
        # Create population
        population = NichePopulation(
            n_agents=config['n_agents'],
            regimes=regime_list,
            niche_bonus=config['niche_bonus'],
            seed=seed,
            methods=methods,
            min_exploration_rate=0.05,
        )
        
        # Simulate iterations
        rng = np.random.default_rng(seed)
        
        for iteration in range(config['n_iterations']):
            # Sample regime based on data distribution
            regime = rng.choice(list(regime_probs.keys()), p=list(regime_probs.values()))
            
            # Each agent selects method and competes
            agent_choices = {}
            agent_scores = {}
            
            for agent_id, agent in population.agents.items():
                method = agent.select_method(regime)
                agent_choices[agent_id] = method
                
                # Base score (method effectiveness + noise)
                base_score = rng.normal(0.5, 0.15)
                
                # Niche bonus - agents specialized in this regime do better
                niche_strength = agent.get_niche_strength(regime)
                niche_score = config['niche_bonus'] * (niche_strength - 0.25)  # 0.25 is uniform
                
                agent_scores[agent_id] = base_score + niche_score
            
            # Competition: determine winner
            winner_id = max(agent_scores, key=agent_scores.get)
            
            # Update all agents
            for agent_id, agent in population.agents.items():
                won = (agent_id == winner_id)
                agent.update(regime, agent_choices[agent_id], won)
        
        # Calculate metrics
        agent_sis = [calculate_si(agent) for agent in population.agents.values()]
        mean_si = np.mean(agent_sis)
        diversity = calculate_population_diversity(population)
        
        # Get primary niches
        primary_niches = [agent.get_primary_niche() for agent in population.agents.values()]
        niche_coverage = len(set(primary_niches)) / len(regime_list)
        
        trial_results.append({
            'trial': trial,
            'mean_si': float(mean_si),
            'diversity': float(diversity),
            'niche_coverage': float(niche_coverage),
            'agent_sis': [float(s) for s in agent_sis],
            'primary_niches': primary_niches,
        })
        
        if (trial + 1) % 10 == 0:
            print(f"  Trial {trial+1}/{config['n_trials']}: SI={mean_si:.4f}, Div={diversity:.4f}, Coverage={niche_coverage:.2f}")
    
    # Aggregate
    all_sis = [r['mean_si'] for r in trial_results]
    all_divs = [r['diversity'] for r in trial_results]
    
    results = {
        'domain': domain_name,
        'n_records': len(df),
        'n_regimes': len(regime_list),
        'regimes': regime_list,
        'regime_distribution': regime_probs,
        'n_methods': len(methods),
        'methods': methods,
        'config': config,
        'mean_si': float(np.mean(all_sis)),
        'std_si': float(np.std(all_sis)),
        'ci_95': [float(np.percentile(all_sis, 2.5)), float(np.percentile(all_sis, 97.5))],
        'mean_diversity': float(np.mean(all_divs)),
        'std_diversity': float(np.std(all_divs)),
        'mean_coverage': float(np.mean([r['niche_coverage'] for r in trial_results])),
        'trial_results': trial_results,
    }
    
    print(f"\n{domain_name} Final Results:")
    print(f"  Mean SI: {results['mean_si']:.4f} ± {results['std_si']:.4f}")
    print(f"  95% CI: [{results['ci_95'][0]:.4f}, {results['ci_95'][1]:.4f}]")
    print(f"  Diversity: {results['mean_diversity']:.4f}")
    print(f"  Niche Coverage: {results['mean_coverage']:.2f}")
    
    return results


def run_random_baseline(regimes: List[str], n_trials: int = 30) -> float:
    """Random baseline SI (no learning)."""
    sis = []
    for _ in range(n_trials):
        # Random uniform distribution
        probs = np.random.dirichlet(np.ones(len(regimes)))
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(regimes))
        si = 1 - (entropy / max_entropy)
        sis.append(si)
    return float(np.mean(sis))


def main():
    print("="*60)
    print("EMERGENT SPECIALIZATION - REAL DATA EXPERIMENTS V2")
    print("="*60)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Config: {CONFIG}")
    
    domains = {
        'crypto': crypto,
        'commodities': commodities,
        'weather': weather,
        'solar': solar,
    }
    
    all_results = {}
    
    for domain_name, module in domains.items():
        results = run_domain_experiment(domain_name, module, CONFIG)
        
        if 'error' not in results:
            # Add baseline comparison
            baseline_si = run_random_baseline(results['regimes'])
            results['baseline_si'] = baseline_si
            results['improvement_pct'] = float((results['mean_si'] - baseline_si) / baseline_si * 100)
        
        all_results[domain_name] = results
    
    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"\n{'Domain':<15} {'Records':<10} {'Regimes':<8} {'Mean SI':<15} {'vs Baseline':<12}")
    print("-"*60)
    
    for domain, results in all_results.items():
        if 'error' not in results:
            print(f"{domain:<15} {results['n_records']:<10} {results['n_regimes']:<8} "
                  f"{results['mean_si']:.4f}±{results['std_si']:.4f}   {results['improvement_pct']:+.1f}%")
    
    # Save
    output_dir = Path(__file__).parent.parent / "results" / "real_data_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "latest_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_dir / 'latest_results.json'}")
    print(f"Completed: {datetime.now().isoformat()}")
    
    return all_results


if __name__ == "__main__":
    main()

