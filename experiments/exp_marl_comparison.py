#!/usr/bin/env python3
"""
Compare Emergent Specialization against MARL Baselines.

Baselines:
- IQL: Independent Q-Learning
- Random: Random method selection
- Homogeneous: All agents use same best method

Comparison metric: Specialization Index (SI)
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.niche_population import NichePopulation
from src.baselines.marl_baselines import IndependentQLearning, MARLConfig
from src.domains import crypto, commodities, weather, solar


CONFIG = {
    'n_agents': 8,
    'n_iterations': 500,
    'n_trials': 20,
    'niche_bonus': 0.3,
    'random_seed': 42,
}


def calculate_si(distribution: Dict[str, float]) -> float:
    """Calculate Specialization Index from a distribution."""
    probs = np.array(list(distribution.values()))
    probs = probs / (probs.sum() + 1e-10)
    probs = probs[probs > 0]
    
    if len(probs) <= 1:
        return 1.0
    
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(len(distribution))
    
    return 1 - (entropy / max_entropy) if max_entropy > 0 else 0


def run_niche_population(regimes: List[str], methods: List[str], 
                         regime_probs: Dict[str, float], seed: int) -> dict:
    """Run our NichePopulation approach."""
    
    population = NichePopulation(
        n_agents=CONFIG['n_agents'],
        regimes=regimes,
        niche_bonus=CONFIG['niche_bonus'],
        seed=seed,
        methods=methods,
    )
    
    rng = np.random.default_rng(seed)
    
    for _ in range(CONFIG['n_iterations']):
        regime = rng.choice(list(regime_probs.keys()), p=list(regime_probs.values()))
        
        agent_scores = {}
        agent_choices = {}
        
        for agent_id, agent in population.agents.items():
            method = agent.select_method(regime)
            agent_choices[agent_id] = method
            
            base_score = rng.normal(0.5, 0.15)
            niche_strength = agent.get_niche_strength(regime)
            niche_score = CONFIG['niche_bonus'] * (niche_strength - 0.25)
            agent_scores[agent_id] = base_score + niche_score
        
        winner_id = max(agent_scores, key=agent_scores.get)
        
        for agent_id, agent in population.agents.items():
            won = (agent_id == winner_id)
            agent.update(regime, agent_choices[agent_id], won)
    
    # Calculate final SI
    agent_sis = []
    for agent in population.agents.values():
        si = calculate_si(agent.niche_affinity)
        agent_sis.append(si)
    
    return {
        'mean_si': float(np.mean(agent_sis)),
        'std_si': float(np.std(agent_sis)),
    }


def run_iql(regimes: List[str], methods: List[str], 
            regime_probs: Dict[str, float], seed: int) -> dict:
    """Run Independent Q-Learning baseline."""
    
    config = MARLConfig(
        n_agents=CONFIG['n_agents'],
        n_methods=len(methods),
        n_regimes=len(regimes),
    )
    
    rng = np.random.default_rng(seed)
    
    # Q-tables for each agent
    q_tables = {}
    for i in range(CONFIG['n_agents']):
        q_tables[f"agent_{i}"] = {
            r: {m: 0.0 for m in methods}
            for r in regimes
        }
    
    epsilon = 0.3
    epsilon_decay = 0.995
    min_epsilon = 0.05
    lr = 0.1
    
    for _ in range(CONFIG['n_iterations']):
        regime = rng.choice(list(regime_probs.keys()), p=list(regime_probs.values()))
        
        choices = {}
        for agent_id in q_tables:
            if rng.random() < epsilon:
                choices[agent_id] = rng.choice(methods)
            else:
                q_vals = q_tables[agent_id][regime]
                choices[agent_id] = max(q_vals, key=q_vals.get)
        
        # Rewards
        scores = {aid: rng.normal(0.5, 0.2) for aid in q_tables}
        winner = max(scores, key=scores.get)
        
        for agent_id in q_tables:
            method = choices[agent_id]
            reward = 1.0 if agent_id == winner else 0.0
            
            old_q = q_tables[agent_id][regime][method]
            q_tables[agent_id][regime][method] = old_q + lr * (reward - old_q)
        
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
    
    # Calculate SI from Q-table preferences
    agent_sis = []
    for agent_id in q_tables:
        # Get regime preferences based on max Q
        regime_prefs = {}
        for r in regimes:
            max_q = max(q_tables[agent_id][r].values())
            regime_prefs[r] = max(0.01, max_q)
        
        si = calculate_si(regime_prefs)
        agent_sis.append(si)
    
    return {
        'mean_si': float(np.mean(agent_sis)),
        'std_si': float(np.std(agent_sis)),
    }


def run_random(regimes: List[str], methods: List[str], seed: int) -> dict:
    """Random baseline (no learning)."""
    rng = np.random.default_rng(seed)
    
    agent_sis = []
    for _ in range(CONFIG['n_agents']):
        # Random uniform preference
        prefs = rng.dirichlet(np.ones(len(regimes)))
        dist = {r: prefs[i] for i, r in enumerate(regimes)}
        agent_sis.append(calculate_si(dist))
    
    return {
        'mean_si': float(np.mean(agent_sis)),
        'std_si': float(np.std(agent_sis)),
    }


def run_homogeneous(regimes: List[str], methods: List[str], seed: int) -> dict:
    """Homogeneous baseline (all agents same)."""
    # All agents have uniform distribution (no specialization)
    uniform_dist = {r: 1.0/len(regimes) for r in regimes}
    si = calculate_si(uniform_dist)
    
    return {
        'mean_si': si,
        'std_si': 0.0,
    }


def run_domain_comparison(domain_name: str, domain_module) -> dict:
    """Run all methods on a domain and compare."""
    
    print(f"\n{'='*60}")
    print(f"DOMAIN: {domain_name.upper()}")
    print(f"{'='*60}")
    
    # Load data
    if domain_name == 'crypto':
        df = domain_module.load_data('BTC')
    else:
        df = domain_module.load_data()
    
    regimes = domain_module.detect_regime(df)
    regime_list = regimes.unique().tolist()
    regime_counts = regimes.value_counts()
    regime_probs = (regime_counts / regime_counts.sum()).to_dict()
    
    methods = list(domain_module.get_prediction_methods().keys())
    
    print(f"Regimes: {regime_list}")
    print(f"Methods: {methods}")
    
    # Run all approaches
    results = {
        'niche_population': [],
        'iql': [],
        'random': [],
        'homogeneous': [],
    }
    
    for trial in range(CONFIG['n_trials']):
        seed = CONFIG['random_seed'] + trial
        
        results['niche_population'].append(
            run_niche_population(regime_list, methods, regime_probs, seed)
        )
        results['iql'].append(
            run_iql(regime_list, methods, regime_probs, seed)
        )
        results['random'].append(
            run_random(regime_list, methods, seed)
        )
        results['homogeneous'].append(
            run_homogeneous(regime_list, methods, seed)
        )
        
        if (trial + 1) % 5 == 0:
            print(f"  Completed trial {trial + 1}/{CONFIG['n_trials']}")
    
    # Aggregate
    summary = {}
    for method_name, trials in results.items():
        sis = [t['mean_si'] for t in trials]
        summary[method_name] = {
            'mean_si': float(np.mean(sis)),
            'std_si': float(np.std(sis)),
            'ci_95': [float(np.percentile(sis, 2.5)), float(np.percentile(sis, 97.5))],
        }
    
    print(f"\nResults for {domain_name}:")
    print(f"  {'Method':<20} {'Mean SI':<15} {'95% CI'}")
    print(f"  {'-'*50}")
    for method_name, stats in summary.items():
        ci = f"[{stats['ci_95'][0]:.3f}, {stats['ci_95'][1]:.3f}]"
        print(f"  {method_name:<20} {stats['mean_si']:.4f}±{stats['std_si']:.4f}   {ci}")
    
    return {
        'domain': domain_name,
        'n_regimes': len(regime_list),
        'n_methods': len(methods),
        'results': summary,
    }


def main():
    print("="*60)
    print("MARL BASELINE COMPARISON")
    print("="*60)
    print(f"Started: {datetime.now().isoformat()}")
    
    domains = {
        'crypto': crypto,
        'commodities': commodities,
        'weather': weather,
        'solar': solar,
    }
    
    all_results = {}
    
    for domain_name, module in domains.items():
        results = run_domain_comparison(domain_name, module)
        all_results[domain_name] = results
    
    # Final summary table
    print("\n" + "="*60)
    print("FINAL COMPARISON TABLE")
    print("="*60)
    print(f"\n{'Domain':<12} {'NichePopulation':<18} {'IQL':<18} {'Random':<18}")
    print("-"*70)
    
    for domain, results in all_results.items():
        r = results['results']
        niche = f"{r['niche_population']['mean_si']:.3f}±{r['niche_population']['std_si']:.3f}"
        iql = f"{r['iql']['mean_si']:.3f}±{r['iql']['std_si']:.3f}"
        rand = f"{r['random']['mean_si']:.3f}±{r['random']['std_si']:.3f}"
        print(f"{domain:<12} {niche:<18} {iql:<18} {rand:<18}")
    
    # Save
    output_dir = Path(__file__).parent.parent / "results" / "marl_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "latest_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {output_dir / 'latest_results.json'}")
    print(f"Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()

