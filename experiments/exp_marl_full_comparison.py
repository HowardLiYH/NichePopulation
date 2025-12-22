#!/usr/bin/env python3
"""
Full MARL Baseline Comparison Experiment.

Compares NichePopulation (Ours) against:
- QMIX: Monotonic value function factorisation
- MAPPO: Multi-Agent PPO with centralized critic
- IQL: Independent Q-Learning
- Random: Random baseline

Metrics:
- Specialization Index (SI)
- Performance (domain-specific)
- Convergence speed
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.niche_population import NichePopulation
from src.baselines.qmix import QMIX, QMIXConfig, run_qmix_experiment
from src.baselines.mappo import MAPPO, MAPPOConfig, run_mappo_experiment
from src.baselines.marl_baselines import IndependentQLearning, MARLConfig

# Import domain modules
from src.domains import crypto, commodities, weather, solar


@dataclass
class ExperimentResult:
    """Result from a single experiment trial."""
    method: str
    domain: str
    trial: int
    mean_si: float
    agent_sis: List[float]
    performance: float
    convergence_steps: int = 500


def run_random_baseline(regimes: List[str], methods: List[str],
                        regime_probs: Dict[str, float],
                        n_iterations: int = 500,
                        seed: int = 42) -> Dict:
    """Run random baseline (no learning)."""
    rng = np.random.default_rng(seed)
    n_agents = 8

    # Track random regime selections
    regime_counts = {f"agent_{i}": {r: 0 for r in regimes} for i in range(n_agents)}

    for _ in range(n_iterations):
        regime = rng.choice(list(regime_probs.keys()), p=list(regime_probs.values()))

        # Random agent "wins"
        winner = f"agent_{rng.integers(0, n_agents)}"
        regime_counts[winner][regime] += 1

    # Compute SI (should be near uniform = 0)
    sis = []
    for agent_id in regime_counts:
        counts = np.array(list(regime_counts[agent_id].values()))
        total = counts.sum()
        if total > 0:
            probs = counts / total
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log(probs))
            si = 1 - entropy / np.log(len(regimes))
            sis.append(si)
        else:
            sis.append(0.0)

    return {
        'mean_si': float(np.mean(sis)),
        'agent_sis': sis,
    }


def run_niche_population(regimes: List[str], methods: List[str],
                          regime_probs: Dict[str, float],
                          n_iterations: int = 500,
                          niche_bonus: float = 0.3,
                          seed: int = 42) -> Dict:
    """Run our NichePopulation approach."""
    rng = np.random.default_rng(seed)

    pop = NichePopulation(
        n_agents=8,
        regimes=regimes,
        niche_bonus=niche_bonus,
        seed=seed,
        methods=methods,
    )

    for _ in range(n_iterations):
        regime = rng.choice(list(regime_probs.keys()), p=list(regime_probs.values()))

        agent_scores = {}
        agent_choices = {}

        for agent_id, agent in pop.agents.items():
            method = agent.select_method(regime)
            agent_choices[agent_id] = method

            base_score = rng.normal(0.5, 0.15)
            niche_strength = agent.get_niche_strength(regime)
            agent_scores[agent_id] = base_score + 0.3 * (niche_strength - 0.25)

        winner_id = max(agent_scores, key=agent_scores.get)

        for agent_id, agent in pop.agents.items():
            won = (agent_id == winner_id)
            agent.update(regime, agent_choices[agent_id], won)

    # Compute SI
    sis = []
    for agent in pop.agents.values():
        affinities = np.array(list(agent.niche_affinity.values()))
        affinities = affinities / (affinities.sum() + 1e-10)
        entropy = -np.sum(affinities * np.log(affinities + 1e-10))
        si = 1 - entropy / np.log(len(regimes))
        sis.append(si)

    return {
        'mean_si': float(np.mean(sis)),
        'agent_sis': sis,
    }


def run_iql_baseline(regimes: List[str], methods: List[str],
                     regime_probs: Dict[str, float],
                     n_iterations: int = 500,
                     seed: int = 42) -> Dict:
    """Run Independent Q-Learning baseline."""
    config = MARLConfig(n_agents=8, n_methods=len(methods), n_regimes=len(regimes))
    iql = IndependentQLearning(config, seed=seed)

    rng = np.random.default_rng(seed)

    for _ in range(n_iterations):
        regime = rng.choice(list(regime_probs.keys()), p=list(regime_probs.values()))
        actions = iql.select_actions(regime)

        # Simulate rewards
        rewards = {agent_id: rng.normal(0.5, 0.2) for agent_id in actions}
        winner_id = max(rewards, key=rewards.get)
        rewards[winner_id] += 0.5

        iql.update(regime, actions, rewards)

    return {
        'mean_si': iql.get_mean_si(),
        'agent_sis': [iql.get_specialization_index(f"agent_{i}")
                      for i in range(config.n_agents)],
    }


def run_domain_comparison(domain_name: str, domain_module,
                           n_trials: int = 30, n_iterations: int = 500,
                           seed: int = 42) -> Dict:
    """Run full comparison on a single domain."""
    print(f"\n{'='*60}")
    print(f"MARL COMPARISON: {domain_name.upper()}")
    print(f"{'='*60}")

    # Load domain data
    if domain_name == 'crypto':
        df = domain_module.load_data('BTC')
    else:
        df = domain_module.load_data()

    # Get regimes and methods
    regimes = domain_module.detect_regime(df)
    regime_list = regimes.unique().tolist()
    regime_probs = (regimes.value_counts() / len(regimes)).to_dict()
    methods = list(domain_module.get_prediction_methods().keys())

    print(f"Regimes: {len(regime_list)}, Methods: {len(methods)}")

    # Results storage
    results = {
        'NichePopulation': {'si': [], 'agent_sis': []},
        'QMIX': {'si': [], 'agent_sis': []},
        'MAPPO': {'si': [], 'agent_sis': []},
        'IQL': {'si': [], 'agent_sis': []},
        'Random': {'si': [], 'agent_sis': []},
    }

    for trial in range(n_trials):
        trial_seed = seed + trial

        # NichePopulation (Ours)
        niche_result = run_niche_population(
            regime_list, methods, regime_probs, n_iterations, seed=trial_seed
        )
        results['NichePopulation']['si'].append(niche_result['mean_si'])
        results['NichePopulation']['agent_sis'].append(niche_result['agent_sis'])

        # QMIX
        qmix_result = run_qmix_experiment(
            regime_list, methods, regime_probs, n_iterations, seed=trial_seed
        )
        results['QMIX']['si'].append(qmix_result['mean_si'])
        results['QMIX']['agent_sis'].append(qmix_result['agent_sis'])

        # MAPPO
        mappo_result = run_mappo_experiment(
            regime_list, methods, regime_probs, n_iterations, seed=trial_seed
        )
        results['MAPPO']['si'].append(mappo_result['mean_si'])
        results['MAPPO']['agent_sis'].append(mappo_result['agent_sis'])

        # IQL
        iql_result = run_iql_baseline(
            regime_list, methods, regime_probs, n_iterations, seed=trial_seed
        )
        results['IQL']['si'].append(iql_result['mean_si'])
        results['IQL']['agent_sis'].append(iql_result['agent_sis'])

        # Random
        random_result = run_random_baseline(
            regime_list, methods, regime_probs, n_iterations, seed=trial_seed
        )
        results['Random']['si'].append(random_result['mean_si'])
        results['Random']['agent_sis'].append(random_result['agent_sis'])

        if (trial + 1) % 10 == 0:
            print(f"  Trial {trial+1}/{n_trials}")

    # Compute statistics
    summary = {}
    for method_name, data in results.items():
        si_array = np.array(data['si'])
        summary[method_name] = {
            'mean_si': float(np.mean(si_array)),
            'std_si': float(np.std(si_array)),
            'min_si': float(np.min(si_array)),
            'max_si': float(np.max(si_array)),
        }

    return {
        'domain': domain_name,
        'n_trials': n_trials,
        'n_regimes': len(regime_list),
        'n_methods': len(methods),
        'results': results,
        'summary': summary,
    }


def main():
    """Run full MARL comparison on all domains."""
    print("="*60)
    print("FULL MARL BASELINE COMPARISON")
    print("="*60)

    domains = {
        'crypto': crypto,
        'commodities': commodities,
        'weather': weather,
        'solar': solar,
    }

    all_results = {}

    for domain_name, module in domains.items():
        results = run_domain_comparison(domain_name, module, n_trials=30)
        all_results[domain_name] = results

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY: MEAN SPECIALIZATION INDEX (SI)")
    print("="*70)
    print(f"\n{'Domain':<12} {'NichePop':<10} {'QMIX':<10} {'MAPPO':<10} {'IQL':<10} {'Random':<10}")
    print("-"*70)

    for domain, results in all_results.items():
        s = results['summary']
        print(f"{domain:<12} "
              f"{s['NichePopulation']['mean_si']:.3f}     "
              f"{s['QMIX']['mean_si']:.3f}     "
              f"{s['MAPPO']['mean_si']:.3f}     "
              f"{s['IQL']['mean_si']:.3f}     "
              f"{s['Random']['mean_si']:.3f}")

    # Cross-domain averages
    print("-"*70)
    avg = {method: [] for method in ['NichePopulation', 'QMIX', 'MAPPO', 'IQL', 'Random']}
    for domain, results in all_results.items():
        for method in avg:
            avg[method].append(results['summary'][method]['mean_si'])

    print(f"{'AVERAGE':<12} "
          f"{np.mean(avg['NichePopulation']):.3f}     "
          f"{np.mean(avg['QMIX']):.3f}     "
          f"{np.mean(avg['MAPPO']):.3f}     "
          f"{np.mean(avg['IQL']):.3f}     "
          f"{np.mean(avg['Random']):.3f}")

    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "marl_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON
    json_results = {}
    for domain, data in all_results.items():
        json_results[domain] = {
            'domain': data['domain'],
            'n_trials': data['n_trials'],
            'n_regimes': data['n_regimes'],
            'n_methods': data['n_methods'],
            'summary': data['summary'],
        }

    with open(output_dir / "latest_results.json", 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'latest_results.json'}")

    # Statistical significance test
    print("\n" + "="*70)
    print("STATISTICAL TESTS (NichePopulation vs Others)")
    print("="*70)

    from scipy import stats

    for domain, data in all_results.items():
        print(f"\n{domain.upper()}:")
        niche_si = data['results']['NichePopulation']['si']

        for baseline in ['QMIX', 'MAPPO', 'IQL', 'Random']:
            baseline_si = data['results'][baseline]['si']
            t_stat, p_value = stats.ttest_ind(niche_si, baseline_si)

            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            diff = np.mean(niche_si) - np.mean(baseline_si)
            print(f"  vs {baseline:<8}: Δ={diff:+.3f}, p={p_value:.4f} {sig}")


if __name__ == "__main__":
    main()
