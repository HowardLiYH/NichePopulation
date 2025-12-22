#!/usr/bin/env python3
"""
Tier-1 Domain Screening Experiment.

Tests all 5 new Tier-1 domains to identify which ones show strong
emergent specialization (SI > 0.40) for inclusion in the paper.

Domains tested:
1. Air Quality (EPA)
2. Wikipedia (Pageviews)
3. Solar (NREL)
4. Water (USGS)
5. Commodities (FRED)

Selection criteria:
- SI > 0.40 (specialization emerges)
- Improvement > 0% vs homogeneous baseline
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.niche_population import NicheAgent, NichePopulation


def compute_specialization_index(niche_distributions: Dict[str, Dict[str, float]]) -> float:
    """
    Compute Specialization Index (SI) from niche affinity distributions.

    SI measures how specialized each agent is (low entropy = specialized).
    Returns mean SI across agents.
    """
    si_values = []
    for agent_id, affinity in niche_distributions.items():
        prefs = np.array(list(affinity.values()))
        prefs = prefs / (prefs.sum() + 1e-8)
        prefs = np.clip(prefs, 1e-8, 1)

        # Shannon entropy
        entropy = -np.sum(prefs * np.log(prefs + 1e-8))
        max_entropy = np.log(len(prefs))

        # SI = 1 - normalized entropy (higher = more specialized)
        si = 1 - (entropy / max_entropy) if max_entropy > 0 else 0
        si_values.append(si)

    return float(np.mean(si_values))


@dataclass
class DomainResult:
    """Results for a single domain."""
    domain: str
    si_mean: float
    si_std: float
    improvement_mean: float
    improvement_std: float
    n_trials: int
    n_regimes: int
    regime_distribution: Dict[str, float]
    verdict: str  # "include", "appendix", "exclude"

    def to_dict(self):
        return asdict(self)


def reward_fn(methods: List[str], prices: np.ndarray) -> float:
    """Simple reward function based on price movement."""
    if len(prices) < 2:
        return 0.0
    return float(prices[-1] - prices[-2])


def run_domain_experiment(
    domain_name: str,
    n_trials: int = 30,
    n_iterations: int = 2000,
    n_agents: int = 8,
    niche_bonus: float = 0.3,
    seed_offset: int = 0,
) -> DomainResult:
    """
    Run screening experiment for a single domain.

    Returns SI and improvement statistics.
    """
    # Import domain-specific environment
    if domain_name == "air_quality":
        from src.domains.air_quality import create_air_quality_environment
        create_env = lambda seed: create_air_quality_environment(n_bars=n_iterations, seed=seed)
    elif domain_name == "wikipedia":
        from src.domains.wikipedia import create_wikipedia_environment
        create_env = lambda seed: create_wikipedia_environment(n_bars=n_iterations, seed=seed)
    elif domain_name == "solar":
        from src.domains.solar import create_solar_environment
        create_env = lambda seed: create_solar_environment(n_bars=n_iterations, seed=seed)
    elif domain_name == "water":
        from src.domains.water import create_water_environment
        create_env = lambda seed: create_water_environment(n_bars=n_iterations, seed=seed)
    elif domain_name == "commodities":
        from src.domains.commodities import create_commodities_environment
        create_env = lambda seed: create_commodities_environment(n_bars=n_iterations, seed=seed)
    else:
        raise ValueError(f"Unknown domain: {domain_name}")

    si_values = []
    improvements = []
    regime_counts = []

    for trial in tqdm(range(n_trials), desc=f"{domain_name}"):
        seed = seed_offset + trial

        # Create environment
        df, regimes, methods = create_env(seed)
        method_names = list(methods.keys())
        regime_names = list(set(regimes))
        n_regimes = len(regime_names)

        # Count regime distribution
        regime_dist = regimes.value_counts(normalize=True).to_dict()
        regime_counts.append(regime_dist)

        # Create population with niche-based competition
        np.random.seed(seed)
        population = NichePopulation(
            n_agents=n_agents,
            regimes=regime_names,
            niche_bonus=niche_bonus,
            seed=seed,
            methods=method_names,
            min_exploration_rate=0.05,
        )

        # Track rewards for diverse vs homogeneous comparison
        diverse_rewards = []
        homogeneous_rewards = []

        # Simulate competition
        for t in range(min(len(df), n_iterations)):
            # Get current regime
            regime = regimes.iloc[t] if t < len(regimes) else regime_names[0]

            # Get observation
            obs = df.iloc[t].values

            # Each agent selects a method
            selections = {}
            for agent_id, agent in population.agents.items():
                method = agent.select_method(regime)
                selections[agent_id] = method

            # Compute raw rewards
            raw_rewards = {}
            for agent_id, method_name in selections.items():
                method = methods[method_name]
                result = method.execute(obs)
                base_reward = 0.5 + 0.5 * result.get('signal', 0)

                # Bonus if method is optimal for regime
                if hasattr(method, 'optimal_regimes') and regime in method.optimal_regimes:
                    base_reward *= 1.5

                raw_rewards[agent_id] = base_reward

            # Track diverse population reward (average)
            diverse_rewards.append(np.mean(list(raw_rewards.values())))

            # Track homogeneous baseline (use first method always)
            homo_method = methods[method_names[0]]
            homo_result = homo_method.execute(obs)
            homo_reward = 0.5 + 0.5 * homo_result.get('signal', 0)
            if hasattr(homo_method, 'optimal_regimes') and regime in homo_method.optimal_regimes:
                homo_reward *= 1.5
            homogeneous_rewards.append(homo_reward)

            # Apply niche bonus and determine winner
            adjusted_rewards = {}
            for agent_id, agent in population.agents.items():
                raw = raw_rewards[agent_id]
                agent_niche = agent.get_primary_niche()
                if agent_niche == regime:
                    bonus = niche_bonus * agent.niche_affinity[regime]
                else:
                    bonus = -niche_bonus * 0.3 * (1 - agent.niche_affinity.get(regime, 0.25))
                adjusted_rewards[agent_id] = raw + bonus

            winner_id = max(adjusted_rewards, key=adjusted_rewards.get)

            # Update all agents
            for agent_id, agent in population.agents.items():
                won = (agent_id == winner_id)
                agent.update(regime, selections[agent_id], won=won)

        # Compute SI from niche distributions
        niche_dist = population.get_niche_distribution()
        si = compute_specialization_index(niche_dist)
        si_values.append(si)

        # Compute improvement vs homogeneous
        diverse_total = np.mean(diverse_rewards)
        homo_total = np.mean(homogeneous_rewards)
        if homo_total > 0:
            improvement = (diverse_total - homo_total) / homo_total * 100
        else:
            improvement = 0
        improvements.append(improvement)

    # Aggregate results
    si_mean = float(np.mean(si_values))
    si_std = float(np.std(si_values))
    improvement_mean = float(np.mean(improvements))
    improvement_std = float(np.std(improvements))

    # Average regime distribution
    avg_regime_dist = {}
    all_keys = set()
    for rc in regime_counts:
        all_keys.update(rc.keys())
    for key in all_keys:
        avg_regime_dist[key] = float(np.mean([rc.get(key, 0) for rc in regime_counts]))

    # Determine verdict
    if si_mean >= 0.40 and improvement_mean > 0:
        verdict = "include"
    elif si_mean >= 0.35:
        verdict = "appendix"
    else:
        verdict = "exclude"

    return DomainResult(
        domain=domain_name,
        si_mean=si_mean,
        si_std=si_std,
        improvement_mean=improvement_mean,
        improvement_std=improvement_std,
        n_trials=n_trials,
        n_regimes=len(avg_regime_dist),
        regime_distribution=avg_regime_dist,
        verdict=verdict,
    )


def run_all_screenings(
    n_trials: int = 30,
    n_iterations: int = 2000,
    n_agents: int = 8,
) -> Dict[str, DomainResult]:
    """Run screening for all 5 Tier-1 domains."""

    domains = ["air_quality", "wikipedia", "solar", "water", "commodities"]
    results = {}

    print("=" * 60)
    print("TIER-1 DOMAIN SCREENING EXPERIMENT")
    print("=" * 60)
    print(f"Trials per domain: {n_trials}")
    print(f"Iterations per trial: {n_iterations}")
    print(f"Agents per population: {n_agents}")
    print("=" * 60)

    for domain in domains:
        print(f"\n>>> Testing domain: {domain}")
        result = run_domain_experiment(
            domain_name=domain,
            n_trials=n_trials,
            n_iterations=n_iterations,
            n_agents=n_agents,
            niche_bonus=0.3,  # Competition incentive
        )
        results[domain] = result

        print(f"    SI: {result.si_mean:.3f} ± {result.si_std:.3f}")
        print(f"    Improvement: {result.improvement_mean:.1f}% ± {result.improvement_std:.1f}%")
        print(f"    Regimes: {result.n_regimes}")
        print(f"    Verdict: {result.verdict.upper()}")

    return results


def print_ranking_table(results: Dict[str, DomainResult]):
    """Print ranking table for domain selection."""

    # Sort by SI
    sorted_domains = sorted(results.values(), key=lambda x: x.si_mean, reverse=True)

    print("\n" + "=" * 80)
    print("DOMAIN RANKING TABLE")
    print("=" * 80)
    print(f"{'Rank':<6} {'Domain':<15} {'SI':<15} {'Improvement':<15} {'Verdict':<10}")
    print("-" * 80)

    for i, result in enumerate(sorted_domains):
        si_str = f"{result.si_mean:.3f} ± {result.si_std:.3f}"
        imp_str = f"{result.improvement_mean:.1f}% ± {result.improvement_std:.1f}%"
        print(f"{i+1:<6} {result.domain:<15} {si_str:<15} {imp_str:<15} {result.verdict.upper():<10}")

    print("-" * 80)

    # Summary
    included = [r for r in sorted_domains if r.verdict == "include"]
    appendix = [r for r in sorted_domains if r.verdict == "appendix"]
    excluded = [r for r in sorted_domains if r.verdict == "exclude"]

    print(f"\nSUMMARY:")
    print(f"  Include in paper: {len(included)} domains")
    for r in included:
        print(f"    - {r.domain} (SI={r.si_mean:.3f}, Imp={r.improvement_mean:.1f}%)")

    if appendix:
        print(f"  Move to appendix: {len(appendix)} domains")
        for r in appendix:
            print(f"    - {r.domain} (SI={r.si_mean:.3f}, Imp={r.improvement_mean:.1f}%)")

    if excluded:
        print(f"  Exclude: {len(excluded)} domains")
        for r in excluded:
            print(f"    - {r.domain} (SI={r.si_mean:.3f}, Imp={r.improvement_mean:.1f}%)")


def save_results(results: Dict[str, DomainResult], output_dir: str = None):
    """Save screening results to JSON."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results" / "tier1_screening"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serializable format
    results_dict = {
        "timestamp": datetime.now().isoformat(),
        "domains": {name: result.to_dict() for name, result in results.items()},
        "summary": {
            "total_domains": len(results),
            "included": len([r for r in results.values() if r.verdict == "include"]),
            "appendix": len([r for r in results.values() if r.verdict == "appendix"]),
            "excluded": len([r for r in results.values() if r.verdict == "exclude"]),
        }
    }

    output_path = output_dir / "screening_results.json"
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    """Main function to run Tier-1 domain screening."""
    import argparse

    parser = argparse.ArgumentParser(description="Tier-1 Domain Screening")
    parser.add_argument("--trials", type=int, default=30, help="Number of trials per domain")
    parser.add_argument("--iterations", type=int, default=2000, help="Iterations per trial")
    parser.add_argument("--agents", type=int, default=8, help="Agents per population")
    args = parser.parse_args()

    # Run screenings
    results = run_all_screenings(
        n_trials=args.trials,
        n_iterations=args.iterations,
        n_agents=args.agents,
    )

    # Print ranking table
    print_ranking_table(results)

    # Save results
    save_results(results)

    print("\n" + "=" * 60)
    print("SCREENING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
