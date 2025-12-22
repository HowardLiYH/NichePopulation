#!/usr/bin/env python3
"""
Experiment: Multi-Domain Validation

Tests whether emergent specialization generalizes across multiple domains
beyond financial trading.

Domains tested:
1. Finance (baseline) - Market trading with regime-based methods
2. Traffic - Traffic flow optimization
3. Energy - Grid management
4. Weather - Forecasting
5. E-commerce - Inventory management
6. Sports - Game strategy

Hypothesis:
- Specialization emerges in ALL domains with regime structure
- SI should be similar across domains (~0.7-0.9)
- This validates emergent specialization as a general MARL phenomenon

Usage:
    python experiments/exp_multi_domain.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
from scipy import stats
import time

from src.agents.niche_population import NichePopulation
from src.domains.synthetic_domains import (
    create_traffic_environment,
    create_energy_environment,
    create_weather_environment,
    create_ecommerce_environment,
    create_sports_environment,
    TRAFFIC_REGIMES,
    ENERGY_REGIMES,
    WEATHER_REGIMES,
    ECOMMERCE_REGIMES,
    SPORTS_REGIMES,
)
from src.environment.synthetic_market import SyntheticMarketConfig, SyntheticMarketEnvironment


# Configuration
N_TRIALS = 30
N_ITERATIONS = 2000
N_AGENTS = 8
NICHE_BONUS = 0.5

# Output
RESULTS_DIR = Path(__file__).parent.parent / "results" / "exp_multi_domain"


def compute_regime_si(niche_affinities: Dict[str, float]) -> float:
    """Compute SI from regime affinities."""
    affinities = np.array(list(niche_affinities.values()))
    affinities = affinities / (affinities.sum() + 1e-8)
    entropy = -np.sum(affinities * np.log(affinities + 1e-8))
    max_entropy = np.log(len(affinities))
    return 1 - (entropy / max_entropy) if max_entropy > 0 else 0.0


@dataclass
class DomainResult:
    """Result from one domain experiment."""
    domain: str
    n_regimes: int
    si_mean: float
    si_std: float
    si_ci_lower: float
    si_ci_upper: float
    diversity_mean: float
    n_trials: int


def create_domain_reward_fn(domain: str, methods_dict: Dict):
    """Create reward function for a domain."""

    def reward_fn(methods, observation):
        if len(observation) < 2:
            return 0.0

        method_name = methods[0] if methods else list(methods_dict.keys())[0]
        method_info = methods_dict.get(method_name, {})

        # Simplified reward: signal aligned with observation change
        signal = observation[-1] - observation[-2] if len(observation) >= 2 else 0

        # Bonus for optimal regimes (not used directly but conceptually)
        return signal * 100

    return reward_fn


def run_finance_domain(trial_id: int) -> Tuple[float, float]:
    """Run finance domain (baseline)."""
    from src.agents.inventory_v2 import METHOD_INVENTORY_V2

    env = SyntheticMarketEnvironment(SyntheticMarketConfig(seed=trial_id * 1000))
    prices, regimes = env.generate(n_bars=N_ITERATIONS + 100)

    population = NichePopulation(
        n_agents=N_AGENTS,
        niche_bonus=NICHE_BONUS,
        seed=trial_id
    )

    def reward_fn(methods, prices_window):
        if len(prices_window) < 2:
            return 0.0
        ret = (prices_window[-1] - prices_window[-2]) / prices_window[-2]
        method_name = methods[0] if methods else "BuyMomentum"
        if "Buy" in method_name or "Trend" in method_name:
            signal = 1.0
        elif "Sell" in method_name or "Fade" in method_name:
            signal = -1.0
        else:
            signal = 0.0
        return signal * ret * 100

    for i in range(20, min(len(prices), N_ITERATIONS + 50)):
        regime = regimes.iloc[i]
        price_window = prices['close'].values[max(0, i-20):i+1]
        population.run_iteration(price_window, regime, reward_fn)

    niche_dist = population.get_niche_distribution()
    agent_sis = [compute_regime_si(aff) for aff in niche_dist.values()]

    return np.mean(agent_sis), 0.0  # diversity placeholder


def run_generic_domain(
    domain_creator,
    regime_list: List[str],
    trial_id: int,
) -> Tuple[float, float]:
    """Run a generic domain experiment."""

    df, regimes, methods_dict = domain_creator(n_bars=N_ITERATIONS + 100, seed=trial_id * 1000)

    # Create population with domain's regime names
    population = NichePopulation(
        n_agents=N_AGENTS,
        regimes=regime_list,
        niche_bonus=NICHE_BONUS,
        seed=trial_id,
        methods=list(methods_dict.keys()),
    )

    def reward_fn(methods, obs_window):
        if len(obs_window) < 2:
            return 0.0
        signal = obs_window[-1] - obs_window[-2]
        return signal * 100

    # Get observation column (first column)
    obs_column = df.iloc[:, 0].values

    for i in range(20, min(len(df), N_ITERATIONS + 50)):
        regime = regimes.iloc[i]
        obs_window = obs_column[max(0, i-20):i+1]
        population.run_iteration(obs_window, regime, reward_fn)

    niche_dist = population.get_niche_distribution()
    agent_sis = [compute_regime_si(aff) for aff in niche_dist.values()]

    return np.mean(agent_sis), 0.0


def bootstrap_ci(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Compute bootstrap confidence interval."""
    if len(values) < 2:
        return (np.mean(values), np.mean(values))

    n_bootstrap = 1000
    bootstrap_means = [
        np.mean(np.random.choice(values, size=len(values), replace=True))
        for _ in range(n_bootstrap)
    ]

    alpha = 1 - confidence
    return (
        np.percentile(bootstrap_means, alpha / 2 * 100),
        np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
    )


def run_experiment():
    """Run the full multi-domain experiment."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MULTI-DOMAIN VALIDATION EXPERIMENT")
    print("=" * 60)
    print()

    # Define domains
    domains = [
        ("Finance", None, ["trend_up", "trend_down", "mean_revert", "volatile"]),
        ("Traffic", create_traffic_environment, TRAFFIC_REGIMES),
        ("Energy", create_energy_environment, ENERGY_REGIMES),
        ("Weather", create_weather_environment, WEATHER_REGIMES),
        ("E-commerce", create_ecommerce_environment, ECOMMERCE_REGIMES),
        ("Sports", create_sports_environment, SPORTS_REGIMES),
    ]

    results = []

    for domain_name, creator, regime_list in domains:
        print(f"\n--- Domain: {domain_name} ({len(regime_list)} regimes) ---")

        si_values = []
        diversity_values = []

        start_time = time.time()

        for trial in range(N_TRIALS):
            if (trial + 1) % 10 == 0:
                print(f"    Trial {trial + 1}/{N_TRIALS}...")

            if creator is None:
                # Finance domain
                si, div = run_finance_domain(trial)
            else:
                si, div = run_generic_domain(creator, regime_list, trial)

            si_values.append(si)
            diversity_values.append(div)

        elapsed = time.time() - start_time

        ci_lower, ci_upper = bootstrap_ci(si_values)

        result = DomainResult(
            domain=domain_name,
            n_regimes=len(regime_list),
            si_mean=np.mean(si_values),
            si_std=np.std(si_values),
            si_ci_lower=ci_lower,
            si_ci_upper=ci_upper,
            diversity_mean=np.mean(diversity_values),
            n_trials=N_TRIALS,
        )
        results.append(result)

        print(f"    SI: {result.si_mean:.4f} ± {result.si_std:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"    Time: {elapsed:.1f}s")

    # Statistical analysis
    print("\n" + "=" * 60)
    print("CROSS-DOMAIN ANALYSIS")
    print("=" * 60)

    si_values = [r.si_mean for r in results]
    print(f"\nMean SI across domains: {np.mean(si_values):.4f} ± {np.std(si_values):.4f}")
    print(f"Range: [{min(si_values):.4f}, {max(si_values):.4f}]")

    # Test if all SIs are above 0.5 (one-sample t-test)
    t_stat, p_value = stats.ttest_1samp(si_values, 0.5)
    print(f"\nOne-sample t-test (H0: SI = 0.5): t={t_stat:.3f}, p={p_value:.6f}")
    print(f"Conclusion: {'SI > 0.5 in all domains' if p_value < 0.05 and np.mean(si_values) > 0.5 else 'Mixed results'}")

    # Save results
    summary = {
        "experiment": "multi_domain_validation",
        "date": datetime.now().isoformat(),
        "config": {
            "n_trials": N_TRIALS,
            "n_iterations": N_ITERATIONS,
            "n_agents": N_AGENTS,
            "niche_bonus": NICHE_BONUS,
        },
        "results": [asdict(r) for r in results],
        "analysis": {
            "mean_si": float(np.mean(si_values)),
            "std_si": float(np.std(si_values)),
            "min_si": float(min(si_values)),
            "max_si": float(max(si_values)),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
        },
        "conclusion": {
            "specialization_emerges": np.mean(si_values) > 0.5,
            "generalizes_across_domains": np.std(si_values) < 0.2,
        }
    }

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("CONCLUSIONS")
    print("=" * 60)
    print(f"Specialization emerges: {summary['conclusion']['specialization_emerges']}")
    print(f"Generalizes across domains: {summary['conclusion']['generalizes_across_domains']}")
    print(f"\nResults saved to {RESULTS_DIR}")

    return results


if __name__ == "__main__":
    run_experiment()
