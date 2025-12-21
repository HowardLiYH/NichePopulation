"""
Experiment 4: Transfer Frequency Effect

Research Question: How does knowledge transfer frequency affect specialization?

Hypothesis H4: There exists an optimal transfer frequency τ* where:
  - Too frequent (τ < τ*): Agents converge to same strategy (no specialization)
  - Too infrequent (τ > τ*): Agents don't benefit from collective learning

Protocol:
1. Test transfer frequencies τ ∈ {1, 5, 10, 25, 50, 100}
2. For each τ, run 50 trials of 500 iterations
3. Measure: Final SI, Population Diversity, Total Reward

Statistical Analysis:
- One-way ANOVA across frequencies
- Identify phase transition point

Expected Results:
- τ* ≈ 10-25 iterations
- SI maximized at intermediate τ
- Performance relatively stable for τ > 10
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.synthetic_market import SyntheticMarketEnvironment, SyntheticMarketConfig
from src.agents.population import Population, PopulationConfig, compute_reward_from_methods
from src.analysis.specialization import SpecializationTracker, compute_population_diversity

from .config import EXP4_CONFIG, ExperimentConfig


TRANSFER_FREQUENCIES = [1, 5, 10, 25, 50, 100]


@dataclass
class FrequencyTrialResult:
    """Result for a single trial with specific transfer frequency."""
    transfer_frequency: int
    trial_id: int
    seed: int
    final_si: float
    population_diversity: float
    total_reward: float
    method_entropy: float  # Entropy of method usage across population


@dataclass
class FrequencyResult:
    """Aggregate results for a transfer frequency."""
    transfer_frequency: int
    n_trials: int

    si_mean: float
    si_std: float
    diversity_mean: float
    diversity_std: float
    reward_mean: float
    reward_std: float
    entropy_mean: float
    entropy_std: float


@dataclass
class ExperimentResult:
    """Full experiment results."""
    experiment_name: str
    transfer_frequencies: List[int]
    frequency_results: Dict[int, FrequencyResult]
    optimal_frequency: int
    trial_results: List[FrequencyTrialResult]


def compute_method_entropy(population: Population) -> float:
    """Compute entropy of method usage across entire population."""
    all_usage = {}
    for agent in population.agents.values():
        usage = agent.get_method_usage_distribution()
        for method, prob in usage.items():
            all_usage[method] = all_usage.get(method, 0) + prob

    # Normalize
    total = sum(all_usage.values())
    if total == 0:
        return 0.0

    probs = [v / total for v in all_usage.values()]
    entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
    return entropy


def run_frequency_trial(
    transfer_frequency: int,
    trial_id: int,
    seed: int,
    config: ExperimentConfig,
) -> FrequencyTrialResult:
    """
    Run a single trial with specific transfer frequency.
    """
    # Create environment
    env_config = SyntheticMarketConfig(
        regime_names=config.regime_names,
        regime_duration_mean=config.regime_duration_mean,
        regime_duration_std=config.regime_duration_std,
        seed=seed,
    )
    env = SyntheticMarketEnvironment(env_config)
    prices, regimes = env.generate(config.n_bars, seed=seed)

    # Create population with specified transfer frequency
    pop_config = PopulationConfig(
        n_agents=config.n_agents,
        max_methods_per_agent=config.max_methods,
        transfer_frequency=transfer_frequency,
        transfer_tau=config.transfer_tau,
        seed=seed,
    )
    population = Population(pop_config)

    window_size = 20
    tracker = SpecializationTracker(n_methods=11)
    total_reward = 0.0

    for i in range(window_size, min(len(prices) - 1, window_size + 500)):
        current_regime = regimes.iloc[i]
        price_window = prices.iloc[i-window_size:i+1]

        result = population.run_iteration(price_window, compute_reward_from_methods, current_regime)
        total_reward += result.winner_reward

    # Final metrics
    final_distributions = population.get_all_method_usage()
    final_metrics = tracker.record(config.n_bars, final_distributions, regime=regimes.iloc[-1])

    diversity = compute_population_diversity(population.get_all_method_usage())
    entropy = compute_method_entropy(population)

    return FrequencyTrialResult(
        transfer_frequency=transfer_frequency,
        trial_id=trial_id,
        seed=seed,
        final_si=final_metrics.avg_specialization,
        population_diversity=diversity,
        total_reward=total_reward,
        method_entropy=entropy,
    )


def run_experiment(
    config: ExperimentConfig = EXP4_CONFIG,
    transfer_frequencies: List[int] = TRANSFER_FREQUENCIES,
    save_results: bool = True,
) -> ExperimentResult:
    """
    Run the full transfer frequency experiment.
    """
    print(f"=" * 60)
    print(f"Experiment 4: Transfer Frequency Effect")
    print(f"Frequencies: {transfer_frequencies}")
    print(f"Trials per frequency: {config.n_trials}")
    print(f"=" * 60)

    all_trials = []
    frequency_results = {}

    for freq in transfer_frequencies:
        print(f"\nTesting τ={freq}...")

        freq_trials = []
        for trial_id in tqdm(range(config.n_trials), desc=f"τ={freq}"):
            seed = config.base_seed + freq * 10000 + trial_id * 1000
            result = run_frequency_trial(freq, trial_id, seed, config)
            freq_trials.append(result)
            all_trials.append(result)

        # Aggregate for this frequency
        sis = [t.final_si for t in freq_trials]
        diversities = [t.population_diversity for t in freq_trials]
        rewards = [t.total_reward for t in freq_trials]
        entropies = [t.method_entropy for t in freq_trials]

        frequency_results[freq] = FrequencyResult(
            transfer_frequency=freq,
            n_trials=config.n_trials,
            si_mean=float(np.mean(sis)),
            si_std=float(np.std(sis)),
            diversity_mean=float(np.mean(diversities)),
            diversity_std=float(np.std(diversities)),
            reward_mean=float(np.mean(rewards)),
            reward_std=float(np.std(rewards)),
            entropy_mean=float(np.mean(entropies)),
            entropy_std=float(np.std(entropies)),
        )

    # Find optimal frequency (by SI, as we want maximum specialization)
    optimal_frequency = max(frequency_results.keys(), key=lambda f: frequency_results[f].si_mean)

    result = ExperimentResult(
        experiment_name=config.experiment_name,
        transfer_frequencies=transfer_frequencies,
        frequency_results=frequency_results,
        optimal_frequency=optimal_frequency,
        trial_results=all_trials,
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"RESULTS: Experiment 4")
    print(f"{'=' * 60}")
    print(f"{'τ':>5} | {'SI':>10} | {'Diversity':>10} | {'Reward':>12} | {'Entropy':>10}")
    print("-" * 60)
    for freq in transfer_frequencies:
        fr = frequency_results[freq]
        opt = "*" if freq == optimal_frequency else " "
        print(f"{freq:>4}{opt} | {fr.si_mean:.3f}±{fr.si_std:.3f} | "
              f"{fr.diversity_mean:.3f}±{fr.diversity_std:.3f} | "
              f"{fr.reward_mean:>8.2f}±{fr.reward_std:.2f} | "
              f"{fr.entropy_mean:.3f}±{fr.entropy_std:.3f}")
    print(f"\nOptimal transfer frequency: τ* = {optimal_frequency}")
    print(f"{'=' * 60}")

    # Save results
    if save_results:
        results_dir = Path(config.results_dir) / config.experiment_name
        results_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "experiment_name": result.experiment_name,
            "transfer_frequencies": transfer_frequencies,
            "optimal_frequency": optimal_frequency,
            "frequency_results": {
                str(f): {
                    "si_mean": fr.si_mean,
                    "si_std": fr.si_std,
                    "diversity_mean": fr.diversity_mean,
                    "diversity_std": fr.diversity_std,
                    "reward_mean": fr.reward_mean,
                    "reward_std": fr.reward_std,
                    "entropy_mean": fr.entropy_mean,
                    "entropy_std": fr.entropy_std,
                }
                for f, fr in frequency_results.items()
            },
        }

        with open(results_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to {results_dir}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Experiment 4: Transfer Frequency Effect")
    parser.add_argument("--trials", type=int, default=50, help="Trials per frequency")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--frequencies", nargs="+", type=int, default=TRANSFER_FREQUENCIES)
    parser.add_argument("--no-save", action="store_true", help="Don't save results")

    args = parser.parse_args()

    config = ExperimentConfig(
        experiment_name="exp4_transfer_frequency",
        n_trials=args.trials,
        base_seed=args.seed,
    )

    result = run_experiment(config, transfer_frequencies=args.frequencies, save_results=not args.no_save)
