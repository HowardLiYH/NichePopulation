"""
Experiment 3: Population Size Effect

Research Question: What is the optimal population size for emergence?

Hypothesis H3: There exists an optimal population size N* where:
  - Too few agents (N < N*): Insufficient coverage of regime space
  - Too many agents (N > N*): Redundancy and slower convergence

Protocol:
1. Test population sizes N ∈ {3, 5, 7, 10, 15, 20}
2. For each N, run 50 trials of 500 iterations
3. Measure: Final SI, Regime Coverage, Total Reward, Convergence Speed

Statistical Analysis:
- One-way ANOVA across population sizes
- Post-hoc Tukey HSD for pairwise comparisons
- Polynomial regression for optimal N

Expected Results:
- Optimal N ≈ 5-7 for 4-regime environment
- Performance plateau after N*
- SI peaks at intermediate N
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
from src.analysis.specialization import SpecializationTracker, compute_regime_coverage

from .config import EXP3_CONFIG, ExperimentConfig


POPULATION_SIZES = [3, 5, 7, 10, 15, 20]


@dataclass
class SizeTrialResult:
    """Result for a single trial with specific population size."""
    population_size: int
    trial_id: int
    seed: int
    final_si: float
    regime_coverage: float
    total_reward: float
    convergence_iteration: int  # When SI first exceeds 0.5


@dataclass
class SizeResult:
    """Aggregate results for a population size."""
    population_size: int
    n_trials: int

    # SI statistics
    si_mean: float
    si_std: float

    # Coverage statistics
    coverage_mean: float
    coverage_std: float

    # Reward statistics
    reward_mean: float
    reward_std: float

    # Convergence statistics
    convergence_mean: float
    convergence_std: float


@dataclass
class ExperimentResult:
    """Full experiment results."""
    experiment_name: str
    population_sizes: List[int]
    size_results: Dict[int, SizeResult]
    optimal_size: int
    optimal_criterion: str  # "reward", "si", "coverage"
    trial_results: List[SizeTrialResult]


def run_size_trial(
    population_size: int,
    trial_id: int,
    seed: int,
    config: ExperimentConfig,
) -> SizeTrialResult:
    """
    Run a single trial with specific population size.
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

    # Create population with specified size
    pop_config = PopulationConfig(
        n_agents=population_size,
        max_methods_per_agent=config.max_methods,
        transfer_frequency=config.transfer_frequency,
        transfer_tau=config.transfer_tau,
        seed=seed,
    )
    population = Population(pop_config)

    window_size = 20
    tracker = SpecializationTracker(n_methods=11)

    convergence_iteration = config.n_bars  # Default: never converged
    total_reward = 0.0

    for i in range(window_size, min(len(prices) - 1, window_size + 500)):
        iteration = i - window_size + 1
        current_regime = regimes.iloc[i]
        price_window = prices.iloc[i-window_size:i+1]

        result = population.run_iteration(price_window, compute_reward_from_methods, current_regime)
        total_reward += result.winner_reward

        # Check for convergence (SI > 0.5)
        if iteration % 50 == 0:
            distributions = population.get_all_method_usage()
            metrics = tracker.record(iteration, distributions, regime=current_regime)

            if metrics.avg_specialization > 0.5 and convergence_iteration == config.n_bars:
                convergence_iteration = iteration

    # Final metrics
    final_distributions = population.get_all_method_usage()
    final_metrics = tracker.record(config.n_bars, final_distributions, regime=regimes.iloc[-1])

    # Compute regime coverage
    regime_coverage = compute_regime_coverage(
        population.agents,
        config.regime_names,
    )

    return SizeTrialResult(
        population_size=population_size,
        trial_id=trial_id,
        seed=seed,
        final_si=final_metrics.avg_specialization,
        regime_coverage=regime_coverage,
        total_reward=total_reward,
        convergence_iteration=convergence_iteration,
    )


def run_experiment(
    config: ExperimentConfig = EXP3_CONFIG,
    population_sizes: List[int] = POPULATION_SIZES,
    save_results: bool = True,
) -> ExperimentResult:
    """
    Run the full population size experiment.
    """
    print(f"=" * 60)
    print(f"Experiment 3: Population Size Effect")
    print(f"Sizes: {population_sizes}")
    print(f"Trials per size: {config.n_trials}")
    print(f"=" * 60)

    all_trials = []
    size_results = {}

    for pop_size in population_sizes:
        print(f"\nTesting N={pop_size}...")

        size_trials = []
        for trial_id in tqdm(range(config.n_trials), desc=f"N={pop_size}"):
            seed = config.base_seed + pop_size * 10000 + trial_id * 1000
            result = run_size_trial(pop_size, trial_id, seed, config)
            size_trials.append(result)
            all_trials.append(result)

        # Aggregate for this size
        sis = [t.final_si for t in size_trials]
        coverages = [t.regime_coverage for t in size_trials]
        rewards = [t.total_reward for t in size_trials]
        convergences = [t.convergence_iteration for t in size_trials]

        size_results[pop_size] = SizeResult(
            population_size=pop_size,
            n_trials=config.n_trials,
            si_mean=float(np.mean(sis)),
            si_std=float(np.std(sis)),
            coverage_mean=float(np.mean(coverages)),
            coverage_std=float(np.std(coverages)),
            reward_mean=float(np.mean(rewards)),
            reward_std=float(np.std(rewards)),
            convergence_mean=float(np.mean(convergences)),
            convergence_std=float(np.std(convergences)),
        )

    # Find optimal size (by reward)
    optimal_size = max(size_results.keys(), key=lambda n: size_results[n].reward_mean)

    result = ExperimentResult(
        experiment_name=config.experiment_name,
        population_sizes=population_sizes,
        size_results=size_results,
        optimal_size=optimal_size,
        optimal_criterion="reward",
        trial_results=all_trials,
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"RESULTS: Experiment 3")
    print(f"{'=' * 60}")
    print(f"{'N':>4} | {'SI':>10} | {'Coverage':>10} | {'Reward':>12} | {'Converge':>10}")
    print("-" * 60)
    for n in population_sizes:
        sr = size_results[n]
        opt = "*" if n == optimal_size else " "
        print(f"{n:>3}{opt} | {sr.si_mean:.3f}±{sr.si_std:.3f} | "
              f"{sr.coverage_mean:.3f}±{sr.coverage_std:.3f} | "
              f"{sr.reward_mean:>8.2f}±{sr.reward_std:.2f} | "
              f"{sr.convergence_mean:>6.0f}±{sr.convergence_std:.0f}")
    print(f"\nOptimal population size: N* = {optimal_size}")
    print(f"{'=' * 60}")

    # Save results
    if save_results:
        results_dir = Path(config.results_dir) / config.experiment_name
        results_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "experiment_name": result.experiment_name,
            "population_sizes": population_sizes,
            "optimal_size": optimal_size,
            "optimal_criterion": "reward",
            "size_results": {
                str(n): {
                    "si_mean": sr.si_mean,
                    "si_std": sr.si_std,
                    "coverage_mean": sr.coverage_mean,
                    "coverage_std": sr.coverage_std,
                    "reward_mean": sr.reward_mean,
                    "reward_std": sr.reward_std,
                    "convergence_mean": sr.convergence_mean,
                    "convergence_std": sr.convergence_std,
                }
                for n, sr in size_results.items()
            },
        }

        with open(results_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to {results_dir}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Experiment 3: Population Size Effect")
    parser.add_argument("--trials", type=int, default=50, help="Trials per size")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--sizes", nargs="+", type=int, default=POPULATION_SIZES, help="Population sizes to test")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")

    args = parser.parse_args()

    config = ExperimentConfig(
        experiment_name="exp3_population_size",
        n_trials=args.trials,
        base_seed=args.seed,
    )

    result = run_experiment(config, population_sizes=args.sizes, save_results=not args.no_save)
