"""
Experiment 2: Value of Diversity

Research Question: Does emergent specialization improve collective performance?

Hypothesis H2: A diverse population of specialists outperforms both:
  (a) A homogeneous population (all agents identical)
  (b) Random method selection
  (c) Single best-method strategies

Protocol:
1. Train diverse population (N=5) for 500 iterations
2. Compare against baselines on SAME test data:
   - Homogeneous population (5 copies of best agent)
   - Random selection
   - Single-method strategies (MomentumFollow, MeanRevert, etc.)
   - Oracle Specialist (upper bound)
3. Repeat 100 times with different seeds

Statistical Analysis:
- Paired t-tests for each comparison
- Bonferroni correction for multiple comparisons
- Effect size for practical significance

Expected Results:
- Diverse > Homogeneous (regime adaptability)
- Diverse > Random (learned specialization)
- Diverse < Oracle (but approaching)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.synthetic_market import SyntheticMarketEnvironment, SyntheticMarketConfig
from src.agents.population import Population, PopulationConfig, compute_reward_from_methods
from src.baselines.oracle import OracleSpecialist
from src.baselines.homogeneous import HomogeneousPopulation
from src.baselines.random_selection import RandomSelectionPopulation as RandomSelector
from src.baselines.simple_strategies import (
    MomentumStrategy,
    MeanReversionStrategy,
    BuyAndHold,
)
from src.analysis.statistical_tests import paired_t_test, bonferroni_correction

from .config import EXP2_CONFIG, ExperimentConfig


@dataclass
class BaselineResult:
    """Result for a single baseline."""
    name: str
    total_reward: float
    avg_reward_per_bar: float
    sharpe_ratio: float
    win_rate: float


@dataclass
class TrialResult:
    """Result of a single trial."""
    trial_id: int
    seed: int
    diverse_reward: float
    baseline_rewards: Dict[str, float]
    diverse_sharpe: float
    baseline_sharpes: Dict[str, float]


@dataclass
class ComparisonResult:
    """Statistical comparison result."""
    baseline_name: str
    diverse_mean: float
    baseline_mean: float
    t_statistic: float
    p_value: float
    p_value_corrected: float
    effect_size: float
    significant: bool
    diverse_wins: bool


@dataclass
class ExperimentResult:
    """Aggregate result of experiment."""
    experiment_name: str
    n_trials: int

    # Performance summary
    diverse_reward_mean: float
    diverse_reward_std: float

    # Baseline comparisons
    comparisons: List[ComparisonResult]

    # Ranking
    performance_ranking: List[str]

    # Raw results
    trial_results: List[TrialResult]


def evaluate_strategy(
    strategy,
    prices: pd.DataFrame,
    regimes: pd.Series,
    window_size: int = 20,
) -> Tuple[float, float, float]:
    """
    Evaluate a strategy on price data.

    Returns:
        Tuple of (total_reward, sharpe_ratio, win_rate)
    """
    rewards = []

    for i in range(window_size, len(prices) - 1):
        current_regime = regimes.iloc[i]
        price_window = prices.iloc[i-window_size:i+1]

        # Get strategy's method selection or signal
        try:
            if hasattr(strategy, 'select'):
                # Baselines like Oracle, Homogeneous
                select_method = strategy.select
                import inspect
                sig = inspect.signature(select_method)
                if len(sig.parameters) > 0:
                    methods = select_method(current_regime)
                else:
                    methods = select_method()
                if isinstance(methods, dict):
                    # HomogeneousPopulation returns dict
                    methods = list(methods.values())[0] if methods else ["Hold"]
            elif hasattr(strategy, 'get_signal'):
                # Simple strategies like BuyAndHold, Momentum
                signal = strategy.get_signal(price_window)
                # Convert signal to reward directly
                next_return = (prices["close"].iloc[min(i+1, len(prices)-1)] / prices["close"].iloc[i]) - 1
                reward = signal.get("signal", 0) * next_return
                rewards.append(reward)
                continue
            else:
                methods = ["Hold"]
        except Exception as e:
            methods = ["Hold"]

        # Compute reward
        reward = compute_reward_from_methods(methods, price_window, current_regime)
        rewards.append(reward)

    if not rewards:
        return 0.0, 0.0, 0.0

    total_reward = sum(rewards)
    sharpe = np.mean(rewards) / (np.std(rewards) + 1e-8) * np.sqrt(252)
    win_rate = sum(1 for r in rewards if r > 0) / len(rewards)

    return total_reward, sharpe, win_rate


def run_single_trial(
    trial_id: int,
    seed: int,
    config: ExperimentConfig = EXP2_CONFIG,
) -> TrialResult:
    """
    Run a single trial comparing diverse population vs baselines.
    """
    # Create environment
    env_config = SyntheticMarketConfig(
        regime_names=config.regime_names,
        regime_duration_mean=config.regime_duration_mean,
        regime_duration_std=config.regime_duration_std,
        seed=seed,
    )
    env = SyntheticMarketEnvironment(env_config)

    # Generate data
    prices, regimes = env.generate(config.n_bars, seed=seed)

    window_size = 20

    # === TRAIN DIVERSE POPULATION ===
    pop_config = PopulationConfig(
        n_agents=config.n_agents,
        max_methods_per_agent=config.max_methods,
        transfer_frequency=config.transfer_frequency,
        transfer_tau=config.transfer_tau,
        seed=seed,
    )
    diverse_pop = Population(pop_config)

    diverse_rewards = []
    for i in range(window_size, len(prices) - 1):
        current_regime = regimes.iloc[i]
        price_window = prices.iloc[i-window_size:i+1]
        result = diverse_pop.run_iteration(price_window, compute_reward_from_methods, current_regime)
        diverse_rewards.append(result.winner_reward)

    diverse_total = sum(diverse_rewards)
    diverse_sharpe = np.mean(diverse_rewards) / (np.std(diverse_rewards) + 1e-8) * np.sqrt(252)

    # === EVALUATE BASELINES ===
    baselines = {
        "Oracle": OracleSpecialist(),
        "Homogeneous": HomogeneousPopulation(n_agents=config.n_agents, seed=seed),
        "Random": RandomSelector(seed=seed),
        "Momentum": MomentumStrategy(),
        "MeanReversion": MeanReversionStrategy(),
        "BuyAndHold": BuyAndHold(),
    }

    baseline_rewards = {}
    baseline_sharpes = {}

    for name, strategy in baselines.items():
        total, sharpe, _ = evaluate_strategy(strategy, prices, regimes, window_size)
        baseline_rewards[name] = total
        baseline_sharpes[name] = sharpe

    return TrialResult(
        trial_id=trial_id,
        seed=seed,
        diverse_reward=diverse_total,
        baseline_rewards=baseline_rewards,
        diverse_sharpe=diverse_sharpe,
        baseline_sharpes=baseline_sharpes,
    )


def run_experiment(
    config: ExperimentConfig = EXP2_CONFIG,
    save_results: bool = True,
) -> ExperimentResult:
    """
    Run the full diversity value experiment.
    """
    print(f"=" * 60)
    print(f"Experiment 2: Value of Diversity")
    print(f"Trials: {config.n_trials}")
    print(f"=" * 60)

    trial_results = []

    # Run trials
    for trial_id in tqdm(range(config.n_trials), desc="Running trials"):
        seed = config.base_seed + trial_id * 1000
        result = run_single_trial(trial_id, seed, config)
        trial_results.append(result)

    # Aggregate diverse performance
    diverse_rewards = [r.diverse_reward for r in trial_results]

    # Statistical comparisons
    baseline_names = list(trial_results[0].baseline_rewards.keys())
    raw_p_values = []
    comparisons = []

    for baseline_name in baseline_names:
        baseline_rewards = [r.baseline_rewards[baseline_name] for r in trial_results]

        test_result = paired_t_test(
            np.array(diverse_rewards),
            np.array(baseline_rewards),
            alpha=0.05,
        )
        raw_p_values.append(test_result.p_value)

        comparisons.append(ComparisonResult(
            baseline_name=baseline_name,
            diverse_mean=float(np.mean(diverse_rewards)),
            baseline_mean=float(np.mean(baseline_rewards)),
            t_statistic=test_result.statistic,
            p_value=test_result.p_value,
            p_value_corrected=0.0,  # Will be updated
            effect_size=test_result.effect_size or 0.0,
            significant=False,  # Will be updated
            diverse_wins=np.mean(diverse_rewards) > np.mean(baseline_rewards),
        ))

    # Apply Bonferroni correction
    corrected_p, corrected_sig = bonferroni_correction(raw_p_values, alpha=0.05)
    for i, comp in enumerate(comparisons):
        comp.p_value_corrected = corrected_p[i]
        comp.significant = corrected_sig[i]

    # Performance ranking
    all_means = {"Diverse": np.mean(diverse_rewards)}
    for comp in comparisons:
        all_means[comp.baseline_name] = comp.baseline_mean

    performance_ranking = sorted(all_means.keys(), key=lambda x: all_means[x], reverse=True)

    result = ExperimentResult(
        experiment_name=config.experiment_name,
        n_trials=config.n_trials,
        diverse_reward_mean=float(np.mean(diverse_rewards)),
        diverse_reward_std=float(np.std(diverse_rewards)),
        comparisons=comparisons,
        performance_ranking=performance_ranking,
        trial_results=trial_results,
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"RESULTS: Experiment 2")
    print(f"{'=' * 60}")
    print(f"Diverse Population: {result.diverse_reward_mean:.4f} ± {result.diverse_reward_std:.4f}")
    print(f"\nComparisons (Bonferroni-corrected):")
    for comp in comparisons:
        status = "✓" if comp.diverse_wins and comp.significant else "✗"
        sig = "*" if comp.significant else ""
        print(f"  vs {comp.baseline_name:20s}: {comp.baseline_mean:8.4f} | "
              f"p={comp.p_value_corrected:.4f}{sig} | d={comp.effect_size:.2f} {status}")

    print(f"\nPerformance Ranking:")
    for i, name in enumerate(performance_ranking):
        print(f"  {i+1}. {name}: {all_means[name]:.4f}")
    print(f"{'=' * 60}")

    # Save results
    if save_results:
        results_dir = Path(config.results_dir) / config.experiment_name
        results_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "experiment_name": result.experiment_name,
            "n_trials": result.n_trials,
            "diverse_reward_mean": result.diverse_reward_mean,
            "diverse_reward_std": result.diverse_reward_std,
            "performance_ranking": result.performance_ranking,
            "comparisons": [
                {
                    "baseline": c.baseline_name,
                    "diverse_mean": c.diverse_mean,
                    "baseline_mean": c.baseline_mean,
                    "p_value": c.p_value,
                    "p_value_corrected": float(c.p_value_corrected),
                    "effect_size": float(c.effect_size),
                    "significant": bool(c.significant),
                    "diverse_wins": bool(c.diverse_wins),
                }
                for c in comparisons
            ],
        }

        with open(results_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to {results_dir}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Experiment 2: Value of Diversity")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")

    args = parser.parse_args()

    config = ExperimentConfig(
        experiment_name="exp2_diversity_value",
        n_trials=args.trials,
        base_seed=args.seed,
    )

    result = run_experiment(config, save_results=not args.no_save)
