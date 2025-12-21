"""
Experiment 1: Emergence of Specialists

Research Question: Do agents naturally specialize without supervision?

Hypothesis H1: After sufficient training, agent populations will exhibit
specialization indices significantly higher than random initialization.

Protocol:
1. Initialize population of N=5 agents with uniform preferences
2. Run 500 training iterations on synthetic data (4 regimes)
3. Track SI at intervals [0, 50, 100, 200, 300, 400, 500]
4. Repeat 100 times with different seeds

Statistical Analysis:
- One-sample t-test: H₀: SI_final = SI_initial
- Effect size: Cohen's d
- Power analysis: n=100 trials provides 95% power for d=0.5

Expected Results:
- SI increases from ~0.1 (uniform) to ~0.6-0.8 (specialized)
- Convergence within 200-300 iterations
- Different agents converge to different methods
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.synthetic_market import SyntheticMarketEnvironment, SyntheticMarketConfig
from src.agents.population import Population, PopulationConfig, compute_reward_from_methods
from src.analysis.specialization import SpecializationTracker, compute_specialization_index
from src.analysis.statistical_tests import paired_t_test, bootstrap_confidence_interval

from .config import EXP1_CONFIG, ExperimentConfig


@dataclass
class TrialResult:
    """Result of a single trial."""
    trial_id: int
    seed: int
    si_trajectory: Dict[str, List[float]]  # agent_id -> SI over time
    avg_si_trajectory: List[float]
    final_avg_si: float
    initial_avg_si: float
    dominant_methods: Dict[str, str]  # agent_id -> dominant method
    total_reward: float
    checkpoint_iterations: List[int]


@dataclass
class ExperimentResult:
    """Aggregate result of experiment."""
    experiment_name: str
    n_trials: int

    # SI statistics
    initial_si_mean: float
    initial_si_std: float
    final_si_mean: float
    final_si_std: float

    # Statistical test
    t_statistic: float
    p_value: float
    effect_size: float
    significant: bool

    # Confidence intervals
    final_si_ci_lower: float
    final_si_ci_upper: float

    # Trajectories (averaged across trials)
    avg_si_trajectory: List[float]
    checkpoint_iterations: List[int]

    # Raw trial results for detailed analysis
    trial_results: List[TrialResult]


def run_single_trial(
    trial_id: int,
    seed: int,
    config: ExperimentConfig = EXP1_CONFIG,
) -> TrialResult:
    """
    Run a single trial of the emergence experiment.

    Args:
        trial_id: Trial identifier
        seed: Random seed for this trial
        config: Experiment configuration

    Returns:
        TrialResult with specialization trajectory
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

    # Create population
    pop_config = PopulationConfig(
        n_agents=config.n_agents,
        max_methods_per_agent=config.max_methods,
        transfer_frequency=config.transfer_frequency,
        transfer_tau=config.transfer_tau,
        seed=seed,
    )
    population = Population(pop_config)

    # Tracking
    tracker = SpecializationTracker(n_methods=11)  # 11 methods in inventory
    si_by_agent = {agent_id: [] for agent_id in population.agents}
    avg_si_list = []
    checkpoint_si = []

    window_size = 20

    # Record initial SI (should be ~0.1 for uniform)
    initial_distributions = population.get_all_method_usage()
    initial_metrics = tracker.record(0, initial_distributions, regime=regimes.iloc[window_size] if len(regimes) > window_size else None)

    for agent_id in population.agents:
        si_by_agent[agent_id].append(initial_metrics.agent_metrics[list(population.agents.keys()).index(agent_id)].specialization_index)
    avg_si_list.append(initial_metrics.avg_specialization)

    # Training loop
    for i in range(window_size, min(config.n_bars - 1, window_size + 500)):
        iteration = i - window_size + 1

        # Get current regime
        current_regime = regimes.iloc[i]

        # Get price window
        price_window = prices.iloc[i-window_size:i+1]

        # Run population iteration
        result = population.run_iteration(
            prices=price_window,
            reward_fn=compute_reward_from_methods,
            regime=current_regime,
        )

        # Record at checkpoints
        if iteration in config.checkpoint_iterations:
            distributions = population.get_all_method_usage()
            metrics = tracker.record(iteration, distributions, regime=current_regime)

            for idx, agent_id in enumerate(population.agents):
                si_by_agent[agent_id].append(metrics.agent_metrics[idx].specialization_index)
            avg_si_list.append(metrics.avg_specialization)
            checkpoint_si.append(iteration)

    # Get final dominant methods
    dominant_methods = {}
    for agent_id, agent in population.agents.items():
        dominant_methods[agent_id] = agent.get_dominant_method() or "None"

    # Compute total reward
    total_reward = sum(population.cumulative_rewards.values())

    return TrialResult(
        trial_id=trial_id,
        seed=seed,
        si_trajectory=si_by_agent,
        avg_si_trajectory=avg_si_list,
        final_avg_si=avg_si_list[-1] if avg_si_list else 0.0,
        initial_avg_si=avg_si_list[0] if avg_si_list else 0.0,
        dominant_methods=dominant_methods,
        total_reward=total_reward,
        checkpoint_iterations=config.checkpoint_iterations[:len(avg_si_list)],
    )


def run_experiment(
    config: ExperimentConfig = EXP1_CONFIG,
    save_results: bool = True,
) -> ExperimentResult:
    """
    Run the full emergence experiment.

    Args:
        config: Experiment configuration
        save_results: Whether to save results to disk

    Returns:
        ExperimentResult with aggregate statistics
    """
    print(f"=" * 60)
    print(f"Experiment 1: Emergence of Specialists")
    print(f"Trials: {config.n_trials}, Iterations: {config.n_bars}")
    print(f"=" * 60)

    trial_results = []

    # Run trials
    for trial_id in tqdm(range(config.n_trials), desc="Running trials"):
        seed = config.base_seed + trial_id * 1000
        result = run_single_trial(trial_id, seed, config)
        trial_results.append(result)

    # Aggregate results
    initial_sis = [r.initial_avg_si for r in trial_results]
    final_sis = [r.final_avg_si for r in trial_results]

    # Statistical test: paired t-test (initial vs final)
    test_result = paired_t_test(
        np.array(final_sis),
        np.array(initial_sis),
        alpha=0.05,
    )

    # Bootstrap CI for final SI
    point_est, ci_lower, ci_upper = bootstrap_confidence_interval(
        np.array(final_sis),
        statistic=np.mean,
        confidence=0.95,
        n_bootstrap=10000,
        seed=config.base_seed,
    )

    # Average trajectory across trials
    n_checkpoints = len(config.checkpoint_iterations)
    avg_trajectory = []
    for i in range(n_checkpoints):
        checkpoint_sis = [
            r.avg_si_trajectory[i]
            for r in trial_results
            if len(r.avg_si_trajectory) > i
        ]
        if checkpoint_sis:
            avg_trajectory.append(np.mean(checkpoint_sis))

    result = ExperimentResult(
        experiment_name=config.experiment_name,
        n_trials=config.n_trials,
        initial_si_mean=float(np.mean(initial_sis)),
        initial_si_std=float(np.std(initial_sis)),
        final_si_mean=float(np.mean(final_sis)),
        final_si_std=float(np.std(final_sis)),
        t_statistic=test_result.statistic,
        p_value=test_result.p_value,
        effect_size=test_result.effect_size or 0.0,
        significant=test_result.significant,
        final_si_ci_lower=ci_lower,
        final_si_ci_upper=ci_upper,
        avg_si_trajectory=avg_trajectory,
        checkpoint_iterations=config.checkpoint_iterations[:len(avg_trajectory)],
        trial_results=trial_results,
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"RESULTS: Experiment 1")
    print(f"{'=' * 60}")
    print(f"Initial SI: {result.initial_si_mean:.4f} ± {result.initial_si_std:.4f}")
    print(f"Final SI:   {result.final_si_mean:.4f} ± {result.final_si_std:.4f}")
    print(f"95% CI:     [{result.final_si_ci_lower:.4f}, {result.final_si_ci_upper:.4f}]")
    print(f"t-statistic: {result.t_statistic:.4f}")
    print(f"p-value:     {result.p_value:.6f}")
    print(f"Effect size: {result.effect_size:.4f} (Cohen's d)")
    print(f"Significant: {result.significant}")
    print(f"{'=' * 60}")

    # Save results
    if save_results:
        results_dir = Path(config.results_dir) / config.experiment_name
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save summary (without full trial results)
        summary = {
            "experiment_name": result.experiment_name,
            "n_trials": result.n_trials,
            "initial_si_mean": float(result.initial_si_mean),
            "initial_si_std": float(result.initial_si_std),
            "final_si_mean": float(result.final_si_mean),
            "final_si_std": float(result.final_si_std),
            "t_statistic": float(result.t_statistic),
            "p_value": float(result.p_value),
            "effect_size": float(result.effect_size),
            "significant": bool(result.significant),
            "final_si_ci_lower": float(result.final_si_ci_lower),
            "final_si_ci_upper": float(result.final_si_ci_upper),
            "avg_si_trajectory": [float(x) for x in result.avg_si_trajectory],
            "checkpoint_iterations": list(result.checkpoint_iterations),
        }

        with open(results_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to {results_dir}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Experiment 1: Emergence of Specialists")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")

    args = parser.parse_args()

    config = ExperimentConfig(
        experiment_name="exp1_emergence",
        n_trials=args.trials,
        base_seed=args.seed,
    )

    result = run_experiment(config, save_results=not args.no_save)
