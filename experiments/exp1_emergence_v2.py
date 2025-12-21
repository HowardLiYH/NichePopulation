"""
Experiment 1 (V2): Emergence of Specialists

Tests whether agents naturally specialize to different regimes
using the NichePopulation with competitive exclusion mechanism.

Hypothesis: After sufficient training, agents will develop distinct
niche preferences with SI > 0.5.
"""

import sys
sys.path.insert(0, '.')

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import List, Dict

from src.environment.synthetic_market import SyntheticMarketEnvironment, SyntheticMarketConfig
from src.agents.niche_population import NichePopulation
from src.agents.inventory_v2 import METHOD_INVENTORY_V2


def compute_reward(methods, prices):
    """Compute reward for selected methods."""
    if len(prices) < 2:
        return 0.0
    signals, confs = [], []
    for m in methods:
        if m in METHOD_INVENTORY_V2:
            result = METHOD_INVENTORY_V2[m].execute(prices)
            signals.append(result['signal'])
            confs.append(result['confidence'])
    if not signals:
        return 0.0
    weights = np.array(confs) / (sum(confs) + 1e-8)
    signal = sum(s * w for s, w in zip(signals, weights))
    price_return = (prices['close'].iloc[-1] / prices['close'].iloc[-2]) - 1
    return float(np.clip(signal * price_return * 10, -1, 1))


def compute_regime_si(niche_affinities: Dict[str, float]) -> float:
    """Compute specialization index from regime affinities."""
    affinities = np.array(list(niche_affinities.values()))
    affinities = affinities / (affinities.sum() + 1e-8)
    entropy = -np.sum(affinities * np.log(affinities + 1e-8))
    max_entropy = np.log(len(affinities))
    return 1 - (entropy / max_entropy) if max_entropy > 0 else 0.0


def run_single_trial(
    trial_id: int,
    n_iterations: int = 3000,
    n_agents: int = 8,
    niche_bonus: float = 0.5,
    checkpoints: List[int] = None,
) -> Dict:
    """Run a single trial and return metrics."""
    checkpoints = checkpoints or [0, 500, 1000, 2000, 3000]

    # Create environment
    env = SyntheticMarketEnvironment(SyntheticMarketConfig(
        regime_duration_mean=100,
        seed=trial_id * 1000
    ))
    prices, regimes = env.generate(n_bars=n_iterations + 100)

    # Create population
    pop = NichePopulation(
        n_agents=n_agents,
        niche_bonus=niche_bonus,
        seed=trial_id
    )

    # Track SI over time
    si_trajectory = [0.0]  # Initial
    window_size = 20

    for i in range(window_size, min(len(prices)-1, n_iterations + 50)):
        price_window = prices.iloc[i-window_size:i+1]
        regime = regimes.iloc[i]
        pop.run_iteration(price_window, regime, compute_reward)

        if pop.iteration in checkpoints[1:]:
            niche_dist = pop.get_niche_distribution()
            agent_sis = [compute_regime_si(aff) for aff in niche_dist.values()]
            avg_si = np.mean(agent_sis)
            si_trajectory.append(avg_si)

    # Final metrics
    niche_dist = pop.get_niche_distribution()
    win_matrix = pop.get_regime_win_matrix()

    agent_sis = [compute_regime_si(aff) for aff in niche_dist.values()]
    final_si = np.mean(agent_sis)

    # Diversity (unique primary niches)
    primary_niches = [max(aff, key=aff.get) for aff in niche_dist.values()]
    diversity = len(set(primary_niches)) / 4

    # Specialist win rates
    specialist_win_rates = []
    for regime in pop.regimes:
        best_agent = max(win_matrix, key=lambda a: win_matrix[a][regime])
        specialist_win_rates.append(win_matrix[best_agent][regime])

    return {
        "trial_id": trial_id,
        "final_si": final_si,
        "si_trajectory": si_trajectory,
        "diversity": diversity,
        "avg_specialist_win_rate": np.mean(specialist_win_rates),
        "checkpoints": checkpoints[:len(si_trajectory)],
    }


def run_experiment(
    n_trials: int = 50,
    n_iterations: int = 3000,
    output_dir: str = "results/exp1_emergence_v2",
):
    """Run full experiment with multiple trials."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []
    checkpoints = [0, 500, 1000, 2000, 3000]

    print(f"Running Experiment 1 V2: Emergence of Specialists")
    print(f"Trials: {n_trials}, Iterations: {n_iterations}")
    print()

    for trial in tqdm(range(n_trials), desc="Trials"):
        result = run_single_trial(
            trial_id=trial,
            n_iterations=n_iterations,
            checkpoints=checkpoints,
        )
        results.append(result)

    # Compute summary statistics
    final_sis = [r["final_si"] for r in results]
    diversities = [r["diversity"] for r in results]
    win_rates = [r["avg_specialist_win_rate"] for r in results]

    # Compute SI trajectory mean/std
    trajectories = np.array([r["si_trajectory"] for r in results])
    trajectory_mean = trajectories.mean(axis=0).tolist()
    trajectory_std = trajectories.std(axis=0).tolist()

    # Statistical test: SI > 0 (one-sample t-test)
    from scipy import stats
    t_stat, p_value = stats.ttest_1samp(final_sis, 0.5)  # Test against 0.5
    effect_size = np.mean(final_sis) / (np.std(final_sis) + 1e-8)

    summary = {
        "experiment_name": "exp1_emergence_v2",
        "n_trials": n_trials,
        "n_iterations": n_iterations,
        "final_si_mean": float(np.mean(final_sis)),
        "final_si_std": float(np.std(final_sis)),
        "final_si_ci_lower": float(np.percentile(final_sis, 2.5)),
        "final_si_ci_upper": float(np.percentile(final_sis, 97.5)),
        "diversity_mean": float(np.mean(diversities)),
        "specialist_win_rate_mean": float(np.mean(win_rates)),
        "specialist_win_rate_std": float(np.std(win_rates)),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "effect_size": float(effect_size),
        "significant": str(p_value < 0.05 and np.mean(final_sis) > 0.5),
        "si_trajectory_mean": trajectory_mean,
        "si_trajectory_std": trajectory_std,
        "checkpoints": checkpoints,
    }

    # Save results
    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(output_path / "trials.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print()
    print("=" * 60)
    print("EXPERIMENT 1 V2 RESULTS: Emergence of Specialists")
    print("=" * 60)
    print(f"Final SI: {summary['final_si_mean']:.3f} ± {summary['final_si_std']:.3f}")
    print(f"95% CI: [{summary['final_si_ci_lower']:.3f}, {summary['final_si_ci_upper']:.3f}]")
    print(f"Diversity: {summary['diversity_mean']:.2f}")
    print(f"Specialist Win Rate: {summary['specialist_win_rate_mean']:.1%} ± {summary['specialist_win_rate_std']:.1%}")
    print(f"Effect Size (Cohen's d): {summary['effect_size']:.2f}")
    print(f"p-value (SI > 0.5): {summary['p_value']:.2e}")
    print(f"Significant: {summary['significant']}")
    print()

    return summary


if __name__ == "__main__":
    run_experiment(n_trials=50, n_iterations=3000)
