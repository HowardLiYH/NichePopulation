#!/usr/bin/env python3
"""
Experiment: Transaction Cost Phase Transition Analysis (v3)

Tests how transaction costs affect specialization and diversity value.

Hypothesis:
- H5: Transaction costs reduce SI (slope < -0.1 per 0.1% cost)

Usage:
    python experiments/exp_cost_transition_v3.py
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

from src.environment.synthetic_market import SyntheticMarketConfig, SyntheticMarketEnvironment
from src.agents.niche_population import NichePopulation
from src.agents.inventory_v2 import METHOD_INVENTORY_V2
from src.analysis.specialization import compute_specialization_index


# Configuration
N_TRIALS = 100
N_ITERATIONS = 2000
COST_LEVELS = np.arange(0.0, 1.05, 0.1).tolist()  # 0% to 1% in 0.1% steps (fewer for speed)
N_AGENTS = 8
NICHE_BONUS = 0.3

# Bonferroni correction
ALPHA_CORRECTED = 0.01

# Output
RESULTS_DIR = Path(__file__).parent.parent / "results" / "exp_cost_transition_v3"


@dataclass
class CostResult:
    """Result for a single cost level."""
    cost_percent: float
    si_mean: float
    si_std: float
    si_ci: Tuple[float, float]
    switching_freq_mean: float
    diverse_mean: float
    n_trials: int


def apply_transaction_cost(reward: float, switched: bool, cost_percent: float) -> float:
    """Apply transaction cost to reward."""
    if switched:
        cost = abs(reward) * (cost_percent / 100)
        return reward - cost
    return reward


def create_reward_fn():
    """Create a simple reward function."""
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
            signal = 0.5
        return signal * ret * 100
    return reward_fn


def run_cost_experiment(cost_percent: float, n_trials: int = N_TRIALS) -> CostResult:
    """Run experiment for a single cost level."""

    si_values = []
    switching_freqs = []
    diverse_rewards = []

    for trial in range(n_trials):
        # Create market
        config = SyntheticMarketConfig(
            regime_names=["trend_up", "trend_down", "mean_revert", "volatile"],
            regime_duration_mean=100,
            regime_duration_std=30,
            seed=trial * 1000
        )
        market = SyntheticMarketEnvironment(config)

        # Generate data
        prices_df, regimes_series = market.generate(n_bars=N_ITERATIONS, seed=trial)
        prices = prices_df["close"].values
        regimes = regimes_series.values

        # Create population
        population = NichePopulation(
            n_agents=N_AGENTS,
            regimes=["trend_up", "trend_down", "mean_revert", "volatile"],
            niche_bonus=NICHE_BONUS,
            seed=trial,
            min_exploration_rate=0.05,
        )

        reward_fn = create_reward_fn()

        trial_rewards = []
        switches = 0
        prev_method = None

        for i in range(len(prices)):
            regime = regimes[i]
            start_idx = max(0, i - 20)
            price_window = prices[start_idx:i+1]

            result = population.run_iteration(price_window, regime, reward_fn)

            current_method = result.get("winner_method", None)
            switched = (prev_method is not None and current_method != prev_method)
            if switched:
                switches += 1
            prev_method = current_method

            if len(price_window) >= 2:
                ret = (price_window[-1] - price_window[-2]) / price_window[-2]
                adjusted_reward = apply_transaction_cost(ret * 100, switched, cost_percent)
                trial_rewards.append(adjusted_reward)

        # Compute SI
        method_usage = population.get_all_method_usage()
        agent_sis = []
        for agent_id, agent_methods in method_usage.items():
            if sum(agent_methods.values()) > 0:
                agent_si = compute_specialization_index(agent_methods)
                agent_sis.append(agent_si)
        si = np.mean(agent_sis) if agent_sis else 0.0

        si_values.append(si)
        switching_freqs.append(switches / len(prices))
        diverse_rewards.append(np.mean(trial_rewards) if trial_rewards else 0)

    # Compute statistics
    def bootstrap_ci(values, confidence=0.95):
        n = len(values)
        if n < 2:
            return (np.mean(values), np.mean(values))
        means = [np.mean(np.random.choice(values, n, replace=True)) for _ in range(1000)]
        alpha = 1 - confidence
        return (np.percentile(means, alpha/2*100), np.percentile(means, (1-alpha/2)*100))

    si_ci = bootstrap_ci(si_values)

    return CostResult(
        cost_percent=cost_percent,
        si_mean=np.mean(si_values),
        si_std=np.std(si_values),
        si_ci=si_ci,
        switching_freq_mean=np.mean(switching_freqs),
        diverse_mean=np.mean(diverse_rewards),
        n_trials=n_trials
    )


def run_experiment():
    """Run the full cost transition experiment."""
    print("=" * 70)
    print("EXPERIMENT: TRANSACTION COST PHASE TRANSITION (v3)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Cost levels: {len(COST_LEVELS)} ({min(COST_LEVELS):.2f}% to {max(COST_LEVELS):.2f}%)")
    print(f"  Trials per level: {N_TRIALS}")
    print(f"  Iterations per trial: {N_ITERATIONS}")

    start_time = time.time()
    results: List[CostResult] = []

    for i, cost in enumerate(COST_LEVELS):
        print(f"\n{'='*50}")
        print(f"Cost level {i+1}/{len(COST_LEVELS)}: {cost:.2f}%")
        print(f"{'='*50}")

        result = run_cost_experiment(cost, n_trials=min(N_TRIALS, 30))
        results.append(result)

        print(f"  SI: {result.si_mean:.3f} [{result.si_ci[0]:.3f}, {result.si_ci[1]:.3f}]")
        print(f"  Switching freq: {result.switching_freq_mean:.3f}")

    # Test hypothesis H5
    print("\n" + "=" * 70)
    print("HYPOTHESIS TESTS")
    print("=" * 70)

    costs = [r.cost_percent for r in results]
    sis = [r.si_mean for r in results]

    slope, intercept, r_value, p_value, std_err = stats.linregress(costs, sis)
    p_one_tailed = p_value / 2 if slope < 0 else 1 - p_value / 2

    h5_result = {
        "hypothesis": "H5: Costs reduce SI",
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_value**2),
        "p_value_one_tailed": float(p_one_tailed),
        "significant": p_one_tailed < ALPHA_CORRECTED,
        "passed": slope < 0 and p_one_tailed < ALPHA_CORRECTED
    }

    print(f"\nH5: Transaction costs reduce SI")
    print(f"  Slope: {h5_result['slope']:.4f}")
    print(f"  R²: {h5_result['r_squared']:.3f}")
    print(f"  p-value: {h5_result['p_value_one_tailed']:.4f}")
    print(f"  Result: {'✓ PASS' if h5_result['passed'] else '✗ FAIL'}")

    # Summary table
    print("\n" + "=" * 70)
    print("RESULTS BY COST LEVEL")
    print("=" * 70)
    print(f"\n{'Cost':>6} {'SI':>10} {'95% CI':>20} {'Switch':>10}")
    print("-" * 50)

    for r in results:
        ci_str = f"[{r.si_ci[0]:.3f}, {r.si_ci[1]:.3f}]"
        print(f"{r.cost_percent:>5.2f}% {r.si_mean:>10.3f} {ci_str:>20} {r.switching_freq_mean:>10.3f}")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed / 60:.1f} minutes")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "cost_levels": COST_LEVELS,
            "n_trials": N_TRIALS,
            "n_iterations": N_ITERATIONS,
            "n_agents": N_AGENTS
        },
        "results": [asdict(r) for r in results],
        "hypotheses": {"H5": h5_result},
        "elapsed_seconds": elapsed
    }

    output_path = RESULTS_DIR / "results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == "__main__":
    run_experiment()
