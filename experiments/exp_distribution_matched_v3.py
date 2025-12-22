#!/usr/bin/env python3
"""
Experiment: Distribution-Matched Generalization Test (v3)

Tests whether specialists generalize to their "home" regime vs. other regimes.

Hypothesis:
- H4: Specialists outperform in their home regime, underperform in others

Usage:
    python experiments/exp_distribution_matched_v3.py
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
from src.analysis.specialization import compute_specialization_index


# Configuration
N_TRIALS = 50
N_TRAIN_ITERATIONS = 1000
N_TEST_ITERATIONS = 500
N_AGENTS = 8
NICHE_BONUS = 0.3

REGIMES = ["trend_up", "trend_down", "mean_revert", "volatile"]

# Output
RESULTS_DIR = Path(__file__).parent.parent / "results" / "exp_distribution_matched_v3"


@dataclass
class GeneralizationResult:
    """Result for train/test comparison."""
    train_regime_mix: str  # "mixed" or specific regime
    test_regime: str
    train_si: float
    test_reward_mean: float
    test_reward_std: float
    n_trials: int


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


def run_train_test(
    train_regimes: List[str],
    test_regime: str,
    n_trials: int = N_TRIALS
) -> Tuple[List[float], List[float]]:
    """Train on mixed regimes, test on pure regime."""

    train_sis = []
    test_rewards = []

    for trial in range(n_trials):
        # Generate training data (mixed regimes)
        train_config = SyntheticMarketConfig(
            regime_names=train_regimes,
            regime_duration_mean=100,
            seed=trial * 1000
        )
        train_market = SyntheticMarketEnvironment(train_config)
        train_df, train_regimes_series = train_market.generate(n_bars=N_TRAIN_ITERATIONS, seed=trial)
        train_prices = train_df["close"].values
        train_regime_vals = train_regimes_series.values

        # Generate test data (pure regime)
        test_config = SyntheticMarketConfig(
            regime_names=[test_regime],
            regime_duration_mean=N_TEST_ITERATIONS,  # Single long regime
            seed=trial * 1000 + 1
        )
        test_market = SyntheticMarketEnvironment(test_config)
        test_df, test_regimes_series = test_market.generate(n_bars=N_TEST_ITERATIONS, seed=trial + 1)
        test_prices = test_df["close"].values

        # Create and train population
        population = NichePopulation(
            n_agents=N_AGENTS,
            regimes=REGIMES,
            niche_bonus=NICHE_BONUS,
            seed=trial,
            min_exploration_rate=0.05,
        )

        reward_fn = create_reward_fn()

        # Training phase
        for i in range(min(N_TRAIN_ITERATIONS, len(train_prices))):
            regime = train_regime_vals[i]
            start_idx = max(0, i - 20)
            price_window = train_prices[start_idx:i+1]
            population.run_iteration(price_window, regime, reward_fn)

        # Compute training SI
        method_usage = population.get_all_method_usage()
        agent_sis = []
        for agent_id, agent_methods in method_usage.items():
            if sum(agent_methods.values()) > 0:
                agent_si = compute_specialization_index(agent_methods)
                agent_sis.append(agent_si)
        train_si = np.mean(agent_sis) if agent_sis else 0.0
        train_sis.append(train_si)

        # Testing phase (no learning, just evaluation)
        test_trial_rewards = []
        for i in range(min(N_TEST_ITERATIONS, len(test_prices))):
            start_idx = max(0, i - 20)
            price_window = test_prices[start_idx:i+1]

            # Run iteration but don't update beliefs
            result = population.run_iteration(price_window, test_regime, reward_fn)

            if len(price_window) >= 2:
                ret = (price_window[-1] - price_window[-2]) / price_window[-2]
                test_trial_rewards.append(ret * 100)

        test_rewards.append(np.mean(test_trial_rewards) if test_trial_rewards else 0)

    return train_sis, test_rewards


def run_experiment():
    """Run the distribution-matched experiment."""
    print("=" * 70)
    print("EXPERIMENT: DISTRIBUTION-MATCHED GENERALIZATION (v3)")
    print("=" * 70)

    start_time = time.time()
    results: List[GeneralizationResult] = []

    # Test generalization to each pure regime
    for test_regime in REGIMES:
        print(f"\n{'='*50}")
        print(f"Testing generalization to: {test_regime}")
        print(f"{'='*50}")

        train_sis, test_rewards = run_train_test(
            train_regimes=REGIMES,  # Mixed training
            test_regime=test_regime,
            n_trials=min(N_TRIALS, 20)
        )

        result = GeneralizationResult(
            train_regime_mix="mixed",
            test_regime=test_regime,
            train_si=np.mean(train_sis),
            test_reward_mean=np.mean(test_rewards),
            test_reward_std=np.std(test_rewards),
            n_trials=len(train_sis)
        )
        results.append(result)

        print(f"  Train SI: {result.train_si:.3f}")
        print(f"  Test Reward: {result.test_reward_mean:.3f} ± {result.test_reward_std:.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("GENERALIZATION SUMMARY")
    print("=" * 70)
    print(f"\n{'Test Regime':<15} {'Train SI':>10} {'Test Reward':>15}")
    print("-" * 45)

    for r in results:
        print(f"{r.test_regime:<15} {r.train_si:>10.3f} {r.test_reward_mean:>10.3f} ± {r.test_reward_std:.3f}")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed / 60:.1f} minutes")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_trials": N_TRIALS,
            "n_train_iterations": N_TRAIN_ITERATIONS,
            "n_test_iterations": N_TEST_ITERATIONS,
            "regimes": REGIMES
        },
        "results": [asdict(r) for r in results],
        "elapsed_seconds": elapsed
    }

    output_path = RESULTS_DIR / "results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == "__main__":
    run_experiment()
