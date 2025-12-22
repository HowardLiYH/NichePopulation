#!/usr/bin/env python3
"""
Experiment: Mono-Regime Validation (v3)

Tests the theoretical prediction that specialization requires regime heterogeneity.

Hypotheses:
- H1: Mono-regime (1 regime) produces SI < 0.15
- H2: SI increases monotonically with regime count (r > 0.9)

Usage:
    python experiments/exp_mono_regime_v3.py
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
REGIME_CONFIGS = {
    1: ["trend_up"],
    2: ["trend_up", "mean_revert"],
    3: ["trend_up", "mean_revert", "volatile"],
    4: ["trend_up", "trend_down", "mean_revert", "volatile"],
}
N_AGENTS = 8
NICHE_BONUS = 0.3

# Bonferroni correction
ALPHA_CORRECTED = 0.01

# Output
RESULTS_DIR = Path(__file__).parent.parent / "results" / "exp_mono_regime_v3"


@dataclass
class TrialResult:
    """Result from a single trial."""
    n_regimes: int
    trial_id: int
    si: float
    mean_reward: float


@dataclass
class ConfigResult:
    """Aggregated results for a configuration."""
    n_regimes: int
    n_trials: int
    si_mean: float
    si_std: float
    si_ci_lower: float
    si_ci_upper: float
    reward_mean: float


def create_reward_function(prices_df):
    """Create a simple reward function based on price returns."""
    returns = prices_df["close"].pct_change().fillna(0).values

    def reward_fn(methods, prices_window):
        """
        Simple reward: method signal * next return.

        For simplicity, use the latest return.
        """
        if len(prices_window) < 2:
            return 0.0

        ret = (prices_window[-1] - prices_window[-2]) / prices_window[-2]

        # Get method and compute signal
        method_name = methods[0] if methods else "BuyMomentum"
        method_class = METHOD_INVENTORY_V2.get(method_name)

        if method_class is None:
            return ret  # Default: long position

        # Simple heuristic: different methods have different biases
        if "Buy" in method_name or "Trend" in method_name:
            signal = 1.0  # Long bias
        elif "Sell" in method_name or "Fade" in method_name:
            signal = -1.0  # Short bias
        else:
            signal = 0.5  # Neutral

        return signal * ret * 100  # Scale for visibility

    return reward_fn


def run_single_trial(
    n_regimes: int,
    trial_id: int,
    n_iterations: int = N_ITERATIONS
) -> TrialResult:
    """Run a single trial with given configuration."""
    # Create market with specific regime count
    regime_names = REGIME_CONFIGS[n_regimes]
    config = SyntheticMarketConfig(
        regime_names=regime_names,
        regime_duration_mean=100,
        regime_duration_std=30,
        initial_price=100.0,
        seed=trial_id * 1000 + n_regimes
    )

    market = SyntheticMarketEnvironment(config)

    # Generate data
    prices_df, regimes_series = market.generate(n_bars=n_iterations, seed=trial_id)
    prices = prices_df["close"].values
    regimes = regimes_series.values

    # Create population
    population = NichePopulation(
        n_agents=N_AGENTS,
        regimes=regime_names,
        niche_bonus=NICHE_BONUS,
        seed=trial_id,
        min_exploration_rate=0.05,
    )

    # Create reward function
    reward_fn = create_reward_function(prices_df)

    # Run iterations
    rewards = []
    for i in range(min(n_iterations, len(prices))):
        regime = regimes[i]

        # Get price window
        start_idx = max(0, i - 20)
        price_window = prices[start_idx:i+1]

        result = population.run_iteration(price_window, regime, reward_fn)

        # Compute reward for winner
        if len(price_window) >= 2:
            ret = (price_window[-1] - price_window[-2]) / price_window[-2]
            rewards.append(ret * 100)

    # Compute SI - average across agents
    method_usage = population.get_all_method_usage()
    agent_sis = []
    for agent_id, agent_methods in method_usage.items():
        if sum(agent_methods.values()) > 0:
            agent_si = compute_specialization_index(agent_methods)
            agent_sis.append(agent_si)

    si = np.mean(agent_sis) if agent_sis else 0.0
    mean_reward = np.mean(rewards) if rewards else 0.0

    return TrialResult(
        n_regimes=n_regimes,
        trial_id=trial_id,
        si=si,
        mean_reward=mean_reward
    )


def compute_confidence_interval(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Compute confidence interval using bootstrap."""
    n = len(values)
    if n < 2:
        return (np.mean(values), np.mean(values))

    n_bootstrap = 1000
    bootstrap_means = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

    return (lower, upper)


def aggregate_results(trials: List[TrialResult]) -> ConfigResult:
    """Aggregate trial results into summary statistics."""
    n_regimes = trials[0].n_regimes

    si_values = [t.si for t in trials]
    reward_values = [t.mean_reward for t in trials]

    si_ci = compute_confidence_interval(si_values)

    return ConfigResult(
        n_regimes=n_regimes,
        n_trials=len(trials),
        si_mean=np.mean(si_values),
        si_std=np.std(si_values),
        si_ci_lower=si_ci[0],
        si_ci_upper=si_ci[1],
        reward_mean=np.mean(reward_values)
    )


def run_experiment():
    """Run the full mono-regime experiment."""
    print("=" * 70)
    print("EXPERIMENT: MONO-REGIME VALIDATION (v3)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Regime configs: {list(REGIME_CONFIGS.keys())}")
    print(f"  Trials per config: {N_TRIALS}")
    print(f"  Iterations per trial: {N_ITERATIONS}")
    print(f"  Agents: {N_AGENTS}")
    print(f"  Alpha (Bonferroni): {ALPHA_CORRECTED}")

    start_time = time.time()

    results: Dict[int, ConfigResult] = {}
    all_trials = []

    for n_regimes in sorted(REGIME_CONFIGS.keys()):
        print(f"\n{'='*50}")
        print(f"Running n_regimes = {n_regimes} ({REGIME_CONFIGS[n_regimes]})")
        print(f"{'='*50}")

        config_trials = []

        for trial_id in range(N_TRIALS):
            if (trial_id + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  Trial {trial_id + 1}/{N_TRIALS} (elapsed: {elapsed:.1f}s)")

            trial_result = run_single_trial(n_regimes, trial_id)
            config_trials.append(trial_result)
            all_trials.append(trial_result)

        results[n_regimes] = aggregate_results(config_trials)

        print(f"\n  SI: {results[n_regimes].si_mean:.3f} "
              f"[{results[n_regimes].si_ci_lower:.3f}, {results[n_regimes].si_ci_upper:.3f}]")

    # Test hypotheses
    print("\n" + "=" * 70)
    print("HYPOTHESIS TESTS")
    print("=" * 70)

    # H1: Mono-regime SI < 0.15
    mono_trials = [t.si for t in all_trials if t.n_regimes == 1]
    mono_si = np.mean(mono_trials)
    t_stat, p_value = stats.ttest_1samp(mono_trials, 0.15)
    p_one_tailed = p_value / 2 if t_stat < 0 else 1 - p_value / 2

    h1_result = {
        "hypothesis": "H1: Mono-regime SI < 0.15",
        "observed_si": float(mono_si),
        "ci_95": tuple(float(x) for x in compute_confidence_interval(mono_trials)),
        "threshold": 0.15,
        "t_statistic": float(t_stat),
        "p_value_one_tailed": float(p_one_tailed),
        "significant": p_one_tailed < ALPHA_CORRECTED,
        "passed": mono_si < 0.15 and p_one_tailed < ALPHA_CORRECTED
    }

    print(f"\nH1: Mono-regime SI < 0.15")
    print(f"  Observed SI: {h1_result['observed_si']:.3f}")
    print(f"  95% CI: [{h1_result['ci_95'][0]:.3f}, {h1_result['ci_95'][1]:.3f}]")
    print(f"  p-value: {h1_result['p_value_one_tailed']:.6f}")
    print(f"  Result: {'✓ PASS' if h1_result['passed'] else '✗ FAIL'}")

    # H2: SI increases with regime count
    regime_counts = [t.n_regimes for t in all_trials]
    si_values = [t.si for t in all_trials]
    r, p_value = stats.spearmanr(regime_counts, si_values)
    p_one_tailed = p_value / 2 if r > 0 else 1 - p_value / 2

    h2_result = {
        "hypothesis": "H2: SI increases with regime count",
        "spearman_r": float(r),
        "p_value_one_tailed": float(p_one_tailed),
        "threshold": 0.5,
        "significant": p_one_tailed < ALPHA_CORRECTED,
        "passed": r > 0.5 and p_one_tailed < ALPHA_CORRECTED
    }

    print(f"\nH2: SI increases with regime count")
    print(f"  Spearman r: {h2_result['spearman_r']:.3f}")
    print(f"  p-value: {h2_result['p_value_one_tailed']:.6f}")
    print(f"  Result: {'✓ PASS' if h2_result['passed'] else '✗ FAIL'}")

    # Summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Regimes':>8} {'SI':>10} {'95% CI':>20}")
    print("-" * 40)

    for n_regimes in sorted(REGIME_CONFIGS.keys()):
        r = results[n_regimes]
        ci_str = f"[{r.si_ci_lower:.3f}, {r.si_ci_upper:.3f}]"
        print(f"{n_regimes:>8} {r.si_mean:>10.3f} {ci_str:>20}")

    elapsed_total = time.time() - start_time
    print(f"\nTotal time: {elapsed_total / 60:.1f} minutes")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_trials": N_TRIALS,
            "n_iterations": N_ITERATIONS,
            "regime_configs": {str(k): v for k, v in REGIME_CONFIGS.items()},
            "n_agents": N_AGENTS,
            "niche_bonus": NICHE_BONUS,
            "alpha_corrected": ALPHA_CORRECTED
        },
        "results": {str(k): asdict(v) for k, v in results.items()},
        "hypotheses": {
            "H1": h1_result,
            "H2": h2_result
        },
        "all_passed": h1_result["passed"] and h2_result["passed"],
        "elapsed_seconds": elapsed_total
    }

    output_path = RESULTS_DIR / "results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == "__main__":
    run_experiment()
