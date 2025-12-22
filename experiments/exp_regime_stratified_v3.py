#!/usr/bin/env python3
"""
Experiment: Regime-Stratified Real Data Analysis (v3)

Tests hypothesis that specialization value depends on regime heterogeneity.

Hypothesis:
- H3: SI-Performance correlation is positive when regime diversity > threshold

Usage:
    python experiments/exp_regime_stratified_v3.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from scipy import stats
import time

from src.environment.regime_classifier import UnifiedRegimeClassifier
from src.agents.niche_population import NichePopulation
from src.analysis.specialization import compute_specialization_index


# Configuration
N_TRIALS = 50
N_ITERATIONS = 500
N_AGENTS = 8
NICHE_BONUS = 0.3

ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
INTERVALS = ["1D", "4H", "1H"]
CLASSIFIERS = ["ma", "volatility", "returns", "combined"]

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "bybit"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "exp_regime_stratified_v3"


@dataclass
class SegmentResult:
    """Result for a market segment."""
    asset: str
    interval: str
    classifier: str
    regime: str
    n_bars: int
    si_mean: float
    si_std: float
    reward_mean: float
    n_trials: int


def bootstrap_ci(values, confidence=0.95):
    """Bootstrap confidence interval."""
    n = len(values)
    if n < 2:
        return (np.mean(values), np.mean(values))
    means = [np.mean(np.random.choice(values, n, replace=True)) for _ in range(1000)]
    alpha = 1 - confidence
    return (np.percentile(means, alpha/2*100), np.percentile(means, (1-alpha/2)*100))


def load_data(asset: str, interval: str) -> Optional[pd.DataFrame]:
    """Load price data."""
    filepath = DATA_DIR / f"{asset}_{interval}.csv"
    if not filepath.exists():
        return None
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


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


def run_on_segment(
    prices: np.ndarray,
    n_trials: int = N_TRIALS
) -> Tuple[List[float], List[float]]:
    """Run experiment on a price segment."""
    si_values = []
    rewards_list = []

    regimes = ["bull", "bear", "sideways", "volatile"]

    for trial in range(n_trials):
        population = NichePopulation(
            n_agents=N_AGENTS,
            regimes=regimes,
            niche_bonus=NICHE_BONUS,
            seed=trial,
            min_exploration_rate=0.05,
        )

        reward_fn = create_reward_fn()
        rewards = []
        n_iters = min(len(prices), N_ITERATIONS)

        for i in range(n_iters):
            idx = i % len(prices)
            start_idx = max(0, idx - 20)
            price_window = prices[start_idx:idx+1]
            regime = regimes[trial % len(regimes)]  # Cycle through regimes

            result = population.run_iteration(price_window, regime, reward_fn)

            if len(price_window) >= 2:
                ret = (price_window[-1] - price_window[-2]) / price_window[-2]
                rewards.append(ret * 100)

        # Compute SI
        method_usage = population.get_all_method_usage()
        agent_sis = []
        for agent_id, agent_methods in method_usage.items():
            if sum(agent_methods.values()) > 0:
                agent_si = compute_specialization_index(agent_methods)
                agent_sis.append(agent_si)
        si = np.mean(agent_sis) if agent_sis else 0.0

        si_values.append(si)
        rewards_list.append(np.mean(rewards) if rewards else 0)

    return si_values, rewards_list


def run_experiment():
    """Run the stratified experiment."""
    print("=" * 70)
    print("EXPERIMENT: REGIME-STRATIFIED REAL DATA (v3)")
    print("=" * 70)

    start_time = time.time()
    results: List[SegmentResult] = []

    classifier = UnifiedRegimeClassifier()

    for asset in ASSETS:
        for interval in INTERVALS:
            print(f"\n{'='*50}")
            print(f"Asset: {asset}, Interval: {interval}")
            print(f"{'='*50}")

            df = load_data(asset, interval)
            if df is None or len(df) < 100:
                print(f"  Skipped (insufficient data)")
                continue

            prices = df["close"].values

            for clf_name in CLASSIFIERS:
                print(f"\n  Classifier: {clf_name}")

                try:
                    clf_result = classifier.classify(prices, method=clf_name)
                    regimes = clf_result.labels
                except Exception as e:
                    print(f"    Error: {e}")
                    continue

                # Get unique regimes
                unique_regimes = list(set(regimes))

                for regime in unique_regimes:
                    # Get indices for this regime
                    regime_mask = np.array(regimes) == regime
                    regime_prices = prices[regime_mask]

                    if len(regime_prices) < 50:
                        continue

                    si_values, reward_values = run_on_segment(regime_prices, n_trials=min(N_TRIALS, 10))

                    result = SegmentResult(
                        asset=asset,
                        interval=interval,
                        classifier=clf_name,
                        regime=str(regime),
                        n_bars=len(regime_prices),
                        si_mean=np.mean(si_values),
                        si_std=np.std(si_values),
                        reward_mean=np.mean(reward_values),
                        n_trials=len(si_values)
                    )
                    results.append(result)

                    print(f"    Regime {regime}: SI={result.si_mean:.3f}, Reward={result.reward_mean:.2f}")

    # Compute correlation between SI and reward
    if len(results) >= 3:
        sis = [r.si_mean for r in results]
        rewards = [r.reward_mean for r in results]
        r, p = stats.pearsonr(sis, rewards)

        print("\n" + "=" * 70)
        print("CORRELATION ANALYSIS")
        print("=" * 70)
        print(f"SI-Reward Correlation: r={r:.3f}, p={p:.4f}")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed / 60:.1f} minutes")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_trials": N_TRIALS,
            "n_iterations": N_ITERATIONS,
            "assets": ASSETS,
            "intervals": INTERVALS,
            "classifiers": CLASSIFIERS
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
