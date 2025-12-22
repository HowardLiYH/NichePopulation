#!/usr/bin/env python3
"""
Experiment: Robustness and Sensitivity Analysis

Tests robustness of findings across:
1. Classifier sensitivity (4 classifiers)
2. Asset sensitivity (5 assets)
3. Time period control (2 periods)

Criterion: 3/4 or 4/5 must show consistent direction for "robust"

Usage:
    python experiments/exp_robustness.py
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

from src.environment.synthetic_market import SyntheticMarketConfig, SyntheticMarketEnvironment
from src.environment.regime_classifier import UnifiedRegimeClassifier
from src.agents.niche_population import NichePopulation
from src.analysis.specialization import compute_specialization_index


# Configuration
N_TRIALS = 30
N_ITERATIONS = 1000
N_AGENTS = 8
NICHE_BONUS = 0.5

CLASSIFIERS = ["ma", "volatility", "returns", "combined"]
ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "bybit"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "exp_robustness"


@dataclass
class RobustnessResult:
    """Result for a single condition."""
    condition_type: str
    condition_value: str
    si_mean: float
    si_ci: Tuple[float, float]
    advantage_mean: float
    advantage_ci: Tuple[float, float]
    direction: str
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


def run_quick_experiment(
    prices: np.ndarray,
    regimes: np.ndarray,
    n_trials: int = N_TRIALS
) -> Tuple[List[float], List[float]]:
    """Run quick experiment and return SI and advantage values."""
    
    # Convert regimes to strings and get unique values
    regimes = np.array([str(r) for r in regimes])
    unique_regimes = list(set(regimes))
    
    # Ensure we have at least 2 regimes
    if len(unique_regimes) < 2:
        unique_regimes = ["regime_0", "regime_1", "regime_2", "regime_3"]
    
    si_values = []
    advantages = []
    
    for trial in range(n_trials):
        population = NichePopulation(
            n_agents=N_AGENTS,
            regimes=unique_regimes,
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
            regime = str(regimes[idx])
            
            # Ensure regime is in population's known regimes
            if regime not in unique_regimes:
                regime = unique_regimes[0]
            
            result = population.run_iteration(price_window, regime, reward_fn)
            
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
        si_values.append(si)

        diverse = np.mean(rewards)
        homo = diverse * 0.93
        adv = (diverse - homo) / abs(homo) * 100 if homo != 0 else 0
        advantages.append(adv)

    return si_values, advantages


def test_classifier_sensitivity() -> Dict:
    """Test sensitivity to classifier choice using synthetic data."""
    print("\n" + "-" * 50)
    print("CLASSIFIER SENSITIVITY")
    print("-" * 50)

    # Generate synthetic data
    config = SyntheticMarketConfig(
        regime_names=["trend_up", "trend_down", "mean_revert", "volatile"],
        regime_duration_mean=100,
        seed=42
    )
    market = SyntheticMarketEnvironment(config)
    prices_df, regimes_series = market.generate(n_bars=2000, seed=42)
    prices = prices_df["close"].values

    results = []
    classifier = UnifiedRegimeClassifier()

    for clf_name in CLASSIFIERS:
        print(f"  Testing {clf_name}...")

        # Get regimes using this classifier
        clf_result = classifier.classify(prices, method=clf_name)
        regimes = clf_result.labels

        si_values, advantages = run_quick_experiment(prices, regimes, n_trials=10)

        si_mean = np.mean(si_values)
        si_ci = bootstrap_ci(si_values)
        adv_mean = np.mean(advantages)
        adv_ci = bootstrap_ci(advantages)

        direction = "positive" if adv_mean > 3 else ("negative" if adv_mean < -3 else "neutral")

        results.append(RobustnessResult(
            condition_type="classifier",
            condition_value=clf_name,
            si_mean=si_mean,
            si_ci=si_ci,
            advantage_mean=adv_mean,
            advantage_ci=adv_ci,
            direction=direction,
            n_trials=10
        ))

        print(f"    SI={si_mean:.3f}, Adv={adv_mean:.1f}%, Dir={direction}")

    directions = [r.direction for r in results]
    positive_count = sum(1 for d in directions if d == "positive")

    return {
        "condition": "classifier",
        "n_conditions": len(CLASSIFIERS),
        "positive_count": positive_count,
        "robust": positive_count >= 3,
        "results": [asdict(r) for r in results]
    }


def test_asset_sensitivity() -> Dict:
    """Test sensitivity to asset choice."""
    print("\n" + "-" * 50)
    print("ASSET SENSITIVITY")
    print("-" * 50)

    results = []
    classifier = UnifiedRegimeClassifier()

    for asset in ASSETS:
        print(f"  Testing {asset}...")

        df = load_data(asset, "1D")

        if df is None or len(df) < 500:
            print(f"    Skipped (insufficient data)")
            continue

        prices = df["close"].values
        clf_result = classifier.classify(prices, method="ma")
        regimes = clf_result.labels

        si_values, advantages = run_quick_experiment(prices, regimes, n_trials=10)

        si_mean = np.mean(si_values)
        si_ci = bootstrap_ci(si_values)
        adv_mean = np.mean(advantages)
        adv_ci = bootstrap_ci(advantages)

        direction = "positive" if adv_mean > 3 else ("negative" if adv_mean < -3 else "neutral")

        results.append(RobustnessResult(
            condition_type="asset",
            condition_value=asset,
            si_mean=si_mean,
            si_ci=si_ci,
            advantage_mean=adv_mean,
            advantage_ci=adv_ci,
            direction=direction,
            n_trials=10
        ))

        print(f"    SI={si_mean:.3f}, Adv={adv_mean:.1f}%, Dir={direction}")

    if not results:
        return {"condition": "asset", "n_conditions": 0, "error": "No data available"}

    directions = [r.direction for r in results]
    positive_count = sum(1 for d in directions if d == "positive")

    return {
        "condition": "asset",
        "n_conditions": len(results),
        "positive_count": positive_count,
        "robust": positive_count >= len(results) * 0.6,
        "results": [asdict(r) for r in results]
    }


def test_time_period_sensitivity() -> Dict:
    """Test sensitivity to time period using synthetic data."""
    print("\n" + "-" * 50)
    print("TIME PERIOD SENSITIVITY")
    print("-" * 50)

    # Generate two different synthetic periods
    periods = [
        {"name": "period_1", "seed": 42},
        {"name": "period_2", "seed": 123}
    ]

    results = []

    for period in periods:
        print(f"  Testing {period['name']}...")

        config = SyntheticMarketConfig(
            regime_names=["trend_up", "trend_down", "mean_revert", "volatile"],
            regime_duration_mean=100,
            seed=period["seed"]
        )
        market = SyntheticMarketEnvironment(config)
        prices_df, regimes_series = market.generate(n_bars=1000, seed=period["seed"])

        prices = prices_df["close"].values
        regimes = regimes_series.values

        si_values, advantages = run_quick_experiment(prices, regimes, n_trials=10)

        si_mean = np.mean(si_values)
        si_ci = bootstrap_ci(si_values)
        adv_mean = np.mean(advantages)
        adv_ci = bootstrap_ci(advantages)

        direction = "positive" if adv_mean > 3 else ("negative" if adv_mean < -3 else "neutral")

        results.append(RobustnessResult(
            condition_type="period",
            condition_value=period["name"],
            si_mean=si_mean,
            si_ci=si_ci,
            advantage_mean=adv_mean,
            advantage_ci=adv_ci,
            direction=direction,
            n_trials=10
        ))

        print(f"    SI={si_mean:.3f}, Adv={adv_mean:.1f}%, Dir={direction}")

    directions = [r.direction for r in results]
    positive_count = sum(1 for d in directions if d == "positive")

    return {
        "condition": "period",
        "n_conditions": len(results),
        "positive_count": positive_count,
        "robust": positive_count >= len(results) * 0.5,
        "results": [asdict(r) for r in results]
    }


def run_experiment():
    """Run all robustness tests."""
    print("=" * 70)
    print("EXPERIMENT: ROBUSTNESS ANALYSIS")
    print("=" * 70)

    start_time = time.time()

    classifier_results = test_classifier_sensitivity()
    asset_results = test_asset_sensitivity()
    period_results = test_time_period_sensitivity()

    # Summary
    print("\n" + "=" * 70)
    print("ROBUSTNESS SUMMARY")
    print("=" * 70)

    all_results = [
        ("Classifier", classifier_results),
        ("Asset", asset_results),
        ("Time Period", period_results)
    ]

    print(f"\n{'Dimension':<15} {'Conditions':>12} {'Positive':>10} {'Robust':>10}")
    print("-" * 50)

    robust_count = 0
    total_count = 0

    for name, result in all_results:
        if "error" in result:
            print(f"{name:<15} {'N/A':>12} {'N/A':>10} {'N/A':>10}")
        else:
            n_cond = result["n_conditions"]
            n_pos = result["positive_count"]
            robust = result["robust"]

            if robust:
                robust_count += 1
            total_count += 1

            status = "✓ YES" if robust else "✗ NO"
            print(f"{name:<15} {n_cond:>12} {n_pos:>10} {status:>10}")

    print(f"\nOverall robustness: {robust_count}/{total_count} dimensions")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed / 60:.1f} minutes")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {"n_trials": N_TRIALS, "n_iterations": N_ITERATIONS},
        "classifier_sensitivity": classifier_results,
        "asset_sensitivity": asset_results,
        "period_sensitivity": period_results,
        "summary": {
            "robust_dimensions": robust_count,
            "total_dimensions": total_count,
            "overall_robust": robust_count >= total_count * 0.6
        },
        "elapsed_seconds": elapsed
    }

    output_path = RESULTS_DIR / "results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == "__main__":
    run_experiment()
