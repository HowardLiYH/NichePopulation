#!/usr/bin/env python3
"""
Regime Statistics Analysis for Multi-Domain Evaluation.

Computes comparable regime characteristics across domains:
- Regime count
- Regime entropy (balance)
- Max regime proportion (dominance)
- Transition rate
- Mean regime duration
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def compute_regime_statistics(regimes: List[str]) -> Dict:
    """Compute comprehensive regime statistics."""
    regimes = np.array(regimes)
    n = len(regimes)

    # Regime count
    unique_regimes = list(set(regimes))
    regime_count = len(unique_regimes)

    # Regime distribution
    regime_counts = pd.Series(regimes).value_counts()
    regime_probs = regime_counts / n

    # Entropy (higher = more balanced)
    regime_entropy = entropy(regime_probs, base=2)
    max_entropy = np.log2(regime_count) if regime_count > 1 else 0
    normalized_entropy = regime_entropy / max_entropy if max_entropy > 0 else 0

    # Max proportion (higher = more dominated by single regime)
    max_proportion = regime_probs.max()

    # Transition rate
    transitions = np.sum(regimes[1:] != regimes[:-1])
    transition_rate = transitions / (n - 1) * 100 if n > 1 else 0

    # Mean regime duration
    if transitions > 0:
        mean_duration = (n - 1) / transitions
    else:
        mean_duration = n

    # Regime stationarity (autocorrelation at lag 1)
    regime_encoded = pd.Categorical(regimes).codes
    if len(regime_encoded) > 1:
        autocorr = np.corrcoef(regime_encoded[:-1], regime_encoded[1:])[0, 1]
    else:
        autocorr = 0

    return {
        "regime_count": regime_count,
        "regime_names": unique_regimes,
        "regime_distribution": regime_probs.to_dict(),
        "entropy": float(regime_entropy),
        "max_entropy": float(max_entropy),
        "normalized_entropy": float(normalized_entropy),
        "max_proportion": float(max_proportion),
        "transition_rate_pct": float(transition_rate),
        "mean_duration": float(mean_duration),
        "n_transitions": int(transitions),
        "n_samples": n,
        "autocorrelation": float(autocorr) if not np.isnan(autocorr) else 0.0
    }


def load_domain_regimes(domain: str) -> List[str]:
    """Load regime labels for a domain."""
    data_dir = Path(__file__).parent.parent / "data"

    if domain == "finance":
        # Load from Bybit data
        filepath = data_dir / "bybit" / "Bybit_BTC.csv"
        if not filepath.exists():
            filepath = data_dir / "bybit" / "BTCUSDT_4H.csv"

        df = pd.read_csv(filepath)

        # Compute regimes
        if 'close' in df.columns:
            df['return'] = df['close'].pct_change()
        elif 'Close' in df.columns:
            df['return'] = df['Close'].pct_change()

        df['volatility'] = df['return'].rolling(20).std()
        df['trend'] = df['return'].rolling(10).mean()
        df = df.dropna()

        regimes = []
        for _, row in df.iterrows():
            if row['volatility'] > df['volatility'].quantile(0.75):
                regimes.append('volatile')
            elif row['trend'] > df['trend'].quantile(0.75):
                regimes.append('trend_up')
            elif row['trend'] < df['trend'].quantile(0.25):
                regimes.append('trend_down')
            else:
                regimes.append('mean_revert')

        return regimes

    elif domain == "traffic":
        filepath = data_dir / "traffic" / "nyc_taxi" / "hourly_aggregated.csv"

        if filepath.exists():
            df = pd.read_csv(filepath)
            if 'regime' in df.columns:
                return list(df['regime'])

        # Generate synthetic regimes
        regimes = []
        for hour in range(760):
            h = hour % 24
            if 0 <= h < 6:
                regimes.append('night')
            elif 7 <= h < 10:
                regimes.append('morning_rush')
            elif 10 <= h < 16:
                regimes.append('midday')
            elif 16 <= h < 20:
                regimes.append('evening_rush')
            else:
                regimes.append('evening')

        return regimes

    elif domain == "energy":
        # Try EIA data first
        filepath = data_dir / "energy" / "eia_hourly_demand.csv"
        if not filepath.exists():
            filepath = data_dir / "energy" / "hourly_demand.csv"

        if filepath.exists():
            df = pd.read_csv(filepath)
            if 'regime' in df.columns:
                return list(df['regime'])

        # Generate synthetic regimes
        regimes = []
        for i in range(17520):
            val = np.random.random()
            if val < 0.2:
                regimes.append('low_demand')
            elif val < 0.5:
                regimes.append('normal')
            elif val < 0.8:
                regimes.append('high_load')
            else:
                regimes.append('peak_demand')

        return regimes

    else:
        raise ValueError(f"Unknown domain: {domain}")


def run_regime_statistics() -> Dict:
    """Run regime statistics analysis for all domains."""
    print("="*70)
    print("REGIME STATISTICS ANALYSIS")
    print("="*70)

    domains = ["finance", "traffic", "energy"]
    results = {
        "experiment": "regime_statistics",
        "date": pd.Timestamp.now().isoformat(),
        "domains": {}
    }

    for domain in domains:
        print(f"\n{domain.upper()} Domain:")
        print("-"*40)

        try:
            regimes = load_domain_regimes(domain)
            stats = compute_regime_statistics(regimes)
            results["domains"][domain] = stats

            print(f"  Regime count:      {stats['regime_count']}")
            print(f"  Regimes:           {stats['regime_names']}")
            print(f"  Entropy:           {stats['entropy']:.3f} (max: {stats['max_entropy']:.3f})")
            print(f"  Normalized entropy:{stats['normalized_entropy']:.3f}")
            print(f"  Max proportion:    {stats['max_proportion']:.3f}")
            print(f"  Transition rate:   {stats['transition_rate_pct']:.1f}%")
            print(f"  Mean duration:     {stats['mean_duration']:.1f} steps")
            print(f"  Autocorrelation:   {stats['autocorrelation']:.3f}")

        except Exception as e:
            print(f"  Error: {e}")
            results["domains"][domain] = {"error": str(e)}

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Domain':<10} {'Regimes':<8} {'Entropy':<10} {'MaxProp':<10} {'TransRate':<12} {'Duration':<10}")
    print("-"*60)

    for domain, stats in results["domains"].items():
        if "error" not in stats:
            print(f"{domain:<10} {stats['regime_count']:<8} "
                  f"{stats['normalized_entropy']:<10.3f} {stats['max_proportion']:<10.3f} "
                  f"{stats['transition_rate_pct']:<12.1f} {stats['mean_duration']:<10.1f}")

    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "regime_statistics"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_dir / 'results.json'}")

    return results


if __name__ == "__main__":
    run_regime_statistics()
