#!/usr/bin/env python3
"""
Power Analysis for Experiment Design

Computes required sample sizes for detecting specified effect sizes
with 80% power at various significance levels (including Bonferroni-corrected).

This justifies our choice of 100 trials per experiment.
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
from pathlib import Path


@dataclass
class PowerResult:
    """Result of power analysis."""
    effect_size: float
    effect_type: str  # "cohen_d" or "correlation"
    alpha: float
    power: float
    required_n: int
    description: str


def compute_required_n_ttest(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.8,
    two_tailed: bool = True
) -> int:
    """
    Compute required sample size for t-test.

    Uses the formula: n = 2 * ((z_alpha + z_beta) / d)^2

    Args:
        effect_size: Cohen's d
        alpha: Significance level
        power: Desired power (1 - beta)
        two_tailed: Whether test is two-tailed

    Returns:
        Required sample size per group
    """
    if effect_size == 0:
        return float('inf')

    # Z-scores
    if two_tailed:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)

    z_beta = stats.norm.ppf(power)

    # Required n per group
    n = 2 * ((z_alpha + z_beta) / effect_size) ** 2

    return int(np.ceil(n))


def compute_required_n_correlation(
    effect_size: float,  # r value
    alpha: float = 0.05,
    power: float = 0.8,
    two_tailed: bool = True
) -> int:
    """
    Compute required sample size for correlation test.

    Uses Fisher's z transformation.

    Args:
        effect_size: Expected correlation r
        alpha: Significance level
        power: Desired power
        two_tailed: Whether test is two-tailed

    Returns:
        Required sample size
    """
    if effect_size == 0:
        return float('inf')

    # Fisher's z transformation
    z_r = 0.5 * np.log((1 + effect_size) / (1 - effect_size))

    # Z-scores
    if two_tailed:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)

    z_beta = stats.norm.ppf(power)

    # Required n
    n = ((z_alpha + z_beta) / z_r) ** 2 + 3

    return int(np.ceil(n))


def compute_observed_power(
    effect_size: float,
    n: int,
    alpha: float = 0.05,
    test_type: str = "ttest"
) -> float:
    """
    Compute observed power given sample size.

    Args:
        effect_size: Observed effect size (Cohen's d or r)
        n: Sample size
        alpha: Significance level
        test_type: "ttest" or "correlation"

    Returns:
        Power (probability of detecting effect)
    """
    if test_type == "ttest":
        # Non-centrality parameter
        ncp = effect_size * np.sqrt(n / 2)

        # Critical t-value
        df = 2 * n - 2
        t_crit = stats.t.ppf(1 - alpha / 2, df)

        # Power
        power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)

    elif test_type == "correlation":
        # Fisher's z
        z_r = 0.5 * np.log((1 + effect_size) / (1 - effect_size))

        # Standard error
        se = 1 / np.sqrt(n - 3)

        # Critical z
        z_crit = stats.norm.ppf(1 - alpha / 2)

        # Power
        power = 1 - stats.norm.cdf(z_crit - z_r / se) + stats.norm.cdf(-z_crit - z_r / se)

    else:
        raise ValueError(f"Unknown test type: {test_type}")

    return power


def run_power_analysis() -> Dict:
    """
    Run comprehensive power analysis for all hypotheses.

    Returns:
        Dictionary with all power analysis results
    """
    results = {}

    # Define hypotheses with expected effect sizes
    hypotheses = [
        {
            "id": "H1",
            "name": "Mono-regime produces low SI",
            "effect_type": "cohen_d",
            "effect_size": 2.0,  # Large effect expected
            "description": "SI difference between 1-regime and 4-regime"
        },
        {
            "id": "H2",
            "name": "SI increases with regime count",
            "effect_type": "correlation",
            "effect_size": 0.9,  # Very strong correlation expected
            "description": "Spearman correlation between n_regimes and SI"
        },
        {
            "id": "H3",
            "name": "Diversity advantage in mixed regimes",
            "effect_type": "cohen_d",
            "effect_size": 0.5,  # Medium effect expected
            "description": "Reward difference (diverse - homogeneous)"
        },
        {
            "id": "H4",
            "name": "SI-entropy positive correlation",
            "effect_type": "correlation",
            "effect_size": 0.3,  # Moderate correlation expected
            "description": "Pearson correlation between regime entropy and SI"
        },
        {
            "id": "H5",
            "name": "Transaction costs reduce SI",
            "effect_type": "correlation",
            "effect_size": -0.5,  # Moderate negative correlation
            "description": "Correlation between cost level and SI"
        },
    ]

    # Alpha levels
    alpha_uncorrected = 0.05
    n_primary_tests = 5
    alpha_bonferroni = alpha_uncorrected / n_primary_tests  # 0.01

    print("=" * 70)
    print("POWER ANALYSIS FOR NEURIPS A+ RIGOR")
    print("=" * 70)
    print(f"\nSettings:")
    print(f"  Target power: 80%")
    print(f"  Uncorrected alpha: {alpha_uncorrected}")
    print(f"  Bonferroni-corrected alpha: {alpha_bonferroni:.4f} ({n_primary_tests} tests)")
    print(f"  Proposed sample size: 100 trials")

    print("\n" + "-" * 70)
    print("REQUIRED SAMPLE SIZES")
    print("-" * 70)

    for hyp in hypotheses:
        effect = abs(hyp["effect_size"])

        if hyp["effect_type"] == "cohen_d":
            n_uncorrected = compute_required_n_ttest(effect, alpha=alpha_uncorrected)
            n_corrected = compute_required_n_ttest(effect, alpha=alpha_bonferroni)
            power_at_100 = compute_observed_power(effect, 100, alpha_bonferroni, "ttest")
        else:
            n_uncorrected = compute_required_n_correlation(effect, alpha=alpha_uncorrected)
            n_corrected = compute_required_n_correlation(effect, alpha=alpha_bonferroni)
            power_at_100 = compute_observed_power(effect, 100, alpha_bonferroni, "correlation")

        result = {
            "id": hyp["id"],
            "name": hyp["name"],
            "effect_type": hyp["effect_type"],
            "effect_size": hyp["effect_size"],
            "n_required_uncorrected": n_uncorrected,
            "n_required_bonferroni": n_corrected,
            "power_at_100_trials": power_at_100,
            "sufficient": n_corrected <= 100
        }
        results[hyp["id"]] = result

        print(f"\n{hyp['id']}: {hyp['name']}")
        print(f"  Effect size: {hyp['effect_type']} = {hyp['effect_size']}")
        print(f"  Required n (α=0.05):    {n_uncorrected}")
        print(f"  Required n (Bonferroni): {n_corrected}")
        print(f"  Power at n=100:          {power_at_100:.1%}")
        print(f"  100 trials sufficient:   {'✓ YES' if n_corrected <= 100 else '✗ NO'}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_sufficient = all(r["sufficient"] for r in results.values())
    min_power = min(r["power_at_100_trials"] for r in results.values())

    print(f"\n100 trials sufficient for all hypotheses: {'✓ YES' if all_sufficient else '✗ NO'}")
    print(f"Minimum power at n=100: {min_power:.1%}")

    if not all_sufficient:
        max_required = max(r["n_required_bonferroni"] for r in results.values())
        print(f"Maximum required n: {max_required}")
        print(f"\nRECOMMENDATION: Increase to {max_required} trials for full power")
    else:
        print(f"\nCONCLUSION: 100 trials provides ≥{min_power:.0%} power for all tests")

    # Effect size interpretation guide
    print("\n" + "-" * 70)
    print("EFFECT SIZE INTERPRETATION")
    print("-" * 70)
    print("\nCohen's d:")
    print("  Small:  d = 0.2")
    print("  Medium: d = 0.5")
    print("  Large:  d = 0.8")
    print("\nCorrelation r:")
    print("  Small:  r = 0.1")
    print("  Medium: r = 0.3")
    print("  Large:  r = 0.5")

    return results


def save_results(results: Dict, output_dir: Path = None):
    """Save power analysis results to JSON."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results" / "power_analysis"

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "power_analysis_results.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    results = run_power_analysis()
    save_results(results)
