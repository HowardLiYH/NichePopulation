#!/usr/bin/env python3
"""
Within-Trial SI-Performance Correlation Analysis.

Uses data from lambda_zero_real experiment to compute correlation
between SI and performance improvement across trials.

Key insight: With 30 trials per domain × 3 domains = 90 data points,
we have sufficient statistical power to detect meaningful correlations.
"""

import json
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_lambda_results() -> Dict:
    """Load results from lambda_zero_real experiment."""
    results_path = Path(__file__).parent.parent / "results" / "lambda_zero_real" / "results_full.json"

    if not results_path.exists():
        # Try the summary file
        results_path = Path(__file__).parent.parent / "results" / "lambda_zero_real" / "results.json"

    if not results_path.exists():
        raise FileNotFoundError(f"Lambda results not found at {results_path}")

    with open(results_path) as f:
        return json.load(f)


def extract_trial_data(results: Dict) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Extract SI and performance data for each trial.

    Returns: (si_values, performance_values, domain_labels)
    """
    si_values = []
    perf_values = []
    domain_labels = []

    domains = ["energy", "weather", "finance"]

    for domain in domains:
        if domain not in results["results"]:
            continue

        # Get λ=0.5 results (the main experiment setting)
        domain_data = results["results"][domain].get("0.5", results["results"][domain].get("0.0"))

        if "si_all" in domain_data and "performance_all" in domain_data:
            si_all = domain_data["si_all"]
            perf_all = domain_data["performance_all"]

            for si, perf in zip(si_all, perf_all):
                si_values.append(si)
                # Convert MSE to improvement metric (lower is better, so negate)
                perf_values.append(-perf)  # Negative because lower MSE = better
                domain_labels.append(domain)

    return np.array(si_values), np.array(perf_values), domain_labels


def compute_correlations(si: np.ndarray, perf: np.ndarray) -> Dict:
    """Compute Pearson and Spearman correlations with CIs."""

    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(si, perf)

    # Spearman correlation (rank-based, more robust)
    spearman_r, spearman_p = stats.spearmanr(si, perf)

    # Bootstrap CI for Spearman
    n_bootstrap = 1000
    bootstrap_correlations = []
    n = len(si)
    np.random.seed(42)

    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        boot_si = si[indices]
        boot_perf = perf[indices]
        r, _ = stats.spearmanr(boot_si, boot_perf)
        bootstrap_correlations.append(r)

    ci_lower = np.percentile(bootstrap_correlations, 2.5)
    ci_upper = np.percentile(bootstrap_correlations, 97.5)

    return {
        "pearson": {
            "r": float(pearson_r),
            "p_value": float(pearson_p),
            "significant_005": bool(pearson_p < 0.05)
        },
        "spearman": {
            "r": float(spearman_r),
            "p_value": float(spearman_p),
            "significant_005": bool(spearman_p < 0.05),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper)
        },
        "n_points": int(len(si))
    }


def create_correlation_plot(si: np.ndarray, perf: np.ndarray,
                           domain_labels: list, correlations: Dict,
                           output_path: Path):
    """Create correlation scatter plot."""

    fig, ax = plt.subplots(figsize=(8, 6))

    # Color by domain
    colors = {"energy": "#2ecc71", "weather": "#3498db", "finance": "#e74c3c"}

    for domain in ["energy", "weather", "finance"]:
        mask = [d == domain for d in domain_labels]
        ax.scatter(
            si[mask], perf[mask],
            c=colors[domain],
            label=domain.capitalize(),
            alpha=0.7,
            s=50
        )

    # Add regression line
    z = np.polyfit(si, perf, 1)
    p = np.poly1d(z)
    x_line = np.linspace(si.min(), si.max(), 100)
    ax.plot(x_line, p(x_line), "k--", alpha=0.5, label="Trend")

    # Labels and title
    spearman_r = correlations["spearman"]["r"]
    spearman_p = correlations["spearman"]["p_value"]
    ax.set_xlabel("Specialization Index (SI)", fontsize=12)
    ax.set_ylabel("Prediction Performance (negative MSE)", fontsize=12)
    ax.set_title(f"Within-Trial SI-Performance Correlation\n"
                 f"Spearman r = {spearman_r:.3f}, p = {spearman_p:.4f}", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()

    print(f"Plot saved to {output_path}")


def main():
    print("="*70)
    print("Within-Trial SI-Performance Correlation Analysis")
    print("="*70)

    # Load data
    try:
        results = load_lambda_results()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run exp_lambda_zero_real.py first.")
        return

    # Extract trial data
    si, perf, domains = extract_trial_data(results)

    if len(si) == 0:
        print("No trial data found. Check if si_all and performance_all are in results.")
        return

    print(f"\nData extracted: {len(si)} data points across {len(set(domains))} domains")
    print(f"  Energy: {sum(1 for d in domains if d == 'energy')} trials")
    print(f"  Weather: {sum(1 for d in domains if d == 'weather')} trials")
    print(f"  Finance: {sum(1 for d in domains if d == 'finance')} trials")

    # Compute correlations
    correlations = compute_correlations(si, perf)

    print("\n" + "-"*70)
    print("CORRELATION RESULTS")
    print("-"*70)
    print(f"Pearson r:  {correlations['pearson']['r']:.3f} (p = {correlations['pearson']['p_value']:.4f})")
    print(f"Spearman r: {correlations['spearman']['r']:.3f} (p = {correlations['spearman']['p_value']:.4f})")
    print(f"95% CI:     [{correlations['spearman']['ci_lower']:.3f}, {correlations['spearman']['ci_upper']:.3f}]")
    print(f"N points:   {correlations['n_points']}")

    if correlations['spearman']['significant_005']:
        print("\n✓ Significant positive correlation detected!")
        print("  Higher SI is associated with better prediction performance.")
    else:
        print("\n✗ No significant correlation detected.")
        print("  SI may be a population property, not individual predictor.")

    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "within_trial_correlation"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(correlations, f, indent=2)

    # Create plot
    figures_dir = Path(__file__).parent.parent / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    create_correlation_plot(
        si, perf, domains, correlations,
        figures_dir / "fig_within_trial_correlation.png"
    )

    print(f"\nResults saved to {output_dir}")

    # Per-domain analysis
    print("\n" + "-"*70)
    print("PER-DOMAIN ANALYSIS")
    print("-"*70)

    for domain in ["energy", "weather", "finance"]:
        mask = np.array([d == domain for d in domains])
        if mask.sum() > 5:
            domain_si = si[mask]
            domain_perf = perf[mask]
            r, p = stats.spearmanr(domain_si, domain_perf)
            print(f"{domain.capitalize():10} r = {r:+.3f} (p = {p:.4f}) {'✓' if p < 0.05 else ''}")

    return correlations


if __name__ == "__main__":
    main()
