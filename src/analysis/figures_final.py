#!/usr/bin/env python3
"""
Generate publication-quality figures for the NeurIPS paper.

Figures:
1. Cross-domain MSE comparison (bar chart)
2. Regime statistics (entropy-SI relationship)
3. Mechanistic analysis (variance box plot)
4. Computational cost comparison
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
OUTPUT_DIR = RESULTS_DIR / "figures"


def load_results(experiment: str) -> Dict:
    """Load experiment results."""
    filepath = RESULTS_DIR / experiment / "results.json"
    if filepath.exists():
        with open(filepath) as f:
            return json.load(f)
    return {}


def fig1_cross_domain_mse():
    """Figure 1: Cross-domain MSE comparison bar chart."""
    print("Generating Figure 1: Cross-domain MSE comparison...")

    results = load_results("unified_prediction")
    if not results or "domains" not in results:
        print("  No unified prediction results found")
        return

    domains = []
    diverse_mses = []
    homo_mses = []
    naive_mses = []
    ma_mses = []

    for domain, data in results["domains"].items():
        if "strategies" in data:
            domains.append(domain.capitalize())
            diverse_mses.append(data["strategies"]["Diverse"]["mse_mean"])
            homo_mses.append(data["strategies"]["Homogeneous"]["mse_mean"])
            naive_mses.append(data["strategies"]["Naive"]["mse_mean"])
            ma_mses.append(data["strategies"]["MA(10)"]["mse_mean"])

    if not domains:
        print("  No valid domain data found")
        return

    # Normalize for visualization (log scale for very different magnitudes)
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(domains))
    width = 0.2

    # Use log scale due to large magnitude differences
    ax.set_yscale('log')

    bars1 = ax.bar(x - 1.5*width, diverse_mses, width, label='Diverse (Ours)', color='#2ecc71', edgecolor='black')
    bars2 = ax.bar(x - 0.5*width, homo_mses, width, label='Homogeneous', color='#e74c3c', edgecolor='black')
    bars3 = ax.bar(x + 0.5*width, naive_mses, width, label='Naive', color='#95a5a6', edgecolor='black')
    bars4 = ax.bar(x + 1.5*width, ma_mses, width, label='MA(10)', color='#3498db', edgecolor='black')

    ax.set_xlabel('Domain')
    ax.set_ylabel('MSE (log scale)')
    ax.set_title('Prediction MSE Across Domains')
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.legend(loc='upper right')

    # Add improvement annotations
    for i, (d, h) in enumerate(zip(diverse_mses, homo_mses)):
        if h != 0:
            improvement = (h - d) / h * 100
            color = 'green' if improvement > 0 else 'red'
            ax.annotate(f'{improvement:+.1f}%',
                       xy=(i, max(d, h)),
                       xytext=(0, 5),
                       textcoords='offset points',
                       ha='center', fontsize=9, color=color)

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / "fig1_cross_domain_mse.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig1_cross_domain_mse.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("  Saved: fig1_cross_domain_mse.pdf")


def fig2_regime_statistics():
    """Figure 2: Regime statistics across domains."""
    print("Generating Figure 2: Regime statistics...")

    results = load_results("regime_statistics")
    if not results or "domains" not in results:
        print("  No regime statistics results found")
        return

    domains = []
    entropies = []
    transition_rates = []
    regime_counts = []

    for domain, data in results["domains"].items():
        if "normalized_entropy" in data:
            domains.append(domain.capitalize())
            entropies.append(data["normalized_entropy"])
            transition_rates.append(data["transition_rate_pct"])
            regime_counts.append(data["regime_count"])

    if not domains:
        print("  No valid regime data found")
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    colors = ['#3498db', '#e74c3c', '#2ecc71']

    # Entropy
    axes[0].bar(domains, entropies, color=colors, edgecolor='black')
    axes[0].set_ylabel('Normalized Entropy')
    axes[0].set_title('Regime Balance')
    axes[0].set_ylim(0, 1)

    # Transition rate
    axes[1].bar(domains, transition_rates, color=colors, edgecolor='black')
    axes[1].set_ylabel('Transition Rate (%)')
    axes[1].set_title('Regime Dynamics')

    # Regime count
    axes[2].bar(domains, regime_counts, color=colors, edgecolor='black')
    axes[2].set_ylabel('Number of Regimes')
    axes[2].set_title('Regime Diversity')

    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "fig2_regime_statistics.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig2_regime_statistics.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("  Saved: fig2_regime_statistics.pdf")


def fig3_mechanistic_variance():
    """Figure 3: Mechanistic analysis - variance comparison."""
    print("Generating Figure 3: Mechanistic variance analysis...")

    results = load_results("mechanistic_analysis")
    if not results or "analyses" not in results:
        print("  No mechanistic analysis results found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Panel A: Variance comparison
    var_data = results["analyses"]["variance_reduction"]
    in_var = var_data["in_niche_variance_mean"]
    out_var = var_data["out_niche_variance_mean"]

    x = [0, 1]
    bars = axes[0].bar(x, [in_var, out_var],
                       color=['#2ecc71', '#e74c3c'],
                       edgecolor='black')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['In-Niche', 'Out-of-Niche'])
    axes[0].set_ylabel('Prediction Variance')
    axes[0].set_title(f'Specialists: {var_data["variance_ratio"]:.1f}x Lower Variance In-Niche')

    # Panel B: MSE decomposition
    bv_data = results["analyses"]["bias_variance"]["summary"]
    spec_mse = bv_data["specialist_mse"]
    gen_mse = bv_data["generalist_mse"]

    x = [0, 1]
    bars = axes[1].bar(x, [spec_mse, gen_mse],
                       color=['#2ecc71', '#95a5a6'],
                       edgecolor='black')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['Specialist', 'Generalist'])
    axes[1].set_ylabel('Mean Squared Error')
    axes[1].set_title(f'MSE Reduction: {bv_data["mse_reduction_pct"]:.1f}%')

    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "fig3_mechanistic.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig3_mechanistic.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("  Saved: fig3_mechanistic.pdf")


def fig4_computational_costs():
    """Figure 4: Computational cost comparison."""
    print("Generating Figure 4: Computational costs...")

    results = load_results("benchmarks")
    if not results or "methods" not in results:
        print("  No benchmark results found")
        return

    methods = []
    train_times = []
    memories = []

    for key, data in results["methods"].items():
        methods.append(data["name"].replace(" (NichePopulation)", ""))
        train_times.append(data["train_time_mean"])
        memories.append(data["memory_peak_mb"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Reorder to put Ours first
    if "Ours" in methods:
        idx = methods.index("Ours")
        methods = [methods[idx]] + [m for i, m in enumerate(methods) if i != idx]
        train_times = [train_times[idx]] + [t for i, t in enumerate(train_times) if i != idx]
        memories = [memories[idx]] + [m for i, m in enumerate(memories) if i != idx]

    colors = ['#2ecc71'] + ['#e74c3c'] * (len(methods) - 1)

    # Training time
    axes[0].barh(methods, train_times, color=colors, edgecolor='black')
    axes[0].set_xlabel('Training Time (seconds)')
    axes[0].set_title('Training Efficiency')
    axes[0].invert_yaxis()

    # Memory
    axes[1].barh(methods, memories, color=colors, edgecolor='black')
    axes[1].set_xlabel('Peak Memory (MB)')
    axes[1].set_title('Memory Efficiency')
    axes[1].invert_yaxis()

    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "fig4_computational.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig4_computational.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("  Saved: fig4_computational.pdf")


def generate_all_figures():
    """Generate all figures."""
    print("="*60)
    print("GENERATING PUBLICATION FIGURES")
    print("="*60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig1_cross_domain_mse()
    fig2_regime_statistics()
    fig3_mechanistic_variance()
    fig4_computational_costs()

    print("\n" + "="*60)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    generate_all_figures()
