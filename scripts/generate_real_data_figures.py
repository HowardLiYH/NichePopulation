#!/usr/bin/env python3
"""
Generate publication-quality figures for NeurIPS paper.

Figures:
1. Cross-Domain SI Comparison (bar chart)
2. MARL Baseline Comparison (grouped bar)
3. SI vs Baseline Improvement (scatter)
4. Regime Distribution by Domain (stacked bar)
5. Summary Heatmap
"""

import os
import sys
import json
from pathlib import Path

import numpy as np

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not available")

sys.path.insert(0, str(Path(__file__).parent.parent))


# Load results
RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_results():
    """Load experiment results."""
    results = {}
    
    # Real data experiments
    real_path = RESULTS_DIR / "real_data_v2" / "latest_results.json"
    if real_path.exists():
        with open(real_path) as f:
            results['real_data'] = json.load(f)
    
    # MARL comparison
    marl_path = RESULTS_DIR / "marl_comparison" / "latest_results.json"
    if marl_path.exists():
        with open(marl_path) as f:
            results['marl'] = json.load(f)
    
    return results


def fig1_cross_domain_si(results: dict, output_dir: Path):
    """Figure 1: Cross-Domain Specialization Index."""
    if not HAS_MPL:
        print("Skipping figure 1 - no matplotlib")
        return
    
    data = results.get('real_data', {})
    if not data:
        return
    
    domains = list(data.keys())
    sis = [data[d]['mean_si'] for d in domains]
    stds = [data[d]['std_si'] for d in domains]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    bars = ax.bar(domains, sis, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=1.2)
    
    ax.axhline(y=0.25, color='gray', linestyle='--', linewidth=1, label='Random Baseline')
    
    ax.set_ylabel('Specialization Index (SI)', fontsize=12)
    ax.set_xlabel('Domain', fontsize=12)
    ax.set_title('Emergent Specialization Across 4 Real Data Domains', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 0.6)
    
    # Add value labels
    for bar, si in zip(bars, sis):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{si:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_cross_domain_si.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig1_cross_domain_si.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: fig1_cross_domain_si.pdf")


def fig2_marl_comparison(results: dict, output_dir: Path):
    """Figure 2: MARL Baseline Comparison."""
    if not HAS_MPL:
        print("Skipping figure 2 - no matplotlib")
        return
    
    data = results.get('marl', {})
    if not data:
        return
    
    domains = list(data.keys())
    
    methods = ['niche_population', 'iql', 'random']
    method_labels = ['Niche Population\n(Ours)', 'IQL', 'Random']
    
    x = np.arange(len(domains))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = ['#2E86AB', '#F18F01', '#888888']
    
    for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
        sis = [data[d]['results'][method]['mean_si'] for d in domains]
        stds = [data[d]['results'][method]['std_si'] for d in domains]
        
        bars = ax.bar(x + i*width, sis, width, label=label, color=color, 
                      yerr=stds, capsize=3, edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Specialization Index (SI)', fontsize=12)
    ax.set_xlabel('Domain', fontsize=12)
    ax.set_title('Comparison with MARL Baselines (4 Real Data Domains)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([d.capitalize() for d in domains])
    ax.set_ylim(0, 0.6)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_marl_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig2_marl_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: fig2_marl_comparison.pdf")


def fig3_improvement_vs_baseline(results: dict, output_dir: Path):
    """Figure 3: SI Improvement vs Random Baseline."""
    if not HAS_MPL:
        print("Skipping figure 3 - no matplotlib")
        return
    
    data = results.get('real_data', {})
    if not data:
        return
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    domains = list(data.keys())
    improvements = [data[d].get('improvement_pct', 0) for d in domains]
    sis = [data[d]['mean_si'] for d in domains]
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    scatter = ax.scatter(sis, improvements, c=colors, s=200, edgecolors='black', linewidth=1.5)
    
    for i, domain in enumerate(domains):
        ax.annotate(domain.capitalize(), (sis[i], improvements[i]), 
                   textcoords="offset points", xytext=(10, 5), fontsize=11)
    
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    
    ax.set_xlabel('Specialization Index (SI)', fontsize=12)
    ax.set_ylabel('Improvement vs Random Baseline (%)', fontsize=12)
    ax.set_title('SI Correlates with Improvement Over Baseline', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_improvement_scatter.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig3_improvement_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: fig3_improvement_scatter.pdf")


def fig4_regime_distribution(results: dict, output_dir: Path):
    """Figure 4: Regime Distribution by Domain."""
    if not HAS_MPL:
        print("Skipping figure 4 - no matplotlib")
        return
    
    data = results.get('real_data', {})
    if not data:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    domains = list(data.keys())
    
    for i, domain in enumerate(domains):
        if i >= len(axes):
            break
            
        regime_dist = data[domain].get('regime_distribution', {})
        if not regime_dist:
            continue
        
        regimes = list(regime_dist.keys())
        probs = list(regime_dist.values())
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(regimes)))
        
        axes[i].pie(probs, labels=regimes, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
        axes[i].set_title(f'{domain.capitalize()}\n({data[domain]["n_regimes"]} regimes)', 
                         fontsize=12, fontweight='bold')
    
    plt.suptitle('Regime Distribution Across Domains', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_regime_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig4_regime_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: fig4_regime_distribution.pdf")


def fig5_summary_heatmap(results: dict, output_dir: Path):
    """Figure 5: Summary Heatmap of All Results."""
    if not HAS_MPL:
        print("Skipping figure 5 - no matplotlib")
        return
    
    marl_data = results.get('marl', {})
    if not marl_data:
        return
    
    domains = list(marl_data.keys())
    methods = ['niche_population', 'iql', 'random']
    method_labels = ['Niche Pop.', 'IQL', 'Random']
    
    # Create matrix
    matrix = np.zeros((len(domains), len(methods)))
    for i, domain in enumerate(domains):
        for j, method in enumerate(methods):
            matrix[i, j] = marl_data[domain]['results'][method]['mean_si']
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(np.arange(len(methods)))
    ax.set_yticks(np.arange(len(domains)))
    ax.set_xticklabels(method_labels)
    ax.set_yticklabels([d.capitalize() for d in domains])
    
    # Add values
    for i in range(len(domains)):
        for j in range(len(methods)):
            text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                          ha='center', va='center', color='black', fontsize=11)
    
    ax.set_title('Specialization Index Heatmap', fontsize=14, fontweight='bold')
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('SI', rotation=-90, va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_summary_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig5_summary_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: fig5_summary_heatmap.pdf")


def main():
    print("="*60)
    print("GENERATING PUBLICATION FIGURES")
    print("="*60)
    
    output_dir = RESULTS_DIR / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = load_results()
    
    if not results:
        print("No results found. Run experiments first.")
        return
    
    fig1_cross_domain_si(results, output_dir)
    fig2_marl_comparison(results, output_dir)
    fig3_improvement_vs_baseline(results, output_dir)
    fig4_regime_distribution(results, output_dir)
    fig5_summary_heatmap(results, output_dir)
    
    print(f"\nAll figures saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()

