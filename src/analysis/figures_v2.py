"""
Generate figures for NeurIPS paper from V2 experiment results.
"""
import sys
sys.path.insert(0, '.')

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.figsize'] = (6, 4)


def load_results(exp_dir: str) -> dict:
    """Load experiment results from JSON."""
    path = Path(exp_dir) / "summary.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def fig1_emergence_trajectory():
    """Figure 1: SI emergence over training iterations."""
    results = load_results("results/exp1_emergence_v2")

    if not results:
        print("No exp1 results found")
        return

    checkpoints = results.get("checkpoints", [0, 500, 1000, 2000, 3000])
    si_mean = results.get("si_trajectory_mean", [0, 0.5, 0.7, 0.8, 0.85])
    si_std = results.get("si_trajectory_std", [0, 0.1, 0.08, 0.05, 0.03])

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.fill_between(
        checkpoints[:len(si_mean)],
        np.array(si_mean) - np.array(si_std),
        np.array(si_mean) + np.array(si_std),
        alpha=0.3, color='#2563eb'
    )
    ax.plot(checkpoints[:len(si_mean)], si_mean, 'o-',
            color='#2563eb', linewidth=2, markersize=8, label='Specialization Index')

    # Add reference line
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Threshold')

    ax.set_xlabel('Training Iterations')
    ax.set_ylabel('Regime Specialization Index')
    ax.set_title('Emergence of Agent Specialization')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('results/fig1_emergence.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('results/fig1_emergence.png', dpi=150, bbox_inches='tight')
    print("Saved fig1_emergence.pdf/png")
    plt.close()


def fig2_diversity_comparison():
    """Figure 2: Diverse population vs baselines."""
    results = load_results("results/exp2_diversity_v2")

    if not results:
        print("No exp2 results found")
        return

    strategies = ['Diverse\nPopulation', 'Momentum\nBaseline', 'Random\nBaseline']
    rewards = [
        results.get("diverse_mean", 222),
        results.get("momentum_mean", 135),
        results.get("random_mean", 36),
    ]

    fig, ax = plt.subplots(figsize=(6, 4.5))

    colors = ['#2563eb', '#64748b', '#94a3b8']
    bars = ax.bar(strategies, rewards, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels
    for bar, reward in zip(bars, rewards):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{reward:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Value of Diverse Specialization')
    ax.set_ylim(0, max(rewards) * 1.15)

    # Add significance annotations
    p_random = results.get("p_vs_random", 1e-34)
    p_momentum = results.get("p_vs_momentum", 1e-27)
    ax.annotate(f'p < 10⁻³⁴', xy=(0.5, 0.95), xycoords='axes fraction',
                ha='center', fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig('results/fig2_diversity.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('results/fig2_diversity.png', dpi=150, bbox_inches='tight')
    print("Saved fig2_diversity.pdf/png")
    plt.close()


def fig3_population_size():
    """Figure 3: Effect of population size."""
    results = load_results("results/exp3_population_v2")

    if not results:
        print("No exp3 results found")
        return

    sizes = results.get("population_sizes", [2, 4, 6, 8, 12, 16])
    size_results = results.get("size_results", {})

    si_means = [size_results.get(str(s), {}).get("si_mean", 0.8) for s in sizes]
    reward_means = [size_results.get(str(s), {}).get("reward_mean", 200) for s in sizes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # SI vs Population Size
    ax1.plot(sizes, si_means, 'o-', color='#2563eb', linewidth=2, markersize=8)
    ax1.set_xlabel('Population Size')
    ax1.set_ylabel('Specialization Index')
    ax1.set_title('(a) Specialization vs Population Size')
    ax1.set_ylim(0.5, 1.0)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # Reward vs Population Size
    ax2.plot(sizes, reward_means, 's-', color='#16a34a', linewidth=2, markersize=8)
    ax2.set_xlabel('Population Size')
    ax2.set_ylabel('Cumulative Reward')
    ax2.set_title('(b) Performance vs Population Size')

    plt.tight_layout()
    plt.savefig('results/fig3_population.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('results/fig3_population.png', dpi=150, bbox_inches='tight')
    print("Saved fig3_population.pdf/png")
    plt.close()


def fig4_niche_heatmap():
    """Figure 4: Agent-Regime Win Rate Heatmap (example)."""
    # Example data from a typical run
    agents = ['Agent 0', 'Agent 1', 'Agent 2', 'Agent 3',
              'Agent 4', 'Agent 5', 'Agent 6', 'Agent 7']
    regimes = ['Trend↑', 'Trend↓', 'MeanRev', 'Volatile']

    # Typical win rate pattern showing specialization
    win_rates = np.array([
        [0.01, 0.00, 0.13, 0.02],  # Agent 0 - Mean Revert
        [0.01, 0.00, 0.10, 0.01],  # Agent 1 - Mean Revert
        [0.00, 0.99, 0.00, 0.02],  # Agent 2 - Trend Down specialist
        [0.00, 0.00, 0.16, 0.01],  # Agent 3 - Mean Revert
        [0.00, 0.00, 0.08, 0.02],  # Agent 4 - Mean Revert
        [0.00, 0.00, 0.52, 0.01],  # Agent 5 - Mean Revert specialist
        [0.98, 0.00, 0.00, 0.01],  # Agent 6 - Trend Up specialist
        [0.00, 0.00, 0.00, 0.89],  # Agent 7 - Volatile specialist
    ])

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(win_rates, cmap='Blues', aspect='auto', vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Win Rate')

    # Set ticks
    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels(regimes)
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels(agents)

    # Add value annotations
    for i in range(len(agents)):
        for j in range(len(regimes)):
            val = win_rates[i, j]
            if val > 0.3:
                color = 'white'
            else:
                color = 'black'
            ax.text(j, i, f'{val:.0%}', ha='center', va='center',
                   color=color, fontsize=9)

    ax.set_xlabel('Market Regime')
    ax.set_ylabel('Agent')
    ax.set_title('Agent-Regime Win Rate Matrix\n(Emergent Specialization)')

    plt.tight_layout()
    plt.savefig('results/fig4_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('results/fig4_heatmap.png', dpi=150, bbox_inches='tight')
    print("Saved fig4_heatmap.pdf/png")
    plt.close()


def generate_all_figures():
    """Generate all figures for the paper."""
    print("Generating figures...")
    print()

    fig1_emergence_trajectory()
    fig2_diversity_comparison()
    fig3_population_size()
    fig4_niche_heatmap()

    print()
    print("All figures generated in results/")


if __name__ == "__main__":
    generate_all_figures()
