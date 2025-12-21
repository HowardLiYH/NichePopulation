"""
Cross-Experiment Analysis.

Aggregates results from all experiments for paper tables and summaries.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ExperimentSummary:
    """Summary of a single experiment."""
    name: str
    status: str
    key_metric: str
    key_value: float
    p_value: Optional[float]
    significant: bool
    notes: str


def load_experiment_results(results_dir: str = "results") -> Dict[str, dict]:
    """Load all experiment results from disk."""
    results = {}
    results_path = Path(results_dir)

    exp_names = [
        "exp1_emergence",
        "exp2_diversity_value",
        "exp3_population_size",
        "exp4_transfer_frequency",
        "exp5_regime_transitions",
        "exp6_real_data",
    ]

    for exp_name in exp_names:
        summary_path = results_path / exp_name / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                results[exp_name] = json.load(f)
        else:
            results[exp_name] = None

    return results


def generate_paper_table1(results: Dict[str, dict]) -> str:
    """
    Generate Table 1: Main Results Summary.

    Returns LaTeX table string.
    """
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Summary of Experimental Results}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Experiment & Metric & Value & $p$-value & Significant \\",
        r"\midrule",
    ]

    # Exp 1
    if results.get("exp1_emergence"):
        exp1 = results["exp1_emergence"]
        lines.append(
            f"Emergence (Exp 1) & SI & {exp1.get('final_si_mean', 0):.3f} $\\pm$ {exp1.get('final_si_std', 0):.3f} & "
            f"{exp1.get('p_value', 1):.4f} & {'\\checkmark' if exp1.get('significant') else '---'} \\\\"
        )

    # Exp 2
    if results.get("exp2_diversity_value"):
        exp2 = results["exp2_diversity_value"]
        lines.append(
            f"Diversity (Exp 2) & Reward & {exp2.get('diverse_reward_mean', 0):.3f} & "
            f"--- & --- \\\\"
        )

    # Exp 3
    if results.get("exp3_population_size"):
        exp3 = results["exp3_population_size"]
        lines.append(
            f"Pop Size (Exp 3) & $N^*$ & {exp3.get('optimal_size', 0)} & "
            f"--- & --- \\\\"
        )

    # Exp 4
    if results.get("exp4_transfer_frequency"):
        exp4 = results["exp4_transfer_frequency"]
        lines.append(
            f"Transfer (Exp 4) & $\\tau^*$ & {exp4.get('optimal_frequency', 0)} & "
            f"--- & --- \\\\"
        )

    # Exp 5
    if results.get("exp5_regime_transitions"):
        exp5 = results["exp5_regime_transitions"]
        lines.append(
            f"Transitions (Exp 5) & Switch Rate & {exp5.get('avg_switch_rate', 0):.3f} & "
            f"{exp5.get('chi2_p_value', 1):.4f} & {'\\checkmark' if exp5.get('chi2_p_value', 1) < 0.05 else '---'} \\\\"
        )

    # Exp 6
    if results.get("exp6_real_data"):
        exp6 = results["exp6_real_data"]
        lines.append(
            f"Real Data (Exp 6) & Train SI & {exp6.get('train_final_si', 0):.3f} & "
            f"--- & {'\\checkmark' if exp6.get('si_matches_synthetic') else '---'} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_paper_table2(results: Dict[str, dict]) -> str:
    """
    Generate Table 2: Baseline Comparison.

    Returns LaTeX table string.
    """
    if not results.get("exp2_diversity_value"):
        return "% No exp2 results available"

    exp2 = results["exp2_diversity_value"]

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Performance Comparison with Baselines}",
        r"\label{tab:baselines}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Reward & $p$-value (corrected) & Effect Size ($d$) \\",
        r"\midrule",
        f"Diverse (Ours) & {exp2.get('diverse_reward_mean', 0):.3f} & --- & --- \\\\",
    ]

    for comp in exp2.get("comparisons", []):
        sig = "*" if comp.get("significant") else ""
        lines.append(
            f"{comp.get('baseline', 'Unknown')} & {comp.get('baseline_mean', 0):.3f}{sig} & "
            f"{comp.get('p_value_corrected', 1):.4f} & {comp.get('effect_size', 0):.2f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\multicolumn{4}{l}{\footnotesize * Statistically significant after Bonferroni correction ($\alpha = 0.05$)}",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_markdown_summary(results: Dict[str, dict]) -> str:
    """Generate Markdown summary for README or reports."""
    lines = [
        "# Experiment Results Summary\n",
        "## Key Findings\n",
    ]

    # Exp 1
    if results.get("exp1_emergence"):
        exp1 = results["exp1_emergence"]
        lines.append(f"### Experiment 1: Emergence of Specialists")
        lines.append(f"- **Initial SI**: {exp1.get('initial_si_mean', 0):.4f} ± {exp1.get('initial_si_std', 0):.4f}")
        lines.append(f"- **Final SI**: {exp1.get('final_si_mean', 0):.4f} ± {exp1.get('final_si_std', 0):.4f}")
        lines.append(f"- **p-value**: {exp1.get('p_value', 1):.6f}")
        lines.append(f"- **Effect size (Cohen's d)**: {exp1.get('effect_size', 0):.2f}")
        lines.append(f"- **Significant**: {'✓' if exp1.get('significant') else '✗'}\n")

    # Exp 2
    if results.get("exp2_diversity_value"):
        exp2 = results["exp2_diversity_value"]
        lines.append(f"### Experiment 2: Value of Diversity")
        lines.append(f"- **Diverse Population Reward**: {exp2.get('diverse_reward_mean', 0):.4f}")
        lines.append(f"- **Performance Ranking**: {', '.join(exp2.get('performance_ranking', []))}\n")

    # Exp 3
    if results.get("exp3_population_size"):
        exp3 = results["exp3_population_size"]
        lines.append(f"### Experiment 3: Population Size Effect")
        lines.append(f"- **Optimal Population Size**: N* = {exp3.get('optimal_size', 0)}\n")

    # Exp 4
    if results.get("exp4_transfer_frequency"):
        exp4 = results["exp4_transfer_frequency"]
        lines.append(f"### Experiment 4: Transfer Frequency")
        lines.append(f"- **Optimal Transfer Frequency**: τ* = {exp4.get('optimal_frequency', 0)}\n")

    # Exp 5
    if results.get("exp5_regime_transitions"):
        exp5 = results["exp5_regime_transitions"]
        lines.append(f"### Experiment 5: Regime Transitions")
        lines.append(f"- **Switch Rate**: {exp5.get('avg_switch_rate', 0):.3f}")
        lines.append(f"- **Avg Transition Delay**: {exp5.get('avg_transition_delay', 0):.1f} bars\n")

    # Exp 6
    if results.get("exp6_real_data"):
        exp6 = results["exp6_real_data"]
        lines.append(f"### Experiment 6: Real Data Validation")
        lines.append(f"- **Train SI**: {exp6.get('train_final_si', 0):.4f}")
        lines.append(f"- **Matches Synthetic**: {'✓' if exp6.get('si_matches_synthetic') else '✗'}")
        lines.append(f"- **Outperforms Baseline**: {'✓' if exp6.get('outperforms_baseline') else '✗'}\n")

    return "\n".join(lines)


def run_cross_analysis(results_dir: str = "results", output_dir: str = "paper/tables"):
    """Run full cross-experiment analysis."""
    print("Loading experiment results...")
    results = load_experiment_results(results_dir)

    available = [k for k, v in results.items() if v is not None]
    print(f"Found results for: {available}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate tables
    table1 = generate_paper_table1(results)
    with open(output_path / "table1_main_results.tex", "w") as f:
        f.write(table1)
    print(f"Saved: {output_path / 'table1_main_results.tex'}")

    table2 = generate_paper_table2(results)
    with open(output_path / "table2_baselines.tex", "w") as f:
        f.write(table2)
    print(f"Saved: {output_path / 'table2_baselines.tex'}")

    # Generate markdown summary
    summary = generate_markdown_summary(results)
    with open(output_path / "results_summary.md", "w") as f:
        f.write(summary)
    print(f"Saved: {output_path / 'results_summary.md'}")

    return results


if __name__ == "__main__":
    run_cross_analysis()
