#!/usr/bin/env python3
"""
Compile and analyze all experiment results with proper statistical rigor.

This script:
1. Loads results from all experiments
2. Applies Bonferroni correction for multiple comparisons
3. Computes effect sizes (Cohen's d)
4. Generates summary tables for the paper
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR = RESULTS_DIR / "compiled_analysis"


def load_json(filepath: Path) -> Dict:
    """Load JSON file."""
    if filepath.exists():
        with open(filepath) as f:
            return json.load(f)
    return {}


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Dict:
    """Apply Bonferroni correction for multiple comparisons."""
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests

    significant = [p < corrected_alpha for p in p_values]

    return {
        "n_tests": n_tests,
        "original_alpha": alpha,
        "corrected_alpha": corrected_alpha,
        "p_values": p_values,
        "significant": significant,
        "n_significant": sum(significant)
    }


def cohens_d(group1_mean: float, group2_mean: float, pooled_std: float) -> float:
    """Compute Cohen's d effect size."""
    if pooled_std == 0:
        return 0.0
    return abs(group1_mean - group2_mean) / pooled_std


def analyze_mono_regime():
    """Analyze mono-regime experiment results."""
    data = load_json(RESULTS_DIR / "exp_mono_regime_v3" / "results.json")
    if not data:
        return {"error": "No data found"}

    h1 = data.get("hypotheses", {}).get("H1", {})
    h2 = data.get("hypotheses", {}).get("H2", {})

    # Effect size for H1 (difference from threshold)
    observed_si = h1.get("observed_si", 0)
    threshold = h1.get("threshold", 0.15)

    return {
        "experiment": "Mono-Regime Validation",
        "hypotheses": {
            "H1": {
                "description": "Mono-regime SI < 0.15",
                "passed": h1.get("passed", False),
                "observed_si": observed_si,
                "threshold": threshold,
                "p_value": h1.get("p_value_one_tailed", 1.0),
                "effect": threshold - observed_si
            },
            "H2": {
                "description": "SI increases with regime count",
                "passed": h2.get("passed", False),
                "spearman_r": h2.get("spearman_r", 0),
                "p_value": h2.get("p_value_one_tailed", 1.0)
            }
        },
        "results_by_regime_count": data.get("results", {})
    }


def analyze_cost_transition():
    """Analyze cost transition experiment results."""
    data = load_json(RESULTS_DIR / "exp_cost_transition_v3" / "results.json")
    if not data:
        return {"error": "No data found"}

    h5 = data.get("hypotheses", {}).get("H5", {})

    return {
        "experiment": "Transaction Cost Phase Transition",
        "hypothesis": {
            "description": "Transaction costs reduce SI",
            "passed": h5.get("passed", False),
            "slope": h5.get("slope", 0),
            "r_squared": h5.get("r_squared", 0),
            "p_value": h5.get("p_value_one_tailed", 1.0)
        },
        "results": data.get("results", [])
    }


def analyze_robustness():
    """Analyze robustness experiment results."""
    data = load_json(RESULTS_DIR / "exp_robustness" / "results.json")
    if not data:
        return {"error": "No data found"}

    return {
        "experiment": "Robustness Analysis",
        "summary": data.get("summary", {}),
        "classifier_sensitivity": data.get("classifier_sensitivity", {}),
        "asset_sensitivity": data.get("asset_sensitivity", {}),
        "period_sensitivity": data.get("period_sensitivity", {})
    }


def analyze_distribution_matched():
    """Analyze distribution-matched experiment results."""
    data = load_json(RESULTS_DIR / "exp_distribution_matched_v3" / "results.json")
    if not data:
        return {"error": "No data found"}

    results = data.get("results", [])

    return {
        "experiment": "Distribution-Matched Generalization",
        "results": results,
        "summary": {
            "mean_train_si": np.mean([r.get("train_si", 0) for r in results]),
            "best_test_regime": max(results, key=lambda x: x.get("test_reward_mean", 0)).get("test_regime", "") if results else "",
            "worst_test_regime": min(results, key=lambda x: x.get("test_reward_mean", 0)).get("test_regime", "") if results else ""
        }
    }


def analyze_stratified():
    """Analyze regime-stratified experiment results."""
    data = load_json(RESULTS_DIR / "exp_regime_stratified_v3" / "results.json")
    if not data:
        return {"error": "No data found"}

    results = data.get("results", [])

    # Group by classifier
    by_classifier = {}
    for r in results:
        clf = r.get("classifier", "unknown")
        if clf not in by_classifier:
            by_classifier[clf] = []
        by_classifier[clf].append(r)

    return {
        "experiment": "Regime-Stratified Real Data",
        "n_segments": len(results),
        "by_classifier": {
            clf: {
                "n_segments": len(segments),
                "mean_si": np.mean([s.get("si_mean", 0) for s in segments]),
                "mean_reward": np.mean([s.get("reward_mean", 0) for s in segments])
            }
            for clf, segments in by_classifier.items()
        }
    }


def compile_all_results():
    """Compile all experiment results."""
    print("=" * 70)
    print("COMPILING EXPERIMENT RESULTS")
    print("=" * 70)

    # Collect all analyses
    mono_regime = analyze_mono_regime()
    cost_transition = analyze_cost_transition()
    robustness = analyze_robustness()
    distribution = analyze_distribution_matched()
    stratified = analyze_stratified()

    # Collect p-values for Bonferroni correction
    p_values = []
    if "hypotheses" in mono_regime:
        p_values.append(mono_regime["hypotheses"]["H1"]["p_value"])
        p_values.append(mono_regime["hypotheses"]["H2"]["p_value"])
    if "hypothesis" in cost_transition:
        p_values.append(cost_transition["hypothesis"]["p_value"])

    bonferroni = bonferroni_correction(p_values) if p_values else {}

    # Summary statistics
    summary = {
        "timestamp": datetime.now().isoformat(),
        "experiments_analyzed": 5,
        "hypotheses_tested": len(p_values),
        "bonferroni_correction": bonferroni,
        "overall_findings": {
            "mono_regime_h1_passed": mono_regime.get("hypotheses", {}).get("H1", {}).get("passed", False),
            "mono_regime_h2_passed": mono_regime.get("hypotheses", {}).get("H2", {}).get("passed", False),
            "cost_h5_passed": cost_transition.get("hypothesis", {}).get("passed", False),
            "robustness_passed": robustness.get("summary", {}).get("overall_robust", False)
        }
    }

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY OF FINDINGS")
    print("=" * 70)

    print("\n1. MONO-REGIME EXPERIMENT:")
    if "hypotheses" in mono_regime:
        h1 = mono_regime["hypotheses"]["H1"]
        h2 = mono_regime["hypotheses"]["H2"]
        print(f"   H1 (Mono-regime SI < 0.15): {'✓ PASS' if h1['passed'] else '✗ FAIL'}")
        print(f"      Observed SI: {h1['observed_si']:.3f}, p={h1['p_value']:.4f}")
        print(f"   H2 (SI increases with regimes): {'✓ PASS' if h2['passed'] else '✗ FAIL'}")
        print(f"      Spearman r: {h2['spearman_r']:.3f}, p={h2['p_value']:.4f}")

    print("\n2. COST TRANSITION EXPERIMENT:")
    if "hypothesis" in cost_transition:
        h5 = cost_transition["hypothesis"]
        print(f"   H5 (Costs reduce SI): {'✓ PASS' if h5['passed'] else '✗ FAIL'}")
        print(f"      Slope: {h5['slope']:.4f}, R²={h5['r_squared']:.3f}, p={h5['p_value']:.4f}")

    print("\n3. ROBUSTNESS EXPERIMENT:")
    rob = robustness.get("summary", {})
    print(f"   Overall robust: {'✓ YES' if rob.get('overall_robust', False) else '✗ NO'}")
    print(f"   Robust dimensions: {rob.get('robust_dimensions', 0)}/{rob.get('total_dimensions', 0)}")

    print("\n4. DISTRIBUTION-MATCHED EXPERIMENT:")
    dist = distribution.get("summary", {})
    print(f"   Mean train SI: {dist.get('mean_train_si', 0):.3f}")
    print(f"   Best test regime: {dist.get('best_test_regime', 'N/A')}")
    print(f"   Worst test regime: {dist.get('worst_test_regime', 'N/A')}")

    print("\n5. REGIME-STRATIFIED EXPERIMENT:")
    strat = stratified
    print(f"   Total segments analyzed: {strat.get('n_segments', 0)}")
    for clf, stats in strat.get("by_classifier", {}).items():
        print(f"   {clf}: {stats['n_segments']} segments, SI={stats['mean_si']:.3f}")

    print("\n" + "=" * 70)
    print("BONFERRONI CORRECTION")
    print("=" * 70)
    print(f"   Number of tests: {bonferroni.get('n_tests', 0)}")
    print(f"   Original α: {bonferroni.get('original_alpha', 0.05)}")
    print(f"   Corrected α: {bonferroni.get('corrected_alpha', 0):.4f}")
    print(f"   Significant after correction: {bonferroni.get('n_significant', 0)}/{bonferroni.get('n_tests', 0)}")

    # Save compiled results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "summary": summary,
        "mono_regime": mono_regime,
        "cost_transition": cost_transition,
        "robustness": robustness,
        "distribution_matched": distribution,
        "stratified": stratified
    }

    output_path = OUTPUT_DIR / "compiled_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == "__main__":
    compile_all_results()
