"""
Experiment Runner.

Unified runner for all experiments with consistent logging and output.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional


def run_all_experiments(
    experiments: Optional[List[str]] = None,
    n_trials: Optional[int] = None,
    seed: int = 42,
    output_dir: str = "results",
):
    """
    Run specified experiments.

    Args:
        experiments: List of experiment names to run (None = all)
        n_trials: Override number of trials (None = use defaults)
        seed: Base random seed
        output_dir: Output directory for results
    """
    from .config import (
        EXP1_CONFIG, EXP2_CONFIG, EXP3_CONFIG,
        EXP4_CONFIG, EXP5_CONFIG, EXP6_CONFIG,
        ExperimentConfig,
    )

    all_experiments = {
        "exp1": ("Emergence of Specialists", EXP1_CONFIG),
        "exp2": ("Value of Diversity", EXP2_CONFIG),
        "exp3": ("Population Size Effect", EXP3_CONFIG),
        "exp4": ("Transfer Frequency", EXP4_CONFIG),
        "exp5": ("Regime Transitions", EXP5_CONFIG),
        "exp6": ("Real Data Validation", EXP6_CONFIG),
    }

    if experiments is None:
        experiments = list(all_experiments.keys())

    print("=" * 60)
    print("EMERGENT SPECIALIZATION EXPERIMENTS")
    print(f"Experiments: {experiments}")
    print(f"Seed: {seed}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    results = {}

    for exp_name in experiments:
        if exp_name not in all_experiments:
            print(f"Unknown experiment: {exp_name}")
            continue

        title, config = all_experiments[exp_name]
        print(f"\n{'=' * 60}")
        print(f"Running: {title}")
        print(f"{'=' * 60}")

        # Create a copy of config with overrides
        exp_config = ExperimentConfig(
            experiment_name=config.experiment_name,
            n_trials=n_trials if n_trials is not None else config.n_trials,
            base_seed=seed,
            n_bars=config.n_bars,
            regime_duration_mean=config.regime_duration_mean,
            regime_duration_std=config.regime_duration_std,
            results_dir=output_dir,
        )

        try:
            if exp_name == "exp1":
                from .exp1_emergence import run_experiment
                result = run_experiment(exp_config)
                results[exp_name] = {
                    "status": "success",
                    "final_si": result.final_si_mean,
                    "p_value": result.p_value,
                    "significant": result.significant,
                }
            elif exp_name == "exp2":
                from .exp2_diversity_value import run_experiment
                result = run_experiment(exp_config)
                results[exp_name] = {
                    "status": "success",
                    "diverse_reward": result.diverse_reward_mean,
                    "ranking": result.performance_ranking,
                }
            elif exp_name == "exp3":
                from .exp3_population_size import run_experiment
                result = run_experiment(exp_config)
                results[exp_name] = {
                    "status": "success",
                    "optimal_size": result.optimal_size,
                }
            elif exp_name == "exp4":
                from .exp4_transfer_frequency import run_experiment
                result = run_experiment(exp_config)
                results[exp_name] = {
                    "status": "success",
                    "optimal_frequency": result.optimal_frequency,
                }
            elif exp_name == "exp5":
                from .exp5_regime_transitions import run_experiment
                result = run_experiment(exp_config)
                results[exp_name] = {
                    "status": "success",
                    "switch_rate": result.avg_switch_rate,
                    "p_value": result.chi2_p_value,
                }
            elif exp_name == "exp6":
                from .exp6_real_data import run_experiment
                result = run_experiment(exp_config)
                results[exp_name] = {
                    "status": "success",
                    "si_matches": result.si_matches_synthetic,
                    "outperforms": result.outperforms_baseline,
                }
            else:
                print(f"Experiment {exp_name} not implemented")
                results[exp_name] = {"status": "not_implemented"}
        except Exception as e:
            print(f"Error running {exp_name}: {e}")
            import traceback
            traceback.print_exc()
            results[exp_name] = {"status": "error", "error": str(e)}

    # Save overall summary
    summary_path = Path(output_dir) / "experiment_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "experiments_run": experiments,
        "seed": seed,
        "results": results,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'=' * 60}")
    for exp_name, res in results.items():
        status = res.get("status", "unknown")
        if status == "success":
            print(f"  {exp_name}: ✅ Success")
        elif status == "not_implemented":
            print(f"  {exp_name}: ⏳ Not implemented")
        else:
            print(f"  {exp_name}: ❌ Error")
    print(f"\nSummary saved to {summary_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run emergent specialization experiments"
    )
    parser.add_argument(
        "--experiments", "-e",
        nargs="+",
        default=None,
        help="Experiments to run (exp1, exp2, etc). Default: all"
    )
    parser.add_argument(
        "--trials", "-n",
        type=int,
        default=None,
        help="Number of trials (overrides config)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Base random seed"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results",
        help="Output directory"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all experiments"
    )

    args = parser.parse_args()

    experiments = None if args.all else args.experiments

    run_all_experiments(
        experiments=experiments,
        n_trials=args.trials,
        seed=args.seed,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
