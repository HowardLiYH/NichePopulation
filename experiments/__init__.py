"""
Experiments module: Core experiments for NeurIPS paper.

Experiments:
1. exp1_emergence - Do agents naturally specialize?
2. exp2_diversity_value - Does specialization improve performance?
3. exp3_population_size - Optimal population size
4. exp4_transfer_frequency - Effect of knowledge transfer
5. exp5_regime_transitions - Regime transition behavior
6. exp6_real_data - Real data validation

Usage:
    # Run all experiments
    python -m experiments.runner --all

    # Run specific experiment
    python -m experiments.runner -e exp1

    # Quick test (10 trials)
    python -m experiments.runner -e exp1 --trials 10
"""

from .config import ExperimentConfig, EXP1_CONFIG, EXP2_CONFIG, EXP3_CONFIG
from .runner import run_all_experiments

__all__ = [
    "ExperimentConfig",
    "run_all_experiments",
    "exp1_emergence",
    "exp2_diversity_value",
    "exp3_population_size",
    "exp4_transfer_frequency",
    "exp5_regime_transitions",
    "exp6_real_data",
]
