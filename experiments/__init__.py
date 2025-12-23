"""
Experiments module: Core experiments for NeurIPS paper.

Main entry point:
    python experiments/exp_unified_pipeline.py

Individual experiments:
    - exp_unified_pipeline.py - Runs all core experiments across all domains
    - exp_hypothesis_tests.py - Statistical hypothesis testing
    - exp_method_specialization.py - Method-level specialization analysis
    - exp_lambda_ablation.py - Lambda parameter ablation study
    - exp_marl_standalone.py - MARL baseline comparison

Usage:
    # Run unified pipeline (all experiments)
    python experiments/exp_unified_pipeline.py

    # Run specific experiment
    python experiments/exp_method_specialization.py
"""

from .config import ExperimentConfig

__all__ = [
    "ExperimentConfig",
]
