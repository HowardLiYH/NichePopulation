"""
Analysis module: Metrics, statistical tests, and figure generation.
"""

from .specialization import (
    compute_specialization_index,
    compute_population_diversity,
    compute_regime_coverage,
    compute_method_coverage,
    SpecializationTracker,
    SpecializationMetrics,
    PopulationMetrics,
    AgentMetrics,
)

from .statistical_tests import (
    paired_t_test,
    independent_t_test,
    bootstrap_confidence_interval,
    bonferroni_correction,
    TestResult,
)

__all__ = [
    # Specialization metrics
    "compute_specialization_index",
    "compute_population_diversity",
    "compute_regime_coverage",
    "compute_method_coverage",
    "SpecializationTracker",
    "SpecializationMetrics",
    "PopulationMetrics",
    "AgentMetrics",
    # Statistical tests
    "paired_t_test",
    "independent_t_test",
    "bootstrap_confidence_interval",
    "bonferroni_correction",
    "TestResult",
]
