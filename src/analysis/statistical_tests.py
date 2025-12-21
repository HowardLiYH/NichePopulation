"""
Statistical Testing Utilities for Experiment Analysis.

Provides rigorous statistical tests for comparing conditions
and establishing significance of results.

References:
- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
- Efron, B. & Tibshirani, R. (1993). An Introduction to the Bootstrap.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from scipy import stats
import warnings


@dataclass
class TestResult:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool  # At alpha level
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    additional_info: Optional[Dict] = None


def paired_t_test(
    a: np.ndarray,
    b: np.ndarray,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TestResult:
    """
    Perform paired t-test with effect size (Cohen's d).

    Tests H0: mean(a) = mean(b)

    Args:
        a: First sample
        b: Second sample (paired with a)
        alpha: Significance level
        alternative: "two-sided", "less", or "greater"

    Returns:
        TestResult with statistic, p-value, and Cohen's d
    """
    a = np.asarray(a)
    b = np.asarray(b)

    # Paired t-test
    result = stats.ttest_rel(a, b, alternative=alternative)

    # Cohen's d for paired samples
    diff = a - b
    cohens_d = np.mean(diff) / np.std(diff, ddof=1)

    return TestResult(
        test_name="Paired t-test",
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        significant=result.pvalue < alpha,
        effect_size=float(cohens_d),
        additional_info={
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
            "mean_diff": float(np.mean(diff)),
            "std_diff": float(np.std(diff, ddof=1)),
            "n": len(a),
        },
    )


def independent_t_test(
    a: np.ndarray,
    b: np.ndarray,
    alpha: float = 0.05,
    equal_var: bool = False,
) -> TestResult:
    """
    Perform independent samples t-test with effect size.

    Uses Welch's t-test by default (unequal variances).

    Args:
        a: First sample
        b: Second sample
        alpha: Significance level
        equal_var: Assume equal variances (False = Welch's test)

    Returns:
        TestResult with statistic, p-value, and Cohen's d
    """
    a = np.asarray(a)
    b = np.asarray(b)

    result = stats.ttest_ind(a, b, equal_var=equal_var)

    # Cohen's d for independent samples
    pooled_std = np.sqrt(
        ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) /
        (len(a) + len(b) - 2)
    )
    cohens_d = (np.mean(a) - np.mean(b)) / pooled_std

    return TestResult(
        test_name="Independent t-test (Welch)" if not equal_var else "Independent t-test",
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        significant=result.pvalue < alpha,
        effect_size=float(cohens_d),
        additional_info={
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
            "std_a": float(np.std(a, ddof=1)),
            "std_b": float(np.std(b, ddof=1)),
            "n_a": len(a),
            "n_b": len(b),
        },
    )


def anova_with_tukey(
    groups: Dict[str, np.ndarray],
    alpha: float = 0.05,
) -> TestResult:
    """
    Perform one-way ANOVA with Tukey HSD post-hoc tests.

    Args:
        groups: Dict mapping group names to arrays of observations
        alpha: Significance level

    Returns:
        TestResult with F-statistic, p-value, and pairwise comparisons
    """
    group_names = list(groups.keys())
    group_data = [np.asarray(groups[name]) for name in group_names]

    # One-way ANOVA
    f_stat, p_value = stats.f_oneway(*group_data)

    # Effect size: eta-squared
    # η² = SS_between / SS_total
    grand_mean = np.mean(np.concatenate(group_data))
    ss_between = sum(
        len(g) * (np.mean(g) - grand_mean) ** 2
        for g in group_data
    )
    ss_total = sum(
        np.sum((g - grand_mean) ** 2)
        for g in group_data
    )
    eta_squared = ss_between / ss_total if ss_total > 0 else 0.0

    # Tukey HSD post-hoc
    pairwise_comparisons = {}

    if p_value < alpha and len(group_names) > 2:
        # Flatten data for Tukey
        all_data = np.concatenate(group_data)
        all_labels = np.concatenate([
            [name] * len(groups[name]) for name in group_names
        ])

        try:
            from scipy.stats import tukey_hsd
            tukey_result = tukey_hsd(*group_data)

            for i, name_i in enumerate(group_names):
                for j, name_j in enumerate(group_names):
                    if i < j:
                        pair_key = f"{name_i} vs {name_j}"
                        pairwise_comparisons[pair_key] = {
                            "p_value": float(tukey_result.pvalue[i, j]),
                            "significant": tukey_result.pvalue[i, j] < alpha,
                            "mean_diff": float(np.mean(groups[name_i]) - np.mean(groups[name_j])),
                        }
        except (ImportError, AttributeError):
            # Fallback: simple pairwise t-tests with Bonferroni correction
            n_comparisons = len(group_names) * (len(group_names) - 1) / 2
            bonferroni_alpha = alpha / n_comparisons

            for i, name_i in enumerate(group_names):
                for j, name_j in enumerate(group_names):
                    if i < j:
                        t_result = independent_t_test(
                            groups[name_i],
                            groups[name_j],
                            alpha=bonferroni_alpha,
                        )
                        pair_key = f"{name_i} vs {name_j}"
                        pairwise_comparisons[pair_key] = {
                            "p_value": t_result.p_value,
                            "significant": t_result.significant,
                            "mean_diff": t_result.additional_info["mean_a"] - t_result.additional_info["mean_b"],
                        }

    return TestResult(
        test_name="One-way ANOVA with Tukey HSD",
        statistic=float(f_stat),
        p_value=float(p_value),
        significant=p_value < alpha,
        effect_size=float(eta_squared),
        additional_info={
            "group_means": {name: float(np.mean(groups[name])) for name in group_names},
            "group_stds": {name: float(np.std(groups[name], ddof=1)) for name in group_names},
            "group_sizes": {name: len(groups[name]) for name in group_names},
            "pairwise_comparisons": pairwise_comparisons,
        },
    )


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Reference: Efron, B. & Tibshirani, R. (1993)

    Args:
        data: Input data array
        statistic: Function to compute on each bootstrap sample
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples
        seed: Random seed

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    n = len(data)

    # Point estimate
    point_estimate = statistic(data)

    # Bootstrap samples
    bootstrap_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_stats[i] = statistic(sample)

    # Percentile method for CI
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return float(point_estimate), float(ci_lower), float(ci_upper)


def compute_power(
    effect_size: float,
    n: int,
    alpha: float = 0.05,
    test_type: str = "t-test",
) -> float:
    """
    Compute statistical power for a given effect size and sample size.

    Power = P(reject H0 | H1 is true)

    Args:
        effect_size: Expected Cohen's d
        n: Sample size
        alpha: Significance level
        test_type: Type of test ("t-test", "paired-t")

    Returns:
        Statistical power (0 to 1)
    """
    # For a two-sided t-test
    # Power ≈ 1 - Φ(z_{1-α/2} - d*√n) + Φ(-z_{1-α/2} - d*√n)

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    ncp = effect_size * np.sqrt(n)  # Non-centrality parameter

    # Using t-distribution approximation
    df = n - 1 if test_type == "paired-t" else 2 * n - 2

    # Critical value
    t_crit = stats.t.ppf(1 - alpha / 2, df)

    # Power using non-central t-distribution
    power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)

    return float(power)


def required_sample_size(
    effect_size: float,
    power: float = 0.8,
    alpha: float = 0.05,
) -> int:
    """
    Compute required sample size for desired power.

    Args:
        effect_size: Expected Cohen's d
        power: Desired power (default 0.8)
        alpha: Significance level

    Returns:
        Required sample size per group
    """
    # Approximate formula for two-sided t-test
    # n ≈ 2 * ((z_{1-α/2} + z_{1-β}) / d)²

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    n = 2 * ((z_alpha + z_beta) / effect_size) ** 2

    return int(np.ceil(n))


def effect_size_interpretation(d: float) -> str:
    """
    Interpret Cohen's d effect size.

    Reference: Cohen, J. (1988)
    """
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> Tuple[List[float], List[bool]]:
    """
    Apply Bonferroni correction for multiple comparisons.
    
    The Bonferroni correction is a conservative method to control
    the family-wise error rate (FWER).
    
    Adjusted p-value = min(1.0, p * n_tests)
    
    Args:
        p_values: List of p-values from individual tests
        alpha: Significance level (default 0.05)
        
    Returns:
        Tuple of:
            - List of corrected p-values
            - List of booleans indicating significance
    """
    n_tests = len(p_values)
    
    corrected_p = [min(1.0, p * n_tests) for p in p_values]
    significant = [p < alpha for p in corrected_p]
    
    return corrected_p, significant
