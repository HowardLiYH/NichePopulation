"""
Rigorous Statistical Analysis Utilities

Provides:
- Bootstrap confidence intervals
- Multiple testing corrections (Bonferroni, FDR)
- Effect size calculations (Cohen's d, correlation r)
- Proper statistical reporting

For NeurIPS A+ quality papers.
"""

import numpy as np
from scipy import stats
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import warnings


@dataclass
class StatisticalResult:
    """Container for statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    p_corrected: Optional[float]
    effect_size: float
    effect_type: str  # "cohen_d", "r", "eta_squared"
    ci_lower: float
    ci_upper: float
    n: int
    significant: bool
    alpha: float


def bootstrap_confidence_interval(
    data: Union[List[float], np.ndarray],
    statistic_func: callable = np.mean,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    random_state: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        data: Sample data
        statistic_func: Function to compute statistic (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default: 0.95)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper)
    """
    if random_state is not None:
        np.random.seed(random_state)

    data = np.asarray(data)
    n = len(data)

    if n < 2:
        point = statistic_func(data)
        return point, point, point

    # Point estimate
    point_estimate = statistic_func(data)

    # Bootstrap
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_func(sample))

    # Percentile method
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)

    return point_estimate, ci_lower, ci_upper


def bootstrap_mean_difference(
    group1: Union[List[float], np.ndarray],
    group2: Union[List[float], np.ndarray],
    n_bootstrap: int = 10000,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute bootstrap CI for difference between means.

    Returns:
        Tuple of (mean_diff, ci_lower, ci_upper)
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    point_diff = np.mean(group1) - np.mean(group2)

    diffs = []
    for _ in range(n_bootstrap):
        sample1 = np.random.choice(group1, size=len(group1), replace=True)
        sample2 = np.random.choice(group2, size=len(group2), replace=True)
        diffs.append(np.mean(sample1) - np.mean(sample2))

    alpha = 1 - confidence
    ci_lower = np.percentile(diffs, alpha / 2 * 100)
    ci_upper = np.percentile(diffs, (1 - alpha / 2) * 100)

    return point_diff, ci_lower, ci_upper


def cohens_d(
    group1: Union[List[float], np.ndarray],
    group2: Union[List[float], np.ndarray],
    pooled: bool = True
) -> float:
    """
    Compute Cohen's d effect size.

    Args:
        group1: First group data
        group2: Second group data
        pooled: Use pooled standard deviation (default: True)

    Returns:
        Cohen's d value
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    n1, n2 = len(group1), len(group2)
    mean_diff = np.mean(group1) - np.mean(group2)

    if pooled:
        # Pooled standard deviation
        var1 = np.var(group1, ddof=1)
        var2 = np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        return mean_diff / pooled_std if pooled_std > 0 else 0.0
    else:
        # Use first group std (for one-sample effect)
        std1 = np.std(group1, ddof=1)
        return mean_diff / std1 if std1 > 0 else 0.0


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def interpret_correlation(r: float) -> str:
    """Interpret correlation strength."""
    r = abs(r)
    if r < 0.1:
        return "negligible"
    elif r < 0.3:
        return "small"
    elif r < 0.5:
        return "medium"
    else:
        return "large"


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05
) -> Tuple[List[float], List[bool], float]:
    """
    Apply Bonferroni correction for multiple testing.

    Args:
        p_values: List of p-values
        alpha: Family-wise error rate

    Returns:
        Tuple of (corrected_p_values, significant_flags, corrected_alpha)
    """
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests

    corrected_p = [min(p * n_tests, 1.0) for p in p_values]
    significant = [p < corrected_alpha for p in p_values]

    return corrected_p, significant, corrected_alpha


def fdr_correction(
    p_values: List[float],
    alpha: float = 0.05
) -> Tuple[List[float], List[bool]]:
    """
    Apply Benjamini-Hochberg FDR correction.

    Args:
        p_values: List of p-values
        alpha: False discovery rate

    Returns:
        Tuple of (adjusted_p_values, significant_flags)
    """
    n = len(p_values)

    # Sort p-values with indices
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    # Compute adjusted p-values
    adjusted = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            adjusted[i] = sorted_p[i]
        else:
            adjusted[i] = min(adjusted[i + 1], sorted_p[i] * n / (i + 1))

    # Ensure monotonicity
    for i in range(1, n):
        adjusted[i] = max(adjusted[i], adjusted[i - 1])

    # Map back to original order
    adjusted_original = np.zeros(n)
    adjusted_original[sorted_indices] = adjusted

    # Determine significance
    significant = [p < alpha for p in adjusted_original]

    return list(adjusted_original), significant


def ttest_with_stats(
    group1: Union[List[float], np.ndarray],
    group2: Union[List[float], np.ndarray],
    alternative: str = "two-sided",
    alpha: float = 0.05,
    n_corrections: int = 1
) -> StatisticalResult:
    """
    Perform t-test with full statistical reporting.

    Args:
        group1: First group data
        group2: Second group data
        alternative: "two-sided", "greater", or "less"
        alpha: Significance level
        n_corrections: Number of tests for Bonferroni correction

    Returns:
        StatisticalResult with all statistics
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    # T-test
    t_stat, p_value = stats.ttest_ind(group1, group2)

    # Adjust for one-tailed if needed
    if alternative == "greater":
        p_value = p_value / 2 if t_stat > 0 else 1 - p_value / 2
    elif alternative == "less":
        p_value = p_value / 2 if t_stat < 0 else 1 - p_value / 2

    # Bonferroni correction
    p_corrected = min(p_value * n_corrections, 1.0)
    alpha_corrected = alpha / n_corrections

    # Effect size
    d = cohens_d(group1, group2)

    # Confidence interval for difference
    diff, ci_lower, ci_upper = bootstrap_mean_difference(group1, group2)

    return StatisticalResult(
        test_name="Independent t-test",
        statistic=t_stat,
        p_value=p_value,
        p_corrected=p_corrected,
        effect_size=d,
        effect_type="cohen_d",
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n=len(group1) + len(group2),
        significant=p_corrected < alpha_corrected,
        alpha=alpha_corrected
    )


def correlation_with_stats(
    x: Union[List[float], np.ndarray],
    y: Union[List[float], np.ndarray],
    method: str = "pearson",
    alternative: str = "two-sided",
    alpha: float = 0.05,
    n_corrections: int = 1
) -> StatisticalResult:
    """
    Compute correlation with full statistical reporting.

    Args:
        x: First variable
        y: Second variable
        method: "pearson" or "spearman"
        alternative: "two-sided", "greater", or "less"
        alpha: Significance level
        n_corrections: Number of tests for Bonferroni correction

    Returns:
        StatisticalResult with all statistics
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if method == "pearson":
        r, p_value = stats.pearsonr(x, y)
    else:
        r, p_value = stats.spearmanr(x, y)

    # Adjust for one-tailed
    if alternative == "greater":
        p_value = p_value / 2 if r > 0 else 1 - p_value / 2
    elif alternative == "less":
        p_value = p_value / 2 if r < 0 else 1 - p_value / 2

    # Bonferroni correction
    p_corrected = min(p_value * n_corrections, 1.0)
    alpha_corrected = alpha / n_corrections

    # Bootstrap CI for correlation
    def corr_func(data):
        mid = len(data) // 2
        if method == "pearson":
            return stats.pearsonr(data[:mid], data[mid:])[0]
        else:
            return stats.spearmanr(data[:mid], data[mid:])[0]

    # Simple CI using Fisher's z transformation
    n = len(x)
    z = 0.5 * np.log((1 + r) / (1 - r)) if abs(r) < 1 else 0
    se = 1 / np.sqrt(n - 3) if n > 3 else 0
    z_crit = stats.norm.ppf(1 - alpha / 2)

    z_lower = z - z_crit * se
    z_upper = z + z_crit * se

    r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
    r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)

    return StatisticalResult(
        test_name=f"{method.capitalize()} correlation",
        statistic=r,
        p_value=p_value,
        p_corrected=p_corrected,
        effect_size=r,
        effect_type="r",
        ci_lower=r_lower,
        ci_upper=r_upper,
        n=n,
        significant=p_corrected < alpha_corrected,
        alpha=alpha_corrected
    )


def format_statistical_table(
    results: List[StatisticalResult],
    include_effect: bool = True
) -> str:
    """
    Format results as a publication-ready table.

    Args:
        results: List of StatisticalResult objects
        include_effect: Include effect size column

    Returns:
        Formatted table string
    """
    # Header
    if include_effect:
        header = f"{'Test':<30} {'Stat':>8} {'p':>10} {'p_adj':>10} {'Effect':>8} {'95% CI':>20} {'Sig':>5}"
    else:
        header = f"{'Test':<30} {'Stat':>8} {'p':>10} {'p_adj':>10} {'95% CI':>20} {'Sig':>5}"

    lines = [header, "-" * len(header)]

    for r in results:
        ci_str = f"[{r.ci_lower:.3f}, {r.ci_upper:.3f}]"
        sig_str = "✓" if r.significant else ""

        p_str = f"{r.p_value:.4f}" if r.p_value >= 0.0001 else "<.0001"
        p_adj_str = f"{r.p_corrected:.4f}" if r.p_corrected >= 0.0001 else "<.0001"

        if include_effect:
            lines.append(
                f"{r.test_name:<30} {r.statistic:>8.3f} {p_str:>10} {p_adj_str:>10} "
                f"{r.effect_size:>8.3f} {ci_str:>20} {sig_str:>5}"
            )
        else:
            lines.append(
                f"{r.test_name:<30} {r.statistic:>8.3f} {p_str:>10} {p_adj_str:>10} "
                f"{ci_str:>20} {sig_str:>5}"
            )

    return "\n".join(lines)


def format_latex_table(
    results: List[StatisticalResult],
    caption: str = "Statistical Results"
) -> str:
    """
    Format results as a LaTeX table.

    Args:
        results: List of StatisticalResult objects
        caption: Table caption

    Returns:
        LaTeX table string
    """
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Test & Statistic & $p$ & $p_{adj}$ & Effect & 95\\% CI \\\\",
        "\\midrule"
    ]

    for r in results:
        p_str = f"{r.p_value:.4f}" if r.p_value >= 0.0001 else "$<$.0001"
        p_adj_str = f"{r.p_corrected:.4f}" if r.p_corrected >= 0.0001 else "$<$.0001"
        ci_str = f"[{r.ci_lower:.3f}, {r.ci_upper:.3f}]"

        sig = "$^*$" if r.significant else ""

        lines.append(
            f"{r.test_name} & {r.statistic:.3f} & {p_str} & {p_adj_str}{sig} & "
            f"{r.effect_size:.3f} & {ci_str} \\\\"
        )

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        f"\\label{{tab:stats}}",
        "\\end{table}"
    ])

    return "\n".join(lines)
