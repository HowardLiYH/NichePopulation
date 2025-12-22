"""
Formal Definitions for Emergent Specialization Theory.

This module provides rigorous mathematical definitions for:
1. Market Regimes
2. Niche Partitioning
3. Specialization Equilibrium

References:
- MacArthur, R.H. (1958). "Population Ecology of Some Warblers of
  Northeastern Coniferous Forests." Ecology, 39(4), 599-619.
- Hardin, G. (1960). "The Competitive Exclusion Principle." Science, 131, 1292-1297.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import stats


@dataclass
class RegimeDefinition:
    """
    Formal Definition of a Market Regime.

    A regime R is defined as a tuple (μ, σ, τ, f*) where:
    - μ: Expected return (mean)
    - σ: Volatility (standard deviation)
    - τ: Expected duration (mean regime length)
    - f*: Optimal strategy for this regime

    Three mathematical properties characterize a valid regime:

    1. STATIONARITY: Within a regime, returns are drawn from a
       stationary distribution D_R with fixed parameters.

       Formally: ∀t₁, t₂ ∈ R: P(r_t₁) = P(r_t₂) = D_R(μ_R, σ_R)

    2. DISTINGUISHABILITY: Different regimes have distinguishable
       optimal strategies with performance gap δ > 0.

       Formally: ∀R_i ≠ R_j: E[r | f*_i, R_i] - E[r | f*_j, R_i] ≥ δ

       (Using strategy optimal for R_j in regime R_i yields lower returns)

    3. PERSISTENCE: Regimes persist for sufficient time to enable learning.

       Formally: E[τ_R] ≥ τ_min where τ_min is the learning horizon
    """

    name: str
    mean_return: float  # μ
    volatility: float   # σ
    duration_mean: float  # E[τ]
    duration_std: float   # Std[τ]
    optimal_strategy: str  # f*

    def __post_init__(self):
        """Validate regime parameters."""
        assert self.volatility >= 0, "Volatility must be non-negative"
        assert self.duration_mean > 0, "Duration must be positive"

    def sample_return(self, rng: np.random.Generator = None) -> float:
        """Sample a return from this regime's distribution."""
        rng = rng or np.random.default_rng()
        return rng.normal(self.mean_return, self.volatility)

    def sample_duration(self, rng: np.random.Generator = None) -> int:
        """Sample regime duration."""
        rng = rng or np.random.default_rng()
        duration = int(rng.normal(self.duration_mean, self.duration_std))
        return max(1, duration)


@dataclass
class NichePartitioningTheory:
    """
    Niche Partitioning Theory Applied to Multi-Agent Systems.

    Key Concepts:

    1. NICHE: A subset of the environment (regime) where an agent
       has competitive advantage.

       Formally: Niche_i = {R : E[r | a_i, R] ≥ E[r | a_j, R] ∀j}

    2. NICHE WIDTH: The number of regimes an agent specializes in.

       Narrow niche → Specialist (high SI)
       Wide niche → Generalist (low SI)

    3. COMPETITIVE EXCLUSION: In competitive environments, no two
       agents can occupy the exact same niche indefinitely.

       This drives differentiation and specialization.

    4. RESOURCE PARTITIONING: When competition exists, agents
       partition resources (regimes) to reduce competition.

       Result: Each agent specializes in different regimes.

    Prediction: With R regimes and N >> R agents, we expect
    approximately N/R specialists per regime at equilibrium.
    """

    n_regimes: int
    n_agents: int
    regime_names: List[str]

    def expected_specialists_per_regime(self) -> float:
        """Expected number of specialists per regime at equilibrium."""
        return self.n_agents / self.n_regimes

    def expected_si_at_equilibrium(self) -> float:
        """
        Expected Specialization Index at equilibrium.

        If each agent specializes perfectly in one regime,
        SI = 1 - 1/R (entropy normalized)

        With R = 4 regimes: SI_max ≈ 0.75
        With R = 2 regimes: SI_max ≈ 0.50
        """
        if self.n_regimes <= 1:
            return 0.0
        return 1.0 - 1.0 / self.n_regimes

    def niche_overlap(
        self,
        affinity_i: Dict[str, float],
        affinity_j: Dict[str, float]
    ) -> float:
        """
        Compute niche overlap between two agents.

        Overlap = Σ min(p_i(R), p_j(R))

        Range: [0, 1] where 0 = no overlap, 1 = identical niches
        """
        overlap = 0.0
        for regime in self.regime_names:
            overlap += min(
                affinity_i.get(regime, 0.0),
                affinity_j.get(regime, 0.0)
            )
        return overlap


def verify_regime_properties(
    returns: np.ndarray,
    regime_labels: np.ndarray,
    method_rewards: Dict[str, Dict[str, List[float]]],
    min_duration: int = 10,
    significance_level: float = 0.05,
) -> Dict[str, Dict[str, bool]]:
    """
    Verify the three regime properties empirically.

    Args:
        returns: Array of returns
        regime_labels: Array of regime labels
        method_rewards: Dict[regime][method] -> list of rewards
        min_duration: Minimum expected regime duration
        significance_level: For statistical tests

    Returns:
        Dict with verification results for each property
    """
    results = {}
    unique_regimes = np.unique(regime_labels)

    # 1. STATIONARITY: Test for constant mean within regime
    stationarity = {}
    for regime in unique_regimes:
        regime_mask = regime_labels == regime
        regime_returns = returns[regime_mask]

        if len(regime_returns) > 20:
            # Split in half and compare
            mid = len(regime_returns) // 2
            first_half = regime_returns[:mid]
            second_half = regime_returns[mid:]

            # Two-sample t-test
            stat, pvalue = stats.ttest_ind(first_half, second_half)
            stationarity[str(regime)] = pvalue > significance_level
        else:
            stationarity[str(regime)] = True  # Insufficient data

    results["stationarity"] = stationarity

    # 2. DISTINGUISHABILITY: Best method differs across regimes
    distinguishability = {}
    for regime in unique_regimes:
        if str(regime) not in method_rewards:
            distinguishability[str(regime)] = False
            continue

        regime_methods = method_rewards[str(regime)]
        if len(regime_methods) < 2:
            distinguishability[str(regime)] = False
            continue

        # Find best method
        method_means = {m: np.mean(r) for m, r in regime_methods.items() if len(r) > 0}
        if len(method_means) < 2:
            distinguishability[str(regime)] = False
            continue

        sorted_methods = sorted(method_means.items(), key=lambda x: x[1], reverse=True)
        best_mean = sorted_methods[0][1]
        second_mean = sorted_methods[1][1]

        # Check if gap is significant
        gap = best_mean - second_mean
        distinguishability[str(regime)] = gap > 0.01  # 1% performance gap

    results["distinguishability"] = distinguishability

    # 3. PERSISTENCE: Check regime durations
    persistence = {}
    for regime in unique_regimes:
        # Find regime runs
        regime_mask = regime_labels == regime
        runs = []
        current_run = 0

        for is_regime in regime_mask:
            if is_regime:
                current_run += 1
            elif current_run > 0:
                runs.append(current_run)
                current_run = 0
        if current_run > 0:
            runs.append(current_run)

        if runs:
            mean_duration = np.mean(runs)
            persistence[str(regime)] = mean_duration >= min_duration
        else:
            persistence[str(regime)] = False

    results["persistence"] = persistence

    return results


def compute_theoretical_si_bound(n_regimes: int, concentration: float = 0.8) -> float:
    """
    Compute theoretical SI bound for perfect specialization.

    With R regimes, if each agent concentrates 'concentration' fraction
    of its affinity on one regime:

    SI = 1 - H(p) / log(R)

    where H(p) = -concentration * log(concentration)
                 - (1-concentration) * log((1-concentration)/(R-1))

    Args:
        n_regimes: Number of regimes
        concentration: Fraction of affinity on primary regime

    Returns:
        Theoretical SI for given concentration
    """
    if n_regimes <= 1:
        return 0.0

    # Compute entropy of concentrated distribution
    remaining = (1 - concentration) / (n_regimes - 1)

    entropy = -concentration * np.log(concentration + 1e-10)
    entropy -= (n_regimes - 1) * remaining * np.log(remaining + 1e-10)

    max_entropy = np.log(n_regimes)

    return 1.0 - entropy / max_entropy
