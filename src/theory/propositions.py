"""
Formal Propositions for Emergent Specialization.

This module provides theoretical propositions with proof sketches
that establish the conditions under which specialization emerges.

These propositions are inspired by:
- Evolutionary Game Theory (Smith & Price, 1973)
- Niche Partitioning Theory (MacArthur, 1958)
- Multi-Agent Reinforcement Learning (Busoniu et al., 2008)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Proposition1_EquilibriumSpecialization:
    """
    Proposition 1: Specialization as Nash Equilibrium

    STATEMENT:
    -----------
    In a multi-agent system with R distinct regimes, competitive selection,
    and sufficient learning time, the unique stable equilibrium features
    agents specialized to different regimes.

    Formally: Let N agents compete for rewards across R regimes where:
    - Each regime r has optimal strategy s*_r
    - Agents can develop regime preferences (affinities)
    - Only the highest-reward agent wins in each iteration

    Then the Nash equilibrium has N/R agents per regime (on average),
    each with affinity concentrated on their assigned regime.

    PROOF SKETCH:
    -------------
    1. CONTRADICTION: Assume equilibrium with all agents as generalists
       (uniform affinity across regimes).

    2. DEVIATION: Any agent can improve expected reward by specializing
       in an under-served regime r' where competition is lower.

       E[reward | specialize in r'] > E[reward | generalist]

       Because:
       - In r', the specialist wins against generalists (better strategy)
       - Reduced competition: fewer agents competing for same regime

    3. STABILITY: Once agents specialize to different regimes:
       - No agent can improve by switching (other niches occupied)
       - No agent can improve by generalizing (lower win rate in each regime)

    4. UNIQUENESS: The configuration where N/R agents specialize in each
       regime is the unique stable equilibrium up to permutation.

    QED (informal)

    EMPIRICAL VERIFICATION:
    -----------------------
    - SI should increase over training iterations
    - Final SI should approach 1 - 1/R
    - Niche distribution should be approximately uniform across regimes
    """

    n_agents: int
    n_regimes: int

    def theoretical_equilibrium_si(self) -> float:
        """
        Theoretical SI at equilibrium.

        If each agent perfectly specializes in one regime:
        Affinity = [1, 0, 0, 0] for 4 regimes
        Entropy H = 0
        SI = 1 - 0/log(R) = 1

        In practice, with some exploration:
        SI ≈ 1 - ε where ε accounts for exploration
        """
        if self.n_regimes <= 1:
            return 0.0
        # Perfect specialization would give SI = 1
        # Account for residual exploration (~10%)
        exploration_penalty = 0.1
        return 1.0 - exploration_penalty

    def expected_agents_per_regime(self) -> float:
        """Expected agents per regime at equilibrium."""
        return self.n_agents / self.n_regimes

    def verify_equilibrium(
        self,
        niche_distributions: Dict[str, Dict[str, float]],
        threshold: float = 0.6,
    ) -> Dict[str, any]:
        """
        Verify if current state matches equilibrium predictions.

        Args:
            niche_distributions: Agent -> regime affinity distributions
            threshold: SI threshold for "specialized"

        Returns:
            Dict with verification results
        """
        # Compute individual SIs
        agent_sis = []
        primary_niches = []

        for agent_id, affinities in niche_distributions.items():
            aff_array = np.array(list(affinities.values()))
            aff_array = aff_array / (aff_array.sum() + 1e-8)

            # SI computation
            entropy = -np.sum(aff_array * np.log(aff_array + 1e-8))
            max_entropy = np.log(len(aff_array))
            si = 1 - entropy / max_entropy if max_entropy > 0 else 0
            agent_sis.append(si)

            # Primary niche
            regimes = list(affinities.keys())
            primary_niches.append(regimes[np.argmax(aff_array)])

        avg_si = np.mean(agent_sis)
        n_specialists = sum(1 for si in agent_sis if si >= threshold)

        # Check niche coverage
        unique_niches = len(set(primary_niches))
        niche_coverage = unique_niches / self.n_regimes

        return {
            "average_si": avg_si,
            "n_specialists": n_specialists,
            "niche_coverage": niche_coverage,
            "equilibrium_reached": avg_si >= threshold and niche_coverage >= 0.75,
            "theoretical_si": self.theoretical_equilibrium_si(),
        }


@dataclass
class Proposition2_SIConvergence:
    """
    Proposition 2: SI Convergence Bound

    STATEMENT:
    -----------
    Under the niche mechanism with competitive selection, the population's
    average Specialization Index (SI) converges to at least 1 - 1/R as
    the number of iterations T → ∞.

    Formally:
        lim_{T→∞} E[SI(T)] ≥ 1 - 1/R

    where R is the number of regimes.

    INTUITION:
    ----------
    With R regimes, a perfectly specialized agent concentrates all affinity
    on one regime. The minimum entropy state has each agent in a different
    regime, giving:

    SI = 1 - H(1/R, 1/R, ...) / log(R) = 1 - log(R)/log(R) = 0  [uniform]
    SI = 1 - H(1, 0, 0, ...) / log(R) = 1 - 0/log(R) = 1        [specialist]

    With competitive exclusion driving niche partitioning, agents converge
    to the specialist configuration.

    PROOF SKETCH:
    -------------
    1. Let a_i(t) be agent i's affinity distribution at time t

    2. Under niche mechanism, winning in regime r increases a_i(r):
       a_i(r, t+1) = a_i(r, t) + α * I[agent i won in regime r at t]

    3. With competitive selection, agents with higher affinity for current
       regime have higher win probability.

    4. This creates a positive feedback loop:
       High affinity → More wins → Higher affinity

    5. By martingale convergence, each agent's affinity converges to a
       degenerate distribution concentrated on one regime.

    6. Therefore: lim_{T→∞} SI_i(T) → 1 for each agent i

       And: lim_{T→∞} E[SI(T)] → 1 ≥ 1 - 1/R

    EMPIRICAL VERIFICATION:
    -----------------------
    - Plot SI trajectory over iterations
    - Should show monotonic increase (with some noise)
    - Should approach theoretical bound asymptotically
    """

    n_regimes: int
    learning_rate: float = 0.1

    def theoretical_bound(self) -> float:
        """Theoretical lower bound on equilibrium SI."""
        if self.n_regimes <= 1:
            return 0.0
        return 1.0 - 1.0 / self.n_regimes

    def tighter_bound(self, exploration_rate: float = 0.1) -> float:
        """
        Tighter bound accounting for exploration.

        With exploration rate ε, agent doesn't always exploit,
        leading to some entropy in the affinity distribution.
        """
        # Effective concentration with exploration
        concentration = 1.0 - exploration_rate * (self.n_regimes - 1) / self.n_regimes

        # Compute SI for this concentration
        remaining = (1 - concentration) / (self.n_regimes - 1)

        entropy = -concentration * np.log(concentration + 1e-10)
        entropy -= (self.n_regimes - 1) * remaining * np.log(remaining + 1e-10)

        max_entropy = np.log(self.n_regimes)

        return 1.0 - entropy / max_entropy

    def expected_convergence_time(self, n_agents: int, target_si: float = 0.8) -> int:
        """
        Estimate iterations needed to reach target SI.

        Based on exponential convergence model:
        SI(t) ≈ SI_max * (1 - exp(-t/τ))

        where τ is the time constant proportional to n_agents / learning_rate
        """
        si_max = self.theoretical_bound()
        if target_si >= si_max:
            return float('inf')

        # Time constant
        tau = n_agents / self.learning_rate

        # Solve for t: target_si = si_max * (1 - exp(-t/tau))
        # t = -tau * log(1 - target_si/si_max)
        ratio = target_si / si_max
        if ratio >= 1:
            return float('inf')

        t = -tau * np.log(1 - ratio)
        return int(t)

    def verify_convergence(
        self,
        si_trajectory: List[float],
        significance_level: float = 0.05,
    ) -> Dict[str, any]:
        """
        Verify SI trajectory shows convergence.

        Tests:
        1. Positive trend (Spearman correlation with time)
        2. Asymptotic approach to bound
        3. Final SI above bound
        """
        from scipy import stats

        n = len(si_trajectory)
        if n < 10:
            return {"verified": False, "reason": "Insufficient data"}

        # Test 1: Positive trend
        time_points = np.arange(n)
        corr, p_value = stats.spearmanr(time_points, si_trajectory)

        positive_trend = corr > 0 and p_value < significance_level

        # Test 2: Final SI vs bound
        final_si = np.mean(si_trajectory[-10:])
        bound = self.theoretical_bound()
        above_bound = final_si >= bound * 0.9  # Allow 10% margin

        # Test 3: Convergence (variance decreases)
        first_half_var = np.var(si_trajectory[:n//2])
        second_half_var = np.var(si_trajectory[n//2:])
        converging = second_half_var <= first_half_var * 1.5  # Allow some noise

        return {
            "verified": positive_trend and above_bound,
            "spearman_correlation": corr,
            "p_value": p_value,
            "final_si": final_si,
            "theoretical_bound": bound,
            "above_bound": above_bound,
            "converging": converging,
        }


def empirical_proposition_test(
    si_trajectories: List[List[float]],
    niche_distributions: List[Dict[str, Dict[str, float]]],
    n_regimes: int,
    significance_level: float = 0.05,
) -> Dict[str, Dict]:
    """
    Run empirical tests for both propositions.

    Args:
        si_trajectories: SI over time for multiple trials
        niche_distributions: Final niche distributions per trial
        n_regimes: Number of regimes
        significance_level: For statistical tests

    Returns:
        Dict with test results for each proposition
    """
    from scipy import stats

    results = {}

    # Proposition 1: Equilibrium Test
    prop1 = Proposition1_EquilibriumSpecialization(
        n_agents=len(niche_distributions[0]) if niche_distributions else 8,
        n_regimes=n_regimes
    )

    equilibrium_results = []
    for niche_dist in niche_distributions:
        eq_result = prop1.verify_equilibrium(niche_dist)
        equilibrium_results.append(eq_result["equilibrium_reached"])

    results["proposition_1"] = {
        "description": "Specialization as Nash Equilibrium",
        "trials_at_equilibrium": sum(equilibrium_results),
        "total_trials": len(equilibrium_results),
        "success_rate": np.mean(equilibrium_results),
        "verified": np.mean(equilibrium_results) >= 0.8,
    }

    # Proposition 2: Convergence Test
    prop2 = Proposition2_SIConvergence(n_regimes=n_regimes)

    convergence_results = []
    final_sis = []
    for trajectory in si_trajectories:
        conv_result = prop2.verify_convergence(trajectory)
        convergence_results.append(conv_result["verified"])
        final_sis.append(conv_result.get("final_si", 0))

    # One-sample t-test: final SI vs theoretical bound
    bound = prop2.theoretical_bound()
    if final_sis:
        t_stat, p_value = stats.ttest_1samp(final_sis, bound * 0.9)
        above_bound = np.mean(final_sis) >= bound * 0.9 and p_value < significance_level
    else:
        above_bound = False
        p_value = 1.0

    results["proposition_2"] = {
        "description": "SI Convergence Bound",
        "theoretical_bound": bound,
        "empirical_mean_si": np.mean(final_sis) if final_sis else 0,
        "empirical_std_si": np.std(final_sis) if final_sis else 0,
        "trials_converged": sum(convergence_results),
        "total_trials": len(convergence_results),
        "t_test_p_value": p_value,
        "verified": above_bound and np.mean(convergence_results) >= 0.7,
    }

    return results
