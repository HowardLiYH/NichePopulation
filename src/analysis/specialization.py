"""
Specialization Metrics for Agent Populations.

This module provides metrics to quantify:
1. Individual agent specialization (how focused on specific methods)
2. Population diversity (how different agents are from each other)
3. Regime coverage (does population cover all market regimes)

Theoretical References:
- Shannon, C.E. (1948). "A Mathematical Theory of Communication."
  Bell System Technical Journal, 27(3), 379-423.
- Lin, J. (1991). "Divergence Measures Based on the Shannon Entropy."
  IEEE Transactions on Information Theory, 37(1), 145-151.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from collections import defaultdict


def compute_entropy(distribution: Dict[str, float]) -> float:
    """
    Compute Shannon entropy of a probability distribution.

    H(p) = -Σ p_i * log(p_i)

    Args:
        distribution: Dict mapping items to probabilities (must sum to 1)

    Returns:
        Entropy value (>= 0)
    """
    probs = np.array(list(distribution.values()))
    probs = probs[probs > 0]  # Ignore zeros (0 * log(0) = 0)

    if len(probs) == 0:
        return 0.0

    return float(-np.sum(probs * np.log(probs)))


def compute_max_entropy(n_items: int) -> float:
    """
    Compute maximum possible entropy (uniform distribution).

    H_max = log(n)
    """
    if n_items <= 1:
        return 0.0
    return np.log(n_items)


def compute_specialization_index(
    method_usage: Dict[str, float],
    n_methods: Optional[int] = None,
) -> float:
    """
    Compute Specialization Index (SI) for an agent.

    SI = 1 - H(p) / H_max

    Where:
    - H(p) is Shannon entropy of method usage distribution
    - H_max is maximum entropy (uniform distribution)

    Range: [0, 1]
    - SI = 0: Pure generalist (uniform usage of all methods)
    - SI = 1: Pure specialist (uses only one method)

    Args:
        method_usage: Dict mapping method names to usage frequencies
        n_methods: Total number of methods (for max entropy calculation)

    Returns:
        Specialization index in [0, 1]
    """
    # Normalize to probabilities
    total = sum(method_usage.values())
    if total == 0:
        return 0.0

    probs = {k: v / total for k, v in method_usage.items()}

    # Compute entropy
    entropy = compute_entropy(probs)

    # Max entropy
    n = n_methods or len(method_usage)
    max_entropy = compute_max_entropy(n)

    if max_entropy == 0:
        return 1.0  # Only one method available

    # Specialization index
    return 1.0 - (entropy / max_entropy)


def compute_jensen_shannon_divergence(
    p: Dict[str, float],
    q: Dict[str, float],
) -> float:
    """
    Compute Jensen-Shannon divergence between two distributions.

    JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q)

    Range: [0, log(2)] for natural log

    Reference: Lin, J. (1991)
    """
    # Get all keys
    all_keys = set(p.keys()) | set(q.keys())

    # Normalize
    p_sum = sum(p.values())
    q_sum = sum(q.values())

    p_norm = {k: p.get(k, 0) / p_sum for k in all_keys}
    q_norm = {k: q.get(k, 0) / q_sum for k in all_keys}

    # Compute M
    m = {k: 0.5 * (p_norm[k] + q_norm[k]) for k in all_keys}

    # Compute KL divergences
    def kl_divergence(a: Dict, b: Dict) -> float:
        kl = 0.0
        for k in a:
            if a[k] > 0 and b[k] > 0:
                kl += a[k] * np.log(a[k] / b[k])
        return kl

    jsd = 0.5 * kl_divergence(p_norm, m) + 0.5 * kl_divergence(q_norm, m)
    return float(jsd)


def compute_population_diversity(
    agent_distributions: Dict[str, Dict[str, float]],
) -> float:
    """
    Compute Population Diversity Index (PDI).

    PDI = mean pairwise Jensen-Shannon divergence between agents

    Higher PDI indicates more diverse population (agents have different preferences).

    Args:
        agent_distributions: Dict mapping agent_id to method usage distribution

    Returns:
        Population diversity index (>= 0)
    """
    agent_ids = list(agent_distributions.keys())
    n = len(agent_ids)

    if n < 2:
        return 0.0

    # Compute pairwise JSD
    total_jsd = 0.0
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            jsd = compute_jensen_shannon_divergence(
                agent_distributions[agent_ids[i]],
                agent_distributions[agent_ids[j]],
            )
            total_jsd += jsd
            count += 1

    return total_jsd / count if count > 0 else 0.0


def compute_method_coverage(
    agent_distributions: Dict[str, Dict[str, float]],
    usage_threshold: float = 0.1,
) -> float:
    """
    Compute method coverage: fraction of methods used by population.

    A method is "used" if any agent uses it above the threshold.

    Args:
        agent_distributions: Dict mapping agent_id to method usage distribution
        usage_threshold: Minimum usage to count as "used"

    Returns:
        Fraction of methods covered [0, 1]
    """
    if not agent_distributions:
        return 0.0

    # Get all method names
    all_methods = set()
    for dist in agent_distributions.values():
        all_methods.update(dist.keys())

    # Count methods used above threshold
    used_methods = set()
    for dist in agent_distributions.values():
        for method, usage in dist.items():
            if usage >= usage_threshold:
                used_methods.add(method)

    return len(used_methods) / len(all_methods) if all_methods else 0.0


def count_specialists(
    agent_distributions: Dict[str, Dict[str, float]],
    specialist_threshold: float = 0.7,
    n_methods: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Count specialists and generalists in population.

    Args:
        agent_distributions: Dict mapping agent_id to method usage distribution
        specialist_threshold: SI threshold to count as specialist
        n_methods: Total number of methods

    Returns:
        Tuple of (n_specialists, n_generalists)
    """
    n_specialists = 0
    n_generalists = 0

    for dist in agent_distributions.values():
        si = compute_specialization_index(dist, n_methods)
        if si >= specialist_threshold:
            n_specialists += 1
        else:
            n_generalists += 1

    return n_specialists, n_generalists


def compute_regime_coverage(
    agents: Dict,
    regime_names: List[str],
) -> float:
    """
    Compute regime coverage: fraction of regimes with a specialist.
    
    A regime is "covered" if at least one agent specializes in it
    (wins more than average in that regime).
    
    Args:
        agents: Dict of agent objects with regime_wins attribute
        regime_names: List of regime names to check
        
    Returns:
        Fraction of regimes covered [0, 1]
    """
    if not regime_names:
        return 0.0
    
    covered_regimes = 0
    
    for regime in regime_names:
        # Check if any agent specializes in this regime
        for agent in agents.values():
            if hasattr(agent, 'regime_wins') and agent.regime_wins.get(regime, 0) > 0:
                covered_regimes += 1
                break
    
    return covered_regimes / len(regime_names)


@dataclass
class AgentMetrics:
    """Metrics for a single agent."""
    agent_id: str
    specialization_index: float
    dominant_method: Optional[str]
    method_usage: Dict[str, float]
    win_rate: float = 0.0
    cumulative_reward: float = 0.0


@dataclass
class PopulationMetrics:
    """Aggregate metrics for the population."""
    diversity_index: float
    method_coverage: float
    n_specialists: int
    n_generalists: int
    avg_specialization: float
    agent_metrics: List[AgentMetrics] = field(default_factory=list)


class SpecializationTracker:
    """
    Tracks specialization metrics over time for analysis.

    Usage:
        tracker = SpecializationTracker()

        for iteration in range(n_iterations):
            # ... run population iteration ...
            tracker.record(iteration, population)

        # Get metrics over time
        df = tracker.to_dataframe()
    """

    def __init__(self, n_methods: Optional[int] = None):
        """
        Initialize tracker.

        Args:
            n_methods: Total number of methods in inventory
        """
        self.n_methods = n_methods
        self.history: List[Dict] = []

    def record(
        self,
        iteration: int,
        agent_distributions: Dict[str, Dict[str, float]],
        regime: Optional[str] = None,
        additional_data: Optional[Dict] = None,
    ) -> PopulationMetrics:
        """
        Record metrics for current state.

        Args:
            iteration: Current iteration number
            agent_distributions: Dict mapping agent_id to method usage
            regime: Optional current regime label
            additional_data: Optional additional data to record

        Returns:
            PopulationMetrics for this iteration
        """
        # Compute agent-level metrics
        agent_metrics = []
        for agent_id, usage in agent_distributions.items():
            si = compute_specialization_index(usage, self.n_methods)

            # Find dominant method
            dominant = max(usage, key=usage.get) if usage else None

            agent_metrics.append(AgentMetrics(
                agent_id=agent_id,
                specialization_index=si,
                dominant_method=dominant,
                method_usage=usage,
            ))

        # Compute population-level metrics
        diversity = compute_population_diversity(agent_distributions)
        coverage = compute_method_coverage(agent_distributions)
        n_spec, n_gen = count_specialists(agent_distributions, n_methods=self.n_methods)
        avg_si = np.mean([m.specialization_index for m in agent_metrics])

        pop_metrics = PopulationMetrics(
            diversity_index=diversity,
            method_coverage=coverage,
            n_specialists=n_spec,
            n_generalists=n_gen,
            avg_specialization=float(avg_si),
            agent_metrics=agent_metrics,
        )

        # Record history
        record = {
            "iteration": iteration,
            "diversity_index": diversity,
            "method_coverage": coverage,
            "n_specialists": n_spec,
            "n_generalists": n_gen,
            "avg_specialization": avg_si,
            "regime": regime,
        }

        # Add per-agent SI
        for m in agent_metrics:
            record[f"si_{m.agent_id}"] = m.specialization_index

        if additional_data:
            record.update(additional_data)

        self.history.append(record)

        return pop_metrics

    def to_dataframe(self) -> pd.DataFrame:
        """Convert history to DataFrame for analysis."""
        return pd.DataFrame(self.history)

    def get_final_metrics(self) -> Optional[PopulationMetrics]:
        """Get metrics from the last recorded iteration."""
        if not self.history:
            return None

        last = self.history[-1]
        return PopulationMetrics(
            diversity_index=last["diversity_index"],
            method_coverage=last["method_coverage"],
            n_specialists=last["n_specialists"],
            n_generalists=last["n_generalists"],
            avg_specialization=last["avg_specialization"],
        )

    def get_specialization_trajectory(self, agent_id: str) -> List[float]:
        """Get SI over time for a specific agent."""
        key = f"si_{agent_id}"
        return [r.get(key, 0.0) for r in self.history]


# Alias for backward compatibility
SpecializationMetrics = PopulationMetrics
