"""
Regime-Conditioned Method Selector.

Key innovation: Agents maintain SEPARATE beliefs for each regime,
enabling natural specialization to emerge.

This is the key mechanism for emergent specialization - agents
learn which methods work best in which regimes, and can then
specialize to regimes where they have high-confidence beliefs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict

from .inventory import get_method_names
from .inventory_v2 import METHOD_INVENTORY_V2


@dataclass
class MethodBelief:
    """Beta-distributed belief about method performance."""
    successes: float = 1.0
    failures: float = 1.0

    @property
    def alpha(self) -> float:
        return self.successes + 1

    @property
    def beta(self) -> float:
        return self.failures + 1

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def uncertainty(self) -> float:
        """Higher uncertainty = more exploration needed."""
        a, b = self.alpha, self.beta
        return np.sqrt((a * b) / ((a + b) ** 2 * (a + b + 1)))

    def sample(self, rng: np.random.Generator) -> float:
        return rng.beta(self.alpha, self.beta)

    def update(self, success: bool, weight: float = 1.0) -> None:
        """Binary update for cleaner learning signal."""
        if success:
            self.successes += weight
        else:
            self.failures += weight


class RegimeConditionedSelector:
    """
    Agent with regime-conditioned beliefs.

    Key difference from MethodSelector:
    - Maintains beliefs[regime][method] instead of beliefs[method]
    - Selects methods based on CURRENT regime's beliefs
    - Updates beliefs in the correct regime context

    This enables agents to naturally specialize to regimes
    where their preferred methods are most effective.
    """

    def __init__(
        self,
        agent_id: str,
        regimes: List[str] = None,
        max_methods: int = 1,
        exploration_rate: float = 0.15,
        seed: Optional[int] = None,
    ):
        self.agent_id = agent_id
        self.regimes = regimes or ["trend_up", "trend_down", "mean_revert", "volatile"]
        self.max_methods = max_methods
        self.exploration_rate = exploration_rate
        self.rng = np.random.default_rng(seed)

        self.method_names = get_method_names()

        # Regime-conditioned beliefs: beliefs[regime][method]
        self.beliefs: Dict[str, Dict[str, MethodBelief]] = {
            regime: {method: MethodBelief() for method in self.method_names}
            for regime in self.regimes
        }

        # Track usage per regime for specialization analysis
        self.regime_method_usage: Dict[str, Dict[str, int]] = {
            regime: defaultdict(int) for regime in self.regimes
        }

        # Track wins per regime
        self.regime_wins: Dict[str, int] = defaultdict(int)
        self.total_wins: int = 0

        # Track overall usage (for SI calculation)
        self.total_usage: Dict[str, int] = defaultdict(int)

    def select(self, regime: str) -> List[str]:
        """
        Select methods based on current regime's beliefs.

        Uses Thompson Sampling within the regime-specific belief space.
        """
        if regime not in self.regimes:
            regime = self.regimes[0]  # Fallback

        if self.rng.random() < self.exploration_rate:
            # Exploration: random method
            methods = [self.rng.choice(self.method_names)]
        else:
            # Thompson Sampling within regime
            regime_beliefs = self.beliefs[regime]
            samples = {
                method: belief.sample(self.rng)
                for method, belief in regime_beliefs.items()
            }
            sorted_methods = sorted(samples.keys(), key=lambda x: samples[x], reverse=True)
            methods = sorted_methods[:self.max_methods]

        # Track usage
        for m in methods:
            self.regime_method_usage[regime][m] += 1
            self.total_usage[m] += 1

        return methods

    def update(self, regime: str, methods: List[str], won: bool) -> None:
        """
        Update beliefs based on outcome (win/loss).

        This is the key difference: we only update beliefs in the
        current regime's context, enabling regime-specific learning.
        """
        if regime not in self.regimes:
            return

        for method in methods:
            if method in self.beliefs[regime]:
                self.beliefs[regime][method].update(success=won, weight=1.0)

        if won:
            self.regime_wins[regime] += 1
            self.total_wins += 1

    def get_regime_preference(self) -> Dict[str, float]:
        """
        Get this agent's preference distribution over regimes.

        Based on where the agent has highest-confidence beliefs.
        """
        regime_confidence = {}

        for regime in self.regimes:
            beliefs = self.beliefs[regime]
            # Confidence = max belief strength in this regime
            max_belief = max(b.mean for b in beliefs.values())
            confidence = max_belief
            regime_confidence[regime] = confidence

        # Normalize to distribution
        total = sum(regime_confidence.values()) + 1e-8
        return {r: c / total for r, c in regime_confidence.items()}

    def get_regime_win_rate(self, regime: str, total_regime_iters: int) -> float:
        """Get this agent's win rate in a specific regime."""
        if total_regime_iters == 0:
            return 0.0
        return self.regime_wins.get(regime, 0) / total_regime_iters

    def get_dominant_method_in_regime(self, regime: str) -> str:
        """Get the method this agent most uses in a regime."""
        usage = self.regime_method_usage.get(regime, {})
        if not usage:
            return self.method_names[0]
        return max(usage, key=usage.get)

    def get_specialization_regime(self) -> Optional[str]:
        """
        Get the regime this agent is specialized in, if any.

        An agent is considered specialized if it wins significantly
        more in one regime than others.
        """
        if not self.regime_wins:
            return None

        total = sum(self.regime_wins.values())
        if total == 0:
            return None

        # Check for specialization (>50% of wins in one regime)
        for regime, wins in self.regime_wins.items():
            if wins / total > 0.4:  # 40% threshold
                return regime

        return None

    def get_method_usage_distribution(self) -> Dict[str, float]:
        """Get overall method usage distribution (for SI calculation)."""
        total = sum(self.total_usage.values())
        if total == 0:
            n = len(self.method_names)
            return {m: 1.0 / n for m in self.method_names}
        return {m: self.total_usage[m] / total for m in self.method_names}

    def copy_beliefs_from(
        self,
        other: "RegimeConditionedSelector",
        regime: str,
        tau: float = 0.1,
    ) -> None:
        """
        Regime-specific knowledge transfer.

        Only transfer beliefs for a specific regime, preserving
        the agent's specialization in other regimes.
        """
        if regime not in self.regimes:
            return

        for method in self.method_names:
            own = self.beliefs[regime][method]
            src = other.beliefs[regime][method]

            own.successes = (1 - tau) * own.successes + tau * src.successes
            own.failures = (1 - tau) * own.failures + tau * src.failures

    def __repr__(self) -> str:
        spec_regime = self.get_specialization_regime()
        return f"RegimeConditionedSelector(id={self.agent_id}, specialized={spec_regime})"


@dataclass
class PopulationResult:
    """Result of one population iteration."""
    iteration: int
    regime: str
    winner_id: str
    winner_methods: List[str]
    all_selected: Dict[str, List[str]]


class RegimeConditionedPopulation:
    """
    Population of regime-conditioned agents.

    Key differences from Population:
    1. Agents select methods based on current regime
    2. Winner is determined within regime context
    3. Knowledge transfer is regime-specific
    4. Tracks regime-conditional statistics for specialization analysis
    """

    def __init__(
        self,
        n_agents: int = 5,
        regimes: List[str] = None,
        transfer_frequency: int = 100,
        transfer_tau: float = 0.05,
        seed: Optional[int] = None,
    ):
        self.n_agents = n_agents
        self.regimes = regimes or ["trend_up", "trend_down", "mean_revert", "volatile"]
        self.transfer_frequency = transfer_frequency
        self.transfer_tau = transfer_tau
        self.rng = np.random.default_rng(seed)

        # Create agents
        self.agents: Dict[str, RegimeConditionedSelector] = {}
        for i in range(n_agents):
            agent_id = f"agent_{i}"
            self.agents[agent_id] = RegimeConditionedSelector(
                agent_id=agent_id,
                regimes=self.regimes,
                seed=seed + i if seed else None,
            )

        self.iteration = 0
        self.regime_iterations: Dict[str, int] = defaultdict(int)
        self.history: List[PopulationResult] = []

    def run_iteration(
        self,
        prices: np.ndarray,
        regime: str,
        reward_fn,
    ) -> PopulationResult:
        """
        Run one iteration with regime-conditioned selection.
        """
        self.iteration += 1
        self.regime_iterations[regime] += 1

        # Each agent selects methods based on regime
        selections: Dict[str, List[str]] = {}
        rewards: Dict[str, float] = {}

        for agent_id, agent in self.agents.items():
            methods = agent.select(regime)
            selections[agent_id] = methods
            reward = reward_fn(methods, prices)
            rewards[agent_id] = reward

        # Determine winner
        winner_id = max(rewards, key=rewards.get)

        # Update all agents (winner gets positive update, others get negative)
        for agent_id, agent in self.agents.items():
            won = (agent_id == winner_id)
            agent.update(regime, selections[agent_id], won=won)

        # Regime-specific knowledge transfer (less frequent)
        if self.iteration % self.transfer_frequency == 0:
            self._transfer_knowledge_in_regime(regime)

        result = PopulationResult(
            iteration=self.iteration,
            regime=regime,
            winner_id=winner_id,
            winner_methods=selections[winner_id],
            all_selected=selections,
        )
        self.history.append(result)

        return result

    def _transfer_knowledge_in_regime(self, regime: str) -> None:
        """Transfer knowledge only for the current regime."""
        # Find best agent in this regime
        win_rates = {
            aid: agent.get_regime_win_rate(regime, self.regime_iterations[regime])
            for aid, agent in self.agents.items()
        }
        best_id = max(win_rates, key=win_rates.get)
        best_agent = self.agents[best_id]

        # Transfer only to worse-performing agents
        for agent_id, agent in self.agents.items():
            if agent_id != best_id and win_rates[agent_id] < win_rates[best_id] * 0.5:
                agent.copy_beliefs_from(best_agent, regime, tau=self.transfer_tau)

    def get_regime_win_matrix(self) -> Dict[str, Dict[str, float]]:
        """Get win rate matrix: agents x regimes."""
        matrix = {}
        for agent_id, agent in self.agents.items():
            matrix[agent_id] = {}
            for regime in self.regimes:
                total = self.regime_iterations.get(regime, 1)
                matrix[agent_id][regime] = agent.get_regime_win_rate(regime, total)
        return matrix

    def get_specialists(self) -> Dict[str, str]:
        """Get mapping of agents to their specialized regime (if any)."""
        return {
            agent_id: agent.get_specialization_regime()
            for agent_id, agent in self.agents.items()
        }

    def get_all_method_usage(self) -> Dict[str, Dict[str, float]]:
        """Get method usage for all agents (for SI calculation)."""
        return {
            agent_id: agent.get_method_usage_distribution()
            for agent_id, agent in self.agents.items()
        }

    def __repr__(self) -> str:
        specialists = self.get_specialists()
        n_spec = sum(1 for s in specialists.values() if s is not None)
        return f"RegimeConditionedPopulation(n={self.n_agents}, specialists={n_spec})"
