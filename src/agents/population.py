"""
Population of Method Selector Agents with Knowledge Transfer.

This module implements the population-based learning mechanism where:
1. Multiple agents compete on the same market data
2. Best-performing agents transfer knowledge to others
3. Population maintains diversity through controlled transfer

Theoretical References:
- Jaderberg, M., et al. (2017). "Population Based Training of Neural Networks."
  arXiv:1711.09846.
- Maynard Smith, J. (1982). Evolution and the Theory of Games.
  Cambridge University Press.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from collections import defaultdict
import pandas as pd

from .method_selector import MethodSelector, SelectionResult
from .inventory import METHOD_INVENTORY


@dataclass
class PopulationConfig:
    """Configuration for population-based learning."""

    # Population size
    n_agents: int = 5

    # Method selection
    max_methods_per_agent: int = 2  # Fewer methods = sharper specialization

    # Knowledge transfer (REDUCED for diversity)
    transfer_frequency: int = 50   # Less frequent (was 10)
    transfer_tau: float = 0.05     # Gentler transfer (was 0.1)

    # Exploration
    min_exploration_rate: float = 0.10  # More exploration (was 0.05)
    forgetting_factor: float = 0.999    # Less forgetting (was 0.995)

    # Random seed
    seed: Optional[int] = None


@dataclass
class IterationResult:
    """Result of one population iteration."""
    iteration: int
    winner_id: str
    winner_methods: List[str]
    winner_reward: float
    all_rewards: Dict[str, float]
    regime: Optional[str] = None


class Population:
    """
    Population of Method Selector agents.

    Key features:
    1. Multiple agents compete on each trading decision
    2. Winner determined by reward (trading performance)
    3. Periodic knowledge transfer from best to others
    4. Diversity maintained through controlled transfer rate

    This implements the evolutionary dynamics that lead to
    emergent specialization.
    """

    def __init__(self, config: Optional[PopulationConfig] = None):
        """Initialize population."""
        self.config = config or PopulationConfig()
        self.rng = np.random.default_rng(self.config.seed)

        # Create agents
        self.agents: Dict[str, MethodSelector] = {}
        for i in range(self.config.n_agents):
            agent_id = f"agent_{i}"
            self.agents[agent_id] = MethodSelector(
                agent_id=agent_id,
                max_methods=self.config.max_methods_per_agent,
                forgetting_factor=self.config.forgetting_factor,
                min_exploration_rate=self.config.min_exploration_rate,
                seed=self.config.seed + i if self.config.seed else None,
            )

        # Track iteration count
        self.iteration = 0

        # Track win history for each agent
        self.win_counts: Dict[str, int] = defaultdict(int)
        self.cumulative_rewards: Dict[str, float] = defaultdict(float)

        # Track regime-conditional wins
        self.regime_wins: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # History for analysis
        self.iteration_history: List[IterationResult] = []

    def run_iteration(
        self,
        prices: pd.DataFrame,
        reward_fn: Callable[[List[str], pd.DataFrame], float],
        regime: Optional[str] = None,
    ) -> IterationResult:
        """
        Run one iteration of population learning.

        Args:
            prices: Price data for this period
            reward_fn: Function that computes reward given methods and prices
            regime: Optional ground-truth regime label (for analysis)

        Returns:
            IterationResult with winner and rewards
        """
        self.iteration += 1

        # Each agent selects methods
        selections: Dict[str, SelectionResult] = {}
        for agent_id, agent in self.agents.items():
            selections[agent_id] = agent.select()

        # Compute rewards for each agent
        rewards: Dict[str, float] = {}
        for agent_id, selection in selections.items():
            reward = reward_fn(selection.methods, prices)
            rewards[agent_id] = reward

        # Determine winner
        winner_id = max(rewards, key=rewards.get)
        winner_reward = rewards[winner_id]
        winner_methods = selections[winner_id].methods

        # Update win counts
        self.win_counts[winner_id] += 1
        if regime:
            self.regime_wins[winner_id][regime] += 1

        # Update all agents with their rewards
        for agent_id, agent in self.agents.items():
            agent.update(selections[agent_id].methods, rewards[agent_id])
            self.cumulative_rewards[agent_id] += rewards[agent_id]

        # Knowledge transfer (if scheduled)
        if self.iteration % self.config.transfer_frequency == 0:
            self._transfer_knowledge()

        # Record result
        result = IterationResult(
            iteration=self.iteration,
            winner_id=winner_id,
            winner_methods=winner_methods,
            winner_reward=winner_reward,
            all_rewards=rewards,
            regime=regime,
        )
        self.iteration_history.append(result)

        return result

    def _transfer_knowledge(self) -> None:
        """
        Transfer knowledge from best agent to others.

        This implements soft knowledge sharing - not full copying,
        but interpolation of beliefs.
        """
        # Find best agent by cumulative reward
        best_id = max(self.cumulative_rewards, key=self.cumulative_rewards.get)
        best_agent = self.agents[best_id]

        # Transfer to other agents
        for agent_id, agent in self.agents.items():
            if agent_id != best_id:
                agent.copy_beliefs_from(best_agent, tau=self.config.transfer_tau)

    def get_agent(self, agent_id: str) -> MethodSelector:
        """Get agent by ID."""
        return self.agents[agent_id]

    def get_all_preferences(self) -> Dict[str, Dict[str, float]]:
        """Get preference distributions for all agents."""
        return {
            agent_id: agent.get_preferences()
            for agent_id, agent in self.agents.items()
        }

    def get_all_method_usage(self) -> Dict[str, Dict[str, float]]:
        """Get method usage distributions for all agents."""
        return {
            agent_id: agent.get_method_usage_distribution()
            for agent_id, agent in self.agents.items()
        }

    def get_regime_win_rates(self) -> pd.DataFrame:
        """
        Get regime-conditional win rate matrix.

        Returns:
            DataFrame with agents as rows, regimes as columns
        """
        # Collect all regimes
        all_regimes = set()
        for agent_wins in self.regime_wins.values():
            all_regimes.update(agent_wins.keys())

        if not all_regimes:
            return pd.DataFrame()

        # Compute win rates
        regime_totals = defaultdict(int)
        for result in self.iteration_history:
            if result.regime:
                regime_totals[result.regime] += 1

        data = {}
        for agent_id in self.agents:
            agent_wins = self.regime_wins[agent_id]
            data[agent_id] = {
                regime: agent_wins[regime] / max(regime_totals[regime], 1)
                for regime in all_regimes
            }

        return pd.DataFrame(data).T

    def get_best_agent(self) -> Tuple[str, MethodSelector]:
        """Get the best performing agent."""
        best_id = max(self.cumulative_rewards, key=self.cumulative_rewards.get)
        return best_id, self.agents[best_id]

    def clone_best_agent(self) -> "Population":
        """
        Create a homogeneous population by cloning the best agent.

        This is used for baseline comparison (Experiment 2).
        """
        best_id, best_agent = self.get_best_agent()

        # Create new population with same config
        new_pop = Population(self.config)

        # Copy beliefs from best agent to all
        for agent in new_pop.agents.values():
            agent.copy_beliefs_from(best_agent, tau=1.0)  # Full copy

        return new_pop

    def reset(self) -> None:
        """Reset population to initial state."""
        for agent in self.agents.values():
            agent.reset_history()
            # Reset beliefs to uniform
            for belief in agent.beliefs.values():
                belief.successes = 1.0
                belief.failures = 1.0

        self.iteration = 0
        self.win_counts = defaultdict(int)
        self.cumulative_rewards = defaultdict(float)
        self.regime_wins = defaultdict(lambda: defaultdict(int))
        self.iteration_history = []

    def __repr__(self) -> str:
        return f"Population(n_agents={len(self.agents)}, iteration={self.iteration})"


def compute_reward_from_methods(
    methods: List[str],
    prices: pd.DataFrame,
    regime: Optional[str] = None,
) -> float:
    """
    Compute reward by executing methods on price data.

    This is a simple reward function that aggregates signals from
    selected methods and computes PnL based on price movement.

    Args:
        methods: List of method names to execute
        prices: Price DataFrame with at least 'close' column
        regime: Optional regime label (for Oracle comparison)

    Returns:
        Normalized reward in [-1, 1]
    """
    if len(prices) < 2:
        return 0.0

    # Get signals from each method
    signals = []
    confidences = []

    for method_name in methods:
        if method_name in METHOD_INVENTORY:
            method = METHOD_INVENTORY[method_name]
            result = method.execute(prices)
            signals.append(result["signal"])
            confidences.append(result["confidence"])

    if not signals:
        return 0.0

    # Weighted average signal
    weights = np.array(confidences)
    weights = weights / (weights.sum() + 1e-8)
    combined_signal = np.sum(np.array(signals) * weights)

    # Compute actual return
    price_return = (prices["close"].iloc[-1] / prices["close"].iloc[-2]) - 1

    # Reward = signal alignment with price movement
    # If signal and return have same sign, positive reward
    reward = combined_signal * price_return * 10  # Scale factor

    # Clip to [-1, 1]
    return float(np.clip(reward, -1, 1))
