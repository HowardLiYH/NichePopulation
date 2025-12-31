"""
Niche-Based Population with Competitive Exclusion.

Key insight: For specialization to emerge, agents must face
COMPETITIVE PRESSURE to occupy different niches.

Mechanism:
1. Each agent develops a "niche preference" based on past success
2. When regimes match their niche, agents have an ADVANTAGE
3. When regimes don't match, agents have a DISADVANTAGE
4. This creates evolutionary pressure to specialize

This is inspired by ecological niche theory:
- MacArthur, R. & Levins, R. (1967). "The Limiting Similarity"

UPDATED: Now uses proper Beta distribution for Thompson Sampling
and winner-only updates as described in the paper.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict

from .inventory_v2 import METHOD_INVENTORY_V2, get_method_names_v2
from .method_selector import MethodBelief  # Proper Beta distribution!


class NicheAgent:
    """
    Agent that develops a niche preference.

    Key mechanism:
    - Tracks success rate in each regime using Beta distributions
    - Develops "niche affinity" - which regime it prefers
    - Gets bonus/penalty based on regime match
    - Only updates beliefs when winning (winner-take-all)
    """

    def __init__(
        self,
        agent_id: str,
        regimes: List[str],
        seed: Optional[int] = None,
        methods: Optional[List[str]] = None,
        min_exploration: float = 0.05,
        learning_rate: float = 0.1,  # η in paper
    ):
        self.agent_id = agent_id
        self.regimes = regimes
        self.method_names = methods or list(METHOD_INVENTORY_V2.keys())
        self.rng = np.random.default_rng(seed)
        self.learning_rate = learning_rate  # η = 0.1 as in paper

        # Beliefs per regime per method - using proper Beta distribution!
        self.beliefs: Dict[str, Dict[str, MethodBelief]] = {
            r: {m: MethodBelief() for m in self.method_names}
            for r in regimes
        }

        # Niche affinity (learned preference for each regime)
        self.niche_affinity: Dict[str, float] = {r: 1.0 / len(regimes) for r in regimes}

        # Success tracking
        self.regime_successes: Dict[str, int] = defaultdict(int)
        self.regime_attempts: Dict[str, int] = defaultdict(int)

        # Total method usage for SI calculation
        self.method_usage: Dict[str, int] = defaultdict(int)

        # Exploration rate (decays over time)
        self.exploration_rate = 0.3
        self.exploration_decay = 0.999
        self.min_exploration = min_exploration

    def select_method(self, regime: str) -> str:
        """Select a method for the current regime using Thompson Sampling."""
        self.regime_attempts[regime] += 1

        # Decay exploration
        self.exploration_rate = max(
            self.min_exploration,
            self.exploration_rate * self.exploration_decay
        )

        if self.rng.random() < self.exploration_rate:
            # Random exploration
            method = self.rng.choice(self.method_names)
        else:
            # Thompson Sampling: sample from Beta distributions
            samples = {
                m: self.beliefs[regime][m].sample(self.rng)  # Proper Beta sampling!
                for m in self.method_names
            }
            method = max(samples, key=samples.get)

        self.method_usage[method] += 1
        return method

    def update(self, regime: str, method: str, won: bool) -> None:
        """
        Update beliefs and niche affinity.

        IMPORTANT: Only called for the WINNER as per paper.
        Losers do not update (winner-take-all dynamics).
        """
        if not won:
            # Losers don't update - this is winner-take-all!
            return

        # Update method belief with success (winner always gets reward=1)
        self.beliefs[regime][method].update(1.0)

        # Update niche affinity
        self.regime_successes[regime] += 1
        self._update_niche_affinity(regime)

    def _update_niche_affinity(self, regime: str) -> None:
        """
        Update niche affinity using the paper's formula.

        Paper formula: α_r ← α_r + η × (1 - α_r)  [for winning regime]
        Other regimes: α_r' ← α_r' - η/(R-1)
        Then normalize.
        """
        eta = self.learning_rate  # η = 0.1 as in paper
        n_regimes = len(self.regimes)

        # Paper formula: α += η × (1 - α) for winning regime
        self.niche_affinity[regime] += eta * (1 - self.niche_affinity[regime])

        # Decrease other regimes
        for r in self.regimes:
            if r != regime:
                self.niche_affinity[r] -= eta / (n_regimes - 1)
                self.niche_affinity[r] = max(0.01, self.niche_affinity[r])

        # Normalize to maintain probability simplex
        total = sum(self.niche_affinity.values())
        self.niche_affinity = {r: a / total for r, a in self.niche_affinity.items()}

    def get_niche_strength(self, regime: str) -> float:
        """
        How strong is this agent in the given regime?

        Based on both affinity and success rate.
        """
        affinity = self.niche_affinity.get(regime, 0.25)

        attempts = self.regime_attempts.get(regime, 0)
        if attempts == 0:
            return affinity

        success_rate = self.regime_successes.get(regime, 0) / attempts
        return 0.5 * affinity + 0.5 * success_rate

    def get_primary_niche(self) -> str:
        """Get the regime this agent is most specialized in."""
        return max(self.niche_affinity, key=self.niche_affinity.get)

    def get_method_distribution(self) -> Dict[str, float]:
        """Get normalized method usage distribution."""
        total = sum(self.method_usage.values())
        if total == 0:
            n = len(self.method_names)
            return {m: 1.0 / n for m in self.method_names}
        return {m: self.method_usage[m] / total for m in self.method_names}


class NichePopulation:
    """
    Population with niche-based competition.

    Key mechanism: When computing winner, apply niche bonus/penalty.
    Agents that specialize in the current regime get a BOOST.
    This creates evolutionary pressure to specialize.

    UPDATED: Winner-take-all dynamics - only winner updates beliefs/affinity.
    """

    def __init__(
        self,
        n_agents: int = 8,
        regimes: List[str] = None,
        niche_bonus: float = 0.3,  # λ - Reward boost for matching niche
        seed: Optional[int] = None,
        methods: Optional[List[str]] = None,
        min_exploration_rate: float = 0.05,
        learning_rate: float = 0.1,  # η in paper
    ):
        self.n_agents = n_agents
        self.regimes = regimes or ["trend_up", "trend_down", "mean_revert", "volatile"]
        self.niche_bonus = niche_bonus
        self.methods = methods
        self.min_exploration_rate = min_exploration_rate
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)

        self.agents: Dict[str, NicheAgent] = {}
        for i in range(n_agents):
            agent_id = f"agent_{i}"
            # Initialize agents with DIFFERENT random seeds for diversity
            self.agents[agent_id] = NicheAgent(
                agent_id=agent_id,
                regimes=self.regimes,
                seed=(seed + i * 100) if seed else None,
                methods=methods,
                min_exploration=min_exploration_rate,
                learning_rate=learning_rate,
            )

        self.iteration = 0
        self.history = []

    def run_iteration(
        self,
        prices,
        regime: str,
        reward_fn,
    ):
        """Run one iteration with niche-based competition."""
        self.iteration += 1

        # Each agent selects a method
        selections: Dict[str, str] = {}
        raw_rewards: Dict[str, float] = {}

        for agent_id, agent in self.agents.items():
            method = agent.select_method(regime)
            selections[agent_id] = method
            reward = reward_fn([method], prices)
            raw_rewards[agent_id] = reward

        # Apply niche bonus: agents whose primary niche matches get boost
        adjusted_rewards: Dict[str, float] = {}
        for agent_id, agent in self.agents.items():
            raw = raw_rewards[agent_id]

            # Compute niche match bonus
            agent_niche = agent.get_primary_niche()
            if agent_niche == regime:
                # Agent is in their preferred niche - boost!
                bonus = self.niche_bonus * agent.niche_affinity[regime]
            else:
                # Agent is outside their niche - slight penalty
                bonus = -self.niche_bonus * 0.3 * (1 - agent.niche_affinity[regime])

            adjusted_rewards[agent_id] = raw + bonus

        # Determine winner based on adjusted rewards
        winner_id = max(adjusted_rewards, key=adjusted_rewards.get)

        # WINNER-TAKE-ALL: Only winner updates!
        winner_agent = self.agents[winner_id]
        winner_agent.update(regime, selections[winner_id], won=True)

        # Losers do NOT update (this is the key difference from before)
        # for agent_id, agent in self.agents.items():
        #     if agent_id != winner_id:
        #         agent.update(regime, selections[agent_id], won=False)  # NO!

        self.history.append({
            "iteration": self.iteration,
            "regime": regime,
            "winner": winner_id,
            "selections": selections.copy(),
        })

        return {
            "winner_id": winner_id,
            "winner_method": selections[winner_id],
            "winner_reward": raw_rewards[winner_id],
            "regime": regime,
        }

    def get_niche_distribution(self) -> Dict[str, Dict[str, float]]:
        """Get niche affinity for all agents."""
        return {
            agent_id: agent.niche_affinity.copy()
            for agent_id, agent in self.agents.items()
        }

    def get_specialization_summary(self) -> Dict:
        """Get summary of specialization."""
        niches = {}
        for agent_id, agent in self.agents.items():
            primary = agent.get_primary_niche()
            affinity = agent.niche_affinity[primary]

            if primary not in niches:
                niches[primary] = []
            niches[primary].append({
                "agent": agent_id,
                "affinity": affinity,
            })

        return {
            "niche_occupancy": {r: len(niches.get(r, [])) for r in self.regimes},
            "specialists": niches,
        }

    def get_regime_win_matrix(self) -> Dict[str, Dict[str, float]]:
        """Get win rate by agent x regime."""
        # Count wins per agent per regime
        wins = defaultdict(lambda: defaultdict(int))
        regime_counts = defaultdict(int)

        for entry in self.history:
            regime = entry["regime"]
            winner = entry["winner"]
            regime_counts[regime] += 1
            wins[winner][regime] += 1

        # Convert to rates
        matrix = {}
        for agent_id in self.agents:
            matrix[agent_id] = {}
            for regime in self.regimes:
                if regime_counts[regime] > 0:
                    matrix[agent_id][regime] = wins[agent_id][regime] / regime_counts[regime]
                else:
                    matrix[agent_id][regime] = 0.0

        return matrix

    def get_all_method_usage(self) -> Dict[str, Dict[str, float]]:
        """Get method usage for SI calculation."""
        return {
            agent_id: agent.get_method_distribution()
            for agent_id, agent in self.agents.items()
        }
