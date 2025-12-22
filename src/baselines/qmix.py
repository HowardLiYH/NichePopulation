"""
QMIX: Monotonic Value Function Factorisation for Multi-Agent RL.

Reference:
    Rashid et al. (2018) "QMIX: Monotonic Value Function Factorisation
    for Deep Multi-Agent Reinforcement Learning"

Key Idea:
    Q_tot = f(Q_1, ..., Q_n, s) with monotonicity constraint:
    ∂Q_tot/∂Q_i ≥ 0 for all i

This ensures decentralized execution while allowing centralized training.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict


@dataclass
class QMIXConfig:
    """Configuration for QMIX."""
    n_agents: int = 8
    n_methods: int = 5
    n_regimes: int = 4
    hidden_dim: int = 64
    mixing_hidden_dim: int = 32
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon: float = 0.3
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.05
    target_update_freq: int = 100
    batch_size: int = 32


class QNetwork:
    """Individual Q-network for each agent."""

    def __init__(self, n_regimes: int, n_methods: int, hidden_dim: int, seed: int = 0):
        self.n_regimes = n_regimes
        self.n_methods = n_methods
        self.hidden_dim = hidden_dim
        self.rng = np.random.default_rng(seed)

        # Simple linear Q-table with feature weights
        # In practice, this would be a neural network
        self.weights = self.rng.normal(0, 0.1, (n_regimes, n_methods))
        self.bias = np.zeros(n_methods)

    def forward(self, regime_idx: int) -> np.ndarray:
        """Compute Q-values for all methods given regime."""
        return self.weights[regime_idx] + self.bias

    def get_q_value(self, regime_idx: int, method_idx: int) -> float:
        """Get Q-value for specific (regime, method) pair."""
        return self.weights[regime_idx, method_idx] + self.bias[method_idx]

    def update(self, regime_idx: int, method_idx: int, target: float, lr: float):
        """Update Q-value toward target."""
        current = self.get_q_value(regime_idx, method_idx)
        error = target - current
        self.weights[regime_idx, method_idx] += lr * error


class MixingNetwork:
    """
    Mixing network that combines individual Q-values monotonically.

    Uses hypernetwork to generate mixing weights that are constrained
    to be non-negative (via absolute value or softmax).
    """

    def __init__(self, n_agents: int, n_regimes: int, hidden_dim: int, seed: int = 0):
        self.n_agents = n_agents
        self.n_regimes = n_regimes
        self.hidden_dim = hidden_dim
        self.rng = np.random.default_rng(seed)

        # Hypernetwork weights (state -> mixing weights)
        # First layer: regime -> hidden
        self.hyper_w1 = self.rng.normal(0, 0.1, (n_regimes, hidden_dim, n_agents))
        self.hyper_b1 = np.zeros((n_regimes, hidden_dim))

        # Second layer: hidden -> 1
        self.hyper_w2 = self.rng.normal(0, 0.1, (n_regimes, 1, hidden_dim))
        self.hyper_b2 = np.zeros((n_regimes, 1))

    def forward(self, q_values: np.ndarray, regime_idx: int) -> float:
        """
        Combine individual Q-values into Q_tot.

        Args:
            q_values: Array of shape (n_agents,) with individual Q-values
            regime_idx: Current regime index (global state)

        Returns:
            Q_tot: Combined Q-value
        """
        # Get mixing weights for this state
        w1 = np.abs(self.hyper_w1[regime_idx])  # Ensure non-negative (monotonicity)
        b1 = self.hyper_b1[regime_idx]
        w2 = np.abs(self.hyper_w2[regime_idx])
        b2 = self.hyper_b2[regime_idx]

        # Forward pass through mixing network
        hidden = np.maximum(0, w1 @ q_values + b1)  # ReLU
        q_tot = (w2 @ hidden + b2)[0]

        return q_tot

    def update(self, q_values: np.ndarray, regime_idx: int, target: float, lr: float):
        """Update mixing network weights."""
        q_tot = self.forward(q_values, regime_idx)
        error = target - q_tot

        # Simplified gradient update (in practice, use backprop)
        w1 = np.abs(self.hyper_w1[regime_idx])
        hidden = np.maximum(0, w1 @ q_values + self.hyper_b1[regime_idx])

        # Update second layer
        self.hyper_w2[regime_idx] += lr * error * hidden.reshape(1, -1)
        self.hyper_b2[regime_idx] += lr * error


class QMIX:
    """
    QMIX: Monotonic value function factorisation.

    Training: Centralized (uses global state)
    Execution: Decentralized (each agent uses local Q-network)
    """

    def __init__(self, config: QMIXConfig, regimes: List[str],
                 methods: List[str], seed: Optional[int] = None):
        self.config = config
        self.regimes = regimes
        self.methods = methods
        self.regime_to_idx = {r: i for i, r in enumerate(regimes)}
        self.method_to_idx = {m: i for i, m in enumerate(methods)}

        self.rng = np.random.default_rng(seed)

        # Initialize individual Q-networks
        self.q_networks: Dict[str, QNetwork] = {}
        for i in range(config.n_agents):
            agent_id = f"agent_{i}"
            self.q_networks[agent_id] = QNetwork(
                len(regimes), len(methods), config.hidden_dim,
                seed=seed + i if seed else i
            )

        # Initialize mixing network
        self.mixer = MixingNetwork(
            config.n_agents, len(regimes), config.mixing_hidden_dim,
            seed=seed if seed else 0
        )

        # Epsilon for exploration
        self.epsilon = config.epsilon

        # Track for SI calculation
        self.niche_affinities: Dict[str, Dict[str, float]] = {
            f"agent_{i}": {r: 1.0/len(regimes) for r in regimes}
            for i in range(config.n_agents)
        }
        self.regime_wins: Dict[str, Dict[str, int]] = {
            f"agent_{i}": defaultdict(int)
            for i in range(config.n_agents)
        }
        self.total_wins: Dict[str, int] = defaultdict(int)

    def select_actions(self, regime: str) -> Dict[str, str]:
        """Each agent selects action using epsilon-greedy on local Q."""
        regime_idx = self.regime_to_idx[regime]
        actions = {}

        for agent_id, q_net in self.q_networks.items():
            if self.rng.random() < self.epsilon:
                # Explore
                action = self.rng.choice(self.methods)
            else:
                # Exploit
                q_values = q_net.forward(regime_idx)
                action_idx = np.argmax(q_values)
                action = self.methods[action_idx]

            actions[agent_id] = action

        return actions

    def update(self, regime: str, actions: Dict[str, str],
               rewards: Dict[str, float], winner_id: str):
        """
        Update Q-networks and mixer based on team reward.

        QMIX uses a shared team reward for coordination.
        """
        regime_idx = self.regime_to_idx[regime]

        # Compute team reward (average or sum)
        team_reward = sum(rewards.values())

        # Get current Q-values for chosen actions
        current_q = np.array([
            self.q_networks[f"agent_{i}"].get_q_value(
                regime_idx,
                self.method_to_idx[actions[f"agent_{i}"]]
            )
            for i in range(self.config.n_agents)
        ])

        # Compute Q_tot
        q_tot = self.mixer.forward(current_q, regime_idx)

        # TD target (simplified: no next state in this episodic setting)
        target = team_reward

        # Update mixing network
        self.mixer.update(current_q, regime_idx, target, self.config.learning_rate)

        # Update individual Q-networks
        td_error = target - q_tot
        for i, (agent_id, q_net) in enumerate(self.q_networks.items()):
            action = actions[agent_id]
            method_idx = self.method_to_idx[action]

            # Individual target (distribute TD error)
            individual_target = current_q[i] + td_error / self.config.n_agents
            q_net.update(regime_idx, method_idx, individual_target, self.config.learning_rate)

        # Update niche affinities based on wins
        self.regime_wins[winner_id][regime] += 1
        self.total_wins[winner_id] += 1

        # Decay epsilon
        self.epsilon = max(self.config.min_epsilon,
                          self.epsilon * self.config.epsilon_decay)

        # Update niche affinities
        self._update_niche_affinities()

    def _update_niche_affinities(self):
        """Update niche affinities based on regime-specific win rates."""
        for agent_id in self.q_networks.keys():
            total = self.total_wins[agent_id]
            if total > 0:
                for regime in self.regimes:
                    wins = self.regime_wins[agent_id].get(regime, 0)
                    # Exponential moving average
                    old_affinity = self.niche_affinities[agent_id][regime]
                    new_affinity = wins / total
                    self.niche_affinities[agent_id][regime] = (
                        0.9 * old_affinity + 0.1 * new_affinity
                    )

    def get_specialization_index(self, agent_id: str) -> float:
        """Compute SI for an agent based on niche affinity."""
        affinities = np.array(list(self.niche_affinities[agent_id].values()))
        affinities = affinities / (affinities.sum() + 1e-10)
        affinities = affinities[affinities > 0]

        if len(affinities) <= 1:
            return 1.0

        entropy = -np.sum(affinities * np.log(affinities + 1e-10))
        max_entropy = np.log(len(self.regimes))

        return 1 - (entropy / max_entropy) if max_entropy > 0 else 0

    def get_mean_si(self) -> float:
        """Get mean SI across all agents."""
        sis = [self.get_specialization_index(f"agent_{i}")
               for i in range(self.config.n_agents)]
        return float(np.mean(sis))


def run_qmix_experiment(regimes: List[str], methods: List[str],
                        regime_probs: Dict[str, float],
                        n_iterations: int = 500,
                        seed: int = 42) -> Dict:
    """Run a single QMIX experiment."""

    config = QMIXConfig(n_agents=8, n_methods=len(methods), n_regimes=len(regimes))
    qmix = QMIX(config, regimes, methods, seed=seed)

    rng = np.random.default_rng(seed)

    for _ in range(n_iterations):
        # Sample regime
        regime = rng.choice(list(regime_probs.keys()), p=list(regime_probs.values()))

        # Agents select actions
        actions = qmix.select_actions(regime)

        # Simulate rewards (simplified)
        rewards = {agent_id: rng.normal(0.5, 0.2) for agent_id in actions}

        # Determine winner
        winner_id = max(rewards, key=rewards.get)
        rewards[winner_id] += 0.5  # Winner bonus

        # Update
        qmix.update(regime, actions, rewards, winner_id)

    return {
        'mean_si': qmix.get_mean_si(),
        'agent_sis': [qmix.get_specialization_index(f"agent_{i}")
                      for i in range(config.n_agents)],
        'epsilon': qmix.epsilon,
    }
