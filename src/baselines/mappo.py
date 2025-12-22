"""
MAPPO: Multi-Agent Proximal Policy Optimization.

Reference:
    Yu et al. (2021) "The Surprising Effectiveness of PPO in
    Cooperative, Multi-Agent Games"

Key Idea:
    - Centralized critic V(s) using global state
    - Decentralized actors π_i(a|o) using local observations
    - PPO clipping for stable policy updates
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict


@dataclass
class MAPPOConfig:
    """Configuration for MAPPO."""
    n_agents: int = 8
    n_methods: int = 5
    n_regimes: int = 4
    hidden_dim: int = 64
    actor_lr: float = 0.0003
    critic_lr: float = 0.001
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    batch_size: int = 32


class PolicyNetwork:
    """
    Actor network: π(a|o) for each agent.

    Uses softmax policy over discrete actions (methods).
    """

    def __init__(self, n_regimes: int, n_methods: int, hidden_dim: int, seed: int = 0):
        self.n_regimes = n_regimes
        self.n_methods = n_methods
        self.rng = np.random.default_rng(seed)

        # Policy parameters (logits)
        self.logits = self.rng.normal(0, 0.1, (n_regimes, n_methods))

    def get_probs(self, regime_idx: int) -> np.ndarray:
        """Get action probabilities for regime."""
        logits = self.logits[regime_idx]
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / exp_logits.sum()

    def sample_action(self, regime_idx: int, rng: np.random.Generator) -> Tuple[int, float]:
        """Sample action and return (action_idx, log_prob)."""
        probs = self.get_probs(regime_idx)
        action_idx = rng.choice(len(probs), p=probs)
        log_prob = np.log(probs[action_idx] + 1e-10)
        return action_idx, log_prob

    def get_log_prob(self, regime_idx: int, action_idx: int) -> float:
        """Get log probability of action."""
        probs = self.get_probs(regime_idx)
        return np.log(probs[action_idx] + 1e-10)

    def get_entropy(self, regime_idx: int) -> float:
        """Compute policy entropy for exploration bonus."""
        probs = self.get_probs(regime_idx)
        return -np.sum(probs * np.log(probs + 1e-10))

    def update(self, regime_idx: int, action_idx: int, advantage: float,
               old_log_prob: float, clip_epsilon: float, lr: float):
        """
        PPO clipped policy update.

        L^CLIP = min(r * A, clip(r, 1-ε, 1+ε) * A)
        where r = π(a|s) / π_old(a|s)
        """
        new_log_prob = self.get_log_prob(regime_idx, action_idx)
        ratio = np.exp(new_log_prob - old_log_prob)

        # Clipped objective
        clipped_ratio = np.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

        # Use minimum (pessimistic bound)
        if advantage >= 0:
            objective = min(ratio, clipped_ratio) * advantage
        else:
            objective = max(ratio, clipped_ratio) * advantage

        # Gradient of softmax policy (simplified)
        probs = self.get_probs(regime_idx)
        grad = np.zeros(self.n_methods)
        grad[action_idx] = (1 - probs[action_idx]) * objective
        for j in range(self.n_methods):
            if j != action_idx:
                grad[j] = -probs[j] * objective

        self.logits[regime_idx] += lr * grad


class ValueNetwork:
    """
    Centralized critic: V(s) using global state.

    In MAPPO, the value function has access to global information
    during training, enabling better credit assignment.
    """

    def __init__(self, n_regimes: int, hidden_dim: int, seed: int = 0):
        self.n_regimes = n_regimes
        self.rng = np.random.default_rng(seed)

        # Value estimates per state (regime)
        self.values = np.zeros(n_regimes)

    def forward(self, regime_idx: int) -> float:
        """Get value estimate for state."""
        return self.values[regime_idx]

    def update(self, regime_idx: int, target: float, lr: float):
        """Update value estimate toward target."""
        error = target - self.values[regime_idx]
        self.values[regime_idx] += lr * error


class MAPPO:
    """
    Multi-Agent PPO with centralized training, decentralized execution.

    Training: Uses global state for value function
    Execution: Each agent uses local policy network
    """

    def __init__(self, config: MAPPOConfig, regimes: List[str],
                 methods: List[str], seed: Optional[int] = None):
        self.config = config
        self.regimes = regimes
        self.methods = methods
        self.regime_to_idx = {r: i for i, r in enumerate(regimes)}
        self.method_to_idx = {m: i for i, m in enumerate(methods)}

        self.rng = np.random.default_rng(seed)

        # Initialize policy networks (one per agent)
        self.policies: Dict[str, PolicyNetwork] = {}
        for i in range(config.n_agents):
            agent_id = f"agent_{i}"
            self.policies[agent_id] = PolicyNetwork(
                len(regimes), len(methods), config.hidden_dim,
                seed=seed + i if seed else i
            )

        # Centralized value network (shared)
        self.value_net = ValueNetwork(
            len(regimes), config.hidden_dim,
            seed=seed if seed else 0
        )

        # Experience buffer
        self.buffer: List[Dict] = []

        # Track for SI calculation
        self.niche_affinities: Dict[str, Dict[str, float]] = {
            f"agent_{i}": {r: 1.0/len(regimes) for r in regimes}
            for i in range(config.n_agents)
        }
        self.action_counts: Dict[str, Dict[str, Dict[str, int]]] = {
            f"agent_{i}": {r: defaultdict(int) for r in regimes}
            for i in range(config.n_agents)
        }

    def select_actions(self, regime: str) -> Tuple[Dict[str, str], Dict[str, float]]:
        """
        Each agent samples action from its policy.

        Returns:
            actions: Dict of agent_id -> action
            log_probs: Dict of agent_id -> log_prob
        """
        regime_idx = self.regime_to_idx[regime]
        actions = {}
        log_probs = {}

        for agent_id, policy in self.policies.items():
            action_idx, log_prob = policy.sample_action(regime_idx, self.rng)
            actions[agent_id] = self.methods[action_idx]
            log_probs[agent_id] = log_prob

            # Track action counts for SI
            self.action_counts[agent_id][regime][self.methods[action_idx]] += 1

        return actions, log_probs

    def store_transition(self, regime: str, actions: Dict[str, str],
                        log_probs: Dict[str, float], rewards: Dict[str, float]):
        """Store transition in buffer for PPO update."""
        regime_idx = self.regime_to_idx[regime]

        self.buffer.append({
            'regime': regime,
            'regime_idx': regime_idx,
            'actions': actions,
            'log_probs': log_probs,
            'rewards': rewards,
            'value': self.value_net.forward(regime_idx),
        })

    def update(self):
        """Perform PPO update using collected experience."""
        if len(self.buffer) < self.config.batch_size:
            return

        # Compute advantages using GAE
        advantages = self._compute_advantages()

        # PPO epochs
        for _ in range(self.config.n_epochs):
            for i, transition in enumerate(self.buffer):
                regime_idx = transition['regime_idx']
                advantage = advantages[i]

                # Update each agent's policy
                for agent_id, policy in self.policies.items():
                    action = transition['actions'][agent_id]
                    action_idx = self.method_to_idx[action]
                    old_log_prob = transition['log_probs'][agent_id]

                    # PPO clipped update
                    policy.update(
                        regime_idx, action_idx, advantage,
                        old_log_prob, self.config.clip_epsilon,
                        self.config.actor_lr
                    )

                # Update centralized value network
                reward = sum(transition['rewards'].values()) / len(transition['rewards'])
                target = reward + self.config.gamma * self.value_net.forward(regime_idx)
                self.value_net.update(regime_idx, target, self.config.critic_lr)

        # Clear buffer
        self.buffer = []

        # Update niche affinities
        self._update_niche_affinities()

    def _compute_advantages(self) -> np.ndarray:
        """Compute advantages using Generalized Advantage Estimation (GAE)."""
        n = len(self.buffer)
        advantages = np.zeros(n)

        # Simple advantage: reward - value baseline
        for i, transition in enumerate(self.buffer):
            reward = sum(transition['rewards'].values()) / len(transition['rewards'])
            advantages[i] = reward - transition['value']

        # Normalize advantages
        if n > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def _update_niche_affinities(self):
        """Update niche affinities based on action distributions per regime."""
        for agent_id in self.policies.keys():
            total_per_regime = {}
            for regime in self.regimes:
                total = sum(self.action_counts[agent_id][regime].values())
                total_per_regime[regime] = total

            grand_total = sum(total_per_regime.values())
            if grand_total > 0:
                for regime in self.regimes:
                    self.niche_affinities[agent_id][regime] = (
                        total_per_regime[regime] / grand_total
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

    def get_mean_entropy(self) -> float:
        """Get mean policy entropy (exploration measure)."""
        entropies = []
        for agent_id, policy in self.policies.items():
            for regime_idx in range(len(self.regimes)):
                entropies.append(policy.get_entropy(regime_idx))
        return float(np.mean(entropies))


def run_mappo_experiment(regimes: List[str], methods: List[str],
                         regime_probs: Dict[str, float],
                         n_iterations: int = 500,
                         seed: int = 42) -> Dict:
    """Run a single MAPPO experiment."""

    config = MAPPOConfig(n_agents=8, n_methods=len(methods), n_regimes=len(regimes))
    mappo = MAPPO(config, regimes, methods, seed=seed)

    rng = np.random.default_rng(seed)

    for iteration in range(n_iterations):
        # Sample regime
        regime = rng.choice(list(regime_probs.keys()), p=list(regime_probs.values()))

        # Agents select actions
        actions, log_probs = mappo.select_actions(regime)

        # Simulate rewards
        rewards = {agent_id: rng.normal(0.5, 0.2) for agent_id in actions}

        # Determine winner and give bonus
        winner_id = max(rewards, key=rewards.get)
        rewards[winner_id] += 0.5

        # Store transition
        mappo.store_transition(regime, actions, log_probs, rewards)

        # Update periodically
        if (iteration + 1) % config.batch_size == 0:
            mappo.update()

    return {
        'mean_si': mappo.get_mean_si(),
        'agent_sis': [mappo.get_specialization_index(f"agent_{i}")
                      for i in range(config.n_agents)],
        'mean_entropy': mappo.get_mean_entropy(),
    }
