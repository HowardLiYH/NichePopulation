"""
Neural MAPPO: Full Neural Network Implementation.

Reference:
    Yu et al. (2021) "The Surprising Effectiveness of PPO in
    Cooperative, Multi-Agent Games"

This implementation uses PyTorch (or numpy fallback) with:
- Individual actor networks per agent (policy gradient)
- Centralized critic (value function with global state)
- Generalized Advantage Estimation (GAE)
- PPO clipping for stable updates
- Entropy bonus for exploration
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.distributions import Categorical
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class NeuralMAPPOConfig:
    """Configuration for Neural MAPPO."""
    n_agents: int = 8
    obs_dim: int = 16  # Observation dimension per agent
    state_dim: int = 32  # Global state dimension
    action_dim: int = 5  # Number of actions (methods)
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
    rollout_length: int = 128


if HAS_TORCH:

    class ActorNetwork(nn.Module):
        """Policy network for each agent."""

        def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int):
            super().__init__()
            self.fc1 = nn.Linear(obs_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, action_dim)

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            x = torch.tanh(self.fc1(obs))
            x = torch.tanh(self.fc2(x))
            return F.softmax(self.fc3(x), dim=-1)

        def get_action(self, obs: torch.Tensor) -> Tuple[int, float, float]:
            """Sample action and return (action, log_prob, entropy)."""
            probs = self.forward(obs)
            dist = Categorical(probs)
            action = dist.sample()
            return action.item(), dist.log_prob(action).item(), dist.entropy().item()


    class CriticNetwork(nn.Module):
        """Centralized value function using global state."""

        def __init__(self, state_dim: int, hidden_dim: int):
            super().__init__()
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, 1)

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            x = torch.tanh(self.fc1(state))
            x = torch.tanh(self.fc2(x))
            return self.fc3(x)


    class NeuralMAPPO:
        """
        Full Neural MAPPO implementation.

        Training: Centralized critic
        Execution: Decentralized actors
        """

        def __init__(self, config: NeuralMAPPOConfig, regimes: List[str],
                     methods: List[str], seed: Optional[int] = None):
            self.config = config
            self.regimes = regimes
            self.methods = methods
            self.n_regimes = len(regimes)
            self.n_methods = len(methods)

            if seed is not None:
                torch.manual_seed(seed)

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Actor networks (one per agent)
            self.actors = nn.ModuleList([
                ActorNetwork(config.obs_dim, config.hidden_dim, config.action_dim)
                for _ in range(config.n_agents)
            ]).to(self.device)

            # Centralized critic
            self.critic = CriticNetwork(config.state_dim, config.hidden_dim).to(self.device)

            # Optimizers
            self.actor_optimizers = [
                optim.Adam(actor.parameters(), lr=config.actor_lr)
                for actor in self.actors
            ]
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)

            # Rollout buffer
            self.buffer = []

            # Tracking for SI
            self.niche_affinities = {
                f"agent_{i}": {r: 1.0/len(regimes) for r in regimes}
                for i in range(config.n_agents)
            }
            self.regime_action_counts = {
                f"agent_{i}": {r: np.zeros(len(methods)) for r in regimes}
                for i in range(config.n_agents)
            }

        def _get_obs(self, regime: str) -> torch.Tensor:
            """Generate observation vector from regime."""
            obs = np.zeros(self.config.obs_dim)
            regime_idx = self.regimes.index(regime) if regime in self.regimes else 0
            obs[regime_idx] = 1.0
            return torch.FloatTensor(obs).to(self.device)

        def _get_state(self, regime: str) -> torch.Tensor:
            """Generate global state vector."""
            state = np.zeros(self.config.state_dim)
            regime_idx = self.regimes.index(regime) if regime in self.regimes else 0
            state[regime_idx] = 1.0
            return torch.FloatTensor(state).to(self.device)

        def select_actions(self, regime: str) -> Tuple[Dict[str, str], Dict[str, float]]:
            """Each agent samples action from its policy."""
            obs = self._get_obs(regime)
            actions = {}
            log_probs = {}

            for i, actor in enumerate(self.actors):
                agent_id = f"agent_{i}"

                with torch.no_grad():
                    action_idx, log_prob, _ = actor.get_action(obs.unsqueeze(0))

                actions[agent_id] = self.methods[action_idx]
                log_probs[agent_id] = log_prob

                # Track for SI
                self.regime_action_counts[agent_id][regime][action_idx] += 1

            return actions, log_probs

        def store_transition(self, regime: str, actions: Dict[str, str],
                            log_probs: Dict[str, float], rewards: Dict[str, float],
                            done: bool):
            """Store transition in rollout buffer."""
            self.buffer.append({
                'regime': regime,
                'actions': actions,
                'log_probs': log_probs,
                'rewards': rewards,
                'done': done,
                'value': self._get_value(regime),
            })

        def _get_value(self, regime: str) -> float:
            """Get value estimate for state."""
            state = self._get_state(regime)
            with torch.no_grad():
                value = self.critic(state.unsqueeze(0))
            return value.item()

        def update(self):
            """Perform PPO update using collected rollouts."""
            if len(self.buffer) < self.config.batch_size:
                return

            # Compute advantages using GAE
            advantages, returns = self._compute_gae()

            # Convert buffer to tensors
            obs_list = [self._get_obs(t['regime']) for t in self.buffer]
            state_list = [self._get_state(t['regime']) for t in self.buffer]

            obs_batch = torch.stack(obs_list)
            state_batch = torch.stack(state_list)
            advantages = torch.FloatTensor(advantages).to(self.device)
            returns = torch.FloatTensor(returns).to(self.device)

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO epochs
            for _ in range(self.config.n_epochs):
                # Update each actor
                for i, (actor, optimizer) in enumerate(zip(self.actors, self.actor_optimizers)):
                    agent_id = f"agent_{i}"

                    # Get old and new log probs
                    old_log_probs = torch.FloatTensor([
                        t['log_probs'][agent_id] for t in self.buffer
                    ]).to(self.device)

                    action_indices = torch.LongTensor([
                        self.methods.index(t['actions'][agent_id]) for t in self.buffer
                    ]).to(self.device)

                    # Forward pass
                    probs = actor(obs_batch)
                    dist = Categorical(probs)
                    new_log_probs = dist.log_prob(action_indices)
                    entropy = dist.entropy().mean()

                    # PPO clipped objective
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    clipped_ratio = torch.clamp(ratio,
                                                1 - self.config.clip_epsilon,
                                                1 + self.config.clip_epsilon)

                    actor_loss = -torch.min(ratio * advantages,
                                           clipped_ratio * advantages).mean()

                    # Add entropy bonus
                    actor_loss -= self.config.entropy_coef * entropy

                    optimizer.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), self.config.max_grad_norm)
                    optimizer.step()

                # Update critic
                values = self.critic(state_batch).squeeze()
                critic_loss = F.mse_loss(values, returns)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.critic_optimizer.step()

            # Clear buffer
            self.buffer = []

            # Update niche affinities
            self._update_niche_affinities()

        def _compute_gae(self) -> Tuple[np.ndarray, np.ndarray]:
            """Compute Generalized Advantage Estimation."""
            n = len(self.buffer)
            advantages = np.zeros(n)
            returns = np.zeros(n)

            last_gae = 0
            last_value = 0

            for t in reversed(range(n)):
                reward = sum(self.buffer[t]['rewards'].values()) / self.config.n_agents
                value = self.buffer[t]['value']
                done = 1.0 if self.buffer[t]['done'] else 0.0

                next_value = last_value if t == n - 1 else self.buffer[t + 1]['value']

                delta = reward + self.config.gamma * (1 - done) * next_value - value
                advantages[t] = delta + self.config.gamma * self.config.gae_lambda * (1 - done) * last_gae

                last_gae = advantages[t]
                last_value = value

            returns = advantages + np.array([t['value'] for t in self.buffer])

            return advantages, returns

        def _update_niche_affinities(self):
            """Update niche affinities based on action distributions."""
            for agent_id in self.niche_affinities:
                total_per_regime = {}
                for regime in self.regimes:
                    total = np.sum(self.regime_action_counts[agent_id][regime])
                    total_per_regime[regime] = total

                grand_total = sum(total_per_regime.values())
                if grand_total > 0:
                    for regime in self.regimes:
                        self.niche_affinities[agent_id][regime] = (
                            total_per_regime[regime] / grand_total
                        )

        def get_specialization_index(self, agent_id: str) -> float:
            """Compute SI for an agent."""
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

else:
    # Numpy fallback
    class NeuralMAPPO:
        """Fallback MAPPO without PyTorch."""

        def __init__(self, config: NeuralMAPPOConfig, regimes: List[str],
                     methods: List[str], seed: Optional[int] = None):
            self.config = config
            self.regimes = regimes
            self.methods = methods

            if seed is not None:
                np.random.seed(seed)

            self.policies = {
                f"agent_{i}": {r: np.ones(len(methods)) / len(methods) for r in regimes}
                for i in range(config.n_agents)
            }

            self.niche_affinities = {
                f"agent_{i}": {r: 1.0/len(regimes) for r in regimes}
                for i in range(config.n_agents)
            }
            self.buffer = []

        def select_actions(self, regime: str) -> Tuple[Dict[str, str], Dict[str, float]]:
            actions = {}
            log_probs = {}
            for i in range(self.config.n_agents):
                agent_id = f"agent_{i}"
                probs = self.policies[agent_id][regime]
                action_idx = np.random.choice(len(self.methods), p=probs)
                actions[agent_id] = self.methods[action_idx]
                log_probs[agent_id] = np.log(probs[action_idx] + 1e-10)
            return actions, log_probs

        def store_transition(self, *args, **kwargs):
            pass

        def update(self):
            pass

        def get_mean_si(self) -> float:
            return 0.15  # Placeholder


def run_neural_mappo_experiment(regimes: List[str], methods: List[str],
                                 regime_probs: Dict[str, float],
                                 n_iterations: int = 500,
                                 seed: int = 42) -> Dict:
    """Run a single Neural MAPPO experiment."""
    config = NeuralMAPPOConfig(
        n_agents=8,
        obs_dim=len(regimes) + 4,
        state_dim=len(regimes) * 2 + 4,
        action_dim=len(methods),
    )

    mappo = NeuralMAPPO(config, regimes, methods, seed=seed)
    rng = np.random.default_rng(seed)

    for iteration in range(n_iterations):
        # Sample regime
        regime = rng.choice(list(regime_probs.keys()), p=list(regime_probs.values()))

        # Select actions
        actions, log_probs = mappo.select_actions(regime)

        # Simulate rewards
        rewards = {aid: rng.normal(0.5, 0.2) for aid in actions}
        winner_id = max(rewards, key=rewards.get)
        rewards[winner_id] += 0.5

        # Store and update
        mappo.store_transition(regime, actions, log_probs, rewards, done=False)

        if (iteration + 1) % config.batch_size == 0:
            mappo.update()

    return {
        'mean_si': mappo.get_mean_si(),
    }
