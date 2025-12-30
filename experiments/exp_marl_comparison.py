"""
Real MARL Comparison Experiment for Rare Regime Resilience

This implements actual MARL algorithms (IQL, VDN, QMIX, MAPPO) and compares them
with NichePopulation on rare regime performance.

Key difference from previous experiment:
- MARL methods actually LEARN through their respective algorithms
- All methods train on the same task with same reward structure
- Evaluation uses raw task performance (no niche bonus)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

# Domain configurations (same as before)
DOMAIN_CONFIGS = {
    'crypto': {
        'regimes': ['bull', 'bear', 'sideways', 'volatile'],
        'methods': ['naive', 'momentum_short', 'momentum_long', 'mean_revert', 'trend'],
        'regime_probs': {'bull': 0.30, 'bear': 0.20, 'sideways': 0.35, 'volatile': 0.15},
        'affinity_matrix': {
            'bull': {'naive': 0.50, 'momentum_short': 0.80, 'momentum_long': 0.90, 'mean_revert': 0.30, 'trend': 0.85},
            'bear': {'naive': 0.30, 'momentum_short': 0.70, 'momentum_long': 0.80, 'mean_revert': 0.40, 'trend': 0.75},
            'sideways': {'naive': 0.60, 'momentum_short': 0.40, 'momentum_long': 0.30, 'mean_revert': 0.90, 'trend': 0.35},
            'volatile': {'naive': 0.40, 'momentum_short': 0.60, 'momentum_long': 0.50, 'mean_revert': 0.70, 'trend': 0.50},
        }
    },
    'weather': {
        'regimes': ['clear', 'cloudy', 'rainy', 'extreme'],
        'methods': ['naive', 'ma3', 'ma7', 'seasonal', 'trend'],
        'regime_probs': {'clear': 0.30, 'cloudy': 0.35, 'rainy': 0.25, 'extreme': 0.10},
        'affinity_matrix': {
            'clear': {'naive': 0.80, 'ma3': 0.60, 'ma7': 0.50, 'seasonal': 0.75, 'trend': 0.55},
            'cloudy': {'naive': 0.60, 'ma3': 0.70, 'ma7': 0.65, 'seasonal': 0.70, 'trend': 0.60},
            'rainy': {'naive': 0.50, 'ma3': 0.75, 'ma7': 0.80, 'seasonal': 0.65, 'trend': 0.70},
            'extreme': {'naive': 0.30, 'ma3': 0.50, 'ma7': 0.55, 'seasonal': 0.40, 'trend': 0.85},
        }
    },
    'commodities': {
        'regimes': ['rising', 'falling', 'stable', 'volatile'],
        'methods': ['naive', 'ma5', 'ma20', 'mean_revert', 'trend'],
        'regime_probs': {'rising': 0.25, 'falling': 0.25, 'stable': 0.35, 'volatile': 0.15},
        'affinity_matrix': {
            'rising': {'naive': 0.50, 'ma5': 0.70, 'ma20': 0.85, 'mean_revert': 0.30, 'trend': 0.90},
            'falling': {'naive': 0.40, 'ma5': 0.65, 'ma20': 0.80, 'mean_revert': 0.35, 'trend': 0.85},
            'stable': {'naive': 0.70, 'ma5': 0.50, 'ma20': 0.40, 'mean_revert': 0.85, 'trend': 0.30},
            'volatile': {'naive': 0.45, 'ma5': 0.60, 'ma20': 0.50, 'mean_revert': 0.70, 'trend': 0.55},
        }
    },
    'traffic': {
        'regimes': ['morning_rush', 'evening_rush', 'midday', 'night', 'weekend', 'transition'],
        'methods': ['persistence', 'hourly_avg', 'weekly_pattern', 'rush_hour', 'exp_smooth'],
        'regime_probs': {'morning_rush': 0.09, 'evening_rush': 0.09, 'midday': 0.21, 'night': 0.18, 'weekend': 0.29, 'transition': 0.14},
        'affinity_matrix': {
            'morning_rush': {'persistence': 0.40, 'hourly_avg': 0.70, 'weekly_pattern': 0.80, 'rush_hour': 0.95, 'exp_smooth': 0.60},
            'evening_rush': {'persistence': 0.45, 'hourly_avg': 0.70, 'weekly_pattern': 0.80, 'rush_hour': 0.90, 'exp_smooth': 0.65},
            'midday': {'persistence': 0.70, 'hourly_avg': 0.80, 'weekly_pattern': 0.70, 'rush_hour': 0.50, 'exp_smooth': 0.75},
            'night': {'persistence': 0.85, 'hourly_avg': 0.60, 'weekly_pattern': 0.65, 'rush_hour': 0.30, 'exp_smooth': 0.70},
            'weekend': {'persistence': 0.60, 'hourly_avg': 0.65, 'weekly_pattern': 0.90, 'rush_hour': 0.40, 'exp_smooth': 0.70},
            'transition': {'persistence': 0.50, 'hourly_avg': 0.70, 'weekly_pattern': 0.60, 'rush_hour': 0.60, 'exp_smooth': 0.80},
        }
    },
}


class RegimeSwitchingEnv:
    """
    Gym-style environment for regime-switching multi-agent task.

    State: Current regime (one-hot encoded)
    Action: Method selection (discrete)
    Reward: Method performance in current regime (from affinity matrix + noise)
    """

    def __init__(self, config: Dict, n_agents: int, seed: int = 42):
        self.config = config
        self.n_agents = n_agents
        self.rng = np.random.default_rng(seed)

        self.regimes = config['regimes']
        self.methods = config['methods']
        self.n_regimes = len(self.regimes)
        self.n_methods = len(self.methods)
        self.regime_probs = config['regime_probs']
        self.affinity_matrix = config['affinity_matrix']

        # Normalize regime probs
        total = sum(self.regime_probs.values())
        self.regime_probs_norm = {r: p/total for r, p in self.regime_probs.items()}

        self.current_regime = None
        self.regime_idx = None

    def reset(self) -> np.ndarray:
        """Reset environment, return initial state."""
        self._sample_regime()
        return self._get_state()

    def _sample_regime(self):
        """Sample a new regime."""
        self.current_regime = self.rng.choice(
            list(self.regime_probs_norm.keys()),
            p=list(self.regime_probs_norm.values())
        )
        self.regime_idx = self.regimes.index(self.current_regime)

    def _get_state(self) -> np.ndarray:
        """Return one-hot encoded regime state."""
        state = np.zeros(self.n_regimes)
        state[self.regime_idx] = 1.0
        return state

    def step(self, actions: List[int]) -> Tuple[np.ndarray, List[float], bool, Dict]:
        """
        Execute actions for all agents.

        Args:
            actions: List of method indices, one per agent

        Returns:
            next_state, rewards, done, info
        """
        rewards = []
        for action in actions:
            method = self.methods[action]
            base_reward = self.affinity_matrix[self.current_regime][method]
            noise = self.rng.normal(0, 0.1)
            reward = base_reward + noise
            rewards.append(reward)

        # Sample next regime
        self._sample_regime()
        next_state = self._get_state()

        return next_state, rewards, False, {'regime': self.current_regime}

    def force_regime(self, regime: str):
        """Force environment into a specific regime (for evaluation)."""
        self.current_regime = regime
        self.regime_idx = self.regimes.index(regime)
        return self._get_state()


class IQLAgent:
    """
    Independent Q-Learning Agent.

    Each agent learns independently using tabular Q-learning.
    No coordination between agents.
    """

    def __init__(self, n_states: int, n_actions: int,
                 learning_rate: float = 0.1,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.995,
                 seed: int = 42):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

        # Q-table: state -> action values
        self.q_table = np.zeros((n_states, n_actions))

    def get_state_idx(self, state: np.ndarray) -> int:
        """Convert one-hot state to index."""
        return int(np.argmax(state))

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection."""
        state_idx = self.get_state_idx(state)

        if training and self.rng.random() < self.epsilon:
            return self.rng.integers(self.n_actions)
        else:
            return int(np.argmax(self.q_table[state_idx]))

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        """Q-learning update."""
        state_idx = self.get_state_idx(state)
        next_state_idx = self.get_state_idx(next_state)

        # Q-learning update
        best_next = np.max(self.q_table[next_state_idx])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state_idx, action]
        self.q_table[state_idx, action] += self.lr * td_error

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


class VDNAgents:
    """
    Value Decomposition Network (simplified tabular version).

    Q_total = sum(Q_i) - agents learn to maximize joint reward.
    Uses shared reward signal to encourage coordination.
    """

    def __init__(self, n_agents: int, n_states: int, n_actions: int,
                 learning_rate: float = 0.1,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.995,
                 seed: int = 42):
        self.n_agents = n_agents
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

        # Each agent has own Q-table
        self.q_tables = [np.zeros((n_states, n_actions)) for _ in range(n_agents)]

    def get_state_idx(self, state: np.ndarray) -> int:
        return int(np.argmax(state))

    def select_actions(self, state: np.ndarray, training: bool = True) -> List[int]:
        """Select actions for all agents."""
        state_idx = self.get_state_idx(state)
        actions = []

        for i in range(self.n_agents):
            if training and self.rng.random() < self.epsilon:
                actions.append(self.rng.integers(self.n_actions))
            else:
                actions.append(int(np.argmax(self.q_tables[i][state_idx])))

        return actions

    def update(self, state: np.ndarray, actions: List[int],
               rewards: List[float], next_state: np.ndarray):
        """VDN update - use mean reward as shared signal."""
        state_idx = self.get_state_idx(state)
        next_state_idx = self.get_state_idx(next_state)

        # VDN: shared reward is sum of individual rewards
        shared_reward = np.mean(rewards)  # Use mean for stability

        for i in range(self.n_agents):
            best_next = np.max(self.q_tables[i][next_state_idx])
            td_target = shared_reward + self.gamma * best_next
            td_error = td_target - self.q_tables[i][state_idx, actions[i]]
            self.q_tables[i][state_idx, actions[i]] += self.lr * td_error

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


class QMIXAgents:
    """
    QMIX-style agents (simplified tabular version).

    Similar to VDN but with non-linear mixing (approximated).
    Encourages more coordination toward joint optimal.
    """

    def __init__(self, n_agents: int, n_states: int, n_actions: int,
                 learning_rate: float = 0.1,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.995,
                 mixing_weight: float = 0.8,  # Weight toward best agent
                 seed: int = 42):
        self.n_agents = n_agents
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.mixing_weight = mixing_weight
        self.rng = np.random.default_rng(seed)

        self.q_tables = [np.zeros((n_states, n_actions)) for _ in range(n_agents)]

    def get_state_idx(self, state: np.ndarray) -> int:
        return int(np.argmax(state))

    def select_actions(self, state: np.ndarray, training: bool = True) -> List[int]:
        state_idx = self.get_state_idx(state)
        actions = []

        for i in range(self.n_agents):
            if training and self.rng.random() < self.epsilon:
                actions.append(self.rng.integers(self.n_actions))
            else:
                actions.append(int(np.argmax(self.q_tables[i][state_idx])))

        return actions

    def update(self, state: np.ndarray, actions: List[int],
               rewards: List[float], next_state: np.ndarray):
        """QMIX-style update - weight toward best performer."""
        state_idx = self.get_state_idx(state)
        next_state_idx = self.get_state_idx(next_state)

        # QMIX: non-linear mixing - emphasize best performer
        max_reward = max(rewards)
        mean_reward = np.mean(rewards)
        mixed_reward = self.mixing_weight * max_reward + (1 - self.mixing_weight) * mean_reward

        for i in range(self.n_agents):
            best_next = np.max(self.q_tables[i][next_state_idx])
            td_target = mixed_reward + self.gamma * best_next
            td_error = td_target - self.q_tables[i][state_idx, actions[i]]
            self.q_tables[i][state_idx, actions[i]] += self.lr * td_error

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


class MAPPOAgents:
    """
    Multi-Agent PPO (simplified tabular version).

    Uses policy gradient with shared advantage estimation.
    Strong coordination through centralized critic.
    """

    def __init__(self, n_agents: int, n_states: int, n_actions: int,
                 learning_rate: float = 0.05,
                 gamma: float = 0.99,
                 seed: int = 42):
        self.n_agents = n_agents
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)

        # Policy tables (log probabilities)
        self.policy_tables = [np.ones((n_states, n_actions)) / n_actions
                              for _ in range(n_agents)]

        # Shared value function (centralized critic)
        self.value_table = np.zeros(n_states)

        # For softmax temperature
        self.temperature = 1.0

    def get_state_idx(self, state: np.ndarray) -> int:
        return int(np.argmax(state))

    def select_actions(self, state: np.ndarray, training: bool = True) -> List[int]:
        state_idx = self.get_state_idx(state)
        actions = []

        for i in range(self.n_agents):
            probs = self.policy_tables[i][state_idx]
            probs = probs / probs.sum()  # Ensure normalized
            actions.append(self.rng.choice(self.n_actions, p=probs))

        return actions

    def update(self, state: np.ndarray, actions: List[int],
               rewards: List[float], next_state: np.ndarray):
        """MAPPO-style update with shared advantage."""
        state_idx = self.get_state_idx(state)
        next_state_idx = self.get_state_idx(next_state)

        # Shared reward (centralized)
        shared_reward = np.mean(rewards)

        # Compute advantage using centralized critic
        td_target = shared_reward + self.gamma * self.value_table[next_state_idx]
        advantage = td_target - self.value_table[state_idx]

        # Update value function
        self.value_table[state_idx] += self.lr * advantage

        # Update policies (simplified policy gradient)
        for i in range(self.n_agents):
            # Increase probability of action if advantage > 0
            if advantage > 0:
                self.policy_tables[i][state_idx, actions[i]] *= (1 + self.lr * advantage)
            else:
                self.policy_tables[i][state_idx, actions[i]] *= (1 + self.lr * advantage * 0.5)

            # Renormalize
            self.policy_tables[i][state_idx] = np.maximum(0.01, self.policy_tables[i][state_idx])
            self.policy_tables[i][state_idx] /= self.policy_tables[i][state_idx].sum()


class NichePopulationAgents:
    """
    NichePopulation with Thompson Sampling and Winner-Take-All.
    """

    def __init__(self, n_agents: int, n_states: int, n_actions: int,
                 lambda_val: float = 0.3,
                 affinity_lr: float = 0.1,
                 seed: int = 42):
        self.n_agents = n_agents
        self.n_states = n_states
        self.n_actions = n_actions
        self.lambda_val = lambda_val
        self.affinity_lr = affinity_lr
        self.rng = np.random.default_rng(seed)

        # Thompson Sampling beliefs: Beta(alpha, beta) per (agent, state, action)
        self.beliefs_alpha = np.ones((n_agents, n_states, n_actions))
        self.beliefs_beta = np.ones((n_agents, n_states, n_actions))

        # Niche affinity: per (agent, state)
        self.affinities = np.ones((n_agents, n_states)) / n_states

    def get_state_idx(self, state: np.ndarray) -> int:
        return int(np.argmax(state))

    def select_actions(self, state: np.ndarray, training: bool = True) -> List[int]:
        state_idx = self.get_state_idx(state)
        actions = []

        for i in range(self.n_agents):
            if training:
                # Thompson Sampling
                samples = self.rng.beta(
                    self.beliefs_alpha[i, state_idx],
                    self.beliefs_beta[i, state_idx]
                )
                actions.append(int(np.argmax(samples)))
            else:
                # Use mean of beliefs
                means = self.beliefs_alpha[i, state_idx] / (
                    self.beliefs_alpha[i, state_idx] + self.beliefs_beta[i, state_idx]
                )
                actions.append(int(np.argmax(means)))

        return actions

    def update(self, state: np.ndarray, actions: List[int],
               rewards: List[float], next_state: np.ndarray):
        """Winner-take-all update."""
        state_idx = self.get_state_idx(state)

        # Compute scores with niche bonus
        scores = []
        for i in range(self.n_agents):
            niche_bonus = self.lambda_val * (self.affinities[i, state_idx] - 1.0 / self.n_states)
            scores.append(rewards[i] + niche_bonus)

        # Winner take all
        winner = int(np.argmax(scores))

        # Update winner's beliefs
        if rewards[winner] > 0.5:  # Success threshold
            self.beliefs_alpha[winner, state_idx, actions[winner]] += 1
        else:
            self.beliefs_beta[winner, state_idx, actions[winner]] += 1

        # Update winner's affinity
        for s in range(self.n_states):
            if s == state_idx:
                self.affinities[winner, s] += self.affinity_lr * (1 - self.affinities[winner, s])
            else:
                self.affinities[winner, s] = max(0.01,
                    self.affinities[winner, s] - self.affinity_lr / (self.n_states - 1))

        # Normalize affinity
        self.affinities[winner] /= self.affinities[winner].sum()


def compute_si(affinities: np.ndarray) -> float:
    """Compute Specialization Index from affinity distribution."""
    affinities = affinities / (affinities.sum() + 1e-10)
    entropy = -np.sum(affinities * np.log(affinities + 1e-10))
    max_entropy = np.log(len(affinities))
    return 1 - entropy / max_entropy


def train_and_evaluate(domain: str, n_agents: int = 8,
                       n_train_episodes: int = 5000,
                       n_eval_episodes: int = 100,
                       seed: int = 42) -> Dict:
    """Train all methods and evaluate on rare regimes."""

    config = DOMAIN_CONFIGS[domain]
    n_states = len(config['regimes'])
    n_actions = len(config['methods'])

    # Identify rare regimes (< 15% probability)
    rare_regimes = [r for r, p in config['regime_probs'].items() if p < 0.15]
    if not rare_regimes:
        min_prob = min(config['regime_probs'].values())
        rare_regimes = [r for r, p in config['regime_probs'].items() if p == min_prob]

    results = {}

    # ============ Train IQL ============
    print(f"  Training IQL...", end=" ", flush=True)
    env = RegimeSwitchingEnv(config, n_agents, seed)
    iql_agents = [IQLAgent(n_states, n_actions, seed=seed+i) for i in range(n_agents)]

    state = env.reset()
    for ep in range(n_train_episodes):
        actions = [agent.select_action(state) for agent in iql_agents]
        next_state, rewards, _, _ = env.step(actions)
        for i, agent in enumerate(iql_agents):
            agent.update(state, actions[i], rewards[i], next_state)
        state = next_state
    print("Done")

    # ============ Train VDN ============
    print(f"  Training VDN...", end=" ", flush=True)
    env = RegimeSwitchingEnv(config, n_agents, seed)
    vdn = VDNAgents(n_agents, n_states, n_actions, seed=seed)

    state = env.reset()
    for ep in range(n_train_episodes):
        actions = vdn.select_actions(state)
        next_state, rewards, _, _ = env.step(actions)
        vdn.update(state, actions, rewards, next_state)
        state = next_state
    print("Done")

    # ============ Train QMIX ============
    print(f"  Training QMIX...", end=" ", flush=True)
    env = RegimeSwitchingEnv(config, n_agents, seed)
    qmix = QMIXAgents(n_agents, n_states, n_actions, seed=seed)

    state = env.reset()
    for ep in range(n_train_episodes):
        actions = qmix.select_actions(state)
        next_state, rewards, _, _ = env.step(actions)
        qmix.update(state, actions, rewards, next_state)
        state = next_state
    print("Done")

    # ============ Train MAPPO ============
    print(f"  Training MAPPO...", end=" ", flush=True)
    env = RegimeSwitchingEnv(config, n_agents, seed)
    mappo = MAPPOAgents(n_agents, n_states, n_actions, seed=seed)

    state = env.reset()
    for ep in range(n_train_episodes):
        actions = mappo.select_actions(state)
        next_state, rewards, _, _ = env.step(actions)
        mappo.update(state, actions, rewards, next_state)
        state = next_state
    print("Done")

    # ============ Train NichePopulation ============
    print(f"  Training NichePopulation...", end=" ", flush=True)
    env = RegimeSwitchingEnv(config, n_agents, seed)
    niche = NichePopulationAgents(n_agents, n_states, n_actions, seed=seed)

    state = env.reset()
    for ep in range(n_train_episodes):
        actions = niche.select_actions(state)
        next_state, rewards, _, _ = env.step(actions)
        niche.update(state, actions, rewards, next_state)
        state = next_state
    print("Done")

    # ============ Evaluate on Rare Regimes ============
    print(f"  Evaluating on rare regimes: {rare_regimes}")

    def evaluate_on_regime(env, agents, regime, agent_type, n_episodes):
        """Evaluate agents on a specific regime."""
        rewards_collected = []

        for _ in range(n_episodes):
            state = env.force_regime(regime)

            if agent_type == 'iql':
                actions = [agent.select_action(state, training=False) for agent in agents]
            elif agent_type in ['vdn', 'qmix', 'mappo', 'niche']:
                actions = agents.select_actions(state, training=False)

            _, rewards, _, _ = env.step(actions)
            rewards_collected.append(max(rewards))  # Best agent's reward

        return np.mean(rewards_collected), np.std(rewards_collected)

    for rare_regime in rare_regimes:
        regime_idx = config['regimes'].index(rare_regime)

        # Evaluate each method
        env = RegimeSwitchingEnv(config, n_agents, seed + 1000)

        iql_mean, iql_std = evaluate_on_regime(env, iql_agents, rare_regime, 'iql', n_eval_episodes)
        vdn_mean, vdn_std = evaluate_on_regime(env, vdn, rare_regime, 'vdn', n_eval_episodes)
        qmix_mean, qmix_std = evaluate_on_regime(env, qmix, rare_regime, 'qmix', n_eval_episodes)
        mappo_mean, mappo_std = evaluate_on_regime(env, mappo, rare_regime, 'mappo', n_eval_episodes)
        niche_mean, niche_std = evaluate_on_regime(env, niche, rare_regime, 'niche', n_eval_episodes)

        # Compute SI for each method
        # IQL: compute from Q-table preferences
        iql_si = np.mean([compute_si(agent.q_table.mean(axis=1)) for agent in iql_agents])

        # VDN/QMIX/MAPPO: compute from Q-tables
        vdn_si = np.mean([compute_si(q.mean(axis=1)) for q in vdn.q_tables])
        qmix_si = np.mean([compute_si(q.mean(axis=1)) for q in qmix.q_tables])
        mappo_si = np.mean([compute_si(p.mean(axis=1)) for p in mappo.policy_tables])

        # NichePopulation: compute from affinities
        niche_si = np.mean([compute_si(niche.affinities[i]) for i in range(n_agents)])

        results[rare_regime] = {
            'iql': {'reward': iql_mean, 'std': iql_std, 'si': iql_si},
            'vdn': {'reward': vdn_mean, 'std': vdn_std, 'si': vdn_si},
            'qmix': {'reward': qmix_mean, 'std': qmix_std, 'si': qmix_si},
            'mappo': {'reward': mappo_mean, 'std': mappo_std, 'si': mappo_si},
            'niche': {'reward': niche_mean, 'std': niche_std, 'si': niche_si},
        }

    return results


def main():
    """Run the real MARL comparison experiment."""
    print("=" * 100)
    print("REAL MARL COMPARISON EXPERIMENT")
    print("Training actual IQL, VDN, QMIX, MAPPO vs NichePopulation")
    print("=" * 100)
    print()

    all_results = {}

    for domain in DOMAIN_CONFIGS.keys():
        print(f"\n{'='*50}")
        print(f"Domain: {domain.upper()}")
        print(f"{'='*50}")

        # Run multiple trials
        n_trials = 10
        trial_results = []

        for trial in range(n_trials):
            print(f"\nTrial {trial+1}/{n_trials}:")
            results = train_and_evaluate(domain, seed=42 + trial * 100)
            trial_results.append(results)

        # Aggregate results
        all_results[domain] = {}
        rare_regimes = list(trial_results[0].keys())

        for regime in rare_regimes:
            all_results[domain][regime] = {}
            for method in ['iql', 'vdn', 'qmix', 'mappo', 'niche']:
                rewards = [t[regime][method]['reward'] for t in trial_results]
                sis = [t[regime][method]['si'] for t in trial_results]
                all_results[domain][regime][method] = {
                    'reward_mean': np.mean(rewards),
                    'reward_std': np.std(rewards),
                    'si_mean': np.mean(sis),
                }

    # Print summary
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)

    print(f"\n{'Domain':<12} {'Rare Regime':<15} {'Niche':>10} {'IQL':>10} {'VDN':>10} {'QMIX':>10} {'MAPPO':>10}")
    print("-" * 80)

    for domain, regimes in all_results.items():
        for regime, methods in regimes.items():
            niche = methods['niche']['reward_mean']
            iql = methods['iql']['reward_mean']
            vdn = methods['vdn']['reward_mean']
            qmix = methods['qmix']['reward_mean']
            mappo = methods['mappo']['reward_mean']

            print(f"{domain:<12} {regime:<15} {niche:>10.3f} {iql:>10.3f} {vdn:>10.3f} {qmix:>10.3f} {mappo:>10.3f}")

    print("\n" + "=" * 100)
    print("IMPROVEMENT vs BASELINES")
    print("=" * 100)

    print(f"\n{'Domain':<12} {'Rare Regime':<15} {'vs IQL':>10} {'vs VDN':>10} {'vs QMIX':>10} {'vs MAPPO':>10}")
    print("-" * 80)

    improvements = {'iql': [], 'vdn': [], 'qmix': [], 'mappo': []}

    for domain, regimes in all_results.items():
        for regime, methods in regimes.items():
            niche = methods['niche']['reward_mean']

            for baseline in ['iql', 'vdn', 'qmix', 'mappo']:
                baseline_val = methods[baseline]['reward_mean']
                if baseline_val > 0:
                    imp = (niche - baseline_val) / baseline_val * 100
                    improvements[baseline].append(imp)

            vs_iql = (niche - methods['iql']['reward_mean']) / methods['iql']['reward_mean'] * 100
            vs_vdn = (niche - methods['vdn']['reward_mean']) / methods['vdn']['reward_mean'] * 100
            vs_qmix = (niche - methods['qmix']['reward_mean']) / methods['qmix']['reward_mean'] * 100
            vs_mappo = (niche - methods['mappo']['reward_mean']) / methods['mappo']['reward_mean'] * 100

            print(f"{domain:<12} {regime:<15} {vs_iql:>+9.1f}% {vs_vdn:>+9.1f}% {vs_qmix:>+9.1f}% {vs_mappo:>+9.1f}%")

    print("-" * 80)
    print(f"{'AVERAGE':<12} {'':<15} {np.mean(improvements['iql']):>+9.1f}% {np.mean(improvements['vdn']):>+9.1f}% {np.mean(improvements['qmix']):>+9.1f}% {np.mean(improvements['mappo']):>+9.1f}%")

    # Print SI comparison
    print("\n" + "=" * 100)
    print("SPECIALIZATION INDEX (SI) COMPARISON")
    print("=" * 100)

    print(f"\n{'Domain':<12} {'Niche SI':>10} {'IQL SI':>10} {'VDN SI':>10} {'QMIX SI':>10} {'MAPPO SI':>10}")
    print("-" * 80)

    for domain, regimes in all_results.items():
        # Get SI from any regime (should be same)
        regime = list(regimes.keys())[0]
        methods = regimes[regime]
        print(f"{domain:<12} {methods['niche']['si_mean']:>10.3f} {methods['iql']['si_mean']:>10.3f} {methods['vdn']['si_mean']:>10.3f} {methods['qmix']['si_mean']:>10.3f} {methods['mappo']['si_mean']:>10.3f}")

    # Save results
    output_dir = Path("results/real_marl_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    print(f"\nResults saved to {output_dir / 'results.json'}")

    return all_results


if __name__ == "__main__":
    main()
