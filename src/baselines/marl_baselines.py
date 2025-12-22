"""
Multi-Agent Reinforcement Learning (MARL) Baselines.

Implements QMIX, MAPPO, and Quality-Diversity (QD) baselines
for comparison against our emergent specialization approach.

Note: These are simplified implementations focused on the
multi-agent regime-selection task. Full implementations would
require more sophisticated infrastructure.

References:
- QMIX: Rashid et al. (2018) "QMIX: Monotonic Value Function Factorisation"
- MAPPO: Yu et al. (2021) "The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games"
- MAP-Elites: Mouret & Clune (2015) "Illuminating search spaces by mapping elites"
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict

from ..agents.inventory_v2 import METHOD_INVENTORY_V2


@dataclass
class MARLConfig:
    """Configuration for MARL baselines."""
    n_agents: int = 8
    n_methods: int = 10
    n_regimes: int = 4
    learning_rate: float = 0.01
    gamma: float = 0.99
    epsilon: float = 0.1
    epsilon_decay: float = 0.999
    min_epsilon: float = 0.01


class IndependentQLearning:
    """
    Independent Q-Learning baseline.

    Each agent learns independently without coordination.
    This is a simple baseline that ignores multi-agent structure.
    """

    def __init__(self, config: MARLConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.methods = list(METHOD_INVENTORY_V2.keys())[:config.n_methods]
        self.regimes = ["trend_up", "trend_down", "mean_revert", "volatile"]

        # Q-table per agent: Q[agent][regime][method]
        self.q_tables: Dict[str, Dict[str, Dict[str, float]]] = {}
        for i in range(config.n_agents):
            agent_id = f"agent_{i}"
            self.q_tables[agent_id] = {
                r: {m: 0.0 for m in self.methods}
                for r in self.regimes
            }

        self.epsilon = config.epsilon
        self.iteration = 0

        # Track method usage for SI calculation
        self.method_usage: Dict[str, Dict[str, int]] = {
            f"agent_{i}": defaultdict(int)
            for i in range(config.n_agents)
        }

        # Track niche affinities
        self.niche_affinities: Dict[str, Dict[str, float]] = {
            f"agent_{i}": {r: 0.25 for r in self.regimes}
            for i in range(config.n_agents)
        }

    def select_actions(self, regime: str) -> Dict[str, str]:
        """Each agent selects a method epsilon-greedily."""
        actions = {}

        for agent_id, q_table in self.q_tables.items():
            if self.rng.random() < self.epsilon:
                action = self.rng.choice(self.methods)
            else:
                q_values = q_table[regime]
                action = max(q_values, key=q_values.get)

            actions[agent_id] = action
            self.method_usage[agent_id][action] += 1

        return actions

    def update(
        self,
        regime: str,
        actions: Dict[str, str],
        rewards: Dict[str, float],
    ):
        """Update Q-tables based on rewards."""
        for agent_id, action in actions.items():
            reward = rewards.get(agent_id, 0.0)

            # Simple Q-learning update (no next state in bandit setting)
            old_q = self.q_tables[agent_id][regime][action]
            self.q_tables[agent_id][regime][action] = old_q + self.config.learning_rate * (reward - old_q)

            # Update niche affinity
            if reward > 0:
                self.niche_affinities[agent_id][regime] += 0.1

        # Normalize niche affinities
        for agent_id in self.niche_affinities:
            total = sum(self.niche_affinities[agent_id].values())
            if total > 0:
                self.niche_affinities[agent_id] = {
                    r: v / total for r, v in self.niche_affinities[agent_id].items()
                }

        # Decay epsilon
        self.epsilon = max(self.config.min_epsilon, self.epsilon * self.config.epsilon_decay)
        self.iteration += 1

    def run_iteration(self, prices, regime: str, reward_fn):
        """Run one iteration."""
        actions = self.select_actions(regime)

        # Compute rewards
        rewards = {}
        for agent_id, method in actions.items():
            reward = reward_fn([method], prices)
            rewards[agent_id] = reward

        self.update(regime, actions, rewards)

        return {"actions": actions, "rewards": rewards}

    def get_niche_distribution(self) -> Dict[str, Dict[str, float]]:
        """Return niche affinities."""
        return self.niche_affinities


class QMIX:
    """
    QMIX-inspired baseline for multi-agent coordination.

    Key idea: Factored value function with monotonic mixing.
    Q_tot = f(Q_1, Q_2, ..., Q_n) where f is monotonic.

    Simplified implementation: We approximate the mixing by
    having agents share information through a common reward signal.
    """

    def __init__(self, config: MARLConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.methods = list(METHOD_INVENTORY_V2.keys())[:config.n_methods]
        self.regimes = ["trend_up", "trend_down", "mean_revert", "volatile"]

        # Individual Q-values
        self.q_tables: Dict[str, Dict[str, Dict[str, float]]] = {}
        for i in range(config.n_agents):
            agent_id = f"agent_{i}"
            self.q_tables[agent_id] = {
                r: {m: 0.0 for m in self.methods}
                for r in self.regimes
            }

        # Mixing network weights (simplified as linear combination)
        self.mixing_weights = {f"agent_{i}": 1.0 / config.n_agents for i in range(config.n_agents)}

        self.epsilon = config.epsilon
        self.iteration = 0

        self.niche_affinities: Dict[str, Dict[str, float]] = {
            f"agent_{i}": {r: 0.25 for r in self.regimes}
            for i in range(config.n_agents)
        }

    def select_actions(self, regime: str) -> Dict[str, str]:
        """Select actions with coordination bonus."""
        actions = {}

        for agent_id, q_table in self.q_tables.items():
            if self.rng.random() < self.epsilon:
                action = self.rng.choice(self.methods)
            else:
                q_values = q_table[regime]
                action = max(q_values, key=q_values.get)

            actions[agent_id] = action

        return actions

    def compute_team_reward(self, individual_rewards: Dict[str, float]) -> float:
        """Compute team reward via mixing."""
        return sum(
            self.mixing_weights[agent_id] * reward
            for agent_id, reward in individual_rewards.items()
        )

    def update(
        self,
        regime: str,
        actions: Dict[str, str],
        rewards: Dict[str, float],
    ):
        """Update with team reward signal."""
        team_reward = self.compute_team_reward(rewards)

        for agent_id, action in actions.items():
            individual_reward = rewards.get(agent_id, 0.0)

            # QMIX-style: blend individual and team reward
            blended_reward = 0.7 * individual_reward + 0.3 * team_reward

            old_q = self.q_tables[agent_id][regime][action]
            self.q_tables[agent_id][regime][action] = old_q + self.config.learning_rate * (blended_reward - old_q)

            # Update niche affinity
            if individual_reward > 0:
                self.niche_affinities[agent_id][regime] += 0.1

        # Normalize
        for agent_id in self.niche_affinities:
            total = sum(self.niche_affinities[agent_id].values())
            if total > 0:
                self.niche_affinities[agent_id] = {
                    r: v / total for r, v in self.niche_affinities[agent_id].items()
                }

        self.epsilon = max(self.config.min_epsilon, self.epsilon * self.config.epsilon_decay)
        self.iteration += 1

    def run_iteration(self, prices, regime: str, reward_fn):
        """Run one iteration."""
        actions = self.select_actions(regime)

        rewards = {}
        for agent_id, method in actions.items():
            reward = reward_fn([method], prices)
            rewards[agent_id] = reward

        self.update(regime, actions, rewards)

        return {"actions": actions, "rewards": rewards}

    def get_niche_distribution(self) -> Dict[str, Dict[str, float]]:
        return self.niche_affinities


class MAPPO:
    """
    MAPPO-inspired baseline for multi-agent learning.

    Key idea: PPO with centralized value function and
    decentralized policies.

    Simplified implementation: Policy gradient with shared critic.
    """

    def __init__(self, config: MARLConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.methods = list(METHOD_INVENTORY_V2.keys())[:config.n_methods]
        self.regimes = ["trend_up", "trend_down", "mean_revert", "volatile"]

        # Policy parameters (softmax over methods per regime)
        self.policies: Dict[str, Dict[str, Dict[str, float]]] = {}
        for i in range(config.n_agents):
            agent_id = f"agent_{i}"
            self.policies[agent_id] = {
                r: {m: 0.0 for m in self.methods}  # Logits
                for r in self.regimes
            }

        # Shared value function (per regime)
        self.value_function: Dict[str, float] = {r: 0.0 for r in self.regimes}

        self.iteration = 0

        self.niche_affinities: Dict[str, Dict[str, float]] = {
            f"agent_{i}": {r: 0.25 for r in self.regimes}
            for i in range(config.n_agents)
        }

    def _softmax(self, logits: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
        """Compute softmax probabilities."""
        max_logit = max(logits.values())
        exp_logits = {k: np.exp((v - max_logit) / temperature) for k, v in logits.items()}
        total = sum(exp_logits.values())
        return {k: v / total for k, v in exp_logits.items()}

    def select_actions(self, regime: str) -> Tuple[Dict[str, str], Dict[str, float]]:
        """Select actions from policy and return log probabilities."""
        actions = {}
        log_probs = {}

        for agent_id, policy in self.policies.items():
            probs = self._softmax(policy[regime])
            action = self.rng.choice(list(probs.keys()), p=list(probs.values()))
            actions[agent_id] = action
            log_probs[agent_id] = np.log(probs[action] + 1e-8)

        return actions, log_probs

    def update(
        self,
        regime: str,
        actions: Dict[str, str],
        log_probs: Dict[str, float],
        rewards: Dict[str, float],
    ):
        """Policy gradient update with shared baseline."""
        # Compute advantage using shared value function
        team_reward = np.mean(list(rewards.values()))
        advantage = team_reward - self.value_function[regime]

        # Update value function
        self.value_function[regime] += 0.1 * advantage

        # Policy gradient update
        for agent_id, action in actions.items():
            individual_reward = rewards.get(agent_id, 0.0)
            individual_advantage = individual_reward - self.value_function[regime]

            # Gradient ascent on log probability weighted by advantage
            grad = self.config.learning_rate * individual_advantage
            self.policies[agent_id][regime][action] += grad

            # Update niche affinity
            if individual_reward > 0:
                self.niche_affinities[agent_id][regime] += 0.1

        # Normalize niche affinities
        for agent_id in self.niche_affinities:
            total = sum(self.niche_affinities[agent_id].values())
            if total > 0:
                self.niche_affinities[agent_id] = {
                    r: v / total for r, v in self.niche_affinities[agent_id].items()
                }

        self.iteration += 1

    def run_iteration(self, prices, regime: str, reward_fn):
        """Run one iteration."""
        actions, log_probs = self.select_actions(regime)

        rewards = {}
        for agent_id, method in actions.items():
            reward = reward_fn([method], prices)
            rewards[agent_id] = reward

        self.update(regime, actions, log_probs, rewards)

        return {"actions": actions, "rewards": rewards}

    def get_niche_distribution(self) -> Dict[str, Dict[str, float]]:
        return self.niche_affinities


class QualityDiversity:
    """
    Quality-Diversity (MAP-Elites) baseline.

    Key idea: Maintain an archive of diverse high-performing solutions.
    This explicitly optimizes for both quality AND diversity.

    Difference from our approach:
    - QD: Explicitly maintains diversity via archive
    - Ours: Diversity emerges from competition without explicit diversity objective
    """

    def __init__(self, config: MARLConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.methods = list(METHOD_INVENTORY_V2.keys())[:config.n_methods]
        self.regimes = ["trend_up", "trend_down", "mean_revert", "volatile"]

        # Archive: one elite per regime cell
        # Each elite is a policy (method distribution)
        self.archive: Dict[str, Dict] = {
            r: {"policy": None, "fitness": float("-inf")}
            for r in self.regimes
        }

        # Current population
        self.population: List[Dict[str, Dict[str, float]]] = []
        for i in range(config.n_agents):
            # Random initial policy per agent
            policy = {
                r: {m: self.rng.random() for m in self.methods}
                for r in self.regimes
            }
            # Normalize
            for r in self.regimes:
                total = sum(policy[r].values())
                policy[r] = {m: v / total for m, v in policy[r].items()}
            self.population.append(policy)

        self.iteration = 0

        # For compatibility, track niche affinities
        self.niche_affinities: Dict[str, Dict[str, float]] = {
            f"agent_{i}": {r: 0.25 for r in self.regimes}
            for i in range(config.n_agents)
        }

    def select_action(self, policy: Dict[str, Dict[str, float]], regime: str) -> str:
        """Sample action from policy."""
        probs = policy[regime]
        return self.rng.choice(list(probs.keys()), p=list(probs.values()))

    def mutate(self, policy: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Mutate policy."""
        new_policy = {}
        for r in self.regimes:
            new_policy[r] = {}
            for m, v in policy[r].items():
                # Add Gaussian noise
                new_v = max(0.01, v + self.rng.normal(0, 0.1))
                new_policy[r][m] = new_v
            # Normalize
            total = sum(new_policy[r].values())
            new_policy[r] = {m: v / total for m, v in new_policy[r].items()}
        return new_policy

    def update(self, regime: str, policies: List[Dict], rewards: List[float]):
        """Update archive with elites."""
        for policy, reward in zip(policies, rewards):
            # Compute behavior descriptor (which regime does this policy prefer?)
            # Simplified: use the regime with highest probability mass
            regime_probs = {r: max(policy[r].values()) for r in self.regimes}
            preferred_regime = max(regime_probs, key=regime_probs.get)

            # Update archive if better
            if reward > self.archive[preferred_regime]["fitness"]:
                self.archive[preferred_regime]["policy"] = policy
                self.archive[preferred_regime]["fitness"] = reward

        # Generate new population from archive + mutation
        new_population = []
        for i in range(self.config.n_agents):
            if self.archive[self.regimes[i % len(self.regimes)]]["policy"] is not None:
                parent = self.archive[self.regimes[i % len(self.regimes)]]["policy"]
            else:
                parent = self.population[i]

            child = self.mutate(parent)
            new_population.append(child)

        self.population = new_population
        self.iteration += 1

        # Update niche affinities from archive
        for i, agent_id in enumerate([f"agent_{i}" for i in range(self.config.n_agents)]):
            policy = self.population[i]
            for r in self.regimes:
                self.niche_affinities[agent_id][r] = max(policy[r].values())
            # Normalize
            total = sum(self.niche_affinities[agent_id].values())
            if total > 0:
                self.niche_affinities[agent_id] = {
                    r: v / total for r, v in self.niche_affinities[agent_id].items()
                }

    def run_iteration(self, prices, regime: str, reward_fn):
        """Run one iteration."""
        actions = []
        rewards_list = []

        for policy in self.population:
            action = self.select_action(policy, regime)
            reward = reward_fn([action], prices)
            actions.append(action)
            rewards_list.append(reward)

        self.update(regime, self.population, rewards_list)

        rewards = {f"agent_{i}": r for i, r in enumerate(rewards_list)}
        return {"actions": actions, "rewards": rewards}

    def get_niche_distribution(self) -> Dict[str, Dict[str, float]]:
        return self.niche_affinities


def compare_marl_baselines(
    n_iterations: int = 1000,
    n_trials: int = 10,
    seed: int = 42,
) -> Dict[str, Dict]:
    """
    Compare all MARL baselines.

    Returns metrics for each baseline:
    - Final SI
    - Diversity
    - Mean reward
    """
    from ..environment.synthetic_market import SyntheticMarketConfig, SyntheticMarketEnvironment

    config = MARLConfig()
    baselines = {
        "IQL": lambda s: IndependentQLearning(config, seed=s),
        "QMIX": lambda s: QMIX(config, seed=s),
        "MAPPO": lambda s: MAPPO(config, seed=s),
        "QD": lambda s: QualityDiversity(config, seed=s),
    }

    results = {}

    for name, create_fn in baselines.items():
        si_values = []

        for trial in range(n_trials):
            env = SyntheticMarketEnvironment(SyntheticMarketConfig(seed=seed + trial))
            prices, regimes = env.generate(n_bars=n_iterations + 100)

            baseline = create_fn(seed + trial)

            def reward_fn(methods, prices_window):
                if len(prices_window) < 2:
                    return 0.0
                ret = (prices_window[-1] - prices_window[-2]) / prices_window[-2]
                return ret * 100

            for i in range(20, min(len(prices), n_iterations + 50)):
                regime = regimes.iloc[i]
                price_window = prices['close'].values[max(0, i-20):i+1]
                baseline.run_iteration(price_window, regime, reward_fn)

            # Compute SI
            niche_dist = baseline.get_niche_distribution()
            agent_sis = []
            for affinities in niche_dist.values():
                aff_array = np.array(list(affinities.values()))
                aff_array = aff_array / (aff_array.sum() + 1e-8)
                entropy = -np.sum(aff_array * np.log(aff_array + 1e-8))
                max_entropy = np.log(len(aff_array))
                si = 1 - entropy / max_entropy if max_entropy > 0 else 0
                agent_sis.append(si)

            si_values.append(np.mean(agent_sis))

        results[name] = {
            "si_mean": np.mean(si_values),
            "si_std": np.std(si_values),
        }

    return results
