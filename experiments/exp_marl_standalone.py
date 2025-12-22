#!/usr/bin/env python3
"""
Standalone MARL Comparison Experiment.

Self-contained implementation that doesn't rely on pandas-heavy imports.
Compares NichePopulation vs QMIX, MAPPO, IQL, Random baselines.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict

import numpy as np

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    n_agents: int = 8
    n_methods: int = 5
    n_regimes: int = 4
    learning_rate: float = 0.01
    niche_bonus: float = 0.3
    epsilon: float = 0.3
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.05


# =============================================================================
# NichePopulation (Our Method)
# =============================================================================

class NicheAgent:
    """Agent with niche affinity for specific regimes."""

    def __init__(self, agent_id: str, regimes: List[str], methods: List[str], seed: int = 0):
        self.agent_id = agent_id
        self.regimes = regimes
        self.methods = methods
        self.rng = np.random.default_rng(seed)

        # Niche affinity (preference for each regime)
        self.niche_affinity = {r: 1.0/len(regimes) for r in regimes}

        # Method preference per regime
        self.method_prefs = {r: {m: 1.0/len(methods) for m in methods} for r in regimes}

        # Win counts
        self.regime_wins = defaultdict(int)
        self.total_wins = 0

    def select_method(self, regime: str) -> str:
        """Select method based on current preferences."""
        prefs = list(self.method_prefs[regime].values())
        prefs = np.array(prefs) / sum(prefs)
        return self.rng.choice(self.methods, p=prefs)

    def get_niche_strength(self, regime: str) -> float:
        """Get strength of affinity for regime."""
        return self.niche_affinity[regime]

    def update(self, regime: str, method: str, won: bool):
        """Update based on competition outcome."""
        lr = 0.1

        if won:
            self.regime_wins[regime] += 1
            self.total_wins += 1

            # Increase affinity for winning regime
            for r in self.regimes:
                if r == regime:
                    self.niche_affinity[r] = min(1.0, self.niche_affinity[r] + lr)
                else:
                    self.niche_affinity[r] = max(0.01, self.niche_affinity[r] - lr / (len(self.regimes) - 1))

            # Normalize
            total = sum(self.niche_affinity.values())
            self.niche_affinity = {r: v/total for r, v in self.niche_affinity.items()}

            # Update method preference
            for m in self.methods:
                if m == method:
                    self.method_prefs[regime][m] = min(1.0, self.method_prefs[regime][m] + lr)
                else:
                    self.method_prefs[regime][m] = max(0.01, self.method_prefs[regime][m] - lr / (len(self.methods) - 1))


class NichePopulationStandalone:
    """Population of niche-specialized agents."""

    def __init__(self, config: Config, regimes: List[str], methods: List[str], seed: int = 0):
        self.config = config
        self.regimes = regimes
        self.methods = methods

        self.agents = {
            f"agent_{i}": NicheAgent(f"agent_{i}", regimes, methods, seed + i)
            for i in range(config.n_agents)
        }

    def get_mean_si(self) -> float:
        """Compute mean Specialization Index across agents."""
        sis = []
        for agent in self.agents.values():
            affinities = np.array(list(agent.niche_affinity.values()))
            affinities = affinities / (affinities.sum() + 1e-10)
            affinities = affinities[affinities > 0]
            entropy = -np.sum(affinities * np.log(affinities + 1e-10))
            si = 1 - entropy / np.log(len(self.regimes))
            sis.append(si)
        return float(np.mean(sis))


# =============================================================================
# QMIX Baseline
# =============================================================================

class QMIXStandalone:
    """QMIX baseline implementation."""

    def __init__(self, config: Config, regimes: List[str], methods: List[str], seed: int = 0):
        self.config = config
        self.regimes = regimes
        self.methods = methods
        self.rng = np.random.default_rng(seed)

        # Q-tables per agent
        self.q_tables = {
            f"agent_{i}": {
                r: {m: 0.0 for m in methods} for r in regimes
            }
            for i in range(config.n_agents)
        }

        # Mixing weights (simplified)
        self.mix_weights = np.abs(self.rng.normal(0, 0.1, config.n_agents))

        self.epsilon = config.epsilon
        self.niche_affinities = {
            f"agent_{i}": {r: 1.0/len(regimes) for r in regimes}
            for i in range(config.n_agents)
        }
        self.regime_wins = {f"agent_{i}": defaultdict(int) for i in range(config.n_agents)}
        self.total_wins = defaultdict(int)

    def select_actions(self, regime: str) -> Dict[str, str]:
        actions = {}
        for agent_id, q_table in self.q_tables.items():
            if self.rng.random() < self.epsilon:
                action = self.rng.choice(self.methods)
            else:
                q_values = q_table[regime]
                action = max(q_values, key=q_values.get)
            actions[agent_id] = action
        return actions

    def update(self, regime: str, actions: Dict[str, str],
               rewards: Dict[str, float], winner_id: str):
        # Get individual Q-values
        q_values = np.array([
            self.q_tables[f"agent_{i}"][regime][actions[f"agent_{i}"]]
            for i in range(self.config.n_agents)
        ])

        # Mixing (Q_tot = sum of weighted Q)
        q_tot = np.sum(self.mix_weights * q_values)

        # Team reward
        team_reward = sum(rewards.values())

        # TD error
        td_error = team_reward - q_tot

        # Update individual Q-values
        for i, agent_id in enumerate([f"agent_{i}" for i in range(self.config.n_agents)]):
            action = actions[agent_id]
            self.q_tables[agent_id][regime][action] += self.config.learning_rate * td_error / self.config.n_agents

        # Update niche affinities
        self.regime_wins[winner_id][regime] += 1
        self.total_wins[winner_id] += 1

        for agent_id in self.q_tables:
            total = self.total_wins[agent_id]
            if total > 0:
                for r in self.regimes:
                    wins = self.regime_wins[agent_id].get(r, 0)
                    old = self.niche_affinities[agent_id][r]
                    new = wins / total
                    self.niche_affinities[agent_id][r] = 0.9 * old + 0.1 * new

        self.epsilon = max(self.config.min_epsilon, self.epsilon * self.config.epsilon_decay)

    def get_mean_si(self) -> float:
        sis = []
        for agent_id in self.q_tables:
            affinities = np.array(list(self.niche_affinities[agent_id].values()))
            affinities = affinities / (affinities.sum() + 1e-10)
            affinities = affinities[affinities > 0]
            entropy = -np.sum(affinities * np.log(affinities + 1e-10))
            si = 1 - entropy / np.log(len(self.regimes))
            sis.append(si)
        return float(np.mean(sis))


# =============================================================================
# MAPPO Baseline
# =============================================================================

class MAPPOStandalone:
    """MAPPO baseline implementation."""

    def __init__(self, config: Config, regimes: List[str], methods: List[str], seed: int = 0):
        self.config = config
        self.regimes = regimes
        self.methods = methods
        self.rng = np.random.default_rng(seed)

        # Policy logits per agent
        self.policies = {
            f"agent_{i}": {
                r: np.zeros(len(methods)) for r in regimes
            }
            for i in range(config.n_agents)
        }

        # Centralized value function
        self.values = {r: 0.0 for r in regimes}

        self.niche_affinities = {
            f"agent_{i}": {r: 1.0/len(regimes) for r in regimes}
            for i in range(config.n_agents)
        }
        self.action_counts = {
            f"agent_{i}": {r: defaultdict(int) for r in regimes}
            for i in range(config.n_agents)
        }

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / exp_logits.sum()

    def select_actions(self, regime: str) -> Tuple[Dict[str, str], Dict[str, float]]:
        actions = {}
        log_probs = {}

        for agent_id, policy in self.policies.items():
            probs = self._softmax(policy[regime])
            action_idx = self.rng.choice(len(self.methods), p=probs)
            actions[agent_id] = self.methods[action_idx]
            log_probs[agent_id] = np.log(probs[action_idx] + 1e-10)

            self.action_counts[agent_id][regime][self.methods[action_idx]] += 1

        return actions, log_probs

    def update(self, regime: str, actions: Dict[str, str],
               log_probs: Dict[str, float], rewards: Dict[str, float]):
        # Compute advantage
        team_reward = sum(rewards.values()) / len(rewards)
        advantage = team_reward - self.values[regime]

        # Update value
        self.values[regime] += 0.01 * advantage

        # Update policies (simplified PPO)
        for agent_id, policy in self.policies.items():
            action_idx = self.methods.index(actions[agent_id])

            # Policy gradient update
            probs = self._softmax(policy[regime])
            grad = np.zeros(len(self.methods))
            grad[action_idx] = (1 - probs[action_idx]) * advantage
            for j in range(len(self.methods)):
                if j != action_idx:
                    grad[j] = -probs[j] * advantage

            policy[regime] += 0.01 * grad

        # Update niche affinities
        self._update_niche_affinities()

    def _update_niche_affinities(self):
        for agent_id in self.policies:
            total_per_regime = {}
            for regime in self.regimes:
                total = sum(self.action_counts[agent_id][regime].values())
                total_per_regime[regime] = total

            grand_total = sum(total_per_regime.values())
            if grand_total > 0:
                for regime in self.regimes:
                    self.niche_affinities[agent_id][regime] = total_per_regime[regime] / grand_total

    def get_mean_si(self) -> float:
        sis = []
        for agent_id in self.policies:
            affinities = np.array(list(self.niche_affinities[agent_id].values()))
            affinities = affinities / (affinities.sum() + 1e-10)
            affinities = affinities[affinities > 0]
            entropy = -np.sum(affinities * np.log(affinities + 1e-10))
            si = 1 - entropy / np.log(len(self.regimes))
            sis.append(si)
        return float(np.mean(sis))


# =============================================================================
# IQL Baseline
# =============================================================================

class IQLStandalone:
    """Independent Q-Learning baseline."""

    def __init__(self, config: Config, regimes: List[str], methods: List[str], seed: int = 0):
        self.config = config
        self.regimes = regimes
        self.methods = methods
        self.rng = np.random.default_rng(seed)

        self.q_tables = {
            f"agent_{i}": {
                r: {m: 0.0 for m in methods} for r in regimes
            }
            for i in range(config.n_agents)
        }

        self.epsilon = config.epsilon
        self.niche_affinities = {
            f"agent_{i}": {r: 1.0/len(regimes) for r in regimes}
            for i in range(config.n_agents)
        }
        self.regime_wins = {f"agent_{i}": defaultdict(int) for i in range(config.n_agents)}
        self.total_wins = defaultdict(int)

    def select_actions(self, regime: str) -> Dict[str, str]:
        actions = {}
        for agent_id, q_table in self.q_tables.items():
            if self.rng.random() < self.epsilon:
                action = self.rng.choice(self.methods)
            else:
                q_values = q_table[regime]
                action = max(q_values, key=q_values.get)
            actions[agent_id] = action
        return actions

    def update(self, regime: str, actions: Dict[str, str],
               rewards: Dict[str, float], winner_id: str):
        # Update each agent's Q independently
        for agent_id, reward in rewards.items():
            action = actions[agent_id]
            old_q = self.q_tables[agent_id][regime][action]
            self.q_tables[agent_id][regime][action] = old_q + self.config.learning_rate * (reward - old_q)

        # Track wins for SI
        self.regime_wins[winner_id][regime] += 1
        self.total_wins[winner_id] += 1

        for agent_id in self.q_tables:
            total = self.total_wins[agent_id]
            if total > 0:
                for r in self.regimes:
                    wins = self.regime_wins[agent_id].get(r, 0)
                    old = self.niche_affinities[agent_id][r]
                    new = wins / total
                    self.niche_affinities[agent_id][r] = 0.9 * old + 0.1 * new

        self.epsilon = max(self.config.min_epsilon, self.epsilon * self.config.epsilon_decay)

    def get_mean_si(self) -> float:
        sis = []
        for agent_id in self.q_tables:
            affinities = np.array(list(self.niche_affinities[agent_id].values()))
            affinities = affinities / (affinities.sum() + 1e-10)
            affinities = affinities[affinities > 0]
            entropy = -np.sum(affinities * np.log(affinities + 1e-10))
            si = 1 - entropy / np.log(len(self.regimes))
            sis.append(si)
        return float(np.mean(sis))


# =============================================================================
# Experiment Runner
# =============================================================================

def run_experiment(method_class, config: Config, regimes: List[str],
                   methods: List[str], regime_probs: Dict[str, float],
                   n_iterations: int = 500, seed: int = 42) -> Dict:
    """Run experiment for a single method."""
    rng = np.random.default_rng(seed)

    model = method_class(config, regimes, methods, seed)

    for _ in range(n_iterations):
        regime = rng.choice(list(regime_probs.keys()), p=list(regime_probs.values()))

        if isinstance(model, MAPPOStandalone):
            actions, log_probs = model.select_actions(regime)
            rewards = {aid: rng.normal(0.5, 0.2) for aid in actions}
            winner_id = max(rewards, key=rewards.get)
            rewards[winner_id] += 0.5
            model.update(regime, actions, log_probs, rewards)
        elif isinstance(model, NichePopulationStandalone):
            agent_scores = {}
            agent_choices = {}

            for agent_id, agent in model.agents.items():
                method = agent.select_method(regime)
                agent_choices[agent_id] = method
                base_score = rng.normal(0.5, 0.15)
                niche_strength = agent.get_niche_strength(regime)
                agent_scores[agent_id] = base_score + config.niche_bonus * (niche_strength - 0.25)

            winner_id = max(agent_scores, key=agent_scores.get)

            for agent_id, agent in model.agents.items():
                agent.update(regime, agent_choices[agent_id], agent_id == winner_id)
        else:
            actions = model.select_actions(regime)
            rewards = {aid: rng.normal(0.5, 0.2) for aid in actions}
            winner_id = max(rewards, key=rewards.get)
            rewards[winner_id] += 0.5
            model.update(regime, actions, rewards, winner_id)

    return {
        'mean_si': model.get_mean_si(),
    }


def run_domain_comparison(domain_name: str, n_trials: int = 30,
                           n_iterations: int = 500, seed: int = 42) -> Dict:
    """Run comparison on a domain with synthetic regime/method setup."""
    print(f"\n{'='*60}")
    print(f"MARL COMPARISON: {domain_name.upper()}")
    print(f"{'='*60}")

    # Domain-specific settings
    domain_settings = {
        'crypto': {
            'regimes': ['bull', 'bear', 'sideways', 'volatile'],
            'probs': {'bull': 0.12, 'bear': 0.10, 'sideways': 0.53, 'volatile': 0.25},
        },
        'commodities': {
            'regimes': ['bull', 'bear', 'sideways', 'volatile'],
            'probs': {'bull': 0.29, 'bear': 0.26, 'sideways': 0.21, 'volatile': 0.24},
        },
        'weather': {
            'regimes': ['stable', 'approaching_storm', 'active_storm', 'stable_hot', 'stable_cold'],
            'probs': {'stable': 0.63, 'approaching_storm': 0.18, 'active_storm': 0.09, 'stable_hot': 0.07, 'stable_cold': 0.03},
        },
        'solar': {
            'regimes': ['clear', 'partly_cloudy', 'overcast', 'storm'],
            'probs': {'clear': 0.28, 'partly_cloudy': 0.35, 'overcast': 0.33, 'storm': 0.04},
        },
    }

    settings = domain_settings.get(domain_name, domain_settings['crypto'])
    regimes = settings['regimes']
    regime_probs = settings['probs']
    methods = ['method_1', 'method_2', 'method_3', 'method_4', 'method_5']

    config = Config(n_agents=8, n_methods=len(methods), n_regimes=len(regimes))

    print(f"Regimes: {len(regimes)}, Methods: {len(methods)}")

    results = {
        'NichePopulation': [],
        'QMIX': [],
        'MAPPO': [],
        'IQL': [],
    }

    methods_map = {
        'NichePopulation': NichePopulationStandalone,
        'QMIX': QMIXStandalone,
        'MAPPO': MAPPOStandalone,
        'IQL': IQLStandalone,
    }

    for trial in range(n_trials):
        trial_seed = seed + trial

        for method_name, method_class in methods_map.items():
            result = run_experiment(
                method_class, config, regimes, methods, regime_probs,
                n_iterations, trial_seed
            )
            results[method_name].append(result['mean_si'])

        if (trial + 1) % 10 == 0:
            print(f"  Trial {trial+1}/{n_trials}")

    # Compute statistics
    summary = {}
    for method_name, sis in results.items():
        si_array = np.array(sis)
        summary[method_name] = {
            'mean_si': float(np.mean(si_array)),
            'std_si': float(np.std(si_array)),
            'min_si': float(np.min(si_array)),
            'max_si': float(np.max(si_array)),
        }

    return {
        'domain': domain_name,
        'n_trials': n_trials,
        'n_regimes': len(regimes),
        'n_methods': len(methods),
        'results': results,
        'summary': summary,
    }


def main():
    """Run full MARL comparison."""
    print("="*60)
    print("FULL MARL BASELINE COMPARISON (STANDALONE)")
    print("="*60)

    domains = ['crypto', 'commodities', 'weather', 'solar']
    all_results = {}

    for domain_name in domains:
        results = run_domain_comparison(domain_name, n_trials=30)
        all_results[domain_name] = results

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY: MEAN SPECIALIZATION INDEX (SI)")
    print("="*70)
    print(f"\n{'Domain':<12} {'NichePop':<12} {'QMIX':<12} {'MAPPO':<12} {'IQL':<12}")
    print("-"*60)

    for domain, results in all_results.items():
        s = results['summary']
        print(f"{domain:<12} "
              f"{s['NichePopulation']['mean_si']:.3f}±{s['NichePopulation']['std_si']:.2f}  "
              f"{s['QMIX']['mean_si']:.3f}±{s['QMIX']['std_si']:.2f}  "
              f"{s['MAPPO']['mean_si']:.3f}±{s['MAPPO']['std_si']:.2f}  "
              f"{s['IQL']['mean_si']:.3f}±{s['IQL']['std_si']:.2f}")

    # Cross-domain averages
    print("-"*60)
    avg = {method: [] for method in ['NichePopulation', 'QMIX', 'MAPPO', 'IQL']}
    for domain, results in all_results.items():
        for method in avg:
            avg[method].append(results['summary'][method]['mean_si'])

    print(f"{'AVERAGE':<12} "
          f"{np.mean(avg['NichePopulation']):.3f}         "
          f"{np.mean(avg['QMIX']):.3f}         "
          f"{np.mean(avg['MAPPO']):.3f}         "
          f"{np.mean(avg['IQL']):.3f}")

    # Statistical significance tests
    print("\n" + "="*70)
    print("STATISTICAL TESTS (NichePopulation vs Others)")
    print("="*70)

    from scipy import stats

    for domain, data in all_results.items():
        print(f"\n{domain.upper()}:")
        niche_si = data['results']['NichePopulation']

        for baseline in ['QMIX', 'MAPPO', 'IQL']:
            baseline_si = data['results'][baseline]
            t_stat, p_value = stats.ttest_ind(niche_si, baseline_si)

            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            diff = np.mean(niche_si) - np.mean(baseline_si)
            print(f"  vs {baseline:<8}: Δ={diff:+.3f}, t={t_stat:.2f}, p={p_value:.4f} {sig}")

    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "marl_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_results = {}
    for domain, data in all_results.items():
        json_results[domain] = {
            'domain': data['domain'],
            'n_trials': data['n_trials'],
            'n_regimes': data['n_regimes'],
            'n_methods': data['n_methods'],
            'summary': data['summary'],
        }

    with open(output_dir / "latest_results.json", 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\n\nResults saved to: {output_dir / 'latest_results.json'}")


if __name__ == "__main__":
    main()
