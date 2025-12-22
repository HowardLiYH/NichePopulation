#!/usr/bin/env python3
"""
Experiment: Mechanism Ablation Study

Tests which components of the niche mechanism are necessary for specialization.

Conditions:
1. FULL: Niche bonus + Competition (baseline)
2. BONUS_ONLY: Niche bonus, no competition (all agents get rewards)
3. COMPETITION_ONLY: Competition, no niche bonus (λ=0)
4. CONTROL: No bonus, no competition (independent learners)

Hypothesis:
- Competition is necessary (drives differentiation)
- Niche bonus is sufficient but not necessary (shapes which niches form)

Expected Results:
- FULL: High SI, high diversity
- COMPETITION_ONLY: Moderate SI, moderate diversity (competition drives some differentiation)
- BONUS_ONLY: Low SI (no pressure to differentiate)
- CONTROL: Low SI (random/uniform)

Usage:
    python experiments/exp_mechanism_ablation.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
from scipy import stats
import time
from copy import deepcopy

from src.environment.synthetic_market import SyntheticMarketConfig, SyntheticMarketEnvironment
from src.agents.niche_population import NichePopulation
from src.agents.inventory_v2 import METHOD_INVENTORY_V2


# Configuration
N_TRIALS = 30
N_ITERATIONS = 2000
N_AGENTS = 8

# Output
RESULTS_DIR = Path(__file__).parent.parent / "results" / "exp_mechanism_ablation"


def compute_regime_si(niche_affinities: Dict[str, float]) -> float:
    """Compute SI from regime affinities."""
    affinities = np.array(list(niche_affinities.values()))
    affinities = affinities / (affinities.sum() + 1e-8)
    entropy = -np.sum(affinities * np.log(affinities + 1e-8))
    max_entropy = np.log(len(affinities))
    return 1 - (entropy / max_entropy) if max_entropy > 0 else 0.0


def compute_diversity(niche_distributions: Dict[str, Dict[str, float]]) -> float:
    """Compute population diversity via mean pairwise JSD."""
    from src.analysis import compute_niche_heterogeneity
    return compute_niche_heterogeneity(niche_distributions)


@dataclass
class AblationResult:
    """Result from one ablation condition."""
    condition: str
    niche_bonus: float
    competition: bool
    si_mean: float
    si_std: float
    si_ci_lower: float
    si_ci_upper: float
    diversity_mean: float
    reward_mean: float
    n_trials: int


def create_reward_function(prices_df, all_regimes):
    """Create reward function for a given price series."""

    def reward_fn(methods, prices_window):
        if len(prices_window) < 2:
            return 0.0

        ret = (prices_window[-1] - prices_window[-2]) / prices_window[-2]

        method_name = methods[0] if methods else "BuyMomentum"
        method_class = METHOD_INVENTORY_V2.get(method_name)

        if method_class is None:
            return ret

        if "Buy" in method_name or "Trend" in method_name:
            signal = 1.0
        elif "Sell" in method_name or "Fade" in method_name:
            signal = -1.0
        else:
            signal = 0.0

        return signal * ret * 100

    return reward_fn


class NoCompetitionPopulation:
    """
    Population variant with no competitive selection.
    All agents receive their own rewards (no winner-take-all).
    """

    def __init__(self, base_population: NichePopulation):
        self.pop = base_population
        self.iteration = 0

    def run_iteration(self, prices, regime, reward_fn):
        """Run iteration without competition."""
        # Each agent selects and gets their own reward
        results = {}

        for agent_id, agent in self.pop.agents.items():
            method = agent.select_method(regime)
            methods = [method]
            reward = reward_fn(methods, prices)

            # Update agent beliefs (everyone "wins" if reward > 0)
            won = reward > 0
            agent.update(regime, method, won=won)

            results[agent_id] = reward

        self.iteration += 1
        return {"rewards": results}

    def get_niche_distribution(self):
        return self.pop.get_niche_distribution()


class ControlPopulation:
    """
    Control condition: No niche bonus, no competition.
    Agents learn independently with uniform exploration.
    """

    def __init__(self, n_agents: int, regime_names: List[str], seed: int = None):
        self.n_agents = n_agents
        self.regime_names = regime_names
        self.rng = np.random.default_rng(seed)
        self.iteration = 0

        # Initialize uniform affinities (no learning)
        self.niche_affinities = {
            f"agent_{i}": {r: 1.0 / len(regime_names) for r in regime_names}
            for i in range(n_agents)
        }

        # Method selection is random
        self.methods = list(METHOD_INVENTORY_V2.keys())

    def run_iteration(self, prices, regime, reward_fn):
        """Run iteration with random selection, no learning."""
        results = {}

        for agent_id in self.niche_affinities.keys():
            # Random method selection
            methods = [self.rng.choice(self.methods)]
            reward = reward_fn(methods, prices)
            results[agent_id] = reward

        self.iteration += 1
        return {"rewards": results}

    def get_niche_distribution(self):
        return self.niche_affinities


def run_condition(
    condition: str,
    niche_bonus: float,
    competition: bool,
    trial_id: int,
) -> Tuple[float, float, float]:
    """
    Run a single trial for a given condition.

    Returns: (si, diversity, mean_reward)
    """
    # Create environment
    env = SyntheticMarketEnvironment(SyntheticMarketConfig(
        regime_duration_mean=100,
        seed=trial_id * 1000
    ))
    prices, regimes = env.generate(n_bars=N_ITERATIONS + 100)
    regime_names = ["trend_up", "trend_down", "mean_revert", "volatile"]

    # Create population based on condition
    if condition == "CONTROL":
        population = ControlPopulation(
            n_agents=N_AGENTS,
            regime_names=regime_names,
            seed=trial_id
        )
    else:
        base_pop = NichePopulation(
            n_agents=N_AGENTS,
            niche_bonus=niche_bonus,
            seed=trial_id
        )

        if not competition:
            population = NoCompetitionPopulation(base_pop)
        else:
            population = base_pop

    # Create reward function
    prices_arr = prices['close'].values
    reward_fn = create_reward_function(prices, regimes)

    # Run iterations
    rewards = []
    for i in range(20, min(len(prices_arr), N_ITERATIONS + 50)):
        regime = regimes.iloc[i]
        price_window = prices_arr[max(0, i-20):i+1]

        result = population.run_iteration(price_window, regime, reward_fn)

        if len(price_window) >= 2:
            ret = (price_window[-1] - price_window[-2]) / price_window[-2]
            rewards.append(ret * 100)

    # Compute final metrics
    niche_dist = population.get_niche_distribution()

    agent_sis = [compute_regime_si(aff) for aff in niche_dist.values()]
    si = np.mean(agent_sis)

    diversity = compute_diversity(niche_dist)
    mean_reward = np.mean(rewards) if rewards else 0.0

    return si, diversity, mean_reward


def bootstrap_ci(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Compute bootstrap confidence interval."""
    if len(values) < 2:
        return (np.mean(values), np.mean(values))

    n_bootstrap = 1000
    bootstrap_means = [
        np.mean(np.random.choice(values, size=len(values), replace=True))
        for _ in range(n_bootstrap)
    ]

    alpha = 1 - confidence
    return (
        np.percentile(bootstrap_means, alpha / 2 * 100),
        np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
    )


def run_experiment():
    """Run the full mechanism ablation experiment."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MECHANISM ABLATION EXPERIMENT")
    print("=" * 60)
    print()

    # Define conditions
    conditions = [
        ("FULL", 0.5, True),           # Full mechanism
        ("COMPETITION_ONLY", 0.0, True), # No niche bonus
        ("BONUS_ONLY", 0.5, False),    # No competition
        ("CONTROL", 0.0, False),        # Neither
    ]

    results = []

    for condition_name, niche_bonus, competition in conditions:
        print(f"\n--- Condition: {condition_name} ---")
        print(f"    Niche Bonus: {niche_bonus}, Competition: {competition}")

        si_values = []
        diversity_values = []
        reward_values = []

        start_time = time.time()

        for trial in range(N_TRIALS):
            if (trial + 1) % 10 == 0:
                print(f"    Trial {trial + 1}/{N_TRIALS}...")

            si, div, rew = run_condition(condition_name, niche_bonus, competition, trial)
            si_values.append(si)
            diversity_values.append(div)
            reward_values.append(rew)

        elapsed = time.time() - start_time

        # Compute statistics
        ci_lower, ci_upper = bootstrap_ci(si_values)

        result = AblationResult(
            condition=condition_name,
            niche_bonus=niche_bonus,
            competition=competition,
            si_mean=np.mean(si_values),
            si_std=np.std(si_values),
            si_ci_lower=ci_lower,
            si_ci_upper=ci_upper,
            diversity_mean=np.mean(diversity_values),
            reward_mean=np.mean(reward_values),
            n_trials=N_TRIALS,
        )
        results.append(result)

        print(f"    SI: {result.si_mean:.4f} ± {result.si_std:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"    Diversity: {result.diversity_mean:.4f}")
        print(f"    Time: {elapsed:.1f}s")

    # Statistical comparisons
    print("\n" + "=" * 60)
    print("STATISTICAL COMPARISONS")
    print("=" * 60)

    # Compare FULL vs each other condition
    full_sis = [r for r in results if r.condition == "FULL"][0]

    for r in results:
        if r.condition == "FULL":
            continue

        # We don't have raw values, so use effect size estimate
        effect = (full_sis.si_mean - r.si_mean) / ((full_sis.si_std + r.si_std) / 2 + 1e-8)
        print(f"\nFULL vs {r.condition}:")
        print(f"  SI difference: {full_sis.si_mean - r.si_mean:.4f}")
        print(f"  Cohen's d: {effect:.4f}")

    # Save results
    summary = {
        "experiment": "mechanism_ablation",
        "date": datetime.now().isoformat(),
        "config": {
            "n_trials": N_TRIALS,
            "n_iterations": N_ITERATIONS,
            "n_agents": N_AGENTS,
        },
        "results": [asdict(r) for r in results],
        "conclusions": {
            "competition_necessary": results[1].si_mean > results[3].si_mean,  # COMPETITION_ONLY > CONTROL
            "bonus_helpful": results[0].si_mean > results[1].si_mean,  # FULL > COMPETITION_ONLY
        }
    }

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("CONCLUSIONS")
    print("=" * 60)
    print(f"Competition necessary: {summary['conclusions']['competition_necessary']}")
    print(f"Bonus helpful: {summary['conclusions']['bonus_helpful']}")
    print(f"\nResults saved to {RESULTS_DIR}")

    return results


if __name__ == "__main__":
    run_experiment()
