"""
Experiment 3 (V2): Population Size Effect

Tests how population size affects specialization and performance.
"""
import sys
sys.path.insert(0, '.')

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.environment.synthetic_market import SyntheticMarketEnvironment, SyntheticMarketConfig
from src.agents.niche_population import NichePopulation
from src.agents.inventory_v2 import METHOD_INVENTORY_V2


def compute_reward(methods, prices):
    if len(prices) < 2:
        return 0.0
    signals, confs = [], []
    for m in methods:
        if m in METHOD_INVENTORY_V2:
            result = METHOD_INVENTORY_V2[m].execute(prices)
            signals.append(result['signal'])
            confs.append(result['confidence'])
    if not signals:
        return 0.0
    weights = np.array(confs) / (sum(confs) + 1e-8)
    signal = sum(s * w for s, w in zip(signals, weights))
    price_return = (prices['close'].iloc[-1] / prices['close'].iloc[-2]) - 1
    return float(np.clip(signal * price_return * 10, -1, 1))


def compute_regime_si(niche_affinities):
    affinities = np.array(list(niche_affinities.values()))
    affinities = affinities / (affinities.sum() + 1e-8)
    entropy = -np.sum(affinities * np.log(affinities + 1e-8))
    max_entropy = np.log(len(affinities))
    return 1 - (entropy / max_entropy) if max_entropy > 0 else 0.0


def run_single_trial(trial_id: int, n_agents: int, n_iterations: int = 2000) -> dict:
    env = SyntheticMarketEnvironment(SyntheticMarketConfig(
        regime_duration_mean=100, seed=trial_id * 1000
    ))
    prices, regimes = env.generate(n_bars=n_iterations + 100)

    pop = NichePopulation(n_agents=n_agents, niche_bonus=0.5, seed=trial_id)

    total_reward = 0.0
    window_size = 20

    for i in range(window_size, min(len(prices)-1, n_iterations + 50)):
        price_window = prices.iloc[i-window_size:i+1]
        regime = regimes.iloc[i]
        result = pop.run_iteration(price_window, regime, compute_reward)
        total_reward += compute_reward([result["winner_method"]], price_window)

    niche_dist = pop.get_niche_distribution()
    agent_sis = [compute_regime_si(aff) for aff in niche_dist.values()]

    primary_niches = [max(aff, key=aff.get) for aff in niche_dist.values()]
    diversity = len(set(primary_niches)) / 4

    return {
        "trial_id": trial_id,
        "n_agents": n_agents,
        "final_si": np.mean(agent_sis),
        "diversity": diversity,
        "total_reward": total_reward,
    }


def run_experiment(n_trials: int = 20, output_dir: str = "results/exp3_population_v2"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    population_sizes = [2, 4, 6, 8, 12, 16]
    results = {size: [] for size in population_sizes}

    print("Running Experiment 3 V2: Population Size Effect")

    for size in population_sizes:
        print(f"\nPopulation size: {size}")
        for trial in tqdm(range(n_trials), desc=f"Size {size}"):
            result = run_single_trial(trial, n_agents=size)
            results[size].append(result)

    summary = {
        "experiment_name": "exp3_population_v2",
        "population_sizes": population_sizes,
        "n_trials": n_trials,
        "size_results": {},
    }

    print()
    print("=" * 60)
    print("EXPERIMENT 3 V2 RESULTS: Population Size Effect")
    print("=" * 60)
    print(f"{'Size':>6} | {'SI':>8} | {'Diversity':>10} | {'Reward':>10}")
    print("-" * 45)

    for size in population_sizes:
        sis = [r["final_si"] for r in results[size]]
        divs = [r["diversity"] for r in results[size]]
        rewards = [r["total_reward"] for r in results[size]]

        summary["size_results"][size] = {
            "si_mean": float(np.mean(sis)),
            "si_std": float(np.std(sis)),
            "diversity_mean": float(np.mean(divs)),
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
        }

        print(f"{size:>6} | {np.mean(sis):>8.3f} | {np.mean(divs):>10.2f} | {np.mean(rewards):>10.2f}")

    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    return summary


if __name__ == "__main__":
    run_experiment(n_trials=20)
