"""
Experiment 2 (V2): Value of Diversity

Compares diverse population vs homogeneous baselines.
"""
import sys
sys.path.insert(0, '.')

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import stats

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


class RandomBaseline:
    """Random method selection."""
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.methods = list(METHOD_INVENTORY_V2.keys())

    def select(self, regime):
        return self.rng.choice(self.methods)


class SingleMethodBaseline:
    """Always use one method."""
    def __init__(self, method_name):
        self.method = method_name

    def select(self, regime):
        return self.method


def run_single_trial(trial_id: int, n_iterations: int = 2000) -> dict:
    """Run one trial comparing diverse vs baselines."""
    env = SyntheticMarketEnvironment(SyntheticMarketConfig(
        regime_duration_mean=100, seed=trial_id * 1000
    ))
    prices, regimes = env.generate(n_bars=n_iterations + 100)

    # Diverse population
    diverse_pop = NichePopulation(n_agents=8, niche_bonus=0.5, seed=trial_id)

    # Baselines
    random_baseline = RandomBaseline(seed=trial_id)
    momentum_baseline = SingleMethodBaseline("BuyMomentum")

    # Track cumulative rewards
    diverse_reward = 0.0
    random_reward = 0.0
    momentum_reward = 0.0

    window_size = 20
    for i in range(window_size, min(len(prices)-1, n_iterations + 50)):
        price_window = prices.iloc[i-window_size:i+1]
        regime = regimes.iloc[i]

        # Diverse population
        result = diverse_pop.run_iteration(price_window, regime, compute_reward)
        diverse_reward += compute_reward([result["winner_method"]], price_window)

        # Random baseline
        random_method = random_baseline.select(regime)
        random_reward += compute_reward([random_method], price_window)

        # Momentum baseline
        mom_method = momentum_baseline.select(regime)
        momentum_reward += compute_reward([mom_method], price_window)

    return {
        "trial_id": trial_id,
        "diverse_reward": diverse_reward,
        "random_reward": random_reward,
        "momentum_reward": momentum_reward,
    }


def run_experiment(n_trials: int = 30, output_dir: str = "results/exp2_diversity_v2"):
    """Run full experiment."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []
    print("Running Experiment 2 V2: Value of Diversity")

    for trial in tqdm(range(n_trials), desc="Trials"):
        result = run_single_trial(trial)
        results.append(result)

    diverse = [r["diverse_reward"] for r in results]
    random_r = [r["random_reward"] for r in results]
    momentum = [r["momentum_reward"] for r in results]

    # Statistical tests
    t_vs_random, p_vs_random = stats.ttest_rel(diverse, random_r)
    t_vs_momentum, p_vs_momentum = stats.ttest_rel(diverse, momentum)

    summary = {
        "experiment_name": "exp2_diversity_v2",
        "n_trials": n_trials,
        "diverse_mean": float(np.mean(diverse)),
        "diverse_std": float(np.std(diverse)),
        "random_mean": float(np.mean(random_r)),
        "momentum_mean": float(np.mean(momentum)),
        "improvement_vs_random": float(np.mean(diverse) - np.mean(random_r)),
        "improvement_vs_momentum": float(np.mean(diverse) - np.mean(momentum)),
        "p_vs_random": float(p_vs_random),
        "p_vs_momentum": float(p_vs_momentum),
        "significant_vs_random": str(p_vs_random < 0.05 and np.mean(diverse) > np.mean(random_r)),
        "significant_vs_momentum": str(p_vs_momentum < 0.05 and np.mean(diverse) > np.mean(momentum)),
    }

    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 60)
    print("EXPERIMENT 2 V2 RESULTS: Value of Diversity")
    print("=" * 60)
    print(f"Diverse Population: {summary['diverse_mean']:.2f} ± {summary['diverse_std']:.2f}")
    print(f"Random Baseline: {summary['random_mean']:.2f}")
    print(f"Momentum Baseline: {summary['momentum_mean']:.2f}")
    print(f"Improvement vs Random: {summary['improvement_vs_random']:.2f} (p={p_vs_random:.2e})")
    print(f"Improvement vs Momentum: {summary['improvement_vs_momentum']:.2f} (p={p_vs_momentum:.2e})")
    print()

    return summary


if __name__ == "__main__":
    run_experiment(n_trials=30)
