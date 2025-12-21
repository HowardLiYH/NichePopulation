"""
Critical Ablation Studies for NeurIPS Submission

Ablation 1: λ (Niche Bonus) Sweep
- Tests whether specialization is truly "emergent" or just incentivized
- If SI remains high at λ=0, specialization is emergent
- If SI drops to ~0 at λ=0, it's incentive-driven (need to reframe claims)

Ablation 2: Homogeneous Population Baseline
- Clone best agent to create homogeneous population
- Compare diverse vs homogeneous performance
- This is the proper baseline for "value of diversity" claims
"""

import sys
sys.path.insert(0, '.')

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from copy import deepcopy

from src.environment.synthetic_market import SyntheticMarketEnvironment, SyntheticMarketConfig
from src.agents.niche_population import NichePopulation, NicheAgent
from src.agents.inventory_v2 import METHOD_INVENTORY_V2


def compute_reward(methods, prices):
    """Compute reward for selected methods."""
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
    """Compute specialization index from regime affinities."""
    affinities = np.array(list(niche_affinities.values()))
    affinities = affinities / (affinities.sum() + 1e-8)
    entropy = -np.sum(affinities * np.log(affinities + 1e-8))
    max_entropy = np.log(len(affinities))
    return 1 - (entropy / max_entropy) if max_entropy > 0 else 0.0


class HomogeneousPopulation:
    """
    Homogeneous Population: All agents use the same best method.
    
    This is the proper baseline for testing "value of diversity."
    If diverse population doesn't beat homogeneous, diversity has no value.
    """
    
    def __init__(self, n_agents: int = 8, best_method: str = "VolScalp", seed: int = None):
        self.n_agents = n_agents
        self.best_method = best_method
        self.rng = np.random.default_rng(seed)
        self.history = []
        self.iteration = 0
        self.regimes = ["trend_up", "trend_down", "mean_revert", "volatile"]
    
    def run_iteration(self, prices, regime: str, reward_fn):
        """All agents select the same method."""
        self.iteration += 1
        
        # All agents use best method, pick random winner
        reward = reward_fn([self.best_method], prices)
        winner_id = f"agent_{self.rng.integers(0, self.n_agents)}"
        
        self.history.append({
            "iteration": self.iteration,
            "regime": regime,
            "winner": winner_id,
            "method": self.best_method,
            "reward": reward,
        })
        
        return {
            "winner_id": winner_id,
            "winner_method": self.best_method,
            "regime": regime,
            "reward": reward,
        }
    
    def get_total_reward(self):
        return sum(h["reward"] for h in self.history)


class OraclePopulation:
    """
    Oracle Population: Perfect regime knowledge, always selects optimal method.
    
    This is the theoretical upper bound - if diverse can't approach oracle,
    there's room for improvement in the specialization mechanism.
    """
    
    # Optimal methods per regime (based on inventory_v2 design)
    OPTIMAL_METHODS = {
        "trend_up": "BuyMomentum",
        "trend_down": "SellMomentum", 
        "mean_revert": "MeanRevert",
        "volatile": "VolScalp",
    }
    
    def __init__(self, n_agents: int = 8, seed: int = None):
        self.n_agents = n_agents
        self.rng = np.random.default_rng(seed)
        self.history = []
        self.iteration = 0
        self.regimes = ["trend_up", "trend_down", "mean_revert", "volatile"]
    
    def run_iteration(self, prices, regime: str, reward_fn):
        """Select optimal method for current regime."""
        self.iteration += 1
        
        optimal_method = self.OPTIMAL_METHODS.get(regime, "VolScalp")
        reward = reward_fn([optimal_method], prices)
        winner_id = f"agent_{self.rng.integers(0, self.n_agents)}"
        
        self.history.append({
            "iteration": self.iteration,
            "regime": regime,
            "method": optimal_method,
            "reward": reward,
        })
        
        return {
            "winner_id": winner_id,
            "winner_method": optimal_method,
            "regime": regime,
            "reward": reward,
        }
    
    def get_total_reward(self):
        return sum(h["reward"] for h in self.history)


def ablation_niche_bonus(n_trials: int = 30, n_iterations: int = 2000):
    """
    Ablation 1: Vary λ (niche bonus coefficient) from 0 to 1.
    
    Key question: Does specialization emerge WITHOUT the bonus (λ=0)?
    """
    print("=" * 60)
    print("ABLATION 1: Niche Bonus Coefficient (λ)")
    print("=" * 60)
    
    lambda_values = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    results = {lam: [] for lam in lambda_values}
    
    for lam in lambda_values:
        print(f"\nλ = {lam}")
        
        for trial in tqdm(range(n_trials), desc=f"λ={lam}"):
            env = SyntheticMarketEnvironment(SyntheticMarketConfig(
                regime_duration_mean=100, seed=trial * 1000
            ))
            prices, regimes = env.generate(n_bars=n_iterations + 100)
            
            pop = NichePopulation(
                n_agents=8,
                niche_bonus=lam,  # Key parameter
                seed=trial
            )
            
            total_reward = 0.0
            window_size = 20
            
            for i in range(window_size, min(len(prices)-1, n_iterations + 50)):
                price_window = prices.iloc[i-window_size:i+1]
                regime = regimes.iloc[i]
                result = pop.run_iteration(price_window, regime, compute_reward)
                total_reward += compute_reward([result["winner_method"]], price_window)
            
            # Compute metrics
            niche_dist = pop.get_niche_distribution()
            agent_sis = [compute_regime_si(aff) for aff in niche_dist.values()]
            primary_niches = [max(aff, key=aff.get) for aff in niche_dist.values()]
            
            results[lam].append({
                "trial": trial,
                "si": np.mean(agent_sis),
                "diversity": len(set(primary_niches)) / 4,
                "reward": total_reward,
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("ABLATION 1 RESULTS")
    print("=" * 60)
    print(f"{'λ':>6} | {'SI':>12} | {'Diversity':>10} | {'Reward':>12}")
    print("-" * 50)
    
    summary = {}
    for lam in lambda_values:
        sis = [r["si"] for r in results[lam]]
        divs = [r["diversity"] for r in results[lam]]
        rewards = [r["reward"] for r in results[lam]]
        
        summary[lam] = {
            "si_mean": float(np.mean(sis)),
            "si_std": float(np.std(sis)),
            "diversity_mean": float(np.mean(divs)),
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
        }
        
        print(f"{lam:>6.2f} | {np.mean(sis):>6.3f}±{np.std(sis):.3f} | {np.mean(divs):>10.2f} | {np.mean(rewards):>6.1f}±{np.std(rewards):.1f}")
    
    # Key insight
    print()
    si_at_zero = summary[0.0]["si_mean"]
    si_at_half = summary[0.5]["si_mean"]
    
    if si_at_zero > 0.5:
        print(f"✅ GOOD: SI at λ=0 is {si_at_zero:.3f} > 0.5")
        print("   Specialization IS emergent (not just incentivized)")
    elif si_at_zero > 0.3:
        print(f"⚠️ PARTIAL: SI at λ=0 is {si_at_zero:.3f}")
        print("   Some emergent specialization, but bonus helps significantly")
    else:
        print(f"❌ CONCERN: SI at λ=0 is {si_at_zero:.3f}")
        print("   Specialization is primarily incentive-driven, not emergent")
    
    return summary


def ablation_baselines(n_trials: int = 30, n_iterations: int = 2000):
    """
    Ablation 2: Compare Diverse vs Homogeneous vs Oracle.
    
    Key question: Does diversity actually help compared to proper baselines?
    """
    print("\n" + "=" * 60)
    print("ABLATION 2: Diverse vs Homogeneous vs Oracle")
    print("=" * 60)
    
    results = {
        "diverse": [],
        "homogeneous_volscalp": [],
        "homogeneous_momentum": [],
        "oracle": [],
        "random": [],
    }
    
    for trial in tqdm(range(n_trials), desc="Trials"):
        env = SyntheticMarketEnvironment(SyntheticMarketConfig(
            regime_duration_mean=100, seed=trial * 1000
        ))
        prices, regimes = env.generate(n_bars=n_iterations + 100)
        
        # Initialize populations
        diverse = NichePopulation(n_agents=8, niche_bonus=0.5, seed=trial)
        homo_vol = HomogeneousPopulation(n_agents=8, best_method="VolScalp", seed=trial)
        homo_mom = HomogeneousPopulation(n_agents=8, best_method="BuyMomentum", seed=trial)
        oracle = OraclePopulation(n_agents=8, seed=trial)
        
        # Track rewards
        diverse_reward = 0.0
        random_reward = 0.0
        
        window_size = 20
        rng = np.random.default_rng(trial)
        method_names = list(METHOD_INVENTORY_V2.keys())
        
        for i in range(window_size, min(len(prices)-1, n_iterations + 50)):
            price_window = prices.iloc[i-window_size:i+1]
            regime = regimes.iloc[i]
            
            # Diverse
            result = diverse.run_iteration(price_window, regime, compute_reward)
            diverse_reward += compute_reward([result["winner_method"]], price_window)
            
            # Homogeneous (VolScalp)
            homo_vol.run_iteration(price_window, regime, compute_reward)
            
            # Homogeneous (Momentum)
            homo_mom.run_iteration(price_window, regime, compute_reward)
            
            # Oracle
            oracle.run_iteration(price_window, regime, compute_reward)
            
            # Random
            random_method = rng.choice(method_names)
            random_reward += compute_reward([random_method], price_window)
        
        results["diverse"].append(diverse_reward)
        results["homogeneous_volscalp"].append(homo_vol.get_total_reward())
        results["homogeneous_momentum"].append(homo_mom.get_total_reward())
        results["oracle"].append(oracle.get_total_reward())
        results["random"].append(random_reward)
    
    # Summary
    print("\n" + "=" * 60)
    print("ABLATION 2 RESULTS")
    print("=" * 60)
    print(f"{'Strategy':<25} | {'Reward':>15} | {'vs Diverse':>12}")
    print("-" * 60)
    
    diverse_mean = np.mean(results["diverse"])
    diverse_std = np.std(results["diverse"])
    
    summary = {}
    strategies = [
        ("Diverse Population", "diverse"),
        ("Homogeneous (VolScalp)", "homogeneous_volscalp"),
        ("Homogeneous (Momentum)", "homogeneous_momentum"),
        ("Oracle (Perfect)", "oracle"),
        ("Random", "random"),
    ]
    
    for name, key in strategies:
        mean = np.mean(results[key])
        std = np.std(results[key])
        diff = mean - diverse_mean
        
        # Statistical test vs diverse
        if key != "diverse":
            t_stat, p_value = stats.ttest_rel(results["diverse"], results[key])
        else:
            t_stat, p_value = 0, 1.0
        
        summary[key] = {
            "mean": float(mean),
            "std": float(std),
            "diff_vs_diverse": float(diff),
            "p_value": float(p_value),
        }
        
        if key == "diverse":
            print(f"{name:<25} | {mean:>7.1f}±{std:>5.1f} | {'---':>12}")
        else:
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"{name:<25} | {mean:>7.1f}±{std:>5.1f} | {diff:>+8.1f} {sig}")
    
    # Key insights
    print()
    homo_best = max(summary["homogeneous_volscalp"]["mean"], summary["homogeneous_momentum"]["mean"])
    
    if diverse_mean > homo_best:
        improvement = (diverse_mean - homo_best) / homo_best * 100
        print(f"✅ GOOD: Diverse beats best Homogeneous by {improvement:.1f}%")
        print("   Diversity has genuine value!")
    else:
        print(f"❌ CONCERN: Homogeneous beats or matches Diverse")
        print("   Need to rethink diversity claims")
    
    oracle_gap = (summary["oracle"]["mean"] - diverse_mean) / summary["oracle"]["mean"] * 100
    print(f"📊 Gap to Oracle: {oracle_gap:.1f}% (room for improvement)")
    
    return summary


def run_all_ablations():
    """Run all ablation studies."""
    output_dir = Path("results/ablations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ablation 1: Niche bonus
    ablation1_results = ablation_niche_bonus(n_trials=30, n_iterations=2000)
    
    with open(output_dir / "ablation1_niche_bonus.json", "w") as f:
        json.dump({"lambda_sweep": {str(k): v for k, v in ablation1_results.items()}}, f, indent=2)
    
    # Ablation 2: Baselines
    ablation2_results = ablation_baselines(n_trials=30, n_iterations=2000)
    
    with open(output_dir / "ablation2_baselines.json", "w") as f:
        json.dump(ablation2_results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("ALL ABLATIONS COMPLETE")
    print("=" * 60)
    print("Results saved to results/ablations/")
    
    return ablation1_results, ablation2_results


if __name__ == "__main__":
    run_all_ablations()

