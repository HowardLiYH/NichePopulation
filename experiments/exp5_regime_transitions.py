"""
Experiment 5: Regime Transition Behavior

Research Question: How do specialized agents behave during regime transitions?

Hypothesis H5: During regime transitions:
  (a) The "winning" agent changes to the new regime's specialist
  (b) Transition speed depends on transfer frequency
  (c) Population maintains diversity even during transitions

Protocol:
1. Generate data with clear regime transitions (longer regimes)
2. Track which agent wins before/during/after transition
3. Measure transition delay (bars until new specialist dominates)
4. Repeat 50 times

Statistical Analysis:
- Chi-square test for agent switching
- Time-series analysis of winner identity

Expected Results:
- Clear agent switching at regime changes
- Transition delay of 5-10 bars
- Specialists maintain performance in their regimes
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from collections import Counter

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.synthetic_market import SyntheticMarketEnvironment, SyntheticMarketConfig
from src.agents.population import Population, PopulationConfig, compute_reward_from_methods
from src.analysis.specialization import SpecializationTracker

from .config import EXP5_CONFIG, ExperimentConfig


@dataclass
class TransitionEvent:
    """A single regime transition event."""
    iteration: int
    from_regime: str
    to_regime: str
    winner_before: str  # Agent ID winning before transition
    winner_after: str   # Agent ID winning after transition
    transition_delay: int  # Bars until new specialist dominates
    switched: bool  # Did the winner change?


@dataclass
class TrialResult:
    """Result of a single trial."""
    trial_id: int
    seed: int
    transitions: List[TransitionEvent]
    n_transitions: int
    switch_rate: float  # Fraction of transitions where winner changed
    avg_transition_delay: float
    regime_specialist_map: Dict[str, str]  # regime -> dominant agent


@dataclass
class ExperimentResult:
    """Full experiment results."""
    experiment_name: str
    n_trials: int

    # Aggregate metrics
    avg_switch_rate: float
    avg_transition_delay: float

    # Chi-square test result
    switch_expected: float  # Expected switch rate if random
    switch_observed: float
    chi2_statistic: float
    chi2_p_value: float

    trial_results: List[TrialResult]


def detect_regime_transitions(regimes: pd.Series) -> List[Tuple[int, str, str]]:
    """
    Detect regime transition points.

    Returns:
        List of (iteration, from_regime, to_regime)
    """
    transitions = []
    prev_regime = regimes.iloc[0]

    for i in range(1, len(regimes)):
        curr_regime = regimes.iloc[i]
        if curr_regime != prev_regime:
            transitions.append((i, prev_regime, curr_regime))
        prev_regime = curr_regime

    return transitions


def run_single_trial(
    trial_id: int,
    seed: int,
    config: ExperimentConfig,
) -> TrialResult:
    """
    Run a single trial analyzing regime transitions.
    """
    # Create environment with longer, more consistent regimes
    env_config = SyntheticMarketConfig(
        regime_names=config.regime_names,
        regime_duration_mean=config.regime_duration_mean,  # Longer regimes
        regime_duration_std=config.regime_duration_std,    # Less variation
        seed=seed,
    )
    env = SyntheticMarketEnvironment(env_config)
    prices, regimes = env.generate(config.n_bars, seed=seed)

    # Create population
    pop_config = PopulationConfig(
        n_agents=config.n_agents,
        max_methods_per_agent=config.max_methods,
        transfer_frequency=config.transfer_frequency,
        transfer_tau=config.transfer_tau,
        seed=seed,
    )
    population = Population(pop_config)

    window_size = 20

    # Track winner at each iteration
    winner_history = []
    regime_history = []

    for i in range(window_size, min(len(prices) - 1, window_size + config.n_bars - 20)):
        current_regime = regimes.iloc[i]
        price_window = prices.iloc[i-window_size:i+1]

        result = population.run_iteration(price_window, compute_reward_from_methods, current_regime)
        winner_history.append(result.winner_id)
        regime_history.append(current_regime)

    # Detect transitions
    transitions = []
    regime_transitions = detect_regime_transitions(pd.Series(regime_history))

    for trans_idx, from_regime, to_regime in regime_transitions:
        if trans_idx < 10 or trans_idx >= len(winner_history) - 10:
            continue  # Skip transitions too close to edges

        # Winner before transition (mode of last 5 bars)
        winners_before = winner_history[max(0, trans_idx-5):trans_idx]
        if not winners_before:
            continue
        winner_before = Counter(winners_before).most_common(1)[0][0]

        # Winner after transition (mode of next 10 bars)
        winners_after = winner_history[trans_idx:min(len(winner_history), trans_idx+10)]
        if not winners_after:
            continue
        winner_after = Counter(winners_after).most_common(1)[0][0]

        # Transition delay: when does new winner first appear?
        delay = 0
        for j, w in enumerate(winners_after):
            if w != winner_before:
                delay = j
                break
        else:
            delay = len(winners_after)

        switched = winner_before != winner_after

        transitions.append(TransitionEvent(
            iteration=trans_idx,
            from_regime=from_regime,
            to_regime=to_regime,
            winner_before=winner_before,
            winner_after=winner_after,
            transition_delay=delay,
            switched=switched,
        ))

    # Compute regime-specialist mapping
    regime_winners = {}
    for regime, winner in zip(regime_history, winner_history):
        if regime not in regime_winners:
            regime_winners[regime] = []
        regime_winners[regime].append(winner)

    regime_specialist_map = {
        regime: Counter(winners).most_common(1)[0][0] if winners else "unknown"
        for regime, winners in regime_winners.items()
    }

    switch_rate = sum(1 for t in transitions if t.switched) / max(1, len(transitions))
    avg_delay = np.mean([t.transition_delay for t in transitions]) if transitions else 0.0

    return TrialResult(
        trial_id=trial_id,
        seed=seed,
        transitions=transitions,
        n_transitions=len(transitions),
        switch_rate=switch_rate,
        avg_transition_delay=avg_delay,
        regime_specialist_map=regime_specialist_map,
    )


def run_experiment(
    config: ExperimentConfig = EXP5_CONFIG,
    save_results: bool = True,
) -> ExperimentResult:
    """
    Run the full regime transition experiment.
    """
    print(f"=" * 60)
    print(f"Experiment 5: Regime Transition Behavior")
    print(f"Trials: {config.n_trials}")
    print(f"Regime duration: {config.regime_duration_mean} ± {config.regime_duration_std}")
    print(f"=" * 60)

    trial_results = []

    for trial_id in tqdm(range(config.n_trials), desc="Running trials"):
        seed = config.base_seed + trial_id * 1000
        result = run_single_trial(trial_id, seed, config)
        trial_results.append(result)

    # Aggregate metrics
    switch_rates = [r.switch_rate for r in trial_results]
    transition_delays = [r.avg_transition_delay for r in trial_results if r.n_transitions > 0]

    avg_switch_rate = float(np.mean(switch_rates))
    avg_transition_delay = float(np.mean(transition_delays)) if transition_delays else 0.0

    # Chi-square test: is switching more frequent than random?
    # Under random assignment, switch probability = 1 - 1/n_agents
    n_agents = config.n_agents
    expected_switch_rate = 1 - 1/n_agents  # If agents were randomly winning

    total_transitions = sum(r.n_transitions for r in trial_results)
    observed_switches = sum(
        sum(1 for t in r.transitions if t.switched)
        for r in trial_results
    )
    expected_switches = total_transitions * expected_switch_rate

    # Simple chi-square
    if expected_switches > 0:
        chi2 = ((observed_switches - expected_switches) ** 2) / expected_switches
        # Approximate p-value (1 df)
        from scipy import stats
        chi2_p = 1 - stats.chi2.cdf(chi2, df=1)
    else:
        chi2 = 0.0
        chi2_p = 1.0

    result = ExperimentResult(
        experiment_name=config.experiment_name,
        n_trials=config.n_trials,
        avg_switch_rate=avg_switch_rate,
        avg_transition_delay=avg_transition_delay,
        switch_expected=expected_switch_rate,
        switch_observed=observed_switches / max(1, total_transitions),
        chi2_statistic=chi2,
        chi2_p_value=chi2_p,
        trial_results=trial_results,
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"RESULTS: Experiment 5")
    print(f"{'=' * 60}")
    print(f"Total transitions analyzed: {total_transitions}")
    print(f"Average switch rate: {avg_switch_rate:.3f}")
    print(f"  Expected if random: {expected_switch_rate:.3f}")
    print(f"  Chi-square: {chi2:.3f}, p={chi2_p:.4f}")
    print(f"Average transition delay: {avg_transition_delay:.1f} bars")
    print(f"\nInterpretation:")
    if avg_switch_rate > expected_switch_rate:
        print("  → Agents DO switch specialists at regime transitions!")
    else:
        print("  → No significant switching (agents may not be specialized)")
    print(f"{'=' * 60}")

    # Save results
    if save_results:
        results_dir = Path(config.results_dir) / config.experiment_name
        results_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "experiment_name": result.experiment_name,
            "n_trials": result.n_trials,
            "total_transitions": total_transitions,
            "avg_switch_rate": avg_switch_rate,
            "avg_transition_delay": avg_transition_delay,
            "switch_expected": expected_switch_rate,
            "switch_observed": result.switch_observed,
            "chi2_statistic": chi2,
            "chi2_p_value": chi2_p,
        }

        with open(results_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to {results_dir}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Experiment 5: Regime Transitions")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")

    args = parser.parse_args()

    config = ExperimentConfig(
        experiment_name="exp5_regime_transitions",
        n_trials=args.trials,
        base_seed=args.seed,
        regime_duration_mean=100,  # Longer regimes for clear transitions
        regime_duration_std=5,
    )

    result = run_experiment(config, save_results=not args.no_save)
