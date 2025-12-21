"""
Run all V2 experiments for NeurIPS paper.
"""
import sys
sys.path.insert(0, '.')

print("=" * 70)
print("RUNNING ALL V2 EXPERIMENTS FOR NEURIPS PAPER")
print("=" * 70)
print()

# Experiment 1: Emergence
print(">>> Experiment 1: Emergence of Specialists")
from experiments.exp1_emergence_v2 import run_experiment as run_exp1
exp1_results = run_exp1(n_trials=50, n_iterations=3000)

# Experiment 2: Value of Diversity
print("\n>>> Experiment 2: Value of Diversity")
from experiments.exp2_diversity_v2 import run_experiment as run_exp2
exp2_results = run_exp2(n_trials=50)

# Experiment 3: Population Size
print("\n>>> Experiment 3: Population Size Effect")
from experiments.exp3_population_v2 import run_experiment as run_exp3
exp3_results = run_exp3(n_trials=30)

print()
print("=" * 70)
print("ALL EXPERIMENTS COMPLETE")
print("=" * 70)
print()
print("Results saved to:")
print("  - results/exp1_emergence_v2/")
print("  - results/exp2_diversity_v2/")
print("  - results/exp3_population_v2/")
