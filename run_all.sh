#!/bin/bash
# Run all experiments for NeurIPS submission

set -e  # Exit on error

echo "================================================"
echo "Emergent Specialization - Full Experiment Suite"
echo "================================================"

# Default settings
TRIALS=${TRIALS:-100}
SEED=${SEED:-42}
OUTPUT_DIR=${OUTPUT_DIR:-results}

echo "Settings:"
echo "  Trials: $TRIALS"
echo "  Seed: $SEED"
echo "  Output: $OUTPUT_DIR"
echo ""

# Create results directory
mkdir -p "$OUTPUT_DIR"

# Run experiments
echo "Running Experiment 1: Emergence..."
python experiments/exp1_emergence.py --trials $TRIALS --seed $SEED

echo ""
echo "Running Experiment 2: Diversity Value..."
python experiments/exp2_diversity_value.py --trials $TRIALS --seed $SEED

echo ""
echo "Running Experiment 3: Population Size..."
python experiments/exp3_population_size.py --trials 50 --seed $SEED

echo ""
echo "Running Experiment 4: Transfer Frequency..."
python experiments/exp4_transfer_frequency.py --trials 50 --seed $SEED

echo ""
echo "Running Experiment 5: Regime Transitions..."
python experiments/exp5_regime_transitions.py --trials 50 --seed $SEED

echo ""
echo "Running Experiment 6: Real Data Validation..."
python experiments/exp6_real_data.py --seed $SEED

echo ""
echo "================================================"
echo "Generating Figures..."
echo "================================================"
python -c "from src.analysis.figures import generate_all_figures; generate_all_figures()"

echo ""
echo "================================================"
echo "Generating Tables..."
echo "================================================"
python -c "from src.analysis.cross_experiment import run_cross_analysis; run_cross_analysis()"

echo ""
echo "================================================"
echo "All experiments complete!"
echo "Results saved to: $OUTPUT_DIR/"
echo "Figures saved to: paper/figures/"
echo "Tables saved to: paper/tables/"
echo "================================================"
