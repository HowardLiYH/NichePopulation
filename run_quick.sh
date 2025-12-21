#!/bin/bash
# Quick test run with reduced trials (for debugging/CI)

set -e

echo "================================================"
echo "Emergent Specialization - Quick Test"
echo "================================================"

# Reduced settings for quick testing
TRIALS=10
SEED=42

echo "Running quick test with $TRIALS trials..."
echo ""

python -m experiments.runner \
    --experiments exp1 exp2 \
    --trials $TRIALS \
    --seed $SEED \
    --output results_quick

echo ""
echo "Quick test complete! Results in results_quick/"
