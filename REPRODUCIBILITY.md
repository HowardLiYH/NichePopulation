# Reproducibility Guide

This document provides instructions for reproducing all experimental results reported in our NeurIPS 2025 paper: **"Emergent Specialization in Multi-Agent Trading"**.

## System Requirements

- **Python**: 3.9+ (tested on 3.10)
- **Memory**: 8GB RAM minimum
- **Disk**: 1GB for results
- **Time**: ~2 hours for full experiments on modern CPU

## Quick Start

### Option 1: Local Installation

```bash
# Clone repository
git clone https://github.com/HowardLiYH/Emergent-Specialization-in-Multi-Agent-Trading.git
cd Emergent-Specialization-in-Multi-Agent-Trading

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -e .

# Run quick test (2-3 minutes)
make quick

# Run full experiments (~2 hours)
make full
```

### Option 2: Docker

```bash
# Build image
docker build -t emergent-specialization .

# Run quick test
docker run emergent-specialization

# Run full experiments
docker run -v $(pwd)/results:/app/results emergent-specialization \
    python -m experiments.runner --all --trials 100
```

## Experiment Details

### Experiment 1: Emergence of Specialists

- **File**: `experiments/exp1_emergence.py`
- **Trials**: 100
- **Duration**: ~20 minutes
- **Expected Result**: Final SI = 0.65 ± 0.10, p < 0.001

```bash
python experiments/exp1_emergence.py --trials 100 --seed 42
```

### Experiment 2: Value of Diversity

- **File**: `experiments/exp2_diversity_value.py`
- **Trials**: 100
- **Duration**: ~30 minutes
- **Expected Result**: Diverse outperforms Homogeneous by 35%

```bash
python experiments/exp2_diversity_value.py --trials 100 --seed 42
```

### Experiment 3: Population Size Effect

- **File**: `experiments/exp3_population_size.py`
- **Trials**: 50 per size
- **Sizes tested**: [3, 5, 7, 10, 15, 20]
- **Duration**: ~45 minutes
- **Expected Result**: Optimal N* = 5-7

```bash
python experiments/exp3_population_size.py --trials 50 --seed 42
```

### Experiment 4: Transfer Frequency

- **File**: `experiments/exp4_transfer_frequency.py`
- **Trials**: 50 per frequency
- **Frequencies tested**: [1, 5, 10, 25, 50, 100]
- **Duration**: ~30 minutes
- **Expected Result**: Optimal τ* = 10-25

```bash
python experiments/exp4_transfer_frequency.py --trials 50 --seed 42
```

### Experiment 5: Regime Transitions

- **File**: `experiments/exp5_regime_transitions.py`
- **Trials**: 50
- **Duration**: ~15 minutes
- **Expected Result**: Switch rate > 0.8, p < 0.05

```bash
python experiments/exp5_regime_transitions.py --trials 50 --seed 42
```

### Experiment 6: Real Data Validation

- **File**: `experiments/exp6_real_data.py`
- **Data**: BTC 2021-2024 (if available)
- **Duration**: ~10 minutes
- **Expected Result**: SI matches synthetic expectations

```bash
python experiments/exp6_real_data.py --seed 42
```

## Generating Paper Artifacts

### Figures

```bash
make figures
# Output: paper/figures/fig1_emergence.pdf, fig2_diversity.pdf, etc.
```

### Tables

```bash
make tables
# Output: paper/tables/table1_main_results.tex, table2_baselines.tex
```

## Random Seeds

All experiments use seed 42 by default for reproducibility. To verify robustness, run with different seeds:

```bash
make full SEED=123
make full SEED=456
```

## Expected Variance

Results may vary slightly due to:
- Floating-point precision differences
- Random number generator implementation
- Platform-specific optimizations

Acceptable variance: ±5% on mean values, same statistical significance conclusions.

## Hardware Used for Paper

Results reported in the paper were generated on:
- CPU: Apple M1 Pro
- RAM: 16GB
- OS: macOS 14.0
- Python: 3.10.12
- NumPy: 1.24.3
- Pandas: 2.0.3

## Contact

For reproducibility issues, please open a GitHub issue.
