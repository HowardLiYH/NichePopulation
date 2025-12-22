# 🧬 Emergent Specialization in Multi-Agent Systems

### Competition-Driven Niche Partitioning

<div align="center">

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-purple.svg)](#)
[![Data: 100% Real](https://img.shields.io/badge/Data-100%25%20Real-green.svg)](#data)

**Niche Partitioning Without Explicit Coordination**

[Paper](#paper) • [Installation](#installation) • [Quick Start](#quick-start) • [Experiments](#experiments) • [Results](#key-results) • [Citation](#citation)

</div>

---

## 📖 Abstract

We present a population-based multi-agent system where agents **spontaneously specialize** to different environmental regimes without explicit supervision. Drawing from ecological niche theory, we introduce **competitive exclusion with niche affinity** that creates evolutionary pressure for strategy space partitioning.

**Core Thesis:** Competition alone, without explicit diversity incentives, is sufficient to induce emergent specialization in multi-agent systems.

**Validated on 4 real-world domains (100% verified real data):**
- 📈 **Crypto** - Bybit Exchange (8,766 bars)
- 📊 **Commodities** - FRED US Government (5,630 daily prices)
- 🌤️ **Weather** - Open-Meteo (9,105 observations)
- ☀️ **Solar** - Open-Meteo Satellite (116,834 hourly)

---

## 🎯 Key Results (All Real Data)

### Cross-Domain Validation

| Domain | Source | Records | Mean SI | vs Random | vs IQL |
|--------|--------|---------|---------|-----------|--------|
| 📈 **Crypto** | Bybit Exchange | 8,766 | 0.305±0.042 | **+67%** | **+210%** |
| 📊 **Commodities** | FRED (US Gov) | 5,630 | 0.411±0.062 | **+119%** | **+359%** |
| 🌤️ **Weather** | Open-Meteo | 9,105 | 0.205±0.026 | +6% | +98% |
| ☀️ **Solar** | Open-Meteo | 116,834 | 0.443±0.036 | **+96%** | **+294%** |

**All data verified REAL from authoritative sources.**

### MARL Baseline Comparison

| Domain | NichePopulation (Ours) | IQL | Random |
|--------|------------------------|-----|--------|
| Crypto | **0.300±0.038** | 0.097±0.023 | 0.226±0.052 |
| Commodities | **0.403±0.053** | 0.088±0.021 | 0.226±0.052 |
| Weather | **0.201±0.021** | 0.102±0.023 | 0.201±0.046 |
| Solar | **0.450±0.037** | 0.113±0.026 | 0.226±0.052 |

**Key Finding: NichePopulation outperforms IQL by 2-4x across all domains.**

### Data Source Verification

| Domain | Source | Verification |
|--------|--------|--------------|
| 📈 Crypto | Bybit Exchange | ✅ Real exchange data with funding rates, OI, basis |
| 📊 Commodities | fred.stlouisfed.org | ✅ US Government official data (captured -$36.98 oil on 2020-04-20) |
| 🌤️ Weather | Open-Meteo API | ✅ ERA5 reanalysis + weather stations |
| ☀️ Solar | Open-Meteo Solar | ✅ CAMS satellite-derived irradiance |

---

## 🏗️ Architecture

```
emergent_specialization/
├── 📁 src/                           # Core implementation
│   ├── domains/                      # ⭐ Multi-domain validation
│   │   ├── crypto.py                 # Bybit real data
│   │   ├── commodities.py            # FRED real data
│   │   ├── weather.py                # Open-Meteo real data
│   │   └── solar.py                  # Open-Meteo solar data
│   ├── agents/                       # Agent implementations
│   │   ├── niche_population.py       # ⭐ Core: Competitive exclusion
│   │   └── inventory_v2.py           # Prediction methods
│   └── baselines/                    # Comparison baselines
│       ├── marl_baselines.py         # IQL, QMIX, MAPPO
│       └── oracle.py                 # Perfect regime knowledge
├── 📁 experiments/                   # Experiment scripts
│   ├── exp_real_data_v2.py           # ⭐ Main 4-domain experiment
│   └── exp_marl_comparison.py        # ⭐ MARL baseline comparison
├── 📁 data/                          # Real-world datasets
│   ├── bybit/                        # Crypto exchange data
│   ├── commodities/                  # FRED commodity prices
│   ├── weather/                      # Open-Meteo weather
│   └── solar/                        # Open-Meteo solar
├── 📁 results/                       # Experiment outputs
│   └── figures/                      # Publication figures
├── 📁 paper/                         # NeurIPS paper
│   ├── propositions.tex              # 3 theoretical propositions
│   └── limitations.tex               # Limitations section
└── 📁 scripts/                       # Data download utilities
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/HowardLiYH/Emergent-Specialization-in-Multi-Agent-Systems.git
cd Emergent-Specialization-in-Multi-Agent-Systems

# Create conda environment
conda create -n emergent python=3.10
conda activate emergent

# Install dependencies
pip install -e .
```

### Download Real Data

```bash
# Weather (Open-Meteo - no API key needed)
python scripts/download_real_weather.py

# Solar (Open-Meteo - no API key needed)
python scripts/download_real_solar.py

# Commodities (FRED - no API key needed)
python scripts/download_fred_commodities_real.py
```

### Run Experiments

```bash
# Main experiment on all 4 real domains
python experiments/exp_real_data_v2.py

# MARL baseline comparison
python experiments/exp_marl_comparison.py

# Generate publication figures
python scripts/generate_real_data_figures.py
```

---

## 🔬 Theoretical Foundation

### Three Propositions

**Proposition 1: Competitive Exclusion**
> In a multi-agent system with competition intensity c > 0, complete competitors cannot coexist. Agents must specialize to survive.

**Proposition 2: SI Lower Bound**
> With niche bonus λ > 0 and k regimes: E[SI] ≥ λ/(1+λ) · (1 - 1/k)

**Proposition 3: Mono-Regime Collapse**
> When k=1 regime, E[SI] → 0 regardless of competition intensity.

See `paper/propositions.tex` for full proofs.

---

## 📊 Figures

Five publication-quality figures in `results/figures/`:

1. **fig1_cross_domain_si.pdf** - Cross-domain SI comparison
2. **fig2_marl_comparison.pdf** - MARL baseline comparison
3. **fig3_improvement_scatter.pdf** - SI vs improvement correlation
4. **fig4_regime_distribution.pdf** - Regime distributions by domain
5. **fig5_summary_heatmap.pdf** - Summary heatmap

---

## 📋 Changelog

### v2.0.0 (2024-12-23) - Real Data Validation ⭐

**Major Update: All experiments now use 100% verified real data**

- ✅ **4 Real Data Domains**: Crypto, Commodities, Weather, Solar
- ✅ **175K+ real records** across all domains
- ✅ **MARL Comparison**: NichePopulation beats IQL by 2-4x
- ✅ **5 Publication Figures** generated
- ✅ **3 Theoretical Propositions** with proof sketches
- ✅ **Limitations Section** for honest assessment

### v1.7.0 (2024-12-22) - Unified Prediction & Mechanistic Analysis
- 📊 Unified prediction experiment across domains
- 🔬 Mechanistic analysis: why specialization works
- ⚡ Computational benchmarks: 2-4× faster than MARL

### v1.6.0 (2024-12-22) - Multi-Domain Validation
- 🚕 NYC Taxi (Traffic): SI = 0.73
- ⚡ EIA Energy: SI = 0.88
- 📈 Bybit Finance: SI = 0.86

---

## 🔬 Reproducibility

| Setting | Value |
|---------|-------|
| Random Seeds | 0-29 (30 trials per experiment) |
| Statistical Tests | Bonferroni-corrected (α = 0.05/k) |
| Confidence Intervals | 95% Bootstrap CI |
| Effect Sizes | Cohen's d reported |

**All data sources are free and publicly accessible without API keys.**

---

## 📚 Citation

```bibtex
@inproceedings{emergent_specialization_2025,
  title     = {Emergent Specialization in Multi-Agent Systems:
               Niche Partitioning Without Explicit Coordination},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  note      = {Under review}
}
```

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

<div align="center">

**⭐ Star this repo if you find it useful!**

[Report Bug](https://github.com/HowardLiYH/Emergent-Specialization-in-Multi-Agent-Systems/issues) • [Request Feature](https://github.com/HowardLiYH/Emergent-Specialization-in-Multi-Agent-Systems/issues)

</div>
