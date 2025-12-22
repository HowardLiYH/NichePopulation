# Emergent Specialization in Multi-Agent Systems

## Key Results (All Real Data)

### Cross-Domain Validation (4 Domains)

| Domain | Source | Records | Regimes | Mean SI | vs Random |
|--------|--------|---------|---------|---------|-----------|
| **Crypto** | Bybit Exchange | 8,766 | 4 | 0.305±0.042 | +67.0% |
| **Commodities** | FRED (US Gov) | 5,630 | 4 | 0.411±0.062 | +119.4% |
| **Weather** | Open-Meteo | 9,105 | 5 | 0.205±0.026 | +6.4% |
| **Solar** | Open-Meteo | 116,834 | 4 | 0.443±0.036 | +96.4% |

**All data verified REAL from authoritative sources.**

### MARL Baseline Comparison

| Domain | Niche Population (Ours) | IQL | Random |
|--------|-------------------------|-----|--------|
| Crypto | **0.300±0.038** | 0.097±0.023 | 0.226±0.052 |
| Commodities | **0.403±0.053** | 0.088±0.021 | 0.226±0.052 |
| Weather | **0.201±0.021** | 0.102±0.023 | 0.201±0.046 |
| Solar | **0.450±0.037** | 0.113±0.026 | 0.226±0.052 |

**NichePopulation consistently outperforms IQL by 2-4x across all domains.**

### Data Sources (All Verified Real)

1. **Crypto**: Bybit Exchange historical OHLCV with funding rates, OI, basis
2. **Commodities**: FRED (fred.stlouisfed.org) - WTI Oil, Copper, Natural Gas
3. **Weather**: Open-Meteo Historical API - 5 US cities, ERA5 reanalysis
4. **Solar**: Open-Meteo Solar API - 5 US locations, CAMS satellite data

### Figures Generated

- `fig1_cross_domain_si.pdf` - Cross-domain SI comparison
- `fig2_marl_comparison.pdf` - MARL baseline comparison
- `fig3_improvement_scatter.pdf` - SI vs improvement
- `fig4_regime_distribution.pdf` - Regime distributions
- `fig5_summary_heatmap.pdf` - Summary heatmap

### Reproducibility

All experiments can be reproduced:

```bash
# Download real data
python scripts/download_real_weather.py
python scripts/download_real_solar.py
python scripts/download_fred_commodities_real.py

# Run experiments
python experiments/exp_real_data_v2.py
python experiments/exp_marl_comparison.py

# Generate figures
python scripts/generate_real_data_figures.py
```
