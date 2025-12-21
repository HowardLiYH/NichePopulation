# One-Page Pitch: Emergent Specialization in Multi-Agent Trading

## The Problem
Financial markets are non-stationary—they switch between trending, mean-reverting, and volatile regimes. Traditional trading systems either use fixed strategies (fail on regime change) or attempt explicit regime detection (difficult and lagged).

## Our Key Insight
**When a population of agents competes for trading profits using shared methods, they naturally specialize to different market regimes—without any supervision.**

This is analogous to niche partitioning in ecology: species evolve to exploit different resources to avoid competition.

## Why It's Novel
- **Not Algorithm Selection**: We study population dynamics, not single-agent selection
- **Not Population-Based Training**: We study method selection, not hyperparameter tuning
- **Not Emergent Communication**: We study emergent specialization

## Theoretical Grounding
- **Evolutionary Stable Strategies (ESS)**: Maynard Smith & Price (1973)
- **Niche Partitioning Theory**: Hutchinson (1957)

## Experiments
1. **Emergence**: Do agents specialize? (Track specialization index over training)
2. **Value**: Does specialization help? (Compare with homogeneous populations)
3. **Population Size**: What's optimal? (Expected: inverted-U curve)
4. **Transfer Frequency**: How does knowledge sharing affect diversity?
5. **Regime Transitions**: Smooth "handoff" between specialists?
6. **Real Data**: Validate on crypto markets

## Why NeurIPS Will Accept This
| ✅ Novelty | First study of emergent specialization in trading |
|-----------|--------------------------------------------------|
| ✅ Theory | ESS grounding, formal problem statement |
| ✅ Rigor | 1000+ synthetic trials, 8 baselines |
| ✅ Reproducibility | No APIs, synthetic data, code release |
| ✅ Significance | Applies to any multi-agent method selection |

## Timeline
- **Jan-May 2025**: Experiments
- **Jun-Aug 2025**: Paper draft + internal review
- **Sep 2025**: Workshop submission (backup)
- **May 2026**: NeurIPS main conference submission

## One-Sentence Summary
> We show that trading agent populations naturally specialize to different market regimes through competition, and this emergent division of labor improves portfolio performance.

---

*Ready for discussion. Questions welcome.*
