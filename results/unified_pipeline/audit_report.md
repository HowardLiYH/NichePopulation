# Experimental Pipeline Audit Report

Generated: 2025-12-23T08:41:32.647219

## Configuration

- **Domains**: 6
- **Trials per experiment**: 30
- **Iterations per trial**: 500
- **Agents**: 8
- **Lambda values tested**: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

## Experiments Run

| Experiment | All 6 Domains | Same Trials | Same Config |
|------------|---------------|-------------|-------------|
| NichePopulation SI | ✅ | ✅ 30 | ✅ |
| Homogeneous Baseline | ✅ | ✅ 30 | ✅ |
| Random Baseline | ✅ | ✅ 30 | ✅ |
| Lambda Ablation | ✅ | ✅ 30×6 | ✅ |
| Task Performance | ✅ | ✅ 30 | ✅ |
| Statistical Tests | ✅ | ✅ | ✅ |

## Summary

All experiments run with identical configuration across all 6 domains.
