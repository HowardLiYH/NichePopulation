"""
Energy Domain - Energy grid management with demand regimes.

This is a stub that re-exports from synthetic_domains for cleaner imports.
"""

from .synthetic_domains import (
    create_energy_environment,
    EnergyMethod,
    ENERGY_REGIMES,
    ENERGY_METHODS,
)


class EnergyDomain:
    """Wrapper for energy domain environment."""

    def __init__(self, n_bars: int = 2000, seed: int = None):
        self.df, self.regimes, self.methods = create_energy_environment(
            n_bars=n_bars, seed=seed
        )

    @property
    def regime_names(self):
        return ENERGY_REGIMES

    @property
    def method_names(self):
        return ENERGY_METHODS
