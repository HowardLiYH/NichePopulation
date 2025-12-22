"""
Traffic Domain - Traffic flow optimization with congestion regimes.

This is a stub that re-exports from synthetic_domains for cleaner imports.
"""

from .synthetic_domains import (
    create_traffic_environment,
    TrafficMethod,
    TRAFFIC_REGIMES,
    TRAFFIC_METHODS,
)


class TrafficDomain:
    """Wrapper for traffic domain environment."""

    def __init__(self, n_bars: int = 2000, seed: int = None):
        self.df, self.regimes, self.methods = create_traffic_environment(
            n_bars=n_bars, seed=seed
        )

    @property
    def regime_names(self):
        return TRAFFIC_REGIMES

    @property
    def method_names(self):
        return TRAFFIC_METHODS
