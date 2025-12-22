"""
Base classes for multi-domain validation.

This provides a common interface for all domain environments,
enabling consistent experiment infrastructure across domains.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class DomainConfig:
    """Configuration for a domain environment."""
    name: str
    regimes: List[str]
    methods: List[str]
    n_bars: int = 2000
    regime_duration_mean: int = 100
    regime_duration_std: int = 30
    seed: Optional[int] = None


class DomainMethod(ABC):
    """Base class for domain-specific methods."""

    name: str
    optimal_regimes: List[str]

    @abstractmethod
    def execute(self, observation: np.ndarray) -> Dict:
        """
        Execute method on current observation.

        Args:
            observation: Current state observation

        Returns:
            Dict with 'signal' (-1 to 1) and 'confidence' (0 to 1)
        """
        pass


class DomainEnvironment(ABC):
    """Base class for domain environments."""

    def __init__(self, config: DomainConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.regimes = config.regimes
        self.methods: Dict[str, DomainMethod] = {}

    @abstractmethod
    def generate(self, n_bars: Optional[int] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate synthetic data for the domain.

        Returns:
            Tuple of (observations DataFrame, regime Series)
        """
        pass

    @abstractmethod
    def compute_reward(self, method_name: str, observation: np.ndarray, regime: str) -> float:
        """
        Compute reward for a method in current state.

        Args:
            method_name: Name of method
            observation: Current observation
            regime: Current regime

        Returns:
            Reward value
        """
        pass

    def get_regime_names(self) -> List[str]:
        """Get list of regime names."""
        return self.regimes

    def get_method_names(self) -> List[str]:
        """Get list of method names."""
        return list(self.methods.keys())


def create_regime_sequence(
    n_bars: int,
    regimes: List[str],
    duration_mean: int,
    duration_std: int,
    rng: np.random.Generator,
) -> pd.Series:
    """
    Create a sequence of regime labels.

    Args:
        n_bars: Total number of time steps
        regimes: List of regime names
        duration_mean: Mean regime duration
        duration_std: Std of regime duration
        rng: Random number generator

    Returns:
        Series of regime labels
    """
    regime_sequence = []
    current_idx = 0

    while len(regime_sequence) < n_bars:
        # Pick a regime
        regime = rng.choice(regimes)

        # Sample duration
        duration = int(rng.normal(duration_mean, duration_std))
        duration = max(10, min(duration, n_bars - len(regime_sequence)))

        regime_sequence.extend([regime] * duration)

    return pd.Series(regime_sequence[:n_bars])
