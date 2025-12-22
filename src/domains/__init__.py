"""
Multi-Domain Validation Framework.

This module provides environments and methods for testing emergent
specialization across multiple domains beyond financial trading.

Supported Domains:
1. Traffic - Traffic flow optimization with congestion regimes
2. Energy - Energy grid management with demand regimes
3. Weather - Weather prediction with meteorological regimes
4. E-commerce - Inventory management with demand regimes
5. Sports - Team strategy with game-state regimes

Each domain has:
- Environment with regime structure
- Domain-specific methods (strategies)
- Reward function
"""

from .base import DomainEnvironment, DomainMethod, DomainConfig
from .traffic import TrafficDomain
from .energy import EnergyDomain
from .synthetic_domains import (
    create_traffic_environment,
    create_energy_environment,
    create_weather_environment,
    create_ecommerce_environment,
    create_sports_environment,
)

__all__ = [
    "DomainEnvironment",
    "DomainMethod",
    "DomainConfig",
    "TrafficDomain",
    "EnergyDomain",
    "create_traffic_environment",
    "create_energy_environment",
    "create_weather_environment",
    "create_ecommerce_environment",
    "create_sports_environment",
]
