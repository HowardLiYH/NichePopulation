"""
Synthetic Domain Generators.

Creates synthetic environments for multiple domains that share
the same regime-based structure as our financial environment.

Each domain has:
- Distinct regimes that require different strategies
- Methods with varying effectiveness per regime
- Observable state that agents can use for decisions

This enables testing whether emergent specialization generalizes
beyond financial trading.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from .base import DomainConfig, DomainEnvironment, DomainMethod, create_regime_sequence


# =============================================================================
# TRAFFIC DOMAIN
# =============================================================================

TRAFFIC_REGIMES = ["free_flow", "congested", "incident", "peak_hour"]
TRAFFIC_METHODS = [
    "RampMetering", "SignalTiming", "SpeedLimit", "LaneAssignment",
    "RouteGuidance", "DynamicPricing", "EmergencyResponse", "PublicTransit"
]

class TrafficMethod:
    """Traffic management method."""

    def __init__(self, name: str, optimal_regimes: List[str]):
        self.name = name
        self.optimal_regimes = optimal_regimes

    def execute(self, observation: np.ndarray) -> Dict:
        """Generate signal based on traffic state."""
        # Simplified: use observation mean as proxy for congestion
        congestion = observation.mean() if len(observation) > 0 else 0.5

        if self.name == "RampMetering":
            signal = 1.0 if congestion > 0.6 else -0.5
        elif self.name == "SignalTiming":
            signal = 0.5 if 0.3 < congestion < 0.7 else -0.3
        elif self.name == "SpeedLimit":
            signal = -1.0 if congestion > 0.8 else 0.3
        elif self.name == "EmergencyResponse":
            signal = 1.0 if congestion > 0.9 else -0.8
        else:
            signal = np.sin(congestion * np.pi)

        return {"signal": signal, "confidence": abs(signal)}


def create_traffic_environment(
    n_bars: int = 2000,
    regime_duration_mean: int = 100,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, TrafficMethod]]:
    """
    Create synthetic traffic environment.

    State: [flow_rate, density, speed, occupancy]

    Regimes:
    - free_flow: Low density, high speed
    - congested: High density, low speed
    - incident: Sudden drop in capacity
    - peak_hour: Predictable high demand
    """
    rng = np.random.default_rng(seed)

    # Generate regime sequence
    regimes = create_regime_sequence(
        n_bars, TRAFFIC_REGIMES, regime_duration_mean, 30, rng
    )

    # Generate state based on regime
    data = []
    for regime in regimes:
        if regime == "free_flow":
            flow = rng.normal(0.3, 0.1)
            density = rng.normal(0.2, 0.05)
            speed = rng.normal(0.8, 0.1)
        elif regime == "congested":
            flow = rng.normal(0.7, 0.1)
            density = rng.normal(0.8, 0.1)
            speed = rng.normal(0.3, 0.1)
        elif regime == "incident":
            flow = rng.normal(0.9, 0.1)
            density = rng.normal(0.9, 0.1)
            speed = rng.normal(0.1, 0.05)
        else:  # peak_hour
            flow = rng.normal(0.6, 0.15)
            density = rng.normal(0.6, 0.1)
            speed = rng.normal(0.5, 0.1)

        data.append([
            np.clip(flow, 0, 1),
            np.clip(density, 0, 1),
            np.clip(speed, 0, 1),
            np.clip((density + flow) / 2, 0, 1),
        ])

    df = pd.DataFrame(data, columns=["flow", "density", "speed", "occupancy"])

    # Create methods
    methods = {
        "RampMetering": TrafficMethod("RampMetering", ["congested", "peak_hour"]),
        "SignalTiming": TrafficMethod("SignalTiming", ["free_flow", "peak_hour"]),
        "SpeedLimit": TrafficMethod("SpeedLimit", ["congested", "incident"]),
        "LaneAssignment": TrafficMethod("LaneAssignment", ["peak_hour"]),
        "RouteGuidance": TrafficMethod("RouteGuidance", ["congested", "incident"]),
        "DynamicPricing": TrafficMethod("DynamicPricing", ["peak_hour"]),
        "EmergencyResponse": TrafficMethod("EmergencyResponse", ["incident"]),
        "PublicTransit": TrafficMethod("PublicTransit", ["peak_hour", "congested"]),
    }

    return df, regimes, methods


# =============================================================================
# ENERGY DOMAIN
# =============================================================================

ENERGY_REGIMES = ["low_demand", "peak_demand", "renewable_surplus", "grid_stress"]
ENERGY_METHODS = [
    "LoadShifting", "PeakShaving", "BatteryStorage", "DemandResponse",
    "RenewableIntegration", "GridStabilization", "EmergencyReserve", "PriceOptimization"
]

class EnergyMethod:
    """Energy management method."""

    def __init__(self, name: str, optimal_regimes: List[str]):
        self.name = name
        self.optimal_regimes = optimal_regimes

    def execute(self, observation: np.ndarray) -> Dict:
        demand = observation[0] if len(observation) > 0 else 0.5
        renewable = observation[1] if len(observation) > 1 else 0.5

        if self.name == "PeakShaving":
            signal = 1.0 if demand > 0.7 else -0.5
        elif self.name == "BatteryStorage":
            signal = 1.0 if renewable > 0.6 and demand < 0.5 else -0.3
        elif self.name == "RenewableIntegration":
            signal = renewable - 0.5
        elif self.name == "EmergencyReserve":
            signal = 1.0 if demand > 0.9 else -0.8
        else:
            signal = np.tanh(demand - renewable)

        return {"signal": signal, "confidence": abs(signal)}


def create_energy_environment(
    n_bars: int = 2000,
    regime_duration_mean: int = 100,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, EnergyMethod]]:
    """
    Create synthetic energy grid environment.

    State: [demand, renewable_output, grid_frequency, price]
    """
    rng = np.random.default_rng(seed)

    regimes = create_regime_sequence(
        n_bars, ENERGY_REGIMES, regime_duration_mean, 30, rng
    )

    data = []
    for regime in regimes:
        if regime == "low_demand":
            demand = rng.normal(0.3, 0.1)
            renewable = rng.normal(0.4, 0.1)
        elif regime == "peak_demand":
            demand = rng.normal(0.9, 0.1)
            renewable = rng.normal(0.3, 0.1)
        elif regime == "renewable_surplus":
            demand = rng.normal(0.4, 0.1)
            renewable = rng.normal(0.9, 0.1)
        else:  # grid_stress
            demand = rng.normal(0.95, 0.05)
            renewable = rng.normal(0.2, 0.1)

        price = np.clip(demand - renewable + 0.5, 0, 1)
        frequency = np.clip(1 - abs(demand - renewable), 0, 1)

        data.append([
            np.clip(demand, 0, 1),
            np.clip(renewable, 0, 1),
            frequency,
            price,
        ])

    df = pd.DataFrame(data, columns=["demand", "renewable", "frequency", "price"])

    methods = {
        "LoadShifting": EnergyMethod("LoadShifting", ["peak_demand", "low_demand"]),
        "PeakShaving": EnergyMethod("PeakShaving", ["peak_demand"]),
        "BatteryStorage": EnergyMethod("BatteryStorage", ["renewable_surplus", "low_demand"]),
        "DemandResponse": EnergyMethod("DemandResponse", ["peak_demand", "grid_stress"]),
        "RenewableIntegration": EnergyMethod("RenewableIntegration", ["renewable_surplus"]),
        "GridStabilization": EnergyMethod("GridStabilization", ["grid_stress"]),
        "EmergencyReserve": EnergyMethod("EmergencyReserve", ["grid_stress"]),
        "PriceOptimization": EnergyMethod("PriceOptimization", ["low_demand", "renewable_surplus"]),
    }

    return df, regimes, methods


# =============================================================================
# WEATHER DOMAIN
# =============================================================================

WEATHER_REGIMES = ["stable", "approaching_storm", "active_storm", "clearing"]
WEATHER_METHODS = [
    "PersistenceModel", "NWPModel", "EnsembleBlend", "NowcastingRadar",
    "StatisticalDownscaling", "MachineLearning", "ExtremeEventAlert", "ClimateAdjust"
]

def create_weather_environment(
    n_bars: int = 2000,
    regime_duration_mean: int = 100,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    Create synthetic weather prediction environment.

    State: [pressure, temperature, humidity, wind_speed]
    """
    rng = np.random.default_rng(seed)

    regimes = create_regime_sequence(
        n_bars, WEATHER_REGIMES, regime_duration_mean, 30, rng
    )

    data = []
    for regime in regimes:
        if regime == "stable":
            pressure = rng.normal(0.6, 0.05)
            temperature = rng.normal(0.5, 0.1)
            humidity = rng.normal(0.4, 0.1)
            wind = rng.normal(0.2, 0.1)
        elif regime == "approaching_storm":
            pressure = rng.normal(0.4, 0.1)
            temperature = rng.normal(0.6, 0.1)
            humidity = rng.normal(0.7, 0.1)
            wind = rng.normal(0.5, 0.1)
        elif regime == "active_storm":
            pressure = rng.normal(0.2, 0.1)
            temperature = rng.normal(0.4, 0.1)
            humidity = rng.normal(0.9, 0.05)
            wind = rng.normal(0.9, 0.1)
        else:  # clearing
            pressure = rng.normal(0.7, 0.1)
            temperature = rng.normal(0.5, 0.1)
            humidity = rng.normal(0.5, 0.1)
            wind = rng.normal(0.4, 0.1)

        data.append([
            np.clip(pressure, 0, 1),
            np.clip(temperature, 0, 1),
            np.clip(humidity, 0, 1),
            np.clip(wind, 0, 1),
        ])

    df = pd.DataFrame(data, columns=["pressure", "temperature", "humidity", "wind_speed"])

    methods = {
        "PersistenceModel": {"optimal_regimes": ["stable"]},
        "NWPModel": {"optimal_regimes": ["approaching_storm", "clearing"]},
        "EnsembleBlend": {"optimal_regimes": ["approaching_storm", "active_storm"]},
        "NowcastingRadar": {"optimal_regimes": ["active_storm"]},
        "StatisticalDownscaling": {"optimal_regimes": ["stable", "clearing"]},
        "MachineLearning": {"optimal_regimes": ["approaching_storm", "active_storm"]},
        "ExtremeEventAlert": {"optimal_regimes": ["active_storm"]},
        "ClimateAdjust": {"optimal_regimes": ["stable", "clearing"]},
    }

    return df, regimes, methods


# =============================================================================
# E-COMMERCE DOMAIN
# =============================================================================

ECOMMERCE_REGIMES = ["normal", "sale_event", "stock_shortage", "demand_surge"]
ECOMMERCE_METHODS = [
    "BaselineForecasting", "PromotionAware", "SeasonalAdjust", "SafetyStock",
    "JustInTime", "DynamicReorder", "EmergencyReplenish", "DemandShaping"
]

def create_ecommerce_environment(
    n_bars: int = 2000,
    regime_duration_mean: int = 100,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    Create synthetic e-commerce inventory environment.

    State: [demand_level, inventory_level, lead_time, competitor_price]
    """
    rng = np.random.default_rng(seed)

    regimes = create_regime_sequence(
        n_bars, ECOMMERCE_REGIMES, regime_duration_mean, 30, rng
    )

    data = []
    for regime in regimes:
        if regime == "normal":
            demand = rng.normal(0.5, 0.1)
            inventory = rng.normal(0.6, 0.1)
        elif regime == "sale_event":
            demand = rng.normal(0.9, 0.1)
            inventory = rng.normal(0.4, 0.1)
        elif regime == "stock_shortage":
            demand = rng.normal(0.6, 0.1)
            inventory = rng.normal(0.1, 0.05)
        else:  # demand_surge
            demand = rng.normal(0.95, 0.05)
            inventory = rng.normal(0.3, 0.1)

        lead_time = rng.normal(0.5, 0.2)
        competitor_price = rng.normal(0.5, 0.1)

        data.append([
            np.clip(demand, 0, 1),
            np.clip(inventory, 0, 1),
            np.clip(lead_time, 0, 1),
            np.clip(competitor_price, 0, 1),
        ])

    df = pd.DataFrame(data, columns=["demand", "inventory", "lead_time", "competitor_price"])

    methods = {
        "BaselineForecasting": {"optimal_regimes": ["normal"]},
        "PromotionAware": {"optimal_regimes": ["sale_event"]},
        "SeasonalAdjust": {"optimal_regimes": ["normal", "demand_surge"]},
        "SafetyStock": {"optimal_regimes": ["stock_shortage", "demand_surge"]},
        "JustInTime": {"optimal_regimes": ["normal"]},
        "DynamicReorder": {"optimal_regimes": ["demand_surge", "sale_event"]},
        "EmergencyReplenish": {"optimal_regimes": ["stock_shortage"]},
        "DemandShaping": {"optimal_regimes": ["sale_event", "demand_surge"]},
    }

    return df, regimes, methods


# =============================================================================
# SPORTS DOMAIN
# =============================================================================

SPORTS_REGIMES = ["leading", "trailing", "tied", "crunch_time"]
SPORTS_METHODS = [
    "Conservative", "Aggressive", "Balanced", "PossessionControl",
    "FastBreak", "DefensiveFocus", "StarPlayer", "RotationManagement"
]

def create_sports_environment(
    n_bars: int = 2000,
    regime_duration_mean: int = 100,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    Create synthetic sports strategy environment.

    State: [score_differential, time_remaining, possession, fatigue]
    """
    rng = np.random.default_rng(seed)

    regimes = create_regime_sequence(
        n_bars, SPORTS_REGIMES, regime_duration_mean, 30, rng
    )

    data = []
    for regime in regimes:
        if regime == "leading":
            score_diff = rng.normal(0.7, 0.1)
            time_remaining = rng.uniform(0.2, 0.8)
        elif regime == "trailing":
            score_diff = rng.normal(0.3, 0.1)
            time_remaining = rng.uniform(0.2, 0.8)
        elif regime == "tied":
            score_diff = rng.normal(0.5, 0.05)
            time_remaining = rng.uniform(0.3, 0.9)
        else:  # crunch_time
            score_diff = rng.normal(0.5, 0.15)
            time_remaining = rng.uniform(0, 0.15)

        possession = rng.uniform(0.4, 0.6)
        fatigue = rng.normal(0.5, 0.15)

        data.append([
            np.clip(score_diff, 0, 1),
            np.clip(time_remaining, 0, 1),
            np.clip(possession, 0, 1),
            np.clip(fatigue, 0, 1),
        ])

    df = pd.DataFrame(data, columns=["score_diff", "time_remaining", "possession", "fatigue"])

    methods = {
        "Conservative": {"optimal_regimes": ["leading"]},
        "Aggressive": {"optimal_regimes": ["trailing", "crunch_time"]},
        "Balanced": {"optimal_regimes": ["tied"]},
        "PossessionControl": {"optimal_regimes": ["leading", "tied"]},
        "FastBreak": {"optimal_regimes": ["trailing"]},
        "DefensiveFocus": {"optimal_regimes": ["leading", "crunch_time"]},
        "StarPlayer": {"optimal_regimes": ["crunch_time", "trailing"]},
        "RotationManagement": {"optimal_regimes": ["leading", "tied"]},
    }

    return df, regimes, methods


# =============================================================================
# UNIFIED DOMAIN INTERFACE
# =============================================================================

DOMAIN_CREATORS = {
    "finance": None,  # Already have financial environment
    "traffic": create_traffic_environment,
    "energy": create_energy_environment,
    "weather": create_weather_environment,
    "ecommerce": create_ecommerce_environment,
    "sports": create_sports_environment,
}

def get_domain_environment(domain: str, **kwargs):
    """Get environment for a specific domain."""
    if domain not in DOMAIN_CREATORS:
        raise ValueError(f"Unknown domain: {domain}. Available: {list(DOMAIN_CREATORS.keys())}")

    creator = DOMAIN_CREATORS[domain]
    if creator is None:
        raise ValueError(f"Domain {domain} uses main financial environment")

    return creator(**kwargs)
