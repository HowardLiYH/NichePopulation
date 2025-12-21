"""
Method Inventory: Shared methods available to all agents.

Each method represents a trading strategy or decision rule.
Methods are categorized by the type of market regime they excel in.

This is the "niche space" in evolutionary terms - agents must
partition these methods to specialize.
"""

from dataclasses import dataclass
from typing import Dict, List, Callable, Optional
from enum import Enum
import numpy as np
import pandas as pd


class MethodCategory(Enum):
    """Category of trading method."""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY = "volatility"
    RISK_MANAGEMENT = "risk_management"
    NEUTRAL = "neutral"


@dataclass
class TradingMethod:
    """Definition of a trading method."""
    name: str
    category: MethodCategory
    description: str
    optimal_regimes: List[str]  # Regimes where this method excels

    def execute(
        self,
        prices: pd.DataFrame,
        position: float = 0.0,
    ) -> Dict:
        """
        Execute the method and return trading signal.

        Returns:
            dict with keys: signal (-1 to 1), confidence (0 to 1)
        """
        # Default implementation - override in specific methods
        return {"signal": 0.0, "confidence": 0.5}


# ============================================================================
# TREND-FOLLOWING METHODS
# ============================================================================

class MomentumMethod(TradingMethod):
    """
    Momentum: Buy recent winners, sell recent losers.

    Reference: Jegadeesh, N. & Titman, S. (1993). "Returns to Buying
              Winners and Selling Losers." Journal of Finance.
    """

    def __init__(self, lookback: int = 20):
        super().__init__(
            name="Momentum",
            category=MethodCategory.TREND_FOLLOWING,
            description="Buy recent winners, sell recent losers",
            optimal_regimes=["trend_up", "trend_down"],
        )
        self.lookback = lookback

    def execute(self, prices: pd.DataFrame, position: float = 0.0) -> Dict:
        if len(prices) < self.lookback:
            return {"signal": 0.0, "confidence": 0.3}

        returns = prices["close"].pct_change(self.lookback).iloc[-1]
        signal = np.clip(returns * 10, -1, 1)  # Scale to [-1, 1]
        confidence = min(0.9, 0.5 + abs(returns) * 5)

        return {"signal": signal, "confidence": confidence}


class BreakoutMethod(TradingMethod):
    """
    Breakout: Enter when price breaks above/below recent range.
    """

    def __init__(self, lookback: int = 20):
        super().__init__(
            name="Breakout",
            category=MethodCategory.TREND_FOLLOWING,
            description="Trade breakouts from price ranges",
            optimal_regimes=["trend_up", "trend_down"],
        )
        self.lookback = lookback

    def execute(self, prices: pd.DataFrame, position: float = 0.0) -> Dict:
        if len(prices) < self.lookback:
            return {"signal": 0.0, "confidence": 0.3}

        recent = prices["close"].iloc[-self.lookback:]
        current = prices["close"].iloc[-1]
        high = recent.max()
        low = recent.min()
        range_size = high - low

        if range_size < 1e-8:
            return {"signal": 0.0, "confidence": 0.3}

        # Position within range
        position_in_range = (current - low) / range_size

        if position_in_range > 0.95:  # Near high - bullish breakout
            signal = 0.8
        elif position_in_range < 0.05:  # Near low - bearish breakout
            signal = -0.8
        else:
            signal = 0.0

        confidence = 0.6 if abs(signal) > 0.5 else 0.4
        return {"signal": signal, "confidence": confidence}


class TrendFollowingMethod(TradingMethod):
    """
    Trend Following: Use moving average crossover.
    """

    def __init__(self, fast: int = 10, slow: int = 30):
        super().__init__(
            name="TrendFollowing",
            category=MethodCategory.TREND_FOLLOWING,
            description="MA crossover trend following",
            optimal_regimes=["trend_up", "trend_down"],
        )
        self.fast = fast
        self.slow = slow

    def execute(self, prices: pd.DataFrame, position: float = 0.0) -> Dict:
        if len(prices) < self.slow:
            return {"signal": 0.0, "confidence": 0.3}

        close = prices["close"]
        fast_ma = close.rolling(self.fast).mean().iloc[-1]
        slow_ma = close.rolling(self.slow).mean().iloc[-1]

        diff = (fast_ma - slow_ma) / slow_ma
        signal = np.clip(diff * 20, -1, 1)
        confidence = min(0.8, 0.5 + abs(diff) * 10)

        return {"signal": signal, "confidence": confidence}


# ============================================================================
# MEAN REVERSION METHODS
# ============================================================================

class MeanReversionMethod(TradingMethod):
    """
    Mean Reversion: Bet on price returning to mean.
    """

    def __init__(self, lookback: int = 20, threshold: float = 2.0):
        super().__init__(
            name="MeanReversion",
            category=MethodCategory.MEAN_REVERSION,
            description="Trade deviations from moving average",
            optimal_regimes=["mean_revert"],
        )
        self.lookback = lookback
        self.threshold = threshold

    def execute(self, prices: pd.DataFrame, position: float = 0.0) -> Dict:
        if len(prices) < self.lookback:
            return {"signal": 0.0, "confidence": 0.3}

        close = prices["close"]
        ma = close.rolling(self.lookback).mean().iloc[-1]
        std = close.rolling(self.lookback).std().iloc[-1]

        if std < 1e-8:
            return {"signal": 0.0, "confidence": 0.3}

        z_score = (close.iloc[-1] - ma) / std

        # Mean revert: sell when high, buy when low
        signal = np.clip(-z_score / self.threshold, -1, 1)
        confidence = min(0.8, 0.4 + abs(z_score) * 0.2)

        return {"signal": signal, "confidence": confidence}


class RSIMethod(TradingMethod):
    """
    RSI: Relative Strength Index.
    """

    def __init__(self, period: int = 14):
        super().__init__(
            name="RSI",
            category=MethodCategory.MEAN_REVERSION,
            description="Trade overbought/oversold conditions",
            optimal_regimes=["mean_revert"],
        )
        self.period = period

    def execute(self, prices: pd.DataFrame, position: float = 0.0) -> Dict:
        if len(prices) < self.period + 1:
            return {"signal": 0.0, "confidence": 0.3}

        delta = prices["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(self.period).mean().iloc[-1]
        loss = (-delta.where(delta < 0, 0)).rolling(self.period).mean().iloc[-1]

        if loss < 1e-8:
            rsi = 100
        else:
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

        # Overbought (>70) = sell, Oversold (<30) = buy
        if rsi > 70:
            signal = -0.8
        elif rsi < 30:
            signal = 0.8
        else:
            signal = 0.0

        confidence = 0.6 if abs(signal) > 0.5 else 0.4
        return {"signal": signal, "confidence": confidence}


class BollingerBandsMethod(TradingMethod):
    """
    Bollinger Bands: Trade touches of bands.
    """

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__(
            name="BollingerBands",
            category=MethodCategory.MEAN_REVERSION,
            description="Trade Bollinger Band touches",
            optimal_regimes=["mean_revert"],
        )
        self.period = period
        self.std_dev = std_dev

    def execute(self, prices: pd.DataFrame, position: float = 0.0) -> Dict:
        if len(prices) < self.period:
            return {"signal": 0.0, "confidence": 0.3}

        close = prices["close"]
        ma = close.rolling(self.period).mean().iloc[-1]
        std = close.rolling(self.period).std().iloc[-1]

        upper = ma + self.std_dev * std
        lower = ma - self.std_dev * std
        current = close.iloc[-1]

        if current > upper:
            signal = -0.7  # Overbought
        elif current < lower:
            signal = 0.7   # Oversold
        else:
            signal = 0.0

        confidence = 0.6 if abs(signal) > 0.5 else 0.4
        return {"signal": signal, "confidence": confidence}


# ============================================================================
# RISK MANAGEMENT / VOLATILITY METHODS
# ============================================================================

class StayFlatMethod(TradingMethod):
    """
    Stay Flat: Don't trade (capital preservation).
    """

    def __init__(self):
        super().__init__(
            name="StayFlat",
            category=MethodCategory.RISK_MANAGEMENT,
            description="Stay out of market",
            optimal_regimes=["volatile", "sideways"],
        )

    def execute(self, prices: pd.DataFrame, position: float = 0.0) -> Dict:
        return {"signal": 0.0, "confidence": 0.9}


class VolatilityScalingMethod(TradingMethod):
    """
    Volatility Scaling: Reduce position size in high vol.
    """

    def __init__(self, target_vol: float = 0.02):
        super().__init__(
            name="VolatilityScaling",
            category=MethodCategory.VOLATILITY,
            description="Scale positions by volatility",
            optimal_regimes=["volatile"],
        )
        self.target_vol = target_vol

    def execute(self, prices: pd.DataFrame, position: float = 0.0) -> Dict:
        if len(prices) < 20:
            return {"signal": 0.0, "confidence": 0.3}

        vol = prices["close"].pct_change().rolling(20).std().iloc[-1]
        scale = min(1.0, self.target_vol / max(vol, 1e-8))

        # Low signal magnitude (scaled down)
        return {"signal": 0.0, "confidence": scale}


class ReduceExposureMethod(TradingMethod):
    """
    Reduce Exposure: Exit positions in uncertainty.
    """

    def __init__(self):
        super().__init__(
            name="ReduceExposure",
            category=MethodCategory.RISK_MANAGEMENT,
            description="Reduce exposure in high uncertainty",
            optimal_regimes=["volatile"],
        )

    def execute(self, prices: pd.DataFrame, position: float = 0.0) -> Dict:
        # If we have a position, signal to reduce it
        if position > 0:
            signal = -0.5
        elif position < 0:
            signal = 0.5
        else:
            signal = 0.0

        return {"signal": signal, "confidence": 0.7}


class RangeTradingMethod(TradingMethod):
    """
    Range Trading: Buy low, sell high within range.
    """

    def __init__(self, lookback: int = 30):
        super().__init__(
            name="RangeTrading",
            category=MethodCategory.NEUTRAL,
            description="Trade within defined range",
            optimal_regimes=["sideways"],
        )
        self.lookback = lookback

    def execute(self, prices: pd.DataFrame, position: float = 0.0) -> Dict:
        if len(prices) < self.lookback:
            return {"signal": 0.0, "confidence": 0.3}

        recent = prices["close"].iloc[-self.lookback:]
        high = recent.max()
        low = recent.min()
        current = prices["close"].iloc[-1]

        range_size = high - low
        if range_size < 1e-8:
            return {"signal": 0.0, "confidence": 0.3}

        position_in_range = (current - low) / range_size

        # Buy at bottom of range, sell at top
        signal = (0.5 - position_in_range) * 1.5
        signal = np.clip(signal, -1, 1)

        confidence = 0.5
        return {"signal": signal, "confidence": confidence}


class WaitForBreakoutMethod(TradingMethod):
    """
    Wait for Breakout: Stay flat until clear signal.
    """

    def __init__(self):
        super().__init__(
            name="WaitForBreakout",
            category=MethodCategory.NEUTRAL,
            description="Wait for clear directional signal",
            optimal_regimes=["sideways"],
        )

    def execute(self, prices: pd.DataFrame, position: float = 0.0) -> Dict:
        return {"signal": 0.0, "confidence": 0.8}


# ============================================================================
# METHOD INVENTORY
# ============================================================================

# Import V2 methods (better regime differentiation)
from .inventory_v2 import METHOD_INVENTORY_V2, get_method_names_v2

# Use V2 inventory as default (better regime differentiation)
METHOD_INVENTORY: Dict[str, TradingMethod] = METHOD_INVENTORY_V2


def get_method_names() -> List[str]:
    """Get list of all method names."""
    return list(METHOD_INVENTORY.keys())


def get_methods_by_category(category: MethodCategory) -> List[str]:
    """Get method names for a given category."""
    return [
        name for name, method in METHOD_INVENTORY.items()
        if method.category == category
    ]


def get_methods_for_regime(regime: str) -> List[str]:
    """Get optimal method names for a given regime."""
    return [
        name for name, method in METHOD_INVENTORY.items()
        if regime in method.optimal_regimes
    ]
