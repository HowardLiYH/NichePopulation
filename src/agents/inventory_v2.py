"""
Method Inventory V2: Fixed methods with clear regime differentiation.

Key fixes:
1. Methods generate STRONG signals in their optimal regimes
2. Methods generate WEAK/OPPOSITE signals in non-optimal regimes
3. Removed "neutral" methods that always return 0
4. Each regime has clearly distinct optimal methods
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
import numpy as np
import pandas as pd


class MethodCategory(Enum):
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    MEAN_REVERT = "mean_revert"
    VOLATILE = "volatile"


@dataclass
class TradingMethod:
    """Definition of a trading method."""
    name: str
    category: MethodCategory
    description: str
    optimal_regimes: List[str]

    def execute(self, prices: pd.DataFrame, position: float = 0.0) -> Dict:
        return {"signal": 0.0, "confidence": 0.5}


# ============================================================================
# TREND UP METHODS - Long bias, work in uptrends
# ============================================================================

class BuyMomentumMethod(TradingMethod):
    """Strong buy signal when price is rising."""

    def __init__(self, lookback: int = 10):
        super().__init__(
            name="BuyMomentum",
            category=MethodCategory.TREND_UP,
            description="Buy when price shows upward momentum",
            optimal_regimes=["trend_up"],
        )
        self.lookback = lookback

    def execute(self, prices: pd.DataFrame, position: float = 0.0) -> Dict:
        if len(prices) < self.lookback:
            return {"signal": 0.0, "confidence": 0.3}

        ret = (prices["close"].iloc[-1] / prices["close"].iloc[-self.lookback]) - 1

        # Strong LONG signal when trending up
        if ret > 0.01:  # 1% gain
            signal = min(1.0, ret * 30)
            confidence = 0.9
        elif ret > 0:
            signal = ret * 20
            confidence = 0.6
        else:
            # Weak or negative signal when trending down
            signal = ret * 10
            confidence = 0.3

        return {"signal": float(signal), "confidence": float(confidence)}


class BreakoutLongMethod(TradingMethod):
    """Buy breakouts above resistance."""

    def __init__(self, lookback: int = 15):
        super().__init__(
            name="BreakoutLong",
            category=MethodCategory.TREND_UP,
            description="Buy breakout above range high",
            optimal_regimes=["trend_up"],
        )
        self.lookback = lookback

    def execute(self, prices: pd.DataFrame, position: float = 0.0) -> Dict:
        if len(prices) < self.lookback:
            return {"signal": 0.0, "confidence": 0.3}

        current = prices["close"].iloc[-1]
        prev_high = prices["close"].iloc[-self.lookback:-1].max()
        prev_low = prices["close"].iloc[-self.lookback:-1].min()

        range_size = prev_high - prev_low
        if range_size < 1e-8:
            return {"signal": 0.0, "confidence": 0.3}

        if current > prev_high:
            # Breakout above - strong long
            signal = min(1.0, (current - prev_high) / range_size * 3)
            confidence = 0.85
        elif current < prev_low:
            # Breakout below - this method performs badly
            signal = -0.3
            confidence = 0.4
        else:
            signal = 0.0
            confidence = 0.3

        return {"signal": float(signal), "confidence": float(confidence)}


class TrendRiderMethod(TradingMethod):
    """Stay long as long as trend continues."""

    def __init__(self, fast: int = 5, slow: int = 15):
        super().__init__(
            name="TrendRider",
            category=MethodCategory.TREND_UP,
            description="Ride trends using MA crossover",
            optimal_regimes=["trend_up"],
        )
        self.fast = fast
        self.slow = slow

    def execute(self, prices: pd.DataFrame, position: float = 0.0) -> Dict:
        if len(prices) < self.slow:
            return {"signal": 0.0, "confidence": 0.3}

        fast_ma = prices["close"].rolling(self.fast).mean().iloc[-1]
        slow_ma = prices["close"].rolling(self.slow).mean().iloc[-1]

        if fast_ma > slow_ma:
            # Uptrend - long
            diff_pct = (fast_ma - slow_ma) / slow_ma
            signal = min(1.0, diff_pct * 50)
            confidence = 0.8
        else:
            # Downtrend - this method underperforms
            diff_pct = (fast_ma - slow_ma) / slow_ma
            signal = max(-0.5, diff_pct * 30)
            confidence = 0.4

        return {"signal": float(signal), "confidence": float(confidence)}


# ============================================================================
# TREND DOWN METHODS - Short bias, work in downtrends
# ============================================================================

class SellMomentumMethod(TradingMethod):
    """Strong sell signal when price is falling."""

    def __init__(self, lookback: int = 10):
        super().__init__(
            name="SellMomentum",
            category=MethodCategory.TREND_DOWN,
            description="Sell when price shows downward momentum",
            optimal_regimes=["trend_down"],
        )
        self.lookback = lookback

    def execute(self, prices: pd.DataFrame, position: float = 0.0) -> Dict:
        if len(prices) < self.lookback:
            return {"signal": 0.0, "confidence": 0.3}

        ret = (prices["close"].iloc[-1] / prices["close"].iloc[-self.lookback]) - 1

        # Strong SHORT signal when trending down
        if ret < -0.01:  # 1% loss
            signal = max(-1.0, ret * 30)
            confidence = 0.9
        elif ret < 0:
            signal = ret * 20
            confidence = 0.6
        else:
            # Weak or positive signal when trending up
            signal = ret * 10
            confidence = 0.3

        return {"signal": float(signal), "confidence": float(confidence)}


class BreakoutShortMethod(TradingMethod):
    """Sell breakdowns below support."""

    def __init__(self, lookback: int = 15):
        super().__init__(
            name="BreakoutShort",
            category=MethodCategory.TREND_DOWN,
            description="Sell breakout below range low",
            optimal_regimes=["trend_down"],
        )
        self.lookback = lookback

    def execute(self, prices: pd.DataFrame, position: float = 0.0) -> Dict:
        if len(prices) < self.lookback:
            return {"signal": 0.0, "confidence": 0.3}

        current = prices["close"].iloc[-1]
        prev_high = prices["close"].iloc[-self.lookback:-1].max()
        prev_low = prices["close"].iloc[-self.lookback:-1].min()

        range_size = prev_high - prev_low
        if range_size < 1e-8:
            return {"signal": 0.0, "confidence": 0.3}

        if current < prev_low:
            # Breakdown below - strong short
            signal = max(-1.0, (current - prev_low) / range_size * 3)
            confidence = 0.85
        elif current > prev_high:
            # Breakout above - this method performs badly
            signal = 0.3
            confidence = 0.4
        else:
            signal = 0.0
            confidence = 0.3

        return {"signal": float(signal), "confidence": float(confidence)}


class TrendFaderMethod(TradingMethod):
    """Short when MA crossover goes bearish."""

    def __init__(self, fast: int = 5, slow: int = 15):
        super().__init__(
            name="TrendFader",
            category=MethodCategory.TREND_DOWN,
            description="Short on bearish MA crossover",
            optimal_regimes=["trend_down"],
        )
        self.fast = fast
        self.slow = slow

    def execute(self, prices: pd.DataFrame, position: float = 0.0) -> Dict:
        if len(prices) < self.slow:
            return {"signal": 0.0, "confidence": 0.3}

        fast_ma = prices["close"].rolling(self.fast).mean().iloc[-1]
        slow_ma = prices["close"].rolling(self.slow).mean().iloc[-1]

        if fast_ma < slow_ma:
            # Downtrend - short
            diff_pct = (slow_ma - fast_ma) / slow_ma
            signal = max(-1.0, -diff_pct * 50)
            confidence = 0.8
        else:
            # Uptrend - this method underperforms
            diff_pct = (fast_ma - slow_ma) / slow_ma
            signal = min(0.5, diff_pct * 30)
            confidence = 0.4

        return {"signal": float(signal), "confidence": float(confidence)}


# ============================================================================
# MEAN REVERSION METHODS - Fade extremes
# ============================================================================

class MeanRevertMethod(TradingMethod):
    """Buy oversold, sell overbought."""

    def __init__(self, lookback: int = 15, threshold: float = 1.5):
        super().__init__(
            name="MeanRevert",
            category=MethodCategory.MEAN_REVERT,
            description="Fade deviations from mean",
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

        # Strong reversal signal at extremes
        if abs(z_score) > self.threshold:
            signal = float(np.clip(-z_score * 0.7, -1, 1))
            confidence = 0.85
        else:
            signal = float(np.clip(-z_score * 0.3, -0.5, 0.5))
            confidence = 0.5

        return {"signal": signal, "confidence": float(confidence)}


class BollingerMeanRevertMethod(TradingMethod):
    """Trade Bollinger Band touches."""

    def __init__(self, period: int = 15, std_dev: float = 1.8):
        super().__init__(
            name="BollingerMR",
            category=MethodCategory.MEAN_REVERT,
            description="Fade Bollinger Band touches",
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
            signal = -0.9
            confidence = 0.85
        elif current < lower:
            signal = 0.9
            confidence = 0.85
        else:
            # Distance from middle
            if std > 1e-8:
                z = (current - ma) / std
                signal = float(-z * 0.3)
            else:
                signal = 0.0
            confidence = 0.4

        return {"signal": float(signal), "confidence": float(confidence)}


class RSIMeanRevertMethod(TradingMethod):
    """Trade RSI extremes."""

    def __init__(self, period: int = 10):
        super().__init__(
            name="RSI_MR",
            category=MethodCategory.MEAN_REVERT,
            description="Fade RSI overbought/oversold",
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

        # Strong signals at extremes
        if rsi > 75:
            signal = -0.9
            confidence = 0.85
        elif rsi < 25:
            signal = 0.9
            confidence = 0.85
        elif rsi > 60:
            signal = -0.4
            confidence = 0.5
        elif rsi < 40:
            signal = 0.4
            confidence = 0.5
        else:
            signal = 0.0
            confidence = 0.3

        return {"signal": float(signal), "confidence": float(confidence)}


# ============================================================================
# VOLATILITY METHODS - Profit from or hedge against volatility
# ============================================================================

class VolBreakoutMethod(TradingMethod):
    """Trade volatility breakouts aggressively."""

    def __init__(self, lookback: int = 10):
        super().__init__(
            name="VolBreakout",
            category=MethodCategory.VOLATILE,
            description="Aggressive breakout trading in volatile markets",
            optimal_regimes=["volatile"],
        )
        self.lookback = lookback

    def execute(self, prices: pd.DataFrame, position: float = 0.0) -> Dict:
        if len(prices) < self.lookback:
            return {"signal": 0.0, "confidence": 0.3}

        close = prices["close"]
        current = close.iloc[-1]
        prev = close.iloc[-2]
        vol = close.pct_change().rolling(self.lookback).std().iloc[-1]

        if vol < 1e-8:
            return {"signal": 0.0, "confidence": 0.3}

        move = (current - prev) / prev
        move_z = move / vol  # Standardized move

        # In volatile markets, follow momentum aggressively
        if abs(move_z) > 1.5:
            signal = float(np.clip(move_z * 0.5, -1, 1))
            confidence = 0.8
        else:
            signal = float(move_z * 0.3)
            confidence = 0.5

        return {"signal": signal, "confidence": float(confidence)}


class VolScalpMethod(TradingMethod):
    """Scalp quick moves in volatile markets."""

    def __init__(self, lookback: int = 5):
        super().__init__(
            name="VolScalp",
            category=MethodCategory.VOLATILE,
            description="Quick scalping in volatile conditions",
            optimal_regimes=["volatile"],
        )
        self.lookback = lookback

    def execute(self, prices: pd.DataFrame, position: float = 0.0) -> Dict:
        if len(prices) < self.lookback:
            return {"signal": 0.0, "confidence": 0.3}

        close = prices["close"]
        ret = (close.iloc[-1] / close.iloc[-self.lookback]) - 1

        # Follow short-term momentum
        signal = float(np.clip(ret * 50, -1, 1))
        confidence = 0.7

        return {"signal": signal, "confidence": float(confidence)}


class VolFadeMethod(TradingMethod):
    """Fade overextensions in volatile markets."""

    def __init__(self, lookback: int = 8):
        super().__init__(
            name="VolFade",
            category=MethodCategory.VOLATILE,
            description="Fade overextensions",
            optimal_regimes=["volatile"],
        )
        self.lookback = lookback

    def execute(self, prices: pd.DataFrame, position: float = 0.0) -> Dict:
        if len(prices) < self.lookback:
            return {"signal": 0.0, "confidence": 0.3}

        close = prices["close"]
        current = close.iloc[-1]
        ma = close.rolling(self.lookback).mean().iloc[-1]
        vol = close.pct_change().rolling(self.lookback).std().iloc[-1]

        if vol < 1e-8:
            return {"signal": 0.0, "confidence": 0.3}

        deviation = (current - ma) / ma
        deviation_z = deviation / vol

        # Fade large moves
        if abs(deviation_z) > 2:
            signal = float(np.clip(-deviation_z * 0.4, -1, 1))
            confidence = 0.75
        else:
            signal = 0.0
            confidence = 0.3

        return {"signal": signal, "confidence": float(confidence)}


# ============================================================================
# METHOD INVENTORY V2
# ============================================================================

METHOD_INVENTORY_V2: Dict[str, TradingMethod] = {
    # Trend Up specialists (3)
    "BuyMomentum": BuyMomentumMethod(),
    "BreakoutLong": BreakoutLongMethod(),
    "TrendRider": TrendRiderMethod(),

    # Trend Down specialists (3)
    "SellMomentum": SellMomentumMethod(),
    "BreakoutShort": BreakoutShortMethod(),
    "TrendFader": TrendFaderMethod(),

    # Mean Reversion specialists (3)
    "MeanRevert": MeanRevertMethod(),
    "BollingerMR": BollingerMeanRevertMethod(),
    "RSI_MR": RSIMeanRevertMethod(),

    # Volatility specialists (3)
    "VolBreakout": VolBreakoutMethod(),
    "VolScalp": VolScalpMethod(),
    "VolFade": VolFadeMethod(),
}


def get_method_names_v2() -> List[str]:
    return list(METHOD_INVENTORY_V2.keys())


def get_methods_for_regime_v2(regime: str) -> List[str]:
    return [
        name for name, method in METHOD_INVENTORY_V2.items()
        if regime in method.optimal_regimes
    ]
