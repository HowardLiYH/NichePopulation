"""
Regime Classification Module

Implements four methods for classifying market regimes:
1. MA Crossover: Bull/Bear/Sideways based on price vs moving average
2. Volatility: Low/Medium/High based on rolling volatility percentiles
3. Returns: Bull/Bear/Sideways based on rolling return thresholds
4. Combined: 6 regimes (MA direction × Volatility level)

Each classifier provides:
- classify(): Label entire series
- get_regime(): Get regime for specific index
- validate(): Check classifier quality
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
from scipy import stats


class MARegime(Enum):
    """Regimes based on MA crossover."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"


class VolatilityRegime(Enum):
    """Regimes based on volatility level."""
    LOW = "low_vol"
    MEDIUM = "med_vol"
    HIGH = "high_vol"


class ReturnRegime(Enum):
    """Regimes based on return direction."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"


class CombinedRegime(Enum):
    """Combined regimes (6 total)."""
    BULL_LOW_VOL = "bull_low_vol"
    BULL_MED_VOL = "bull_med_vol"
    BULL_HIGH_VOL = "bull_high_vol"
    BEAR_LOW_VOL = "bear_low_vol"
    BEAR_MED_VOL = "bear_med_vol"
    BEAR_HIGH_VOL = "bear_high_vol"


@dataclass
class RegimeClassification:
    """Result of regime classification."""
    labels: np.ndarray  # Regime labels for each bar
    regime_names: List[str]  # Unique regime names
    regime_counts: Dict[str, int]  # Count per regime
    transitions: int  # Number of regime transitions

    def get_segments(self) -> List[Tuple[int, int, str]]:
        """Get contiguous regime segments as (start, end, regime) tuples."""
        segments = []
        if len(self.labels) == 0:
            return segments

        current_regime = self.labels[0]
        start_idx = 0

        for i, label in enumerate(self.labels):
            if label != current_regime:
                segments.append((start_idx, i - 1, current_regime))
                current_regime = label
                start_idx = i

        # Add final segment
        segments.append((start_idx, len(self.labels) - 1, current_regime))

        return segments

    def compute_entropy(self) -> float:
        """Compute regime distribution entropy (higher = more diverse)."""
        total = sum(self.regime_counts.values())
        if total == 0:
            return 0.0

        probs = [count / total for count in self.regime_counts.values() if count > 0]
        return -sum(p * np.log2(p) for p in probs)


class MAClassifier:
    """
    Moving Average Crossover Classifier.

    Regimes:
    - BULL: Price > MA * (1 + band)
    - BEAR: Price < MA * (1 - band)
    - SIDEWAYS: Price within band around MA
    """

    def __init__(self, window: int = 50, band: float = 0.02):
        """
        Args:
            window: MA window period
            band: Threshold band around MA (0.02 = 2%)
        """
        self.window = window
        self.band = band
        self.name = "ma_crossover"

    def classify(self, prices: Union[pd.Series, np.ndarray]) -> RegimeClassification:
        """
        Classify regimes based on MA crossover.

        Args:
            prices: Close prices series

        Returns:
            RegimeClassification with labels
        """
        prices = np.asarray(prices)
        n = len(prices)

        # Compute MA
        ma = pd.Series(prices).rolling(window=self.window, min_periods=1).mean().values

        # Classify
        labels = np.array(["sideways"] * n)

        upper_band = ma * (1 + self.band)
        lower_band = ma * (1 - self.band)

        labels[prices > upper_band] = "bull"
        labels[prices < lower_band] = "bear"

        # Compute stats
        regime_names = ["bull", "bear", "sideways"]
        regime_counts = {r: np.sum(labels == r) for r in regime_names}
        transitions = np.sum(labels[1:] != labels[:-1])

        return RegimeClassification(
            labels=labels,
            regime_names=regime_names,
            regime_counts=regime_counts,
            transitions=transitions
        )


class VolatilityClassifier:
    """
    Volatility-Based Classifier.

    Regimes:
    - LOW: Volatility below 33rd percentile
    - MEDIUM: Volatility between 33rd and 66th percentile
    - HIGH: Volatility above 66th percentile
    """

    def __init__(self, window: int = 20, percentiles: Tuple[int, int] = (33, 66)):
        """
        Args:
            window: Volatility calculation window
            percentiles: (low, high) percentile thresholds
        """
        self.window = window
        self.percentiles = percentiles
        self.name = "volatility"

    def classify(self, returns: Union[pd.Series, np.ndarray] = None,
                 prices: Union[pd.Series, np.ndarray] = None) -> RegimeClassification:
        """
        Classify regimes based on volatility.

        Args:
            returns: Return series (preferred)
            prices: Price series (will compute returns if returns not provided)

        Returns:
            RegimeClassification with labels
        """
        if returns is None:
            if prices is None:
                raise ValueError("Must provide either returns or prices")
            prices = np.asarray(prices)
            returns = np.diff(prices) / prices[:-1]
            returns = np.concatenate([[0], returns])

        returns = np.asarray(returns)
        n = len(returns)

        # Compute rolling volatility
        vol = pd.Series(returns).rolling(window=self.window, min_periods=1).std().values

        # Compute percentile thresholds
        low_thresh = np.nanpercentile(vol, self.percentiles[0])
        high_thresh = np.nanpercentile(vol, self.percentiles[1])

        # Classify
        labels = np.array(["med_vol"] * n)
        labels[vol < low_thresh] = "low_vol"
        labels[vol > high_thresh] = "high_vol"

        # Compute stats
        regime_names = ["low_vol", "med_vol", "high_vol"]
        regime_counts = {r: np.sum(labels == r) for r in regime_names}
        transitions = np.sum(labels[1:] != labels[:-1])

        return RegimeClassification(
            labels=labels,
            regime_names=regime_names,
            regime_counts=regime_counts,
            transitions=transitions
        )


class ReturnsClassifier:
    """
    Return-Based Classifier.

    Regimes:
    - BULL: Rolling return > threshold
    - BEAR: Rolling return < -threshold
    - SIDEWAYS: Rolling return within [-threshold, threshold]
    """

    def __init__(self, window: int = 20, threshold: float = 0.02):
        """
        Args:
            window: Rolling return window
            threshold: Bull/bear threshold (0.02 = 2%)
        """
        self.window = window
        self.threshold = threshold
        self.name = "returns"

    def classify(self, prices: Union[pd.Series, np.ndarray]) -> RegimeClassification:
        """
        Classify regimes based on rolling returns.

        Args:
            prices: Close prices series

        Returns:
            RegimeClassification with labels
        """
        prices = np.asarray(prices)
        n = len(prices)

        # Compute rolling returns
        rolling_ret = pd.Series(prices).pct_change(periods=self.window).values
        rolling_ret = np.nan_to_num(rolling_ret, nan=0.0)

        # Classify
        labels = np.array(["sideways"] * n)
        labels[rolling_ret > self.threshold] = "bull"
        labels[rolling_ret < -self.threshold] = "bear"

        # Compute stats
        regime_names = ["bull", "bear", "sideways"]
        regime_counts = {r: np.sum(labels == r) for r in regime_names}
        transitions = np.sum(labels[1:] != labels[:-1])

        return RegimeClassification(
            labels=labels,
            regime_names=regime_names,
            regime_counts=regime_counts,
            transitions=transitions
        )


class CombinedClassifier:
    """
    Combined MA + Volatility Classifier.

    Creates 6 regimes by combining:
    - MA direction (bull/bear) with
    - Volatility level (low/med/high)

    Note: Sideways MA is mapped to the dominant trend.
    """

    def __init__(self, ma_window: int = 50, ma_band: float = 0.02,
                 vol_window: int = 20, vol_percentiles: Tuple[int, int] = (33, 66)):
        """
        Args:
            ma_window: MA window for trend detection
            ma_band: MA band threshold
            vol_window: Volatility calculation window
            vol_percentiles: Volatility percentile thresholds
        """
        self.ma_classifier = MAClassifier(window=ma_window, band=ma_band)
        self.vol_classifier = VolatilityClassifier(window=vol_window, percentiles=vol_percentiles)
        self.name = "combined"

    def classify(self, prices: Union[pd.Series, np.ndarray]) -> RegimeClassification:
        """
        Classify regimes using combined MA and volatility.

        Args:
            prices: Close prices series

        Returns:
            RegimeClassification with 6 regime labels
        """
        prices = np.asarray(prices)
        n = len(prices)

        # Get MA regimes
        ma_result = self.ma_classifier.classify(prices)
        ma_labels = ma_result.labels

        # Get volatility regimes
        vol_result = self.vol_classifier.classify(prices=prices)
        vol_labels = vol_result.labels

        # Combine
        labels = np.array([""] * n)

        for i in range(n):
            ma = ma_labels[i]
            vol = vol_labels[i]

            # Map sideways to bull for simplicity (can be customized)
            if ma == "sideways":
                ma = "bull"

            labels[i] = f"{ma}_{vol}"

        # Compute stats
        regime_names = [
            "bull_low_vol", "bull_med_vol", "bull_high_vol",
            "bear_low_vol", "bear_med_vol", "bear_high_vol"
        ]
        regime_counts = {r: np.sum(labels == r) for r in regime_names}
        transitions = np.sum(labels[1:] != labels[:-1])

        return RegimeClassification(
            labels=labels,
            regime_names=regime_names,
            regime_counts=regime_counts,
            transitions=transitions
        )


class UnifiedRegimeClassifier:
    """
    Unified interface for all regime classifiers.

    Provides:
    - Access to all 4 classification methods
    - Validation utilities
    - Cross-classifier comparison
    """

    def __init__(
        self,
        ma_window: int = 50,
        ma_band: float = 0.02,
        vol_window: int = 20,
        vol_percentiles: Tuple[int, int] = (33, 66),
        return_window: int = 20,
        return_threshold: float = 0.02
    ):
        """Initialize all classifiers with given parameters."""
        self.classifiers = {
            "ma": MAClassifier(window=ma_window, band=ma_band),
            "volatility": VolatilityClassifier(window=vol_window, percentiles=vol_percentiles),
            "returns": ReturnsClassifier(window=return_window, threshold=return_threshold),
            "combined": CombinedClassifier(
                ma_window=ma_window, ma_band=ma_band,
                vol_window=vol_window, vol_percentiles=vol_percentiles
            )
        }

    def classify(self, prices: Union[pd.Series, np.ndarray],
                 method: str = "ma") -> RegimeClassification:
        """
        Classify regimes using specified method.

        Args:
            prices: Close prices series
            method: Classification method ("ma", "volatility", "returns", "combined")

        Returns:
            RegimeClassification with labels
        """
        if method not in self.classifiers:
            raise ValueError(f"Unknown method: {method}. Choose from {list(self.classifiers.keys())}")

        return self.classifiers[method].classify(prices)

    def classify_all(self, prices: Union[pd.Series, np.ndarray]) -> Dict[str, RegimeClassification]:
        """
        Classify using all methods.

        Args:
            prices: Close prices series

        Returns:
            Dict mapping method name to RegimeClassification
        """
        return {name: clf.classify(prices) for name, clf in self.classifiers.items()}

    def compute_cross_agreement(self, prices: Union[pd.Series, np.ndarray]) -> pd.DataFrame:
        """
        Compute pairwise agreement (Cohen's Kappa) between classifiers.

        Args:
            prices: Close prices series

        Returns:
            DataFrame with pairwise kappa values
        """
        results = self.classify_all(prices)
        methods = list(results.keys())
        n_methods = len(methods)

        # Compute pairwise kappa
        kappa_matrix = np.zeros((n_methods, n_methods))

        for i, m1 in enumerate(methods):
            for j, m2 in enumerate(methods):
                if i == j:
                    kappa_matrix[i, j] = 1.0
                elif i < j:
                    # Simplify labels to common categories for comparison
                    labels1 = self._simplify_labels(results[m1].labels)
                    labels2 = self._simplify_labels(results[m2].labels)

                    kappa = self._cohens_kappa(labels1, labels2)
                    kappa_matrix[i, j] = kappa
                    kappa_matrix[j, i] = kappa

        return pd.DataFrame(kappa_matrix, index=methods, columns=methods)

    def _simplify_labels(self, labels: np.ndarray) -> np.ndarray:
        """Simplify labels to 3 categories for cross-comparison."""
        simplified = np.array(labels, dtype=str)

        # Map to bull/bear/neutral
        bull_patterns = ["bull", "bull_low_vol", "bull_med_vol", "bull_high_vol"]
        bear_patterns = ["bear", "bear_low_vol", "bear_med_vol", "bear_high_vol"]

        result = np.array(["neutral"] * len(labels))

        for pattern in bull_patterns:
            result[simplified == pattern] = "bull"
        for pattern in bear_patterns:
            result[simplified == pattern] = "bear"

        # Map volatility regimes
        result[simplified == "high_vol"] = "neutral"
        result[simplified == "med_vol"] = "neutral"
        result[simplified == "low_vol"] = "neutral"
        result[simplified == "sideways"] = "neutral"

        return result

    def _cohens_kappa(self, labels1: np.ndarray, labels2: np.ndarray) -> float:
        """Compute Cohen's Kappa between two label arrays."""
        # Get unique labels
        all_labels = list(set(labels1) | set(labels2))
        n_labels = len(all_labels)

        # Build confusion matrix
        label_to_idx = {l: i for i, l in enumerate(all_labels)}
        confusion = np.zeros((n_labels, n_labels))

        for l1, l2 in zip(labels1, labels2):
            i, j = label_to_idx[l1], label_to_idx[l2]
            confusion[i, j] += 1

        n = len(labels1)
        if n == 0:
            return 0.0

        # Observed agreement
        p_o = np.trace(confusion) / n

        # Expected agreement
        row_sums = confusion.sum(axis=1)
        col_sums = confusion.sum(axis=0)
        p_e = np.sum(row_sums * col_sums) / (n * n)

        # Kappa
        if p_e == 1:
            return 1.0

        kappa = (p_o - p_e) / (1 - p_e)
        return kappa

    def validate_bootstrap_stability(
        self,
        prices: Union[pd.Series, np.ndarray],
        method: str = "ma",
        n_bootstrap: int = 100
    ) -> float:
        """
        Compute bootstrap stability of classifier (Fleiss' Kappa approximation).

        Args:
            prices: Close prices series
            method: Classification method
            n_bootstrap: Number of bootstrap samples

        Returns:
            Mean pairwise kappa across bootstrap samples
        """
        prices = np.asarray(prices)
        n = len(prices)

        # Generate bootstrap samples and classify
        all_labels = []

        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(n, size=n, replace=True)
            sample_prices = prices[indices]

            result = self.classify(sample_prices, method=method)
            all_labels.append(result.labels)

        # Compute pairwise kappa
        kappas = []
        for i in range(n_bootstrap):
            for j in range(i + 1, min(i + 10, n_bootstrap)):  # Sample pairs for efficiency
                kappa = self._cohens_kappa(all_labels[i], all_labels[j])
                kappas.append(kappa)

        return np.mean(kappas) if kappas else 0.0


# Convenience function
def classify_regimes(
    prices: Union[pd.Series, np.ndarray],
    method: str = "ma",
    **kwargs
) -> RegimeClassification:
    """
    Convenience function to classify regimes.

    Args:
        prices: Close prices series
        method: "ma", "volatility", "returns", or "combined"
        **kwargs: Classifier parameters

    Returns:
        RegimeClassification with labels
    """
    classifier = UnifiedRegimeClassifier(**kwargs)
    return classifier.classify(prices, method=method)
