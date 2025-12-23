"""
Traffic Domain Module - NYC Taxi Trip Analysis.

This module provides regime detection and prediction methods
for NYC taxi traffic data with clear temporal patterns.

Regimes:
- morning_rush: 7-9 AM weekdays
- evening_rush: 5-7 PM weekdays
- midday: 10 AM - 4 PM weekdays
- night: 12 AM - 6 AM
- weekend: All day Saturday/Sunday

Prediction Task: Hourly trip count forecasting
Metric: MAPE (Mean Absolute Percentage Error)
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple, Callable
from datetime import datetime
from collections import defaultdict

import numpy as np

# Data directory
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "traffic"


def load_data() -> Dict:
    """
    Load NYC taxi hourly data.

    Returns:
        Dict with 'timestamps', 'trip_counts', 'regimes' arrays
    """
    data_file = DATA_DIR / "nyc_taxi_hourly.csv"

    if not data_file.exists():
        raise FileNotFoundError(f"Traffic data not found: {data_file}")

    timestamps = []
    trip_counts = []
    regimes = []

    with open(data_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(datetime.fromisoformat(row['timestamp']))
            trip_counts.append(int(row['trip_count']))
            regimes.append(row['regime'])

    return {
        'timestamps': np.array(timestamps),
        'trip_counts': np.array(trip_counts),
        'regimes': np.array(regimes),
    }


def detect_regime(timestamps: np.ndarray) -> np.ndarray:
    """
    Detect regime based on temporal patterns.

    Args:
        timestamps: Array of datetime objects

    Returns:
        Array of regime labels
    """
    regimes = []

    for ts in timestamps:
        hour = ts.hour
        weekday = ts.weekday()

        if weekday >= 5:  # Weekend
            regimes.append('weekend')
        elif 7 <= hour <= 9:
            regimes.append('morning_rush')
        elif 17 <= hour <= 19:
            regimes.append('evening_rush')
        elif 0 <= hour <= 5:
            regimes.append('night')
        else:
            regimes.append('midday')

    return np.array(regimes)


def get_prediction_methods() -> Dict[str, Callable]:
    """
    Get prediction methods for traffic domain.

    Returns:
        Dict mapping method name to prediction function
    """
    return {
        'persistence': persistence_predictor,
        'hourly_average': hourly_average_predictor,
        'weekly_pattern': weekly_pattern_predictor,
        'rush_hour_model': rush_hour_model,
        'exponential_smoothing': exponential_smoothing,
    }


def persistence_predictor(history: np.ndarray, horizon: int = 1) -> np.ndarray:
    """
    Naive persistence: predict last value.

    Good for short-term, stable periods.
    """
    if len(history) == 0:
        return np.zeros(horizon)
    return np.full(horizon, history[-1])


def hourly_average_predictor(history: np.ndarray, horizon: int = 1,
                              timestamps: np.ndarray = None) -> np.ndarray:
    """
    Hourly average: predict based on historical average for each hour.

    Good for capturing daily patterns.
    """
    if len(history) < 24:
        return persistence_predictor(history, horizon)

    # Compute hourly averages
    hourly_avg = defaultdict(list)
    if timestamps is not None:
        for val, ts in zip(history, timestamps):
            hourly_avg[ts.hour].append(val)
    else:
        for i, val in enumerate(history):
            hourly_avg[i % 24].append(val)

    hourly_avg = {h: np.mean(vals) for h, vals in hourly_avg.items()}

    # Predict
    predictions = []
    last_hour = len(history) % 24
    for h in range(horizon):
        hour = (last_hour + h) % 24
        predictions.append(hourly_avg.get(hour, np.mean(history)))

    return np.array(predictions)


def weekly_pattern_predictor(history: np.ndarray, horizon: int = 1,
                              timestamps: np.ndarray = None) -> np.ndarray:
    """
    Weekly pattern: predict based on same hour/day of week.

    Good for capturing weekly seasonality.
    """
    if len(history) < 168:  # One week of hourly data
        return hourly_average_predictor(history, horizon, timestamps)

    # Use value from 7 days ago
    if len(history) >= 168:
        predictions = history[-168:-168+horizon] if horizon <= 168 else np.tile(history[-168:], horizon // 168 + 1)[:horizon]
    else:
        predictions = persistence_predictor(history, horizon)

    return np.array(predictions)


def rush_hour_model(history: np.ndarray, horizon: int = 1,
                    timestamps: np.ndarray = None) -> np.ndarray:
    """
    Rush hour model: different predictions for rush vs non-rush.

    Best for morning/evening rush regimes.
    """
    if len(history) < 24:
        return persistence_predictor(history, horizon)

    # Compute rush vs non-rush averages
    if timestamps is not None:
        rush_vals = [v for v, ts in zip(history, timestamps)
                     if 7 <= ts.hour <= 9 or 17 <= ts.hour <= 19]
        non_rush_vals = [v for v, ts in zip(history, timestamps)
                        if not (7 <= ts.hour <= 9 or 17 <= ts.hour <= 19)]
    else:
        rush_vals = history[history > np.percentile(history, 75)]
        non_rush_vals = history[history <= np.percentile(history, 75)]

    rush_avg = np.mean(rush_vals) if rush_vals else np.mean(history)
    non_rush_avg = np.mean(non_rush_vals) if non_rush_vals else np.mean(history)

    # Simple prediction: alternate based on expected pattern
    predictions = []
    last_hour = len(history) % 24
    for h in range(horizon):
        hour = (last_hour + h) % 24
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            predictions.append(rush_avg)
        else:
            predictions.append(non_rush_avg)

    return np.array(predictions)


def exponential_smoothing(history: np.ndarray, horizon: int = 1,
                          alpha: float = 0.3) -> np.ndarray:
    """
    Exponential smoothing: weighted average with decay.

    Good for trend-following in stable conditions.
    """
    if len(history) == 0:
        return np.zeros(horizon)

    # Compute smoothed value
    smoothed = history[0]
    for val in history[1:]:
        smoothed = alpha * val + (1 - alpha) * smoothed

    return np.full(horizon, smoothed)


def compute_mape(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Compute Mean Absolute Percentage Error.

    MAPE = mean(|pred - actual| / |actual|) × 100
    """
    if len(predictions) == 0 or len(actuals) == 0:
        return float('inf')

    # Avoid division by zero
    mask = np.abs(actuals) > 10  # Minimum threshold
    if not np.any(mask):
        return float('inf')

    ape = np.abs(predictions[mask] - actuals[mask]) / np.abs(actuals[mask])
    return float(np.mean(ape) * 100)


def get_regime_list() -> List[str]:
    """Get list of regimes for this domain."""
    return ['morning_rush', 'evening_rush', 'midday', 'night', 'weekend']


def get_metric_name() -> str:
    """Get primary metric name."""
    return 'MAPE'


def is_lower_better() -> bool:
    """Whether lower metric values are better."""
    return True
