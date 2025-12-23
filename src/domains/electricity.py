"""
Electricity Domain Module - US Grid Demand Analysis.

This module provides regime detection and prediction methods
for US electricity demand data with seasonal and daily patterns.

Regimes:
- peak: High demand periods (top 25%)
- off_peak: Low demand periods (bottom 25%, typically night)
- morning_ramp: 6-9 AM demand increase
- shoulder: Moderate demand periods
- evening_decline: Post-peak decline

Prediction Task: Hourly demand forecasting
Metric: RMSE (Root Mean Square Error)
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple, Callable
from datetime import datetime
from collections import defaultdict

import numpy as np

# Data directory
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "electricity"


def load_data() -> Dict:
    """
    Load EIA hourly electricity demand data.
    
    Returns:
        Dict with 'timestamps', 'demand_mw', 'regimes' arrays
    """
    data_file = DATA_DIR / "eia_hourly_demand.csv"
    
    if not data_file.exists():
        raise FileNotFoundError(f"Electricity data not found: {data_file}")
    
    timestamps = []
    demand_mw = []
    regimes = []
    
    with open(data_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(datetime.fromisoformat(row['timestamp']))
            demand_mw.append(int(row['demand_mw']))
            regimes.append(row['regime'])
    
    return {
        'timestamps': np.array(timestamps),
        'demand_mw': np.array(demand_mw),
        'regimes': np.array(regimes),
    }


def detect_regime(demand: np.ndarray, timestamps: np.ndarray = None) -> np.ndarray:
    """
    Detect regime based on demand levels and temporal patterns.
    
    Args:
        demand: Array of demand values (MW)
        timestamps: Optional array of datetime objects
    
    Returns:
        Array of regime labels
    """
    high_thresh = np.percentile(demand, 75)
    low_thresh = np.percentile(demand, 25)
    
    regimes = []
    
    for i, d in enumerate(demand):
        if timestamps is not None:
            hour = timestamps[i].hour
            month = timestamps[i].month
        else:
            hour = i % 24
            month = (i // 720) % 12 + 1
        
        if d > high_thresh:
            regimes.append('peak')
        elif d < low_thresh:
            regimes.append('off_peak')
        elif 6 <= hour <= 9:
            regimes.append('morning_ramp')
        elif 21 <= hour <= 23:
            regimes.append('evening_decline')
        else:
            regimes.append('shoulder')
    
    return np.array(regimes)


def get_prediction_methods() -> Dict[str, Callable]:
    """
    Get prediction methods for electricity domain.
    
    Returns:
        Dict mapping method name to prediction function
    """
    return {
        'persistence': persistence_predictor,
        'hourly_average': hourly_average_predictor,
        'seasonal_naive': seasonal_naive_predictor,
        'peak_model': peak_demand_model,
        'load_forecast': load_forecast_model,
    }


def persistence_predictor(history: np.ndarray, horizon: int = 1) -> np.ndarray:
    """
    Naive persistence: predict last value.
    
    Good for short-term, stable demand.
    """
    if len(history) == 0:
        return np.zeros(horizon)
    return np.full(horizon, history[-1])


def hourly_average_predictor(history: np.ndarray, horizon: int = 1,
                              timestamps: np.ndarray = None) -> np.ndarray:
    """
    Hourly average: predict based on historical average for each hour.
    
    Good for capturing daily load patterns.
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


def seasonal_naive_predictor(history: np.ndarray, horizon: int = 1,
                              period: int = 24) -> np.ndarray:
    """
    Seasonal naive: predict same value from one period ago.
    
    Good for daily seasonality (period=24).
    """
    if len(history) < period:
        return persistence_predictor(history, horizon)
    
    # Use values from one period ago
    predictions = []
    for h in range(horizon):
        idx = len(history) - period + (h % period)
        if idx >= 0:
            predictions.append(history[idx])
        else:
            predictions.append(history[-1])
    
    return np.array(predictions)


def peak_demand_model(history: np.ndarray, horizon: int = 1,
                      timestamps: np.ndarray = None) -> np.ndarray:
    """
    Peak demand model: different predictions for peak vs off-peak.
    
    Best for high/low demand regimes.
    """
    if len(history) < 24:
        return persistence_predictor(history, horizon)
    
    # Compute peak vs off-peak averages
    high_thresh = np.percentile(history, 75)
    low_thresh = np.percentile(history, 25)
    
    peak_vals = history[history > high_thresh]
    offpeak_vals = history[history < low_thresh]
    mid_vals = history[(history >= low_thresh) & (history <= high_thresh)]
    
    peak_avg = np.mean(peak_vals) if len(peak_vals) > 0 else np.mean(history)
    offpeak_avg = np.mean(offpeak_vals) if len(offpeak_vals) > 0 else np.mean(history)
    mid_avg = np.mean(mid_vals) if len(mid_vals) > 0 else np.mean(history)
    
    # Predict based on hour
    predictions = []
    last_hour = len(history) % 24
    for h in range(horizon):
        hour = (last_hour + h) % 24
        if 14 <= hour <= 20:  # Peak hours
            predictions.append(peak_avg)
        elif 0 <= hour <= 5:  # Off-peak
            predictions.append(offpeak_avg)
        else:
            predictions.append(mid_avg)
    
    return np.array(predictions)


def load_forecast_model(history: np.ndarray, horizon: int = 1,
                        alpha: float = 0.3, beta: float = 0.1) -> np.ndarray:
    """
    Load forecast with trend: Holt's linear exponential smoothing.
    
    Good for capturing trends in demand.
    """
    if len(history) < 2:
        return persistence_predictor(history, horizon)
    
    # Initialize
    level = history[0]
    trend = history[1] - history[0] if len(history) > 1 else 0
    
    # Update through history
    for val in history[1:]:
        prev_level = level
        level = alpha * val + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
    
    # Forecast
    predictions = []
    for h in range(1, horizon + 1):
        predictions.append(level + h * trend)
    
    return np.array(predictions)


def compute_rmse(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Compute Root Mean Square Error.
    
    RMSE = sqrt(mean((pred - actual)^2))
    """
    if len(predictions) == 0 or len(actuals) == 0:
        return float('inf')
    
    mse = np.mean((predictions - actuals) ** 2)
    return float(np.sqrt(mse))


def get_regime_list() -> List[str]:
    """Get list of regimes for this domain."""
    return ['peak', 'off_peak', 'morning_ramp', 'shoulder', 'evening_decline']


def get_metric_name() -> str:
    """Get primary metric name."""
    return 'RMSE'


def is_lower_better() -> bool:
    """Whether lower metric values are better."""
    return True

