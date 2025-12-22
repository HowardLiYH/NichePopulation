"""
Weather Domain - Using REAL Open-Meteo data.

Data Source: Open-Meteo Historical Weather API
- 5 US cities: Chicago, Houston, Los Angeles, New York, Phoenix
- Variables: Temperature, precipitation, wind speed
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "weather"


def load_data() -> pd.DataFrame:
    """Load REAL weather data from Open-Meteo."""

    # Use verified real Open-Meteo data
    real_path = DATA_DIR / "openmeteo_real_weather.csv"

    if real_path.exists():
        df = pd.read_csv(real_path, parse_dates=['date'])
        print(f"[Weather] Loaded {len(df)} REAL records from Open-Meteo")
        return df

    raise FileNotFoundError(
        f"Real weather data not found at {real_path}. "
        "Run scripts/download_real_weather.py first."
    )


def detect_regime(df: pd.DataFrame) -> pd.Series:
    """
    Detect weather regimes.

    Regimes:
    - stable: Normal conditions
    - stable_cold: Temperature below 0°C
    - stable_hot: Temperature above 30°C
    - approaching_storm: Moderate precipitation/wind
    - active_storm: High precipitation or wind
    """

    if 'regime' in df.columns:
        return df['regime']

    def get_regime(row):
        temp = row.get('temperature', 15)
        precip = row.get('precipitation', 0) or 0
        wind = row.get('wind_speed', 0) or 0

        if pd.isna(temp):
            return 'stable'

        if precip > 10 or wind > 50:
            return 'active_storm'
        elif precip > 2 or wind > 30:
            return 'approaching_storm'
        elif temp < 0:
            return 'stable_cold'
        elif temp > 30:
            return 'stable_hot'
        else:
            return 'stable'

    return df.apply(get_regime, axis=1)


def get_prediction_methods():
    """
    Return weather-specific prediction methods.

    Methods:
    - Naive: Persistence (tomorrow = today)
    - MA3: 3-day moving average
    - MA7: 7-day moving average
    - Seasonal: Same day last week
    - Trend: Recent trend extrapolation
    """

    def naive_predict(df, idx):
        """Persistence forecast."""
        if idx < 1:
            return df['temperature'].iloc[0]
        return df['temperature'].iloc[idx - 1]

    def ma3_predict(df, idx):
        """3-day moving average."""
        if idx < 3:
            return df['temperature'].iloc[:max(idx, 1)].mean()
        return df['temperature'].iloc[idx-3:idx].mean()

    def ma7_predict(df, idx):
        """7-day moving average."""
        if idx < 7:
            return df['temperature'].iloc[:max(idx, 1)].mean()
        return df['temperature'].iloc[idx-7:idx].mean()

    def seasonal_predict(df, idx):
        """Same day last week."""
        if idx < 7:
            return df['temperature'].iloc[max(idx-1, 0)]
        return df['temperature'].iloc[idx - 7]

    def trend_predict(df, idx):
        """Recent trend extrapolation."""
        if idx < 3:
            return df['temperature'].iloc[max(idx-1, 0)]

        temps = df['temperature'].iloc[idx-3:idx].values
        trend = (temps[-1] - temps[0]) / 3

        return temps[-1] + trend

    return {
        'naive': naive_predict,
        'ma3': ma3_predict,
        'ma7': ma7_predict,
        'seasonal': seasonal_predict,
        'trend': trend_predict,
    }


def evaluate_prediction(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Evaluate prediction performance."""

    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
    }


def get_regime_counts() -> dict:
    """Get regime distribution from data."""
    df = load_data()
    return df['regime'].value_counts().to_dict()
