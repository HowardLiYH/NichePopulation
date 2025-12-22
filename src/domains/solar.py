"""
Solar Irradiance Domain - Using REAL Open-Meteo data.

Data Source: Open-Meteo Historical Solar API
- 5 US locations: Phoenix AZ, Las Vegas NV, Denver CO, Miami FL, Seattle WA
- Variables: GHI, DNI, DHI (Global/Direct/Diffuse Horizontal Irradiance)
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "solar"


def load_data() -> pd.DataFrame:
    """Load REAL solar irradiance data from Open-Meteo."""

    # Use verified real Open-Meteo data
    real_path = DATA_DIR / "openmeteo_real_irradiance.csv"

    if real_path.exists():
        df = pd.read_csv(real_path, parse_dates=['datetime'])
        print(f"[Solar] Loaded {len(df)} REAL records from Open-Meteo")
        return df

    raise FileNotFoundError(
        f"Real solar data not found at {real_path}. "
        "Run scripts/download_real_solar.py first."
    )


def detect_regime(df: pd.DataFrame) -> pd.Series:
    """
    Detect solar irradiance regimes.

    Regimes (based on clear sky index):
    - clear: CSI > 0.8 (clear sky)
    - partly_cloudy: 0.5 < CSI <= 0.8
    - overcast: 0.2 < CSI <= 0.5
    - storm: CSI <= 0.2 (heavy clouds/storm)
    """

    if 'regime' in df.columns:
        return df['regime']

    def get_regime(row):
        csi = row.get('clear_sky_index', 0.5)
        ghi = row.get('ghi', 0)

        if pd.isna(csi) or ghi < 50:
            return 'overcast'
        elif csi > 0.8:
            return 'clear'
        elif csi > 0.5:
            return 'partly_cloudy'
        elif csi > 0.2:
            return 'overcast'
        else:
            return 'storm'

    return df.apply(get_regime, axis=1)


def get_prediction_methods():
    """
    Return solar-specific prediction methods.

    Methods:
    - Naive: Persistence (next hour = current hour)
    - MA6: 6-hour moving average
    - ClearSky: Use clear sky model
    - Seasonal: Same hour yesterday
    - Hybrid: Blend of methods
    """

    def naive_predict(df, idx):
        """Persistence forecast."""
        if idx < 1:
            return df['ghi'].iloc[0]
        return df['ghi'].iloc[idx - 1]

    def ma6_predict(df, idx):
        """6-hour moving average."""
        if idx < 6:
            return df['ghi'].iloc[:max(idx, 1)].mean()
        return df['ghi'].iloc[idx-6:idx].mean()

    def clear_sky_predict(df, idx):
        """Clear sky model prediction."""
        if 'clear_sky_ghi' in df.columns:
            if idx < len(df):
                return df['clear_sky_ghi'].iloc[idx]
        # Fallback to persistence
        return naive_predict(df, idx)

    def seasonal_predict(df, idx):
        """Same hour yesterday (24 hours ago)."""
        if idx < 24:
            return df['ghi'].iloc[max(idx-1, 0)]
        return df['ghi'].iloc[idx - 24]

    def hybrid_predict(df, idx):
        """Blend persistence and clear sky."""
        persist = naive_predict(df, idx)
        clear = clear_sky_predict(df, idx)

        # Weight by recent accuracy (simplified)
        return 0.6 * persist + 0.4 * clear

    return {
        'naive': naive_predict,
        'ma6': ma6_predict,
        'clear_sky': clear_sky_predict,
        'seasonal': seasonal_predict,
        'hybrid': hybrid_predict,
    }


def evaluate_prediction(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Evaluate prediction performance."""

    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))

    # For solar, skill score vs persistence is common
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
    }


def get_regime_counts() -> dict:
    """Get regime distribution from data."""
    df = load_data()
    return df['regime'].value_counts().to_dict()
