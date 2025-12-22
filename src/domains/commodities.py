"""
Commodities Domain - Using REAL FRED data.

Data Source: Federal Reserve Economic Data (FRED)
- WTI Crude Oil (DCOILWTICO)
- Copper (PCOPPUSDM)
- Natural Gas (DHHNGSP)
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "commodities"


def load_data() -> pd.DataFrame:
    """Load REAL commodity price data from FRED."""

    # Use verified real FRED data
    real_path = DATA_DIR / "fred_real_prices.csv"

    if real_path.exists():
        df = pd.read_csv(real_path, parse_dates=['date'])
        print(f"[Commodities] Loaded {len(df)} REAL records from FRED")
        return df

    raise FileNotFoundError(
        f"Real commodity data not found at {real_path}. "
        "Run scripts/download_fred_commodities_real.py first."
    )


def detect_regime(df: pd.DataFrame) -> pd.Series:
    """
    Detect market regimes based on price trends and volatility.

    Regimes:
    - bull: Price above 20-day MA, positive momentum
    - bear: Price below 20-day MA, negative momentum
    - volatile: High volatility (>75th percentile)
    - sideways: Low volatility, no clear trend
    """

    if 'regime' in df.columns:
        return df['regime']

    # Calculate if not present
    regimes = []

    for commodity in df['commodity'].unique():
        comm_df = df[df['commodity'] == commodity].copy()

        vol_75 = comm_df['vol_20'].quantile(0.75)

        for idx, row in comm_df.iterrows():
            if pd.isna(row.get('vol_20')) or pd.isna(row.get('price_ma20')):
                regimes.append('sideways')
            elif row['vol_20'] > vol_75:
                regimes.append('volatile')
            elif row['price'] > row['price_ma20'] * 1.02:
                regimes.append('bull')
            elif row['price'] < row['price_ma20'] * 0.98:
                regimes.append('bear')
            else:
                regimes.append('sideways')

    return pd.Series(regimes, index=df.index)


def get_prediction_methods():
    """
    Return commodity-specific prediction methods.

    Methods:
    - Naive: Last observed price
    - MA5: 5-day moving average
    - MA20: 20-day moving average
    - MeanRevert: Mean-reverting model
    - Trend: Trend-following model
    """

    def naive_predict(df, idx):
        """Predict next value = current value."""
        if idx < 1:
            return df['price'].iloc[0]
        return df['price'].iloc[idx - 1]

    def ma5_predict(df, idx):
        """5-day moving average prediction."""
        if idx < 5:
            return df['price'].iloc[:max(idx, 1)].mean()
        return df['price'].iloc[idx-5:idx].mean()

    def ma20_predict(df, idx):
        """20-day moving average prediction."""
        if idx < 20:
            return df['price'].iloc[:max(idx, 1)].mean()
        return df['price'].iloc[idx-20:idx].mean()

    def mean_revert_predict(df, idx):
        """Mean-reverting prediction (fade extremes)."""
        if idx < 20:
            return df['price'].iloc[:max(idx, 1)].mean()

        ma20 = df['price'].iloc[idx-20:idx].mean()
        current = df['price'].iloc[idx - 1] if idx > 0 else ma20

        # Predict reversion toward mean
        alpha = 0.3  # Reversion speed
        return current + alpha * (ma20 - current)

    def trend_predict(df, idx):
        """Trend-following prediction."""
        if idx < 5:
            return df['price'].iloc[max(idx-1, 0)]

        # Calculate recent trend
        prices = df['price'].iloc[idx-5:idx].values
        trend = (prices[-1] - prices[0]) / 5

        return prices[-1] + trend

    return {
        'naive': naive_predict,
        'ma5': ma5_predict,
        'ma20': ma20_predict,
        'mean_revert': mean_revert_predict,
        'trend': trend_predict,
    }


def evaluate_prediction(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Evaluate prediction performance."""

    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))

    # Relative errors
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
    }


def get_regime_counts() -> dict:
    """Get regime distribution from data."""
    df = load_data()
    return df['regime'].value_counts().to_dict()
