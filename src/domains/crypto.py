"""
Crypto/Finance Domain - Using REAL Bybit exchange data.

Data Source: Bybit Historical OHLCV Data
- Coins: BTC, ETH, SOL, DOGE, XRP
- ~8,767 records per coin
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "bybit"


def load_data(symbol: str = "BTC") -> pd.DataFrame:
    """Load REAL crypto data from Bybit."""

    file_path = DATA_DIR / f"Bybit_{symbol}.csv"

    if file_path.exists():
        df = pd.read_csv(file_path)

        # Parse timestamp if present
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        elif 'date' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'])

        print(f"[Crypto] Loaded {len(df)} REAL records for {symbol} from Bybit")
        return df

    raise FileNotFoundError(
        f"Real crypto data not found at {file_path}. "
        "Bybit data should be pre-downloaded."
    )


def load_all_coins() -> pd.DataFrame:
    """Load all available coins."""
    coins = ['BTC', 'ETH', 'SOL', 'DOGE', 'XRP']
    all_data = []

    for coin in coins:
        try:
            df = load_data(coin)
            df['symbol'] = coin
            all_data.append(df)
        except FileNotFoundError:
            continue

    if all_data:
        return pd.concat(all_data, ignore_index=True)

    raise FileNotFoundError("No crypto data found")


def detect_regime(df: pd.DataFrame) -> pd.Series:
    """
    Detect market regimes for crypto.

    Regimes:
    - bull: Strong uptrend (price > MA20, positive momentum)
    - bear: Strong downtrend (price < MA20, negative momentum)
    - volatile: High volatility periods
    - sideways: Low volatility, range-bound
    """

    if 'regime' in df.columns:
        return df['regime']

    # Calculate regime indicators
    df = df.copy()

    close_col = 'close' if 'close' in df.columns else df.columns[4]  # Assume OHLCV format

    df['returns'] = df[close_col].pct_change()
    df['ma20'] = df[close_col].rolling(20).mean()
    df['vol_20'] = df['returns'].rolling(20).std()

    vol_75 = df['vol_20'].quantile(0.75)

    def get_regime(row):
        if pd.isna(row['vol_20']) or pd.isna(row['ma20']):
            return 'sideways'

        if row['vol_20'] > vol_75:
            return 'volatile'
        elif row[close_col] > row['ma20'] * 1.02:
            return 'bull'
        elif row[close_col] < row['ma20'] * 0.98:
            return 'bear'
        else:
            return 'sideways'

    return df.apply(get_regime, axis=1)


def get_prediction_methods():
    """
    Return crypto-specific prediction/trading methods.

    Methods:
    - Naive: Buy and hold
    - MomentumShort: 5-period momentum
    - MomentumLong: 20-period momentum
    - MeanRevert: Fade moves from MA
    - Trend: Trend following
    """

    def naive_predict(df, idx):
        """Persistence (last price)."""
        close_col = 'close' if 'close' in df.columns else df.columns[4]
        if idx < 1:
            return df[close_col].iloc[0]
        return df[close_col].iloc[idx - 1]

    def momentum_short_predict(df, idx):
        """5-period momentum."""
        close_col = 'close' if 'close' in df.columns else df.columns[4]
        if idx < 5:
            return df[close_col].iloc[max(idx-1, 0)]

        # Predict continuation
        prices = df[close_col].iloc[idx-5:idx].values
        momentum = prices[-1] - prices[0]

        return prices[-1] + momentum * 0.1

    def momentum_long_predict(df, idx):
        """20-period momentum."""
        close_col = 'close' if 'close' in df.columns else df.columns[4]
        if idx < 20:
            return df[close_col].iloc[max(idx-1, 0)]

        prices = df[close_col].iloc[idx-20:idx].values
        momentum = prices[-1] - prices[0]

        return prices[-1] + momentum * 0.05

    def mean_revert_predict(df, idx):
        """Mean reversion toward MA20."""
        close_col = 'close' if 'close' in df.columns else df.columns[4]
        if idx < 20:
            return df[close_col].iloc[:max(idx, 1)].mean()

        ma20 = df[close_col].iloc[idx-20:idx].mean()
        current = df[close_col].iloc[idx - 1]

        return current + 0.2 * (ma20 - current)

    def trend_predict(df, idx):
        """Trend following."""
        close_col = 'close' if 'close' in df.columns else df.columns[4]
        if idx < 10:
            return df[close_col].iloc[max(idx-1, 0)]

        prices = df[close_col].iloc[idx-10:idx].values
        trend = np.polyfit(range(10), prices, 1)[0]

        return prices[-1] + trend

    return {
        'naive': naive_predict,
        'momentum_short': momentum_short_predict,
        'momentum_long': momentum_long_predict,
        'mean_revert': mean_revert_predict,
        'trend': trend_predict,
    }


def evaluate_prediction(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Evaluate prediction performance."""

    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))

    # For finance, also calculate directional accuracy
    if len(y_true) > 1:
        actual_dir = np.sign(np.diff(y_true))
        pred_dir = np.sign(y_pred[:-1] - y_true[:-1])
        dir_acc = np.mean(actual_dir == pred_dir)
    else:
        dir_acc = 0.5

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'directional_accuracy': dir_acc,
    }


def get_regime_counts() -> dict:
    """Get regime distribution from data."""
    df = load_data('BTC')
    regimes = detect_regime(df)
    return regimes.value_counts().to_dict()
