#!/usr/bin/env python3
"""
Bybit Historical Data Collection Script

Collects OHLCV data for multiple assets and intervals from Bybit API.
Saves to local CSV files for fast backtesting.

Usage:
    python scripts/collect_bybit_data.py
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Bybit API configuration
BYBIT_KLINE_URL = "https://api.bybit.com/v5/market/kline"
MAX_LIMIT = 1000  # Bybit max candles per request
RATE_LIMIT_DELAY = 0.1  # seconds between requests

# Assets and intervals to collect
ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]
INTERVALS = {
    "1D": {"bybit": "D", "minutes": 1440},
    "4H": {"bybit": "240", "minutes": 240},
    "1H": {"bybit": "60", "minutes": 60},
    "15m": {"bybit": "15", "minutes": 15},
    "5m": {"bybit": "5", "minutes": 5},
}

# Date range
START_DATE = datetime(2021, 1, 1)
END_DATE = datetime(2024, 12, 31, 23, 59, 59)

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "bybit"


def timestamp_to_ms(dt: datetime) -> int:
    """Convert datetime to milliseconds timestamp."""
    return int(dt.timestamp() * 1000)


def ms_to_datetime(ms: int) -> datetime:
    """Convert milliseconds timestamp to datetime."""
    return datetime.fromtimestamp(ms / 1000)


def fetch_klines(
    symbol: str,
    interval: str,
    start_time: int,
    end_time: int,
    limit: int = MAX_LIMIT
) -> Optional[List]:
    """
    Fetch kline data from Bybit API.

    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        interval: Bybit interval code (e.g., "D", "240", "60")
        start_time: Start timestamp in milliseconds
        end_time: End timestamp in milliseconds
        limit: Max number of candles to fetch

    Returns:
        List of kline data or None if error
    """
    params = {
        "category": "linear",  # Perpetual contracts
        "symbol": symbol,
        "interval": interval,
        "start": start_time,
        "end": end_time,
        "limit": limit,
    }

    try:
        response = requests.get(BYBIT_KLINE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data.get("retCode") != 0:
            logger.error(f"API error: {data.get('retMsg')}")
            return None

        return data.get("result", {}).get("list", [])

    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return None


def collect_asset_interval(
    symbol: str,
    interval_name: str,
    interval_config: dict,
    start_dt: datetime,
    end_dt: datetime
) -> pd.DataFrame:
    """
    Collect all data for one asset and interval.

    Args:
        symbol: Trading pair
        interval_name: Human-readable interval (e.g., "1H")
        interval_config: Config dict with bybit code and minutes
        start_dt: Start datetime
        end_dt: End datetime

    Returns:
        DataFrame with OHLCV data
    """
    bybit_interval = interval_config["bybit"]
    interval_minutes = interval_config["minutes"]

    all_data = []
    current_end = end_dt

    logger.info(f"  Collecting {symbol} {interval_name}...")

    # Bybit returns data in reverse chronological order
    # We paginate backwards from end_dt to start_dt
    iterations = 0
    max_iterations = 10000  # Safety limit

    while current_end > start_dt and iterations < max_iterations:
        iterations += 1

        # Fetch batch
        klines = fetch_klines(
            symbol=symbol,
            interval=bybit_interval,
            start_time=timestamp_to_ms(start_dt),
            end_time=timestamp_to_ms(current_end),
            limit=MAX_LIMIT
        )

        if not klines:
            logger.warning(f"  No data returned for {symbol} {interval_name} at {current_end}")
            break

        # Bybit returns [[timestamp, open, high, low, close, volume, turnover], ...]
        # Data is in reverse chronological order
        all_data.extend(klines)

        # Get oldest timestamp from this batch
        oldest_ts = int(klines[-1][0])
        oldest_dt = ms_to_datetime(oldest_ts)

        # Move end pointer before oldest data
        current_end = oldest_dt - timedelta(minutes=1)

        # Log progress every 10 iterations
        if iterations % 10 == 0:
            logger.info(f"    Progress: {len(all_data)} candles, oldest: {oldest_dt.strftime('%Y-%m-%d')}")

        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)

        # Break if we've gone past start
        if oldest_dt <= start_dt:
            break

    if not all_data:
        logger.warning(f"  No data collected for {symbol} {interval_name}")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])

    # Convert types
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove turnover column (not needed)
    df = df.drop(columns=["turnover"])

    # Sort by timestamp ascending
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Remove duplicates
    df = df.drop_duplicates(subset=["timestamp"], keep="first")

    # Filter to date range
    df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)]

    logger.info(f"  Collected {len(df)} candles for {symbol} {interval_name}")

    return df


def save_to_csv(df: pd.DataFrame, symbol: str, interval: str) -> str:
    """
    Save DataFrame to CSV file.

    Args:
        df: OHLCV DataFrame
        symbol: Trading pair
        interval: Interval name

    Returns:
        Path to saved file
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"{symbol}_{interval}.csv"
    filepath = OUTPUT_DIR / filename

    df.to_csv(filepath, index=False)
    logger.info(f"  Saved to {filepath}")

    return str(filepath)


def collect_all():
    """Collect data for all assets and intervals."""
    logger.info("=" * 60)
    logger.info("Bybit Data Collection")
    logger.info(f"Assets: {ASSETS}")
    logger.info(f"Intervals: {list(INTERVALS.keys())}")
    logger.info(f"Date range: {START_DATE.date()} to {END_DATE.date()}")
    logger.info("=" * 60)

    summary = []

    for asset in ASSETS:
        logger.info(f"\nCollecting {asset}...")

        for interval_name, interval_config in INTERVALS.items():
            try:
                df = collect_asset_interval(
                    symbol=asset,
                    interval_name=interval_name,
                    interval_config=interval_config,
                    start_dt=START_DATE,
                    end_dt=END_DATE
                )

                if not df.empty:
                    filepath = save_to_csv(df, asset, interval_name)
                    summary.append({
                        "asset": asset,
                        "interval": interval_name,
                        "rows": len(df),
                        "start": df["timestamp"].min(),
                        "end": df["timestamp"].max(),
                        "file": filepath,
                        "status": "OK"
                    })
                else:
                    summary.append({
                        "asset": asset,
                        "interval": interval_name,
                        "rows": 0,
                        "start": None,
                        "end": None,
                        "file": None,
                        "status": "EMPTY"
                    })

            except Exception as e:
                logger.error(f"Error collecting {asset} {interval_name}: {e}")
                summary.append({
                    "asset": asset,
                    "interval": interval_name,
                    "rows": 0,
                    "start": None,
                    "end": None,
                    "file": None,
                    "status": f"ERROR: {e}"
                })

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("COLLECTION SUMMARY")
    logger.info("=" * 60)

    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))

    # Save summary
    summary_path = OUTPUT_DIR / "collection_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"\nSummary saved to {summary_path}")

    # Stats
    total_rows = summary_df["rows"].sum()
    ok_count = len(summary_df[summary_df["status"] == "OK"])
    total_count = len(summary_df)

    logger.info(f"\nTotal: {ok_count}/{total_count} collections successful")
    logger.info(f"Total rows: {total_rows:,}")

    return summary_df


if __name__ == "__main__":
    collect_all()
