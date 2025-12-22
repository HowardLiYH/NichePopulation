#!/usr/bin/env python3
"""
Download REAL commodity price data - Alternative approach with delays.

Uses pandas_datareader or direct CSV sources.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import time

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def download_with_yfinance_retry(symbols: dict, start_date: str, end_date: str) -> pd.DataFrame:
    """Download with retries and delays."""
    try:
        import yfinance as yf
    except ImportError:
        os.system(f"{sys.executable} -m pip install yfinance -q")
        import yfinance as yf

    all_data = []

    for name, symbol in symbols.items():
        print(f"  Downloading {name} ({symbol})...")

        for attempt in range(3):
            try:
                time.sleep(2)  # Wait between requests

                df = yf.download(symbol, start=start_date, end=end_date, progress=False)

                if len(df) > 0:
                    df = df.reset_index()
                    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
                    df['commodity'] = name
                    df['price'] = df['close']

                    print(f"    SUCCESS: {len(df)} records")
                    all_data.append(df)
                    break

            except Exception as e:
                print(f"    Attempt {attempt+1} failed: {e}")
                time.sleep(5)  # Wait longer between retries

    return pd.concat(all_data, ignore_index=True) if all_data else None


def download_from_stooq() -> pd.DataFrame:
    """Download from Stooq (alternative free source)."""
    print("\n>>> Trying Stooq.com as alternative source...")

    # Stooq provides free historical data
    symbols = {
        'WTI_Oil': 'CL.F',
        'Gold': 'GC.F',
        'Corn': 'ZC.F',
        'Copper': 'HG.F',
    }

    all_data = []

    for name, symbol in symbols.items():
        try:
            url = f"https://stooq.com/q/d/l/?s={symbol}&d1=20150101&d2=20241231"
            print(f"  Downloading {name} from Stooq...")

            df = pd.read_csv(url)

            if len(df) > 0:
                df.columns = [c.lower() for c in df.columns]
                df['date'] = pd.to_datetime(df['date'])
                df['commodity'] = name
                df['price'] = df['close']

                print(f"    Downloaded {len(df)} records")
                all_data.append(df)

            time.sleep(1)

        except Exception as e:
            print(f"    Failed: {e}")

    return pd.concat(all_data, ignore_index=True) if all_data else None


def process_commodity_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add features and regimes to commodity data."""

    df = df.sort_values(['commodity', 'date']).reset_index(drop=True)

    features = []
    for comm in df['commodity'].unique():
        comm_df = df[df['commodity'] == comm].copy()

        comm_df['returns'] = comm_df['close'].pct_change()
        comm_df['price_ma5'] = comm_df['close'].rolling(5).mean()
        comm_df['price_ma20'] = comm_df['close'].rolling(20).mean()
        comm_df['returns_ma5'] = comm_df['returns'].rolling(5).mean()
        comm_df['vol_20'] = comm_df['returns'].rolling(20).std()

        comm_df['month'] = comm_df['date'].dt.month

        # Regime detection
        def get_regime(row):
            if pd.isna(row['vol_20']) or pd.isna(row['price_ma20']):
                return 'sideways'

            high_vol = row['vol_20'] > comm_df['vol_20'].quantile(0.75)
            trend_up = row['close'] > row['price_ma20']
            momentum = row['close'] > row['price_ma5']

            if high_vol:
                return 'volatile'
            elif trend_up and momentum:
                return 'bull'
            elif not trend_up and not momentum:
                return 'bear'
            else:
                return 'sideways'

        comm_df['regime'] = comm_df.apply(get_regime, axis=1)
        features.append(comm_df)

    result = pd.concat(features, ignore_index=True)
    result = result.dropna(subset=['close', 'returns']).reset_index(drop=True)

    return result


def main():
    print("=" * 60)
    print("DOWNLOADING REAL COMMODITY DATA (WITH RETRIES)")
    print("=" * 60)

    symbols = {
        'WTI_Oil': 'CL=F',
        'Gold': 'GC=F',
        'Corn': 'ZC=F',
        'Copper': 'HG=F',
    }

    # Try yfinance first with delays
    df = download_with_yfinance_retry(symbols, "2015-01-01", "2024-12-31")

    # If that fails, try Stooq
    if df is None or len(df) == 0:
        df = download_from_stooq()

    if df is None or len(df) == 0:
        print("\nERROR: Could not download commodity data from any source!")
        print("Please try again later or download manually.")
        return

    # Process data
    df = process_commodity_data(df)

    # Save
    output_path = Path(__file__).parent.parent / "data" / "commodities" / "real_prices.csv"
    os.makedirs(output_path.parent, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Total records: {len(df)}")
    print(f"Commodities: {df['commodity'].nunique()}")
    for comm in df['commodity'].unique():
        count = len(df[df['commodity'] == comm])
        print(f"  {comm}: {count} records")
    print(f"\nRegime distribution:")
    print(df['regime'].value_counts())
    print(f"\nSaved to: {output_path}")
    print("\n>>> REAL COMMODITY DATA READY <<<")


if __name__ == "__main__":
    main()
