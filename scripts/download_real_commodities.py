#!/usr/bin/env python3
"""
Download REAL commodity price data from Yahoo Finance.

Uses yfinance library - NO API KEY REQUIRED.

Commodities:
- CL=F: Crude Oil (WTI)
- GC=F: Gold
- ZC=F: Corn
- HG=F: Copper
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Install yfinance if not available
try:
    import yfinance as yf
except ImportError:
    print("Installing yfinance...")
    os.system(f"{sys.executable} -m pip install yfinance")
    import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))


def download_commodity_data(
    symbols: dict,
    start_date: str = "2015-01-01",
    end_date: str = "2024-12-31",
    output_path: str = None,
) -> pd.DataFrame:
    """
    Download REAL commodity data from Yahoo Finance.

    Args:
        symbols: Dict mapping commodity name to Yahoo symbol
        start_date: Start date
        end_date: End date
        output_path: Path to save CSV

    Returns:
        DataFrame with commodity prices and regimes
    """
    print("=" * 60)
    print("DOWNLOADING REAL COMMODITY DATA FROM YAHOO FINANCE")
    print("=" * 60)

    all_data = []

    for name, symbol in symbols.items():
        print(f"\n>>> Downloading {name} ({symbol})...")

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)

            if len(df) == 0:
                print(f"    WARNING: No data for {symbol}, trying download method...")
                df = yf.download(symbol, start=start_date, end=end_date, progress=False)

            if len(df) == 0:
                print(f"    ERROR: Could not download {name}")
                continue

            print(f"    Downloaded {len(df)} records")
            print(f"    Date range: {df.index.min()} to {df.index.max()}")

            # Process data
            df = df.reset_index()
            df.columns = [c.lower().replace(' ', '_') for c in df.columns]

            # Rename date column if needed
            if 'date' not in df.columns and 'datetime' in df.columns:
                df = df.rename(columns={'datetime': 'date'})
            elif 'date' not in df.columns:
                df = df.rename(columns={df.columns[0]: 'date'})

            df['commodity'] = name
            df['price'] = df['close']

            # Calculate returns
            df['returns'] = df['close'].pct_change()

            # Add features
            df['price_ma5'] = df['close'].rolling(5).mean()
            df['price_ma20'] = df['close'].rolling(20).mean()
            df['returns_ma5'] = df['returns'].rolling(5).mean()
            df['vol_20'] = df['returns'].rolling(20).std()

            # Calculate regimes based on trend and volatility
            df['trend'] = np.where(df['close'] > df['price_ma20'], 'up', 'down')
            df['high_vol'] = df['vol_20'] > df['vol_20'].quantile(0.75)

            def get_regime(row):
                if pd.isna(row['vol_20']):
                    return 'normal'
                if row['high_vol']:
                    return 'volatile'
                elif row['trend'] == 'up' and row['close'] > row['price_ma5']:
                    return 'bull'
                elif row['trend'] == 'down' and row['close'] < row['price_ma5']:
                    return 'bear'
                else:
                    return 'sideways'

            df['regime'] = df.apply(get_regime, axis=1)

            all_data.append(df)

        except Exception as e:
            print(f"    ERROR downloading {name}: {e}")
            continue

    if len(all_data) == 0:
        print("\nERROR: Could not download any commodity data!")
        return None

    # Combine all commodities
    combined = pd.concat(all_data, ignore_index=True)

    # Select columns
    columns = ['date', 'commodity', 'price', 'open', 'high', 'low', 'close', 'volume',
               'returns', 'price_ma5', 'price_ma20', 'returns_ma5', 'vol_20', 'regime']
    combined = combined[[c for c in columns if c in combined.columns]]

    # Drop NaN rows
    combined = combined.dropna(subset=['price', 'returns']).reset_index(drop=True)

    # Save
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined.to_csv(output_path, index=False)
        print(f"\n>>> Saved {len(combined)} records to: {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    for name in symbols.keys():
        count = len(combined[combined['commodity'] == name])
        if count > 0:
            print(f"  {name}: {count} records")

    print(f"\nRegime distribution:")
    print(combined['regime'].value_counts())

    print("\n>>> REAL COMMODITY DATA READY <<<")

    return combined


def main():
    """Download real commodity data."""

    # Yahoo Finance commodity symbols
    symbols = {
        'WTI_Oil': 'CL=F',      # Crude Oil WTI
        'Gold': 'GC=F',          # Gold
        'Corn': 'ZC=F',          # Corn
        'Copper': 'HG=F',        # Copper
    }

    output_path = Path(__file__).parent.parent / "data" / "commodities" / "yahoo_real_prices.csv"

    df = download_commodity_data(
        symbols=symbols,
        start_date="2015-01-01",
        end_date="2024-12-31",
        output_path=str(output_path),
    )

    if df is not None:
        # Also save individual commodity files
        for name in symbols.keys():
            commodity_df = df[df['commodity'] == name]
            if len(commodity_df) > 0:
                individual_path = output_path.parent / f"yahoo_{name.lower()}.csv"
                commodity_df.to_csv(individual_path, index=False)
                print(f"  Saved {name} to: {individual_path}")


if __name__ == "__main__":
    main()
