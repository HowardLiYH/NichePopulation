#!/usr/bin/env python3
"""
Download REAL commodity price data from FRED (Federal Reserve).

NO API KEY REQUIRED for basic access.

FRED has real commodity prices:
- DCOILWTICO: WTI Crude Oil
- GOLDAMGBD228NLBM: Gold (London Fixing)
- PCOPPUSDM: Copper
- PMAIZMTUSDM: Corn (Maize)
"""

import os
import sys
from pathlib import Path
import time

import pandas as pd
import numpy as np

try:
    import requests
except ImportError:
    os.system(f"{sys.executable} -m pip install requests")
    import requests

sys.path.insert(0, str(Path(__file__).parent.parent))


# FRED commodity series IDs
FRED_SERIES = {
    'WTI_Oil': 'DCOILWTICO',           # Daily WTI Crude Oil
    'Gold': 'GOLDAMGBD228NLBM',         # Daily Gold Price (London)
    'Copper': 'PCOPPUSDM',              # Monthly Copper
    'Natural_Gas': 'DHHNGSP',           # Daily Natural Gas (Henry Hub)
}


def download_fred_series(series_id: str, name: str) -> pd.DataFrame:
    """
    Download a FRED time series without API key.
    Uses the public CSV download endpoint.
    """
    print(f"  Downloading {name} ({series_id})...")

    # FRED public CSV download
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Parse CSV from response
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))

        if len(df) == 0:
            print(f"    No data returned")
            return None

        # FRED CSVs have DATE and the series name as columns
        df.columns = ['date', 'price']
        df['date'] = pd.to_datetime(df['date'])

        # Remove missing values (FRED uses '.' for missing)
        df = df[df['price'] != '.'].copy()
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df.dropna(subset=['price'])

        # Filter to 2015+
        df = df[df['date'] >= '2015-01-01'].copy()

        df['commodity'] = name

        print(f"    Downloaded {len(df)} records")

        return df

    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def process_commodity_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add features and regimes."""

    df = df.sort_values(['commodity', 'date']).reset_index(drop=True)

    features = []
    for comm in df['commodity'].unique():
        comm_df = df[df['commodity'] == comm].copy()

        comm_df['returns'] = comm_df['price'].pct_change()
        comm_df['price_ma5'] = comm_df['price'].rolling(5).mean()
        comm_df['price_ma20'] = comm_df['price'].rolling(20).mean()
        comm_df['vol_20'] = comm_df['returns'].rolling(20).std()

        comm_df['month'] = comm_df['date'].dt.month

        # Regime based on trend and volatility
        vol_75 = comm_df['vol_20'].quantile(0.75)

        def get_regime(row):
            if pd.isna(row['vol_20']) or pd.isna(row['price_ma20']):
                return 'sideways'
            if row['vol_20'] > vol_75:
                return 'volatile'
            elif row['price'] > row['price_ma20'] * 1.02:
                return 'bull'
            elif row['price'] < row['price_ma20'] * 0.98:
                return 'bear'
            else:
                return 'sideways'

        comm_df['regime'] = comm_df.apply(get_regime, axis=1)
        features.append(comm_df)

    result = pd.concat(features, ignore_index=True)
    result = result.dropna(subset=['price', 'returns']).reset_index(drop=True)

    return result


def main():
    print("=" * 60)
    print("DOWNLOADING REAL COMMODITY DATA FROM FRED")
    print("=" * 60)

    all_data = []

    for name, series_id in FRED_SERIES.items():
        df = download_fred_series(series_id, name)

        if df is not None and len(df) > 0:
            all_data.append(df)

        time.sleep(1)  # Be polite

    if len(all_data) == 0:
        print("\nERROR: Could not download any FRED data!")
        return

    combined = pd.concat(all_data, ignore_index=True)
    combined = process_commodity_data(combined)

    # Save
    output_path = Path(__file__).parent.parent / "data" / "commodities" / "fred_real_prices.csv"
    os.makedirs(output_path.parent, exist_ok=True)
    combined.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Total records: {len(combined)}")
    for comm in combined['commodity'].unique():
        count = len(combined[combined['commodity'] == comm])
        date_range = combined[combined['commodity'] == comm]['date']
        print(f"  {comm}: {count} records ({date_range.min().date()} to {date_range.max().date()})")
    print(f"\nRegime distribution:")
    print(combined['regime'].value_counts())
    print(f"\nSaved to: {output_path}")
    print("\n>>> REAL COMMODITY DATA FROM FRED READY <<<")


if __name__ == "__main__":
    main()
