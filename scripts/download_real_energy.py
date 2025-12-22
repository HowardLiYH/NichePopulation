#!/usr/bin/env python3
"""
Download REAL electricity demand data from Open Power System Data.

NO API KEY REQUIRED - Direct CSV download.

Source: https://data.open-power-system-data.org/time_series/
"""

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

try:
    import requests
except ImportError:
    os.system(f"{sys.executable} -m pip install requests")
    import requests

sys.path.insert(0, str(Path(__file__).parent.parent))


# Open Power System Data URLs
OPSD_URLS = {
    # Hourly time series for electricity consumption
    'hourly': 'https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv',
    # Alternative: use a smaller subset
    'daily': 'https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv',
}


def download_opsd_data(url: str, output_path: str) -> pd.DataFrame:
    """
    Download electricity demand data from Open Power System Data.
    """
    print("=" * 60)
    print("DOWNLOADING REAL ELECTRICITY DEMAND DATA")
    print("=" * 60)
    print(f"Source: Open Power System Data")
    print(f"URL: {url[:80]}...")

    try:
        print("\n>>> Downloading (this may take a minute for large files)...")

        # Download with streaming for large files
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        # Save to temp file first
        temp_path = Path(output_path).parent / "temp_opsd.csv"
        os.makedirs(temp_path.parent, exist_ok=True)

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    print(f"\r    Progress: {pct:.1f}%", end='', flush=True)

        print(f"\n    Downloaded {downloaded / 1024 / 1024:.1f} MB")

        # Read and process
        print(">>> Processing data...")
        df = pd.read_csv(temp_path, low_memory=False)

        print(f"    Raw data: {len(df)} rows, {len(df.columns)} columns")

        # Find load columns (electricity demand)
        load_cols = [c for c in df.columns if 'load' in c.lower() and 'actual' in c.lower()]

        if not load_cols:
            # Try alternative column names
            load_cols = [c for c in df.columns if 'DE_load' in c or 'GB_load' in c or 'FR_load' in c]

        if not load_cols:
            print(f"    Available columns: {df.columns[:20].tolist()}")
            # Use first numeric column that looks like load
            for col in df.columns:
                if df[col].dtype in ['float64', 'int64'] and 'load' in col.lower():
                    load_cols = [col]
                    break

        if not load_cols:
            print("    WARNING: Could not find load columns, using first numeric column")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                load_cols = [numeric_cols[0]]

        print(f"    Using load columns: {load_cols[:5]}")

        # Process the data
        # Use UTC timestamp
        if 'utc_timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['utc_timestamp'])
        elif 'cet_cest_timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['cet_cest_timestamp'])
        else:
            df['datetime'] = pd.to_datetime(df.iloc[:, 0])

        # Filter to recent years
        df = df[df['datetime'] >= '2020-01-01'].copy()

        # Use Germany load if available, otherwise first load column
        if 'DE_load_actual_entsoe_transparency' in df.columns:
            df['demand_mw'] = df['DE_load_actual_entsoe_transparency']
            df['country'] = 'Germany'
        elif len(load_cols) > 0:
            df['demand_mw'] = df[load_cols[0]]
            df['country'] = load_cols[0].split('_')[0]

        # Remove NaN
        df = df.dropna(subset=['demand_mw'])

        # Normalize demand to 0-1 range
        df['demand'] = (df['demand_mw'] - df['demand_mw'].min()) / (df['demand_mw'].max() - df['demand_mw'].min())

        # Add features
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month

        df['demand_lag1'] = df['demand'].shift(1)
        df['demand_lag24'] = df['demand'].shift(24)
        df['demand_ma24'] = df['demand'].rolling(24).mean()
        df['demand_std24'] = df['demand'].rolling(24).std()

        # Assign regimes
        def get_regime(row):
            if pd.isna(row['demand']):
                return 'normal'
            if row['demand'] > 0.8:
                return 'peak_demand'
            elif row['demand'] < 0.3:
                return 'low_demand'
            elif row['month'] in [6, 7, 8] and row['demand'] > 0.6:
                return 'high_load'
            else:
                return 'normal'

        df['regime'] = df.apply(get_regime, axis=1)

        # Select columns
        result = df[['datetime', 'demand', 'demand_mw', 'hour', 'day_of_week', 'month',
                     'demand_lag1', 'demand_lag24', 'demand_ma24', 'demand_std24', 'regime']].copy()
        result = result.dropna().reset_index(drop=True)

        # Save
        result.to_csv(output_path, index=False)

        # Cleanup temp file
        if temp_path.exists():
            os.remove(temp_path)

        print(f"\n>>> Processed {len(result)} records")
        print(f">>> Date range: {result['datetime'].min()} to {result['datetime'].max()}")
        print(f">>> Regime distribution:")
        print(result['regime'].value_counts())
        print(f"\n>>> Saved to: {output_path}")

        return result

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Download real energy demand data."""
    output_path = Path(__file__).parent.parent / "data" / "energy" / "opsd_real_demand.csv"

    df = download_opsd_data(
        url=OPSD_URLS['hourly'],
        output_path=str(output_path),
    )

    if df is not None:
        print("\n>>> REAL ENERGY DATA READY <<<")
    else:
        print("\nFailed to download energy data. Trying alternative source...")
        # Could add fallback to EIA or other sources here


if __name__ == "__main__":
    main()
