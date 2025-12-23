#!/usr/bin/env python3
"""
Download NYC Taxi TLC Trip Record Data.

Source: NYC Taxi & Limousine Commission
URL: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
Format: Parquet files
License: Public domain

This script downloads Yellow Taxi trip data and aggregates it
into hourly trip counts for regime-based analysis.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json

# Try to import required libraries
try:
    import requests
    import pandas as pd
    import pyarrow.parquet as pq
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("Warning: Some dependencies missing. Will generate synthetic data.")

# Output directory
DATA_DIR = Path(__file__).parent.parent / "data" / "traffic"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_taxi_parquet(year: int, month: int) -> pd.DataFrame:
    """Download a single month of Yellow Taxi data."""
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"

    print(f"Downloading: {url}")

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        # Save temporarily
        temp_file = DATA_DIR / f"temp_{year}_{month:02d}.parquet"
        with open(temp_file, 'wb') as f:
            f.write(response.content)

        # Read parquet
        df = pd.read_parquet(temp_file)

        # Clean up
        temp_file.unlink()

        return df

    except Exception as e:
        print(f"  Error downloading {year}-{month:02d}: {e}")
        return None


def aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trip data to hourly counts."""
    # Use pickup datetime
    pickup_col = None
    for col in ['tpep_pickup_datetime', 'pickup_datetime', 'lpep_pickup_datetime']:
        if col in df.columns:
            pickup_col = col
            break

    if pickup_col is None:
        raise ValueError(f"No pickup datetime column found. Columns: {df.columns.tolist()}")

    df['pickup_datetime'] = pd.to_datetime(df[pickup_col])
    df['hour'] = df['pickup_datetime'].dt.floor('H')

    # Aggregate
    hourly = df.groupby('hour').agg({
        pickup_col: 'count',  # trip count
    }).reset_index()

    hourly.columns = ['timestamp', 'trip_count']

    return hourly


def detect_regime(row) -> str:
    """Detect regime based on temporal patterns."""
    hour = row['timestamp'].hour
    weekday = row['timestamp'].weekday()

    if weekday >= 5:  # Weekend
        return 'weekend'
    elif 7 <= hour <= 9:
        return 'morning_rush'
    elif 17 <= hour <= 19:
        return 'evening_rush'
    elif 0 <= hour <= 5:
        return 'night'
    else:
        return 'midday'


def generate_synthetic_taxi_data(n_days: int = 365) -> pd.DataFrame:
    """
    Generate synthetic but realistic taxi trip data.

    Based on known NYC taxi patterns:
    - Morning rush: 7-9 AM, ~5000 trips/hour
    - Evening rush: 5-7 PM, ~6000 trips/hour
    - Midday: 10 AM - 4 PM, ~3000 trips/hour
    - Night: 10 PM - 6 AM, ~1000 trips/hour
    - Weekend: 30% lower than weekday
    """
    import numpy as np

    np.random.seed(42)

    # Generate hourly timestamps
    start_date = datetime(2023, 1, 1)
    timestamps = pd.date_range(start=start_date, periods=n_days * 24, freq='H')

    trip_counts = []
    regimes = []

    for ts in timestamps:
        hour = ts.hour
        weekday = ts.weekday()

        # Base trip count by hour
        if 7 <= hour <= 9:
            base = 5000
            regime = 'morning_rush'
        elif 17 <= hour <= 19:
            base = 6000
            regime = 'evening_rush'
        elif 10 <= hour <= 16:
            base = 3000
            regime = 'midday'
        elif 0 <= hour <= 5:
            base = 800
            regime = 'night'
        else:
            base = 2000
            regime = 'midday'

        # Weekend adjustment
        if weekday >= 5:
            base = int(base * 0.7)
            regime = 'weekend'

        # Add noise
        noise = np.random.normal(0, base * 0.15)
        trip_count = max(100, int(base + noise))

        trip_counts.append(trip_count)
        regimes.append(regime)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'trip_count': trip_counts,
        'regime': regimes,
    })

    return df


def main():
    """Download or generate NYC Taxi data."""
    print("="*60)
    print("NYC TAXI TLC DATA DOWNLOAD")
    print("="*60)

    output_file = DATA_DIR / "nyc_taxi_hourly.csv"

    if HAS_DEPS:
        print("\nAttempting to download real data from NYC TLC...")

        # Try to download 2023 data (most recent complete year)
        all_data = []

        for month in range(1, 13):
            df = download_taxi_parquet(2023, month)
            if df is not None:
                hourly = aggregate_hourly(df)
                all_data.append(hourly)
                print(f"  2023-{month:02d}: {len(hourly)} hours")

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            combined['regime'] = combined.apply(detect_regime, axis=1)
            combined.to_csv(output_file, index=False)

            print(f"\n✅ Downloaded {len(combined)} hours of real data")
            print(f"   Saved to: {output_file}")

            # Print regime distribution
            print("\nRegime distribution:")
            for regime, count in combined['regime'].value_counts().items():
                pct = count / len(combined) * 100
                print(f"  {regime}: {count} ({pct:.1f}%)")

            return

    # Fall back to synthetic data
    print("\nGenerating synthetic taxi data based on NYC patterns...")
    df = generate_synthetic_taxi_data(n_days=365)
    df.to_csv(output_file, index=False)

    print(f"\n✅ Generated {len(df)} hours of synthetic data")
    print(f"   Saved to: {output_file}")

    # Print regime distribution
    print("\nRegime distribution:")
    for regime, count in df['regime'].value_counts().items():
        pct = count / len(df) * 100
        print(f"  {regime}: {count} ({pct:.1f}%)")

    # Create manifest
    manifest = {
        'source': 'NYC TLC (synthetic patterns)',
        'url': 'https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page',
        'records': len(df),
        'columns': df.columns.tolist(),
        'regimes': df['regime'].unique().tolist(),
        'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
        'generated': datetime.now().isoformat(),
    }

    with open(DATA_DIR / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2, default=str)


if __name__ == "__main__":
    main()
