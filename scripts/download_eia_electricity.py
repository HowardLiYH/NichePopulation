#!/usr/bin/env python3
"""
Download US Electricity Demand Data from EIA.

Source: US Energy Information Administration (EIA)
URL: https://www.eia.gov/opendata/
API: Free with registration (or use public datasets)
License: Public domain (US Government)

This script downloads hourly electricity demand data from EIA
for regime-based analysis (peak, off-peak, shoulder periods).
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

# Try to import required libraries
try:
    import requests
    import pandas as pd
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("Warning: Some dependencies missing. Will generate synthetic data.")

# Output directory
DATA_DIR = Path(__file__).parent.parent / "data" / "electricity"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# EIA API key (optional - can use without for some endpoints)
EIA_API_KEY = os.environ.get('EIA_API_KEY', '')


def download_eia_hourly_demand() -> pd.DataFrame:
    """
    Download hourly electricity demand from EIA API.

    Uses the EIA Open Data API v2.
    Endpoint: electricity/rto/region-data
    """
    if not EIA_API_KEY:
        print("No EIA_API_KEY found. Using alternative source...")
        return download_eia_bulk_data()

    url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"

    params = {
        'api_key': EIA_API_KEY,
        'frequency': 'hourly',
        'data[0]': 'value',
        'facets[respondent][]': 'US48',  # Lower 48 states
        'facets[type][]': 'D',  # Demand
        'start': '2023-01-01',
        'end': '2023-12-31',
        'sort[0][column]': 'period',
        'sort[0][direction]': 'asc',
        'length': 5000,
    }

    print(f"Fetching from EIA API...")

    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()

        data = response.json()

        if 'response' in data and 'data' in data['response']:
            records = data['response']['data']
            df = pd.DataFrame(records)

            # Rename columns
            df = df.rename(columns={
                'period': 'timestamp',
                'value': 'demand_mw',
            })

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['demand_mw'] = pd.to_numeric(df['demand_mw'], errors='coerce')

            return df

    except Exception as e:
        print(f"  Error fetching from EIA API: {e}")

    return None


def download_eia_bulk_data() -> pd.DataFrame:
    """
    Try to download from EIA bulk data files.

    These are publicly available without API key.
    """
    # EIA provides bulk downloads - try a known dataset
    bulk_urls = [
        "https://api.eia.gov/bulk/EBA.zip",  # Electricity balancing authority
    ]

    # For now, fall back to synthetic
    return None


def generate_synthetic_electricity_data(n_days: int = 365) -> pd.DataFrame:
    """
    Generate synthetic but realistic electricity demand data.

    Based on typical US grid patterns:
    - Peak: Summer afternoons (AC), Winter mornings/evenings (heating)
    - Off-peak: Night (12 AM - 6 AM)
    - Shoulder: Morning ramp, evening decline

    Demand range: ~300,000 MW (night) to ~700,000 MW (summer peak)
    """
    import numpy as np

    np.random.seed(42)

    # Generate hourly timestamps
    start_date = datetime(2023, 1, 1)
    timestamps = pd.date_range(start=start_date, periods=n_days * 24, freq='H')

    demands = []
    regimes = []

    for ts in timestamps:
        hour = ts.hour
        month = ts.month
        weekday = ts.weekday()

        # Base demand (MW) - typical for US lower 48
        base = 450000

        # Seasonal adjustment
        if month in [6, 7, 8]:  # Summer (AC load)
            seasonal = 100000
        elif month in [12, 1, 2]:  # Winter (heating)
            seasonal = 50000
        else:  # Spring/Fall
            seasonal = 0

        # Hourly pattern
        if 0 <= hour <= 5:
            hourly = -150000
            regime = 'off_peak'
        elif 6 <= hour <= 9:
            hourly = 50000
            regime = 'morning_ramp'
        elif 10 <= hour <= 15:
            hourly = 100000 if month in [6, 7, 8] else 50000
            regime = 'peak' if month in [6, 7, 8] else 'shoulder'
        elif 16 <= hour <= 20:
            hourly = 150000 if month in [6, 7, 8] else 80000
            regime = 'peak'
        else:
            hourly = -50000
            regime = 'evening_decline'

        # Weekend adjustment (-15%)
        if weekday >= 5:
            base *= 0.85

        # Calculate demand
        demand = base + seasonal + hourly

        # Add noise
        noise = np.random.normal(0, demand * 0.05)
        demand = max(250000, demand + noise)

        demands.append(int(demand))
        regimes.append(regime)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'demand_mw': demands,
        'regime': regimes,
    })

    return df


def detect_regime(df: pd.DataFrame) -> pd.Series:
    """Detect regime based on demand percentiles and time."""
    high_thresh = df['demand_mw'].quantile(0.75)
    low_thresh = df['demand_mw'].quantile(0.25)

    def classify(row):
        demand = row['demand_mw']
        hour = row['timestamp'].hour
        month = row['timestamp'].month

        if demand > high_thresh:
            return 'peak'
        elif demand < low_thresh:
            return 'off_peak'
        elif month in [6, 7, 8]:
            return 'summer_shoulder'
        elif month in [12, 1, 2]:
            return 'winter_shoulder'
        else:
            return 'shoulder'

    return df.apply(classify, axis=1)


def main():
    """Download or generate electricity demand data."""
    print("="*60)
    print("EIA ELECTRICITY DEMAND DATA DOWNLOAD")
    print("="*60)

    output_file = DATA_DIR / "eia_hourly_demand.csv"

    if HAS_DEPS:
        print("\nAttempting to download real data from EIA...")

        df = download_eia_hourly_demand()

        if df is not None and len(df) > 0:
            df['regime'] = detect_regime(df)
            df.to_csv(output_file, index=False)

            print(f"\n✅ Downloaded {len(df)} hours of real data")
            print(f"   Saved to: {output_file}")

            # Print regime distribution
            print("\nRegime distribution:")
            for regime, count in df['regime'].value_counts().items():
                pct = count / len(df) * 100
                print(f"  {regime}: {count} ({pct:.1f}%)")

            return

    # Fall back to synthetic data
    print("\nGenerating synthetic electricity data based on US grid patterns...")
    df = generate_synthetic_electricity_data(n_days=365)
    df.to_csv(output_file, index=False)

    print(f"\n✅ Generated {len(df)} hours of synthetic data")
    print(f"   Saved to: {output_file}")

    # Print regime distribution
    print("\nRegime distribution:")
    for regime, count in df['regime'].value_counts().items():
        pct = count / len(df) * 100
        print(f"  {regime}: {count} ({pct:.1f}%)")

    # Print demand statistics
    print(f"\nDemand statistics:")
    print(f"  Min: {df['demand_mw'].min():,.0f} MW")
    print(f"  Max: {df['demand_mw'].max():,.0f} MW")
    print(f"  Mean: {df['demand_mw'].mean():,.0f} MW")

    # Create manifest
    manifest = {
        'source': 'EIA (synthetic patterns)',
        'url': 'https://www.eia.gov/opendata/',
        'records': len(df),
        'columns': df.columns.tolist(),
        'regimes': df['regime'].unique().tolist(),
        'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
        'demand_range_mw': f"{df['demand_mw'].min():,.0f} - {df['demand_mw'].max():,.0f}",
        'generated': datetime.now().isoformat(),
    }

    with open(DATA_DIR / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2, default=str)


if __name__ == "__main__":
    main()
