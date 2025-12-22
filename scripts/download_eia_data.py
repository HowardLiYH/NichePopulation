#!/usr/bin/env python3
"""
Download real electricity demand data from EIA (U.S. Energy Information Administration).

Uses the EIA Open Data API to fetch hourly electricity demand data.
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_synthetic_eia_data(output_path: str, start_date: str = "2022-01-01",
                                  end_date: str = "2024-12-31") -> pd.DataFrame:
    """
    Generate realistic synthetic electricity demand data with regime labels.

    This creates EIA-style data with realistic patterns:
    - Daily seasonality (peak during day, low at night)
    - Weekly seasonality (lower on weekends)
    - Seasonal patterns (higher in summer/winter for AC/heating)
    - Random noise and occasional spikes
    """
    print("Generating realistic EIA-style electricity demand data...")

    # Create hourly timestamps
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    n_hours = len(date_range)

    # Base demand (normalized 0-1)
    base_demand = 0.5

    # Initialize arrays
    demand = np.zeros(n_hours)
    regimes = []

    for i, dt in enumerate(date_range):
        hour = dt.hour
        day_of_week = dt.dayofweek
        month = dt.month

        # Daily pattern: peak at 9-11am and 6-8pm, low at 2-5am
        if 2 <= hour <= 5:
            daily_factor = 0.6
        elif 9 <= hour <= 11:
            daily_factor = 1.1
        elif 18 <= hour <= 20:
            daily_factor = 1.15
        elif 12 <= hour <= 17:
            daily_factor = 0.95
        else:
            daily_factor = 0.8

        # Weekly pattern: lower on weekends
        if day_of_week >= 5:
            weekly_factor = 0.85
        else:
            weekly_factor = 1.0

        # Seasonal pattern: higher in summer (AC) and winter (heating)
        if month in [6, 7, 8]:  # Summer
            seasonal_factor = 1.15
        elif month in [12, 1, 2]:  # Winter
            seasonal_factor = 1.1
        else:
            seasonal_factor = 0.95

        # Random noise
        noise = np.random.normal(0, 0.05)

        # Occasional spikes (heat waves, cold snaps)
        if np.random.random() < 0.01:
            spike = np.random.uniform(0.1, 0.3)
        else:
            spike = 0

        # Calculate demand
        demand[i] = base_demand * daily_factor * weekly_factor * seasonal_factor + noise + spike
        demand[i] = np.clip(demand[i], 0.1, 1.0)

        # Assign regime based on demand level and context
        if demand[i] > 0.8:
            regime = "peak_demand"
        elif demand[i] < 0.4:
            regime = "low_demand"
        elif seasonal_factor > 1.0 and daily_factor > 1.0:
            regime = "high_load"
        else:
            regime = "normal"

        regimes.append(regime)

    # Create DataFrame
    df = pd.DataFrame({
        'datetime': date_range,
        'demand': demand,
        'hour': [dt.hour for dt in date_range],
        'day_of_week': [dt.dayofweek for dt in date_range],
        'month': [dt.month for dt in date_range],
        'regime': regimes
    })

    # Add features for prediction
    df['demand_lag1'] = df['demand'].shift(1)
    df['demand_lag24'] = df['demand'].shift(24)
    df['demand_ma24'] = df['demand'].rolling(24).mean()
    df['demand_std24'] = df['demand'].rolling(24).std()

    # Drop NaN rows
    df = df.dropna().reset_index(drop=True)

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Generated {len(df)} hours of data")
    print(f"Regime distribution:")
    print(df['regime'].value_counts())
    print(f"Saved to: {output_path}")

    return df


def try_download_eia_api(api_key: str = None) -> pd.DataFrame:
    """
    Attempt to download real EIA data via API.

    Note: EIA API requires registration for an API key.
    https://www.eia.gov/opendata/register.php
    """
    try:
        import requests
    except ImportError:
        print("requests library not available, using synthetic data")
        return None

    if not api_key:
        api_key = os.environ.get('EIA_API_KEY')

    if not api_key:
        print("No EIA API key found. To use real data:")
        print("1. Register at https://www.eia.gov/opendata/register.php")
        print("2. Set EIA_API_KEY environment variable")
        print("Using synthetic EIA-style data instead...")
        return None

    # EIA API endpoint for hourly electricity demand
    url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"

    params = {
        'api_key': api_key,
        'frequency': 'hourly',
        'data[0]': 'value',
        'facets[respondent][]': 'US48',  # Lower 48 states
        'facets[type][]': 'D',  # Demand
        'start': '2022-01-01',
        'end': '2024-12-31',
        'sort[0][column]': 'period',
        'sort[0][direction]': 'asc',
        'length': 50000
    }

    try:
        print("Attempting to download from EIA API...")
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        if 'response' in data and 'data' in data['response']:
            records = data['response']['data']
            df = pd.DataFrame(records)
            print(f"Downloaded {len(df)} records from EIA API")
            return df
        else:
            print("Unexpected API response format")
            return None

    except Exception as e:
        print(f"EIA API error: {e}")
        return None


def main():
    """Main function to download or generate EIA data."""
    output_path = Path(__file__).parent.parent / "data" / "energy" / "eia_hourly_demand.csv"

    # Try to download real data first
    real_data = try_download_eia_api()

    if real_data is not None:
        # Process real data
        print("Processing real EIA data...")
        # Add regime labels based on demand percentiles
        real_data['regime'] = pd.qcut(
            real_data['value'],
            q=4,
            labels=['low_demand', 'normal', 'high_load', 'peak_demand']
        )
        real_data.to_csv(output_path, index=False)
        print(f"Saved real EIA data to: {output_path}")
    else:
        # Generate synthetic data
        generate_synthetic_eia_data(str(output_path))

    print("\n=== EIA Data Ready ===")


if __name__ == "__main__":
    main()
