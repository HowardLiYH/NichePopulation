#!/usr/bin/env python3
"""
Generate domain data for Traffic and Electricity using only numpy.

This script generates realistic synthetic data based on known patterns
without requiring pandas (to avoid numpy compatibility issues).
"""

import os
import csv
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np


# Output directories
BASE_DIR = Path(__file__).parent.parent / "data"
TRAFFIC_DIR = BASE_DIR / "traffic"
ELECTRICITY_DIR = BASE_DIR / "electricity"


def generate_taxi_data(n_days: int = 365) -> None:
    """
    Generate synthetic but realistic NYC taxi trip data.

    Based on known NYC taxi patterns:
    - Morning rush: 7-9 AM, ~5000 trips/hour
    - Evening rush: 5-7 PM, ~6000 trips/hour
    - Midday: 10 AM - 4 PM, ~3000 trips/hour
    - Night: 10 PM - 6 AM, ~1000 trips/hour
    - Weekend: 30% lower than weekday
    """
    print("="*60)
    print("GENERATING NYC TAXI DATA")
    print("="*60)

    TRAFFIC_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    # Generate hourly timestamps
    start_date = datetime(2023, 1, 1)

    data = []

    for day in range(n_days):
        for hour in range(24):
            ts = start_date + timedelta(days=day, hours=hour)
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

            data.append({
                'timestamp': ts.isoformat(),
                'trip_count': trip_count,
                'regime': regime,
            })

    # Write to CSV
    output_file = TRAFFIC_DIR / "nyc_taxi_hourly.csv"
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'trip_count', 'regime'])
        writer.writeheader()
        writer.writerows(data)

    # Count regimes
    regime_counts = {}
    for row in data:
        r = row['regime']
        regime_counts[r] = regime_counts.get(r, 0) + 1

    print(f"\n✅ Generated {len(data)} hours of taxi data")
    print(f"   Saved to: {output_file}")
    print("\nRegime distribution:")
    for regime, count in sorted(regime_counts.items(), key=lambda x: -x[1]):
        pct = count / len(data) * 100
        print(f"  {regime}: {count} ({pct:.1f}%)")


def generate_electricity_data(n_days: int = 365) -> None:
    """
    Generate synthetic but realistic US electricity demand data.

    Based on typical US grid patterns:
    - Peak: Summer afternoons (AC), Winter mornings/evenings (heating)
    - Off-peak: Night (12 AM - 6 AM)
    - Shoulder: Morning ramp, evening decline

    Demand range: ~300,000 MW (night) to ~700,000 MW (summer peak)
    """
    print("\n" + "="*60)
    print("GENERATING EIA ELECTRICITY DATA")
    print("="*60)

    ELECTRICITY_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    # Generate hourly timestamps
    start_date = datetime(2023, 1, 1)

    data = []

    for day in range(n_days):
        for hour in range(24):
            ts = start_date + timedelta(days=day, hours=hour)
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
            weekend_mult = 0.85 if weekday >= 5 else 1.0

            # Calculate demand
            demand = (base + seasonal + hourly) * weekend_mult

            # Add noise
            noise = np.random.normal(0, demand * 0.05)
            demand = max(250000, demand + noise)

            data.append({
                'timestamp': ts.isoformat(),
                'demand_mw': int(demand),
                'regime': regime,
            })

    # Write to CSV
    output_file = ELECTRICITY_DIR / "eia_hourly_demand.csv"
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'demand_mw', 'regime'])
        writer.writeheader()
        writer.writerows(data)

    # Count regimes
    regime_counts = {}
    demands = []
    for row in data:
        r = row['regime']
        regime_counts[r] = regime_counts.get(r, 0) + 1
        demands.append(row['demand_mw'])

    print(f"\n✅ Generated {len(data)} hours of electricity data")
    print(f"   Saved to: {output_file}")
    print("\nRegime distribution:")
    for regime, count in sorted(regime_counts.items(), key=lambda x: -x[1]):
        pct = count / len(data) * 100
        print(f"  {regime}: {count} ({pct:.1f}%)")

    print(f"\nDemand statistics:")
    print(f"  Min: {min(demands):,} MW")
    print(f"  Max: {max(demands):,} MW")
    print(f"  Mean: {int(np.mean(demands)):,} MW")


def main():
    """Generate both datasets."""
    generate_taxi_data(n_days=365)
    generate_electricity_data(n_days=365)

    print("\n" + "="*60)
    print("DATA GENERATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
