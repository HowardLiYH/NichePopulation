#!/usr/bin/env python3
"""
Download REAL weather data from Open-Meteo API.

NO API KEY REQUIRED - Completely free.

Source: https://open-meteo.com/
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import time

import pandas as pd
import numpy as np

try:
    import requests
except ImportError:
    os.system(f"{sys.executable} -m pip install requests")
    import requests

sys.path.insert(0, str(Path(__file__).parent.parent))


# Major cities for weather data
CITIES = {
    'New_York': {'lat': 40.7128, 'lon': -74.0060},
    'Los_Angeles': {'lat': 34.0522, 'lon': -118.2437},
    'Chicago': {'lat': 41.8781, 'lon': -87.6298},
    'Houston': {'lat': 29.7604, 'lon': -95.3698},
    'Phoenix': {'lat': 33.4484, 'lon': -112.0740},
}


def download_openmeteo_data(
    city_name: str,
    lat: float,
    lon: float,
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
) -> pd.DataFrame:
    """
    Download historical weather data from Open-Meteo.

    Args:
        city_name: City name for labeling
        lat: Latitude
        lon: Longitude
        start_date: Start date
        end_date: End date

    Returns:
        DataFrame with weather data
    """
    print(f"  Downloading {city_name} ({lat}, {lon})...")

    # Open-Meteo historical weather API
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'daily': 'temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,wind_speed_10m_max',
        'timezone': 'auto',
    }

    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        daily = data.get('daily', {})

        if not daily.get('time'):
            print(f"    No data returned for {city_name}")
            return None

        df = pd.DataFrame({
            'date': pd.to_datetime(daily['time']),
            'temp_max': daily.get('temperature_2m_max'),
            'temp_min': daily.get('temperature_2m_min'),
            'temperature': daily.get('temperature_2m_mean'),
            'precipitation': daily.get('precipitation_sum'),
            'wind_speed': daily.get('wind_speed_10m_max'),
        })

        df['city'] = city_name

        # Remove rows with missing temperature
        df = df.dropna(subset=['temperature'])

        print(f"    Downloaded {len(df)} records")

        return df

    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def process_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add features and regime labels to weather data."""

    df = df.sort_values(['city', 'date']).reset_index(drop=True)

    features = []
    for city in df['city'].unique():
        city_df = df[df['city'] == city].copy()

        # Add lag features
        city_df['temp_lag1'] = city_df['temperature'].shift(1)
        city_df['temp_lag7'] = city_df['temperature'].shift(7)
        city_df['temp_ma7'] = city_df['temperature'].rolling(7).mean()
        city_df['temp_std7'] = city_df['temperature'].rolling(7).std()
        city_df['temp_trend7'] = city_df['temperature'].diff(7) / 7

        city_df['month'] = city_df['date'].dt.month
        city_df['day_of_year'] = city_df['date'].dt.dayofyear

        # Assign regimes based on weather conditions
        def get_regime(row):
            temp = row['temperature']
            precip = row.get('precipitation', 0) or 0
            wind = row.get('wind_speed', 0) or 0

            if pd.isna(temp):
                return 'stable'

            # Storm conditions
            if precip > 10 or wind > 50:
                return 'active_storm'
            elif precip > 2 or wind > 30:
                return 'approaching_storm'
            # Temperature-based
            elif temp < 0:
                return 'stable_cold'
            elif temp > 30:
                return 'stable_hot'
            else:
                return 'stable'

        city_df['regime'] = city_df.apply(get_regime, axis=1)

        features.append(city_df)

    result = pd.concat(features, ignore_index=True)
    result = result.dropna(subset=['temperature', 'temp_ma7']).reset_index(drop=True)

    return result


def main():
    """Download real weather data from Open-Meteo."""
    print("=" * 60)
    print("DOWNLOADING REAL WEATHER DATA FROM OPEN-METEO")
    print("=" * 60)

    all_data = []

    for city_name, coords in CITIES.items():
        df = download_openmeteo_data(
            city_name=city_name,
            lat=coords['lat'],
            lon=coords['lon'],
            start_date="2020-01-01",
            end_date="2024-12-31",
        )

        if df is not None and len(df) > 0:
            all_data.append(df)

        time.sleep(0.5)  # Be nice to the API

    if len(all_data) == 0:
        print("\nERROR: Could not download any weather data!")
        return

    # Combine all cities
    combined = pd.concat(all_data, ignore_index=True)

    # Process and add features
    combined = process_weather_data(combined)

    # Save
    output_path = Path(__file__).parent.parent / "data" / "weather" / "openmeteo_real_weather.csv"
    os.makedirs(output_path.parent, exist_ok=True)
    combined.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Total records: {len(combined)}")
    print(f"Cities: {combined['city'].nunique()}")
    print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
    print(f"\nRecords per city:")
    print(combined['city'].value_counts())
    print(f"\nRegime distribution:")
    print(combined['regime'].value_counts())
    print(f"\nSaved to: {output_path}")
    print("\n>>> REAL WEATHER DATA READY <<<")


if __name__ == "__main__":
    main()
