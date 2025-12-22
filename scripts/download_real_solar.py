#!/usr/bin/env python3
"""
Download REAL solar irradiance data from Open-Meteo or PVGIS.

NO API KEY REQUIRED - Completely free.

Source: https://open-meteo.com/ (historical solar radiation)
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


# Locations with good solar data
LOCATIONS = {
    'Phoenix_AZ': {'lat': 33.4484, 'lon': -112.0740},
    'Las_Vegas_NV': {'lat': 36.1699, 'lon': -115.1398},
    'Denver_CO': {'lat': 39.7392, 'lon': -104.9903},
    'Miami_FL': {'lat': 25.7617, 'lon': -80.1918},
    'Seattle_WA': {'lat': 47.6062, 'lon': -122.3321},
}


def download_solar_data(
    location_name: str,
    lat: float,
    lon: float,
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
) -> pd.DataFrame:
    """
    Download historical solar radiation data from Open-Meteo.
    """
    print(f"  Downloading {location_name} ({lat}, {lon})...")

    # Open-Meteo historical weather API with solar radiation
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': 'shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance',
        'timezone': 'auto',
    }

    try:
        response = requests.get(url, params=params, timeout=120)
        response.raise_for_status()
        data = response.json()

        hourly = data.get('hourly', {})

        if not hourly.get('time'):
            print(f"    No data returned for {location_name}")
            return None

        df = pd.DataFrame({
            'datetime': pd.to_datetime(hourly['time']),
            'ghi': hourly.get('shortwave_radiation'),  # Global Horizontal Irradiance
            'dni': hourly.get('direct_normal_irradiance'),  # Direct Normal Irradiance
            'dhi': hourly.get('diffuse_radiation'),  # Diffuse Horizontal Irradiance
        })

        df['location'] = location_name

        # Remove rows with missing GHI
        df = df.dropna(subset=['ghi'])

        # Filter to daylight hours only (GHI > 0)
        df = df[df['ghi'] > 0].copy()

        print(f"    Downloaded {len(df)} records")

        return df

    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def process_solar_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add features and regime labels to solar data."""

    df = df.sort_values(['location', 'datetime']).reset_index(drop=True)

    features = []
    for loc in df['location'].unique():
        loc_df = df[df['location'] == loc].copy()

        # Add time features
        loc_df['hour'] = loc_df['datetime'].dt.hour
        loc_df['month'] = loc_df['datetime'].dt.month
        loc_df['day_of_year'] = loc_df['datetime'].dt.dayofyear

        # Add lag features
        loc_df['ghi_lag1'] = loc_df['ghi'].shift(1)
        loc_df['ghi_lag24'] = loc_df['ghi'].shift(24)
        loc_df['ghi_ma6'] = loc_df['ghi'].rolling(6).mean()
        loc_df['ghi_std6'] = loc_df['ghi'].rolling(6).std()

        # Calculate clear sky index (simplified)
        # Clear sky GHI approximation based on hour
        def approx_clear_sky(row):
            hour = row['hour']
            doy = row['day_of_year']
            if hour < 6 or hour > 18:
                return 0
            solar_angle = np.sin(np.pi * (hour - 6) / 12)
            seasonal = 1 + 0.3 * np.cos(2 * np.pi * (doy - 172) / 365)
            return 1000 * solar_angle * seasonal

        loc_df['clear_sky_ghi'] = loc_df.apply(approx_clear_sky, axis=1)
        loc_df['clear_sky_index'] = loc_df['ghi'] / (loc_df['clear_sky_ghi'] + 1)

        # Assign regimes based on clear sky index
        def get_regime(row):
            csi = row['clear_sky_index']
            ghi = row['ghi']

            if pd.isna(csi) or ghi < 50:
                return 'overcast'
            elif csi > 0.8:
                return 'clear'
            elif csi > 0.5:
                return 'partly_cloudy'
            elif csi > 0.2:
                return 'overcast'
            else:
                return 'storm'

        loc_df['regime'] = loc_df.apply(get_regime, axis=1)

        features.append(loc_df)

    result = pd.concat(features, ignore_index=True)
    result = result.dropna(subset=['ghi', 'ghi_ma6']).reset_index(drop=True)

    return result


def main():
    """Download real solar irradiance data."""
    print("=" * 60)
    print("DOWNLOADING REAL SOLAR IRRADIANCE DATA FROM OPEN-METEO")
    print("=" * 60)

    all_data = []

    for loc_name, coords in LOCATIONS.items():
        df = download_solar_data(
            location_name=loc_name,
            lat=coords['lat'],
            lon=coords['lon'],
            start_date="2020-01-01",
            end_date="2024-12-31",
        )

        if df is not None and len(df) > 0:
            all_data.append(df)

        time.sleep(1)  # Be nice to the API

    if len(all_data) == 0:
        print("\nERROR: Could not download any solar data!")
        return

    # Combine all locations
    combined = pd.concat(all_data, ignore_index=True)

    # Process and add features
    combined = process_solar_data(combined)

    # Save
    output_path = Path(__file__).parent.parent / "data" / "solar" / "openmeteo_real_irradiance.csv"
    os.makedirs(output_path.parent, exist_ok=True)
    combined.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Total records: {len(combined)}")
    print(f"Locations: {combined['location'].nunique()}")
    print(f"Date range: {combined['datetime'].min()} to {combined['datetime'].max()}")
    print(f"\nRecords per location:")
    print(combined['location'].value_counts())
    print(f"\nRegime distribution:")
    print(combined['regime'].value_counts())
    print(f"\nSaved to: {output_path}")
    print("\n>>> REAL SOLAR DATA READY <<<")


if __name__ == "__main__":
    main()
