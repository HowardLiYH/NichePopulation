#!/usr/bin/env python3
"""
Download REAL streamflow data from USGS Water Services.

NO API KEY REQUIRED - Free public data.

USGS Water Services: https://waterservices.usgs.gov/rest/
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


# Major USGS streamflow gauges
USGS_GAUGES = {
    'Colorado_River_Lees_Ferry': '09380000',  # Colorado River at Lees Ferry, AZ
    'Mississippi_River_StLouis': '07010000',  # Mississippi River at St. Louis, MO
    'Columbia_River_Dalles': '14105700',      # Columbia River at The Dalles, OR
    'Missouri_River_Hermann': '06934500',     # Missouri River at Hermann, MO
    'Ohio_River_Louisville': '03294500',      # Ohio River at Louisville, KY
}


def download_usgs_data(
    site_id: str,
    site_name: str,
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
) -> pd.DataFrame:
    """
    Download daily streamflow data from USGS.

    Args:
        site_id: USGS site ID
        site_name: Human-readable name
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with streamflow data
    """
    print(f"  Downloading {site_name} ({site_id})...")

    # USGS REST API for daily values
    url = "https://waterservices.usgs.gov/nwis/dv/"

    params = {
        'format': 'json',
        'sites': site_id,
        'startDT': start_date,
        'endDT': end_date,
        'parameterCd': '00060',  # Discharge (cfs)
        'siteStatus': 'all',
    }

    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        # Parse JSON response
        time_series = data.get('value', {}).get('timeSeries', [])

        if not time_series:
            print(f"    No data returned for {site_name}")
            return None

        values = time_series[0].get('values', [{}])[0].get('value', [])

        if not values:
            print(f"    No values found for {site_name}")
            return None

        # Convert to DataFrame
        records = []
        for v in values:
            records.append({
                'date': v['dateTime'][:10],
                'flow_cfs': float(v['value']) if v['value'] != '-999999' else np.nan,
            })

        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        df['gauge'] = site_name

        # Remove invalid readings
        df = df[df['flow_cfs'] > 0].copy()

        print(f"    Downloaded {len(df)} records")

        return df

    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def process_streamflow_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add features and regime labels to streamflow data."""

    # Sort by date
    df = df.sort_values(['gauge', 'date']).reset_index(drop=True)

    # Add features per gauge
    features = []
    for gauge in df['gauge'].unique():
        gauge_df = df[df['gauge'] == gauge].copy()

        gauge_df['flow_lag1'] = gauge_df['flow_cfs'].shift(1)
        gauge_df['flow_lag7'] = gauge_df['flow_cfs'].shift(7)
        gauge_df['flow_ma7'] = gauge_df['flow_cfs'].rolling(7).mean()
        gauge_df['flow_ma30'] = gauge_df['flow_cfs'].rolling(30).mean()
        gauge_df['flow_std7'] = gauge_df['flow_cfs'].rolling(7).std()

        gauge_df['month'] = gauge_df['date'].dt.month
        gauge_df['day_of_year'] = gauge_df['date'].dt.dayofyear

        # Calculate regime based on flow percentiles
        p10 = gauge_df['flow_cfs'].quantile(0.10)
        p25 = gauge_df['flow_cfs'].quantile(0.25)
        p75 = gauge_df['flow_cfs'].quantile(0.75)
        p95 = gauge_df['flow_cfs'].quantile(0.95)

        def get_regime(flow):
            if pd.isna(flow):
                return 'normal'
            if flow >= p95:
                return 'flood'
            elif flow >= p75:
                return 'high_flow'
            elif flow <= p10:
                return 'low_flow'
            elif flow <= p25:
                return 'below_normal'
            else:
                return 'normal'

        gauge_df['regime'] = gauge_df['flow_cfs'].apply(get_regime)

        features.append(gauge_df)

    result = pd.concat(features, ignore_index=True)
    result = result.dropna(subset=['flow_cfs', 'flow_ma7']).reset_index(drop=True)

    return result


def main():
    """Download real USGS streamflow data."""
    print("=" * 60)
    print("DOWNLOADING REAL USGS STREAMFLOW DATA")
    print("=" * 60)

    all_data = []

    for name, site_id in USGS_GAUGES.items():
        df = download_usgs_data(
            site_id=site_id,
            site_name=name,
            start_date="2020-01-01",
            end_date="2024-12-31",
        )

        if df is not None and len(df) > 0:
            all_data.append(df)

        # Be nice to the API
        time.sleep(1)

    if len(all_data) == 0:
        print("\nERROR: Could not download any USGS data!")
        return

    # Combine all gauges
    combined = pd.concat(all_data, ignore_index=True)

    # Process and add features
    combined = process_streamflow_data(combined)

    # Save
    output_path = Path(__file__).parent.parent / "data" / "water" / "usgs_real_streamflow.csv"
    os.makedirs(output_path.parent, exist_ok=True)
    combined.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Total records: {len(combined)}")
    print(f"Gauges: {combined['gauge'].nunique()}")
    print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
    print(f"\nRecords per gauge:")
    print(combined['gauge'].value_counts())
    print(f"\nRegime distribution:")
    print(combined['regime'].value_counts())
    print(f"\nSaved to: {output_path}")
    print("\n>>> REAL USGS DATA READY <<<")


if __name__ == "__main__":
    main()
