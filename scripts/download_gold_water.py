#!/usr/bin/env python3
"""Download Gold from alternative FRED series and Water from USGS with retry."""

import os
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np
import requests

# Gold from London PM Fix
GOLD_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GOLDPMGBD228NLBM"

# USGS Water - try different approach
USGS_SITES = [
    ('09380000', 'Colorado_River'),
    ('07010000', 'Mississippi_River'),
    ('14105700', 'Columbia_River'),
]

def download_gold():
    print("Downloading Gold (London PM Fix)...")
    try:
        response = requests.get(GOLD_URL, timeout=30)
        if response.status_code == 200:
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            df.columns = ['date', 'price']
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['price'] != '.']
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df = df.dropna()
            df = df[df['date'] >= '2015-01-01']
            df['commodity'] = 'Gold'
            print(f"  Downloaded {len(df)} Gold records")
            return df
    except Exception as e:
        print(f"  Gold failed: {e}")
    return None

def download_water():
    print("\nDownloading USGS Water Data...")
    
    all_data = []
    
    for site_id, name in USGS_SITES:
        print(f"  Trying {name} ({site_id})...")
        
        # Use RDB format which is more reliable
        url = f"https://waterdata.usgs.gov/nwis/dv?format=rdb&site_no={site_id}&period=&begin_date=2020-01-01&end_date=2024-12-31"
        
        try:
            response = requests.get(url, timeout=60, verify=True)
            
            if response.status_code == 200 and len(response.text) > 500:
                # Parse RDB format (tab-separated with # comments)
                lines = response.text.split('\n')
                data_lines = [l for l in lines if l and not l.startswith('#') and not l.startswith('5s')]
                
                if len(data_lines) > 2:
                    # Skip header lines
                    from io import StringIO
                    clean_text = '\n'.join(data_lines)
                    
                    try:
                        df = pd.read_csv(StringIO(clean_text), sep='\t', low_memory=False)
                        
                        # Find date and discharge columns
                        date_col = [c for c in df.columns if 'date' in c.lower()][0]
                        flow_cols = [c for c in df.columns if df[c].dtype in ['float64', 'int64']]
                        
                        if flow_cols:
                            df = df[[date_col, flow_cols[0]]].copy()
                            df.columns = ['date', 'flow_cfs']
                            df['date'] = pd.to_datetime(df['date'], errors='coerce')
                            df['flow_cfs'] = pd.to_numeric(df['flow_cfs'], errors='coerce')
                            df = df.dropna()
                            df['gauge'] = name
                            
                            if len(df) > 100:
                                print(f"    SUCCESS: {len(df)} records")
                                all_data.append(df)
                    except:
                        pass
                        
        except Exception as e:
            print(f"    Failed: {e}")
        
        time.sleep(1)
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        return combined
    return None

def main():
    print("=" * 60)
    print("DOWNLOADING GOLD AND WATER DATA")
    print("=" * 60)
    
    output_dir = Path(__file__).parent.parent / "data"
    
    # Gold
    gold_df = download_gold()
    if gold_df is not None and len(gold_df) > 0:
        # Append to existing commodities
        comm_path = output_dir / "commodities" / "fred_real_prices.csv"
        if comm_path.exists():
            existing = pd.read_csv(comm_path, parse_dates=['date'])
            combined = pd.concat([existing, gold_df], ignore_index=True)
            combined.to_csv(comm_path, index=False)
            print(f"\nAppended Gold to commodities ({len(combined)} total records)")
    
    # Water
    water_df = download_water()
    if water_df is not None and len(water_df) > 0:
        water_path = output_dir / "water" / "usgs_real_streamflow.csv"
        os.makedirs(water_path.parent, exist_ok=True)
        water_df.to_csv(water_path, index=False)
        print(f"\nSaved {len(water_df)} water records to {water_path}")
    else:
        print("\nCould not download USGS water data - network issues")
        
    print("\nDONE")

if __name__ == "__main__":
    main()
