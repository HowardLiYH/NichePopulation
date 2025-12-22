#!/usr/bin/env python3
"""
Data Validation Script

Validates collected Bybit data for:
1. Completeness: All expected files present
2. Continuity: No large gaps in timestamps
3. Quality: No anomalous values

Usage:
    python scripts/validate_data.py
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json

# Expected data
ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]
INTERVALS = ["1D", "4H", "1H", "15m", "5m"]

# Interval durations in minutes
INTERVAL_MINUTES = {
    "1D": 1440,
    "4H": 240,
    "1H": 60,
    "15m": 15,
    "5m": 5,
}

# Maximum allowed gap (in intervals)
MAX_GAP_INTERVALS = 24  # Allow up to 24 missing bars

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data" / "bybit"


def validate_file_exists(asset: str, interval: str) -> Tuple[bool, str]:
    """Check if data file exists."""
    filepath = DATA_DIR / f"{asset}_{interval}.csv"
    if filepath.exists():
        return True, str(filepath)
    return False, f"File not found: {filepath}"


def validate_file_content(filepath: Path) -> Dict:
    """
    Validate contents of a data file.

    Returns:
        Dict with validation results
    """
    result = {
        "file": str(filepath),
        "exists": True,
        "rows": 0,
        "start": None,
        "end": None,
        "gaps": [],
        "max_gap_hours": 0,
        "anomalies": [],
        "valid": True,
        "issues": []
    }

    try:
        df = pd.read_csv(filepath)
        result["rows"] = len(df)

        if len(df) == 0:
            result["valid"] = False
            result["issues"].append("Empty file")
            return result

        # Parse timestamps
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        result["start"] = df["timestamp"].min().isoformat()
        result["end"] = df["timestamp"].max().isoformat()

        # Check for gaps
        time_diffs = df["timestamp"].diff().dropna()

        # Get expected interval from filename
        interval = filepath.stem.split("_")[-1]
        expected_minutes = INTERVAL_MINUTES.get(interval, 60)
        expected_delta = timedelta(minutes=expected_minutes)
        max_allowed = expected_delta * MAX_GAP_INTERVALS

        # Find gaps
        gaps = []
        for i, diff in enumerate(time_diffs):
            if diff > max_allowed:
                gap_start = df["timestamp"].iloc[i]
                gap_end = df["timestamp"].iloc[i + 1]
                gaps.append({
                    "start": gap_start.isoformat(),
                    "end": gap_end.isoformat(),
                    "hours": diff.total_seconds() / 3600
                })

        result["gaps"] = gaps
        if gaps:
            result["max_gap_hours"] = max(g["hours"] for g in gaps)
            if result["max_gap_hours"] > 168:  # > 1 week
                result["issues"].append(f"Large gap: {result['max_gap_hours']:.1f} hours")

        # Check for anomalies
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                # Check for zeros
                zeros = (df[col] == 0).sum()
                if zeros > 0:
                    result["anomalies"].append(f"{zeros} zero values in {col}")

                # Check for negative values
                negatives = (df[col] < 0).sum()
                if negatives > 0:
                    result["anomalies"].append(f"{negatives} negative values in {col}")

                # Check for NaN
                nans = df[col].isna().sum()
                if nans > 0:
                    result["anomalies"].append(f"{nans} NaN values in {col}")

        # Check OHLC consistency
        invalid_ohlc = ((df["high"] < df["low"]) |
                        (df["high"] < df["open"]) |
                        (df["high"] < df["close"]) |
                        (df["low"] > df["open"]) |
                        (df["low"] > df["close"])).sum()
        if invalid_ohlc > 0:
            result["anomalies"].append(f"{invalid_ohlc} invalid OHLC bars")

        if result["anomalies"]:
            result["issues"].extend(result["anomalies"])

        result["valid"] = len(result["issues"]) == 0

    except Exception as e:
        result["valid"] = False
        result["issues"].append(f"Error reading file: {e}")

    return result


def validate_all() -> Dict:
    """
    Validate all expected data files.

    Returns:
        Dict with validation summary
    """
    print("=" * 70)
    print("DATA VALIDATION")
    print("=" * 70)

    results = {
        "timestamp": datetime.now().isoformat(),
        "data_dir": str(DATA_DIR),
        "expected_files": len(ASSETS) * len(INTERVALS),
        "found_files": 0,
        "valid_files": 0,
        "total_rows": 0,
        "files": [],
        "missing": [],
        "issues": []
    }

    for asset in ASSETS:
        print(f"\n{asset}:")

        for interval in INTERVALS:
            exists, filepath = validate_file_exists(asset, interval)

            if not exists:
                results["missing"].append(f"{asset}_{interval}")
                print(f"  {interval}: ❌ MISSING")
                continue

            results["found_files"] += 1

            # Validate content
            file_result = validate_file_content(Path(filepath))
            results["files"].append(file_result)
            results["total_rows"] += file_result["rows"]

            if file_result["valid"]:
                results["valid_files"] += 1
                status = "✓"
            else:
                status = "⚠"
                results["issues"].extend([f"{asset}_{interval}: {i}" for i in file_result["issues"]])

            print(f"  {interval}: {status} {file_result['rows']:,} rows "
                  f"({file_result['start'][:10] if file_result['start'] else 'N/A'} to "
                  f"{file_result['end'][:10] if file_result['end'] else 'N/A'})")

            if file_result["gaps"]:
                print(f"       Gaps: {len(file_result['gaps'])} (max {file_result['max_gap_hours']:.1f}h)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nExpected files: {results['expected_files']}")
    print(f"Found files:    {results['found_files']}")
    print(f"Valid files:    {results['valid_files']}")
    print(f"Total rows:     {results['total_rows']:,}")

    if results["missing"]:
        print(f"\nMissing files ({len(results['missing'])}):")
        for m in results["missing"]:
            print(f"  - {m}")

    if results["issues"]:
        print(f"\nIssues ({len(results['issues'])}):")
        for issue in results["issues"][:10]:  # Show first 10
            print(f"  - {issue}")
        if len(results["issues"]) > 10:
            print(f"  ... and {len(results['issues']) - 10} more")

    # Overall status
    all_valid = (results["found_files"] == results["expected_files"] and
                 results["valid_files"] == results["found_files"])

    print(f"\nOverall status: {'✓ PASS' if all_valid else '⚠ ISSUES FOUND'}")

    # Save results
    output_path = DATA_DIR / "validation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return results


def check_date_coverage() -> pd.DataFrame:
    """
    Check date coverage for each asset.

    Returns:
        DataFrame with coverage info
    """
    coverage = []

    for asset in ASSETS:
        for interval in INTERVALS:
            filepath = DATA_DIR / f"{asset}_{interval}.csv"
            if not filepath.exists():
                continue

            try:
                df = pd.read_csv(filepath)
                df["timestamp"] = pd.to_datetime(df["timestamp"])

                coverage.append({
                    "asset": asset,
                    "interval": interval,
                    "start": df["timestamp"].min(),
                    "end": df["timestamp"].max(),
                    "rows": len(df),
                    "days": (df["timestamp"].max() - df["timestamp"].min()).days
                })
            except:
                pass

    if coverage:
        return pd.DataFrame(coverage)
    return pd.DataFrame()


if __name__ == "__main__":
    results = validate_all()

    print("\n" + "=" * 70)
    print("DATE COVERAGE")
    print("=" * 70)

    coverage = check_date_coverage()
    if not coverage.empty:
        # Pivot to show coverage matrix
        pivot = coverage.pivot(index="asset", columns="interval", values="days")
        print("\nDays of data per asset/interval:")
        print(pivot.to_string())
