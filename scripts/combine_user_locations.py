#!/usr/bin/env python3
"""
Combine worker results from parallel user location fetching.

Usage:
    python scripts/combine_user_locations.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd


def main():
    data_dir = Path("data/output")
    temp_dir = data_dir / "temp_user_locations"

    print("Combining user location worker results...")
    print("=" * 60)

    # Combine location files
    all_records = []
    for i in range(1, 7):
        worker_file = temp_dir / f"locations_worker_{i}.parquet"
        if worker_file.exists():
            df = pd.read_parquet(worker_file)
            all_records.append(df)
            print(f"  Worker {i}: {len(df):,} records")
        else:
            print(f"  Worker {i}: FILE NOT FOUND - {worker_file}")

    if not all_records:
        print("\nERROR: No location files found!")
        print("Make sure all workers have completed.")
        sys.exit(1)

    # Combine all DataFrames
    combined_df = pd.concat(all_records, ignore_index=True)
    print(f"\n  Total records: {len(combined_df):,}")

    # Save combined parquet
    parquet_path = data_dir / "user_locations.parquet"
    combined_df.to_parquet(parquet_path, index=False)
    print(f"\nSaved combined data to: {parquet_path}")

    # Save combined CSV
    csv_path = data_dir / "user_locations.csv"
    combined_df.to_csv(csv_path, index=False)
    print(f"Saved combined data to: {csv_path}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total users: {len(combined_df):,}")

    with_location = (combined_df['location'] != '').sum()
    without_location = (combined_df['location'] == '').sum()
    pct_with = 100 * with_location / len(combined_df) if len(combined_df) > 0 else 0

    print(f"  Users with location: {with_location:,} ({pct_with:.1f}%)")
    print(f"  Users without location: {without_location:,}")

    # Top locations
    print("\nTop 30 locations:")
    location_counts = combined_df[combined_df['location'] != '']['location'].value_counts().head(30)
    for loc, count in location_counts.items():
        print(f"  {count:5,} - {loc}")

    # Top countries (rough extraction)
    print("\n" + "=" * 60)
    print("Location samples for country identification:")
    print("=" * 60)
    sample_locs = combined_df[combined_df['location'] != '']['location'].sample(min(50, with_location)).tolist()
    for loc in sample_locs[:50]:
        print(f"  - {loc}")


if __name__ == "__main__":
    main()
