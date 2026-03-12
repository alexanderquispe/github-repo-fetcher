#!/usr/bin/env python3
"""
Combine worker results from parallel contributor fetching.

Usage:
    python scripts/combine_worker_results.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd


def main():
    data_dir = Path("data/output")
    temp_dir = data_dir / "temp_contributors"

    print("Combining contributor worker results...")
    print("=" * 60)

    # Combine contributor files
    all_records = []
    for i in range(1, 7):
        worker_file = temp_dir / f"contributors_worker_{i}.parquet"
        if worker_file.exists():
            df = pd.read_parquet(worker_file)
            all_records.append(df)
            print(f"  Worker {i}: {len(df):,} records")
        else:
            print(f"  Worker {i}: FILE NOT FOUND - {worker_file}")

    if not all_records:
        print("\nERROR: No contributor files found!")
        print("Make sure all workers have completed.")
        sys.exit(1)

    # Combine all DataFrames
    combined_df = pd.concat(all_records, ignore_index=True)
    print(f"\n  Total records: {len(combined_df):,}")

    # Save combined file
    output_path = data_dir / "contributors.parquet"
    combined_df.to_parquet(output_path, index=False)
    print(f"\nSaved combined data to: {output_path}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total repos: {combined_df['repo_nwo'].nunique():,}")
    print(f"  Total contributor records: {len(combined_df):,}")
    print(f"  Unique contributors: {combined_df['user_login'].nunique():,}")

    print("\nNow you can run the analysis:")
    print("  python scripts/run_simplified_contagion.py --analyze")


if __name__ == "__main__":
    main()
