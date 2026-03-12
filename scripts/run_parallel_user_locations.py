#!/usr/bin/env python3
"""
Run parallel workers to fetch user locations from GitHub profiles.

Usage:
    python scripts/run_parallel_user_locations.py --csv <path_to_csv>
    python scripts/run_parallel_user_locations.py --csv <path_to_csv> --test 100

Requires environment variables:
    GITHUB_TOKEN_1 through GITHUB_TOKEN_6
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import pandas as pd


def split_users_for_workers(users: list, worker_ids: list) -> dict:
    """Split users into chunks for each worker ID."""
    num_workers = len(worker_ids)
    chunk_size = len(users) // num_workers
    remainder = len(users) % num_workers

    chunks = {}
    start = 0
    for idx, worker_id in enumerate(worker_ids):
        # Distribute remainder among first workers
        extra = 1 if idx < remainder else 0
        end = start + chunk_size + extra
        chunks[worker_id] = users[start:end]
        start = end

    return chunks


def main():
    parser = argparse.ArgumentParser(description="Fetch user locations in parallel")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file with usernames")
    parser.add_argument("--column", type=str, default="repo_owner", help="Column name with usernames")
    parser.add_argument("--test", type=int, default=0, help="Test with N users only (0 = all)")
    parser.add_argument("--workers", type=str, default="", help="Comma-separated list of worker IDs to use (e.g., '1,2,3,5')")
    args = parser.parse_args()

    # Determine which workers to use
    if args.workers:
        worker_ids = [int(w.strip()) for w in args.workers.split(",")]
    else:
        worker_ids = list(range(1, 7))

    # Check for tokens
    tokens = {}
    for i in worker_ids:
        token = os.getenv(f"GITHUB_TOKEN_{i}")
        if token:
            tokens[i] = token
        else:
            print(f"WARNING: GITHUB_TOKEN_{i} not set")

    if not tokens:
        print("ERROR: No tokens found. Please set GITHUB_TOKEN_1 through GITHUB_TOKEN_6")
        sys.exit(1)

    print(f"Found {len(tokens)} tokens")

    # Load users from CSV
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        sys.exit(1)

    print(f"\nLoading users from {csv_path}...")
    df = pd.read_csv(csv_path)
    users = df[args.column].dropna().unique().tolist()
    print(f"  Found {len(users):,} unique users")

    # Limit for testing
    if args.test > 0:
        users = users[:args.test]
        print(f"  TEST MODE: Using only {len(users)} users")

    # Split users for workers (use actual token keys)
    user_chunks = split_users_for_workers(users, list(tokens.keys()))

    # Create temp directory and save user chunks
    data_dir = Path("data/output")
    temp_dir = data_dir / "temp_user_locations"
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSplitting users into {len(tokens)} chunks...")
    for worker_id, chunk in user_chunks.items():
        chunk_file = temp_dir / f"users_worker_{worker_id}.parquet"
        chunk_df = pd.DataFrame({"username": chunk})
        chunk_df.to_parquet(chunk_file, index=False)
        print(f"  Worker {worker_id}: {len(chunk):,} users -> {chunk_file}")

    # Start workers
    print(f"\nStarting {len(tokens)} workers...")
    processes = []
    for worker_id, token in tokens.items():
        cmd = [
            sys.executable,
            "scripts/fetch_user_locations_worker.py",
            "--worker", str(worker_id),
            "--token", token
        ]

        print(f"  Starting worker {worker_id}...")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        processes.append((worker_id, proc))

    # Wait for all workers and print output
    for worker_id, proc in processes:
        print(f"\n{'='*60}")
        print(f"Worker {worker_id} output:")
        print('='*60)
        stdout, _ = proc.communicate()
        print(stdout)

        if proc.returncode != 0:
            print(f"Worker {worker_id} failed with return code {proc.returncode}")

    # Combine results
    print("\n" + "="*60)
    print("Combining worker results...")
    print("="*60)

    all_records = []
    for i in tokens.keys():
        worker_file = temp_dir / f"locations_worker_{i}.parquet"
        if worker_file.exists():
            worker_df = pd.read_parquet(worker_file)
            all_records.append(worker_df)
            print(f"  Worker {i}: {len(worker_df):,} users")

    if all_records:
        combined_df = pd.concat(all_records, ignore_index=True)

        # Save as parquet
        combined_parquet = data_dir / "user_locations.parquet"
        combined_df.to_parquet(combined_parquet, index=False)
        print(f"\nSaved {len(combined_df):,} users to {combined_parquet}")

        # Also save as CSV for easy viewing
        combined_csv = data_dir / "user_locations.csv"
        combined_df.to_csv(combined_csv, index=False)
        print(f"Saved {len(combined_df):,} users to {combined_csv}")

        # Summary statistics
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"  Total users: {len(combined_df):,}")
        print(f"  Users with location: {(combined_df['location'] != '').sum():,}")
        print(f"  Users without location: {(combined_df['location'] == '').sum():,}")

        # Show top locations
        print("\nTop 20 locations:")
        location_counts = combined_df[combined_df['location'] != '']['location'].value_counts().head(20)
        for loc, count in location_counts.items():
            print(f"  {count:5,} - {loc}")
    else:
        print("No results to combine")


if __name__ == "__main__":
    main()
