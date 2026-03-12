#!/usr/bin/env python3
"""
Run parallel workers to fetch user repos.

Usage:
    python scripts/run_parallel_user_repos.py

Requires environment variables:
    GITHUB_TOKEN_1 through GITHUB_TOKEN_6
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    # Check for tokens
    tokens = {}
    for i in range(1, 7):
        token = os.getenv(f"GITHUB_TOKEN_{i}")
        if token:
            tokens[i] = token
        else:
            print(f"WARNING: GITHUB_TOKEN_{i} not set")

    if not tokens:
        print("ERROR: No tokens found. Please set GITHUB_TOKEN_1 through GITHUB_TOKEN_6")
        sys.exit(1)

    print(f"Found {len(tokens)} tokens")

    # Start workers
    processes = []
    for worker_id, token in tokens.items():
        cmd = [
            sys.executable,
            "scripts/fetch_user_repos_worker.py",
            "--worker", str(worker_id),
            "--token", token
        ]

        print(f"Starting worker {worker_id}...")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        processes.append((worker_id, proc))

    # Wait for all workers and print output
    for worker_id, proc in processes:
        print(f"\n{'='*50}")
        print(f"Worker {worker_id} output:")
        print('='*50)
        stdout, _ = proc.communicate()
        print(stdout)

        if proc.returncode != 0:
            print(f"Worker {worker_id} failed with return code {proc.returncode}")

    # Combine results
    print("\n" + "="*50)
    print("Combining worker results...")
    print("="*50)

    data_dir = Path("data/output")
    all_records = []

    for i in range(1, 7):
        worker_file = data_dir / f"temp_user_repos/repos_worker_{i}.parquet"
        if worker_file.exists():
            df = pd.read_parquet(worker_file)
            all_records.extend(df.to_dict('records'))
            print(f"  Worker {i}: {len(df)} users")

    if all_records:
        import pandas as pd
        combined_df = pd.DataFrame(all_records)
        combined_path = data_dir / "user_repos.parquet"
        combined_df.to_parquet(combined_path, index=False)
        print(f"\nCombined {len(combined_df)} users to {combined_path}")
    else:
        print("No results to combine")


if __name__ == "__main__":
    import pandas as pd
    main()
