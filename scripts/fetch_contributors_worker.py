#!/usr/bin/env python3
"""
Parallel worker for fetching repo contributors.

Usage:
    python scripts/fetch_contributors_worker.py --worker 1
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from src.github_fetcher.contributor_fetcher import ContributorFetcher


def main():
    parser = argparse.ArgumentParser(description="Fetch contributors for repos (worker)")
    parser.add_argument("--worker", type=int, required=True, help="Worker number (1-6)")
    parser.add_argument("--token", type=str, help="GitHub token (or use GITHUB_TOKEN_N env var)")
    args = parser.parse_args()

    worker_id = args.worker

    # Get token
    token = args.token or os.getenv(f"GITHUB_TOKEN_{worker_id}") or os.getenv("GITHUB_TOKEN")
    if not token:
        print(f"ERROR: No token found. Set GITHUB_TOKEN_{worker_id} or pass --token")
        sys.exit(1)

    # Paths
    data_dir = Path("data/output")
    repos_file = data_dir / f"temp_contributors/repos_worker_{worker_id}.parquet"
    output_file = data_dir / f"temp_contributors/contributors_worker_{worker_id}.parquet"

    if not repos_file.exists():
        print(f"ERROR: Repos file not found: {repos_file}")
        sys.exit(1)

    # Load repos
    repos_df = pd.read_parquet(repos_file)
    repos = repos_df['repo_nwo'].tolist()

    print(f"Worker {worker_id}: Fetching contributors for {len(repos):,} repos")

    # Fetch contributors
    fetcher = ContributorFetcher(token)
    fetcher.fetch_all_repos(
        repos=repos,
        output_path=output_file,
        resume_from=output_file if output_file.exists() else None
    )

    print(f"\nWorker {worker_id}: Done!")


if __name__ == "__main__":
    main()
