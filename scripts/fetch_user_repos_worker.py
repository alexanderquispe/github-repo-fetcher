#!/usr/bin/env python3
"""
Parallel worker for fetching user repos.

Usage:
    python scripts/fetch_user_repos_worker.py --worker 1
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from src.github_fetcher.user_repo_fetcher import UserRepoFetcher


def main():
    parser = argparse.ArgumentParser(description="Fetch repos for users (worker)")
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
    users_file = data_dir / f"temp_user_repos/users_worker_{worker_id}.parquet"
    output_file = data_dir / f"temp_user_repos/repos_worker_{worker_id}.parquet"

    if not users_file.exists():
        print(f"ERROR: Users file not found: {users_file}")
        sys.exit(1)

    # Load users
    users_df = pd.read_parquet(users_file)
    users = users_df['user_login'].tolist()

    print(f"Worker {worker_id}: Fetching repos for {len(users):,} users")

    # Fetch repos
    fetcher = UserRepoFetcher(token)
    user_repos = fetcher.fetch_repos_for_users(
        users=users,
        output_path=output_file,
        resume_from=output_file if output_file.exists() else None
    )

    print(f"\nWorker {worker_id}: Done!")
    print(f"  Processed {len(user_repos):,} users")
    print(f"  Total repos: {sum(len(r) for r in user_repos.values()):,}")


if __name__ == "__main__":
    main()
