#!/usr/bin/env python3
"""Parallel fetcher for GitHub Copilot PRs.

Searches for PRs with branches starting with `copilot/` across GitHub.
Uses 6 parallel workers with load-balanced date ranges.

Usage:
    python scripts/fetch_copilot_parallel.py --dry-run   # show plan only
    python scripts/fetch_copilot_parallel.py             # full download
"""

import argparse
import json
import multiprocessing
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
import pandas as pd


# Date ranges balanced by estimated PR volume (~1.09M total)
# Based on monthly counts:
# 2025-01 to 2025-04: ~805
# 2025-05: 5,598
# 2025-06: 14,498
# 2025-07: 41,960
# 2025-08: 71,947
# 2025-09: 101,181
# 2025-10: 135,585
# 2025-11: 194,420
# 2025-12: 200,365
# 2026-01: 219,627
# 2026-02: 86,552

# Target: ~182K PRs per worker for 6 workers
DATE_RANGES = [
    # Worker 1: Jan 2025 - Aug 2025 (~135K PRs)
    (date(2025, 1, 1), date(2025, 8, 31)),
    # Worker 2: Sep 2025 (~101K PRs)
    (date(2025, 9, 1), date(2025, 9, 30)),
    # Worker 3: Oct 2025 (~136K PRs)
    (date(2025, 10, 1), date(2025, 10, 31)),
    # Worker 4: Nov 2025 (~194K PRs)
    (date(2025, 11, 1), date(2025, 11, 30)),
    # Worker 5: Dec 2025 (~200K PRs)
    (date(2025, 12, 1), date(2025, 12, 31)),
    # Worker 6: Jan 2026 - Feb 2026 (~306K PRs)
    (date(2026, 1, 1), date(2026, 2, 10)),
]


def get_tokens_from_env() -> List[Tuple[str, str]]:
    """Get all available tokens from environment."""
    load_dotenv()
    tokens = []
    for i in range(1, 20):
        token = os.environ.get(f"GITHUB_TOKEN_{i}")
        if token:
            tokens.append((f"GITHUB_TOKEN_{i}", token))
    return tokens


def worker_process(
    worker_id: int,
    token: str,
    start_date: date,
    end_date: date,
    output_path: Path,
) -> Tuple[int, int, str]:
    """
    Worker process that fetches Copilot PRs for a specific date range.

    Returns:
        Tuple of (worker_id, pr_count, output_file_path)
    """
    try:
        print(f"[Worker {worker_id}] Starting: {start_date} to {end_date}")
        print(f"[Worker {worker_id}] Output: {output_path}")

        from github_fetcher.copilot_fetcher import CopilotPRFetcher

        fetcher = CopilotPRFetcher(token)

        # Fetch PRs without prefix filtering (branch info not available in search API)
        prs = fetcher.fetch_copilot_prs(
            start_date=start_date,
            end_date=end_date,
            output_path=output_path,
            filter_prefix=False  # Can't filter - branch info not in search results
        )

        print(f"[Worker {worker_id}] Completed: {len(prs):,} PRs")
        return (worker_id, len(prs), str(output_path))

    except Exception as e:
        print(f"[Worker {worker_id}] Error: {e}")
        import traceback
        traceback.print_exc()
        return (worker_id, 0, str(output_path))


def merge_files(temp_dir: Path, output_file: Path) -> Tuple[int, int]:
    """Merge worker files into final output."""
    print(f"\nMerging worker files from {temp_dir}...")

    dfs = []
    for jsonl_file in sorted(temp_dir.glob("copilot_prs_worker_*.jsonl")):
        prs = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        prs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        if prs:
            df = pd.DataFrame(prs)
            print(f"  {jsonl_file.name}: {len(df):,} PRs")
            dfs.append(df)

    if not dfs:
        print("No data to merge!")
        return 0, 0

    # Concatenate and deduplicate
    merged = pd.concat(dfs, ignore_index=True)
    original_count = len(merged)

    merged = merged.drop_duplicates(subset=["pr_id"], keep="first")
    final_count = len(merged)

    if original_count != final_count:
        print(f"  Removed {original_count - final_count:,} duplicates")

    # Sort by created_at
    if "created_at" in merged.columns:
        merged = merged.sort_values("created_at", ascending=False)

    # Save merged output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_file, index=False)

    jsonl_output = output_file.with_suffix(".jsonl")
    with open(jsonl_output, "w", encoding="utf-8") as f:
        for _, row in merged.iterrows():
            f.write(json.dumps(row.to_dict(), default=str, ensure_ascii=False) + "\n")

    print(f"\nMerged output: {output_file}")
    print(f"Total unique PRs: {final_count:,}")

    # Count unique repos
    unique_repos = 0
    if "repo_nwo" in merged.columns:
        unique_repos = merged["repo_nwo"].nunique()
        print(f"Unique repositories: {unique_repos:,}")

        # Save repos summary
        repos_output = output_file.parent / "copilot_repos.csv"
        repo_stats = merged.groupby("repo_nwo").agg(
            copilot_pr_count=("pr_id", "count"),
            first_copilot_pr=("created_at", "min"),
            last_copilot_pr=("created_at", "max")
        ).reset_index()
        repo_stats = repo_stats.sort_values("copilot_pr_count", ascending=False)
        repo_stats.to_csv(repos_output, index=False)
        print(f"Repos summary: {repos_output}")

    return final_count, unique_repos


def main():
    parser = argparse.ArgumentParser(
        description="Fetch GitHub Copilot PRs in parallel"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/output/copilot_prs.parquet",
        help="Output file path"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan only, don't fetch"
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip final merge step"
    )
    args = parser.parse_args()

    # Get tokens
    tokens = get_tokens_from_env()
    num_workers = min(len(tokens), len(DATE_RANGES))

    if num_workers < len(DATE_RANGES):
        print(f"Warning: Only {len(tokens)} tokens available, need {len(DATE_RANGES)}")
        print(f"Will use {num_workers} workers")

    if num_workers == 0:
        print("Error: No GitHub tokens found in .env")
        sys.exit(1)

    print(f"Found {len(tokens)} tokens (using {num_workers})")

    output_path = Path(args.output)
    temp_dir = output_path.parent / "temp_copilot_workers"

    # Print plan
    print("\n" + "=" * 70)
    print("COPILOT PR DOWNLOAD PLAN")
    print("=" * 70)
    print(f"Temp directory: {temp_dir}")
    print(f"Final output: {output_path}")
    print(f"\nWorker date ranges:")
    print("-" * 70)

    for i, (start_date, end_date) in enumerate(DATE_RANGES[:num_workers]):
        print(f"  Worker {i+1}: {start_date} to {end_date}")
        print(f"           Token: {tokens[i][0]}")

    print("-" * 70)
    print(f"\nEstimated total: ~1,087,196 PRs")
    print(f"Estimated time: ~4-5 hours")

    if args.dry_run:
        print("\n[DRY RUN] Would start parallel download with above configuration")
        return

    # Create temp directory
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Prepare worker arguments
    worker_args = []
    for i, (start_date, end_date) in enumerate(DATE_RANGES[:num_workers]):
        temp_file = temp_dir / f"copilot_prs_worker_{i+1}.parquet"
        worker_args.append((
            i + 1,
            tokens[i][1],
            start_date,
            end_date,
            temp_file,
        ))

    print(f"\nStarting {num_workers} parallel workers...")
    print("=" * 70)

    start_time = time.time()

    # Run workers in parallel
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.starmap(worker_process, worker_args)

    elapsed = time.time() - start_time

    # Print results
    print("\n" + "=" * 70)
    print("WORKER RESULTS")
    print("=" * 70)

    total_prs = 0
    for worker_id, pr_count, path in results:
        print(f"  Worker {worker_id}: {pr_count:,} PRs")
        total_prs += pr_count

    print(f"\nTotal from workers: {total_prs:,}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")

    # Merge files
    if not args.skip_merge:
        final_count, unique_repos = merge_files(temp_dir, output_path)

        print("\n" + "=" * 70)
        print("DOWNLOAD COMPLETE")
        print("=" * 70)
        print(f"Output file: {output_path}")
        print(f"Total PRs: {final_count:,}")
        print(f"Unique repos: {unique_repos:,}")
        print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
