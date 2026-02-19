#!/usr/bin/env python3
"""Resume download from where each worker stopped."""

import multiprocessing
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
import pandas as pd

from github_fetcher.claude_fetcher import ClaudeCommitFetcher


def get_tokens_from_env() -> List[Tuple[str, str]]:
    """Get all available tokens from environment."""
    load_dotenv()
    tokens = []
    for i in range(1, 20):
        token = os.environ.get(f"GITHUB_TOKEN_{i}")
        if token:
            tokens.append((f"GITHUB_TOKEN_{i}", token))
    return tokens


def get_resume_ranges(temp_dir: Path) -> Dict[int, Tuple[date, date]]:
    """Return hardcoded resume ranges based on where each worker stopped."""

    # Based on analysis - where each worker stopped and needs to resume from
    # Worker 1: COMPLETE (finished Jun 30, 2025)
    # Worker 2: Stopped at Aug 17, needs Aug 18 - Sep 15
    # Worker 3: Stopped at Oct 17, needs Oct 18 - Oct 31
    # Worker 4: Stopped at Nov 16, needs Nov 17 - Nov 30
    # Worker 5: Stopped at Dec 14, needs Dec 15 - Dec 31
    # Worker 6: Stopped at Jan 10, needs Jan 11 - Jan 27

    resume_ranges = {
        1: None,  # Complete
        2: (date(2025, 8, 17), date(2025, 9, 15)),   # Start from Aug 17
        3: (date(2025, 10, 17), date(2025, 10, 31)), # Start from Oct 17
        4: (date(2025, 11, 16), date(2025, 11, 30)), # Start from Nov 16
        5: (date(2025, 12, 14), date(2025, 12, 31)), # Start from Dec 14
        6: (date(2026, 1, 10), date(2026, 1, 27)),   # Start from Jan 10
    }

    return resume_ranges


def worker_process(
    worker_id: int,
    token: str,
    start_date: date,
    end_date: date,
    output_path: Path
) -> Tuple[int, int, str]:
    """Worker process that fetches commits for a specific date range."""
    try:
        print(f"[Worker {worker_id}] Resuming: {start_date} to {end_date}")
        print(f"[Worker {worker_id}] Output: {output_path}")

        fetcher = ClaudeCommitFetcher(token)
        commits = fetcher.fetch_claude_commits(
            start_date=start_date,
            end_date=end_date,
            output_path=output_path
        )

        print(f"[Worker {worker_id}] Completed: {len(commits):,} commits")
        return (worker_id, len(commits), str(output_path))

    except Exception as e:
        print(f"[Worker {worker_id}] Error: {e}")
        import traceback
        traceback.print_exc()
        return (worker_id, 0, str(output_path))


def merge_all_data(temp_dir: Path, output_file: Path) -> int:
    """Merge original data with resumed data."""
    import json

    print(f"\nMerging all data...")

    dfs = []

    # Load original worker files
    for i in range(1, 7):
        pq_file = temp_dir / f"commits_worker_{i}.parquet"
        if pq_file.exists():
            try:
                df = pd.read_parquet(pq_file)
                print(f"  Worker {i} original: {len(df):,} commits")
                dfs.append(df)
            except:
                pass

    # Load resumed worker files
    for i in range(1, 7):
        pq_file = temp_dir / f"commits_worker_{i}_resumed.parquet"
        if pq_file.exists():
            try:
                df = pd.read_parquet(pq_file)
                print(f"  Worker {i} resumed: {len(df):,} commits")
                dfs.append(df)
            except:
                # Try jsonl fallback
                jsonl_file = temp_dir / f"commits_worker_{i}_resumed.jsonl"
                if jsonl_file.exists():
                    try:
                        commits = []
                        with open(jsonl_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                if line.strip():
                                    commits.append(json.loads(line))
                        if commits:
                            df = pd.DataFrame(commits)
                            print(f"  Worker {i} resumed (jsonl): {len(df):,} commits")
                            dfs.append(df)
                    except:
                        pass

    if not dfs:
        print("No data to merge!")
        return 0

    # Concatenate and deduplicate
    merged = pd.concat(dfs, ignore_index=True)
    original_count = len(merged)

    merged = merged.drop_duplicates(subset=['sha'], keep='first')
    final_count = len(merged)

    if original_count != final_count:
        print(f"  Removed {original_count - final_count:,} duplicates")

    # Sort by date
    merged = merged.sort_values('committed_date', ascending=False)

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_file, index=False)

    print(f"\nFinal output: {output_file}")
    print(f"Total unique commits: {final_count:,}")

    return final_count


def main():
    temp_dir = Path("data/output/temp_workers")
    output_file = Path("data/output/claude_commits_full.parquet")

    # Get tokens
    tokens = get_tokens_from_env()
    print(f"Found {len(tokens)} tokens")

    # Get resume ranges
    resume_ranges = get_resume_ranges(temp_dir)

    print("\n" + "=" * 70)
    print("RESUME PLAN")
    print("=" * 70)

    workers_to_run = []
    for i, rng in resume_ranges.items():
        if rng:
            print(f"  Worker {i}: {rng[0]} to {rng[1]}")
            workers_to_run.append((i, rng))
        else:
            print(f"  Worker {i}: Already complete - skip")

    if not workers_to_run:
        print("\nAll workers complete! Just merging...")
        final_count = merge_all_data(temp_dir, output_file)
        return

    print("=" * 70)
    print(f"\nStarting {len(workers_to_run)} workers...")

    start_time = time.time()

    # Prepare worker arguments
    worker_args = []
    for i, (start, end) in workers_to_run:
        token = tokens[i - 1][1] if i <= len(tokens) else tokens[0][1]
        temp_file = temp_dir / f"commits_worker_{i}_resumed.parquet"
        worker_args.append((i, token, start, end, temp_file))

    # Run workers in parallel
    with multiprocessing.Pool(processes=len(worker_args)) as pool:
        results = pool.starmap(worker_process, worker_args)

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("WORKER RESULTS")
    print("=" * 70)

    for worker_id, count, path in results:
        print(f"  Worker {worker_id}: {count:,} commits")

    print(f"\nTime elapsed: {elapsed/60:.1f} minutes")

    # Merge all data
    final_count = merge_all_data(temp_dir, output_file)

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"Output file: {output_file}")
    print(f"Total commits: {final_count:,}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
