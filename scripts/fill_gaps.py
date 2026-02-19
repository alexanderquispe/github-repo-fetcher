#!/usr/bin/env python3
"""Fill in missing date ranges with 1-day buffer before each gap."""

import multiprocessing
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

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


def worker_process(
    worker_id: int,
    token: str,
    start_date: date,
    end_date: date,
    output_path: Path
) -> Tuple[int, int, str]:
    """Worker process that fetches commits for a specific date range."""
    try:
        print(f"[Worker {worker_id}] Fetching: {start_date} to {end_date}")
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


def main():
    temp_dir = Path("data/output/temp_workers")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Get tokens
    tokens = get_tokens_from_env()
    print(f"Found {len(tokens)} tokens")

    # Gap ranges with 1-day buffer before each gap
    # Format: (start_date, end_date, output_name)
    gap_ranges = [
        # Worker 4 gaps (Nov 4-14 missing, with buffer start Nov 3)
        (date(2025, 11, 3), date(2025, 11, 14), "gap_nov03_14"),

        # Worker 4 gap (Nov 29-30 missing, with buffer start Nov 28)
        (date(2025, 11, 28), date(2025, 11, 30), "gap_nov28_30"),

        # Worker 5 gap (Dec 26-31 missing, with buffer start Dec 25)
        (date(2025, 12, 25), date(2025, 12, 31), "gap_dec25_31"),

        # Worker 6 gap (Jan 18-27 missing, with buffer start Jan 17)
        (date(2026, 1, 17), date(2026, 1, 27), "gap_jan17_27"),

        # Worker 1 early gaps (scattered dates in Jan-Feb 2025)
        # Group 1: Dec 31 2024 to Feb 5 (covers Jan 1, 5, 7-8, 11, 13-17, 19-23, 25-27, 29 - Feb 5)
        (date(2024, 12, 31), date(2025, 2, 5), "gap_dec31_feb05"),

        # Group 2: Feb 10-23 (covers Feb 11-16, 18, 20, 23)
        (date(2025, 2, 10), date(2025, 2, 23), "gap_feb10_23"),
    ]

    print("\n" + "=" * 70)
    print("GAP FILL PLAN (with 1-day buffer)")
    print("=" * 70)

    for i, (start, end, name) in enumerate(gap_ranges, 1):
        days = (end - start).days + 1
        print(f"  Worker {i}: {start} to {end} ({days} days) -> {name}")

    print("=" * 70)

    if len(gap_ranges) > len(tokens):
        print(f"\nWARNING: {len(gap_ranges)} gaps but only {len(tokens)} tokens.")
        print("Some workers will share tokens (slower due to rate limits).")

    print(f"\nStarting {len(gap_ranges)} workers...")

    start_time = time.time()

    # Prepare worker arguments
    worker_args = []
    for i, (start, end, name) in enumerate(gap_ranges):
        token = tokens[i % len(tokens)][1]
        temp_file = temp_dir / f"commits_{name}.parquet"
        worker_args.append((i + 1, token, start, end, temp_file))

    # Run workers in parallel
    with multiprocessing.Pool(processes=min(len(worker_args), len(tokens))) as pool:
        results = pool.starmap(worker_process, worker_args)

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    total = 0
    for worker_id, count, path in results:
        print(f"  Worker {worker_id}: {count:,} commits -> {Path(path).name}")
        total += count

    print(f"\nTotal new commits: {total:,}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print("\nGap files saved to:", temp_dir)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
