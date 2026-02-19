#!/usr/bin/env python3
"""Parallel downloader for Claude Code co-authored commits using multiple API tokens.

This script splits the date range across multiple workers, each using a different
GitHub token from a different account to maximize throughput.

Usage:
    python scripts/fetch_claude_parallel.py -o data/output/claude_commits.parquet

    # Custom date range
    python scripts/fetch_claude_parallel.py --start-date 2025-02-01 -o output.parquet

    # Dry run (show plan without executing)
    python scripts/fetch_claude_parallel.py --dry-run -o output.parquet
"""

import argparse
import multiprocessing
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
import pandas as pd

from github_fetcher.claude_fetcher import ClaudeCommitFetcher


def parse_date(date_str: str) -> date:
    """Parse a date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid date format: {date_str}. Use YYYY-MM-DD."
        )


def get_tokens_from_env() -> List[Tuple[str, str]]:
    """Get all available tokens from environment."""
    load_dotenv()

    tokens = []

    # Check numbered tokens (GITHUB_TOKEN_1, GITHUB_TOKEN_2, etc.)
    for i in range(1, 20):
        token = os.environ.get(f"GITHUB_TOKEN_{i}")
        if token:
            tokens.append((f"GITHUB_TOKEN_{i}", token))

    # If no numbered tokens, use the default
    if not tokens:
        default_token = os.environ.get("GITHUB_TOKEN")
        if default_token:
            tokens.append(("GITHUB_TOKEN", default_token))

    return tokens


def get_monthly_commit_counts(token: str) -> Dict[str, int]:
    """Get commit counts by month to help with load balancing."""
    from github_fetcher.rest_client import GitHubRESTClient

    client = GitHubRESTClient(token)
    query_base = '"Co-Authored-By" "noreply@anthropic.com"'

    # Define months from Feb 2025 to Jan 2026
    months = [
        ("2025-02-01", "2025-02-28", "Feb 2025"),
        ("2025-03-01", "2025-03-31", "Mar 2025"),
        ("2025-04-01", "2025-04-30", "Apr 2025"),
        ("2025-05-01", "2025-05-31", "May 2025"),
        ("2025-06-01", "2025-06-30", "Jun 2025"),
        ("2025-07-01", "2025-07-31", "Jul 2025"),
        ("2025-08-01", "2025-08-31", "Aug 2025"),
        ("2025-09-01", "2025-09-30", "Sep 2025"),
        ("2025-10-01", "2025-10-31", "Oct 2025"),
        ("2025-11-01", "2025-11-30", "Nov 2025"),
        ("2025-12-01", "2025-12-31", "Dec 2025"),
        ("2026-01-01", "2026-01-31", "Jan 2026"),
    ]

    counts = {}
    for start, end, label in months:
        query = f'{query_base} committer-date:{start}..{end}'
        try:
            count = client.get_search_count(query)
            counts[label] = count
        except Exception as e:
            print(f"Error getting count for {label}: {e}")
            counts[label] = 0

    return counts


def calculate_balanced_ranges(
    start_date: date,
    end_date: date,
    num_workers: int
) -> List[Tuple[date, date]]:
    """
    Calculate date ranges that roughly balance the workload.

    Based on known commit distribution (heavier in recent months),
    we split to balance the load across workers.

    Commit volume distribution (approximate):
    - Feb-Apr 2025: ~500K
    - May-Jun 2025: ~800K
    - Jul-Aug 2025: ~900K
    - Sep-Oct 2025: ~1.5M
    - Nov 2025: ~1.2M
    - Dec 2025: ~1.5M
    - Jan 2026: ~2.1M
    """
    if num_workers == 6:
        ranges = [
            (date(2025, 1, 1), date(2025, 6, 30)),   # ~1.3M commits (includes Jan with ~21)
            (date(2025, 7, 1), date(2025, 9, 15)),   # ~1.2M commits
            (date(2025, 9, 16), date(2025, 10, 31)), # ~1.3M commits
            (date(2025, 11, 1), date(2025, 11, 30)), # ~1.2M commits
            (date(2025, 12, 1), date(2025, 12, 31)), # ~1.5M commits
            (date(2026, 1, 1), end_date),            # ~2.1M+ commits
        ]
    elif num_workers == 5:
        ranges = [
            (date(2025, 1, 1), date(2025, 7, 31)),   # ~1.5M commits (includes Jan with ~21)
            (date(2025, 8, 1), date(2025, 10, 15)),  # ~1.5M commits
            (date(2025, 10, 16), date(2025, 11, 30)), # ~1.5M commits
            (date(2025, 12, 1), date(2025, 12, 31)), # ~1.5M commits
            (date(2026, 1, 1), end_date),            # ~2.1M+ commits
        ]
    elif num_workers == 4:
        ranges = [
            (date(2025, 1, 1), date(2025, 9, 30)),   # ~2M commits (includes Jan with ~21)
            (date(2025, 10, 1), date(2025, 11, 30)), # ~2.2M commits
            (date(2025, 12, 1), date(2025, 12, 31)), # ~1.5M commits
            (date(2026, 1, 1), end_date),            # ~2.1M+ commits
        ]
    elif num_workers == 3:
        ranges = [
            (date(2025, 1, 1), date(2025, 9, 30)),
            (date(2025, 10, 1), date(2025, 12, 15)),
            (date(2025, 12, 16), end_date),
        ]
    elif num_workers == 2:
        ranges = [
            (date(2025, 1, 1), date(2025, 10, 31)),
            (date(2025, 11, 1), end_date),
        ]
    else:
        # Single worker or unknown - use full range
        ranges = [(start_date, end_date)]

    # Filter ranges to be within the requested date range
    filtered_ranges = []
    for s, e in ranges:
        if e < start_date or s > end_date:
            continue
        s = max(s, start_date)
        e = min(e, end_date)
        filtered_ranges.append((s, e))

    return filtered_ranges


def worker_process(
    worker_id: int,
    token: str,
    start_date: date,
    end_date: date,
    output_path: Path
) -> Tuple[int, int, str]:
    """
    Worker process that fetches commits for a specific date range.

    Returns:
        Tuple of (worker_id, commit_count, output_file_path)
    """
    try:
        print(f"[Worker {worker_id}] Starting: {start_date} to {end_date}")
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


def merge_parquet_files(input_files: List[Path], output_file: Path) -> int:
    """Merge multiple parquet/jsonl files into one, removing duplicates."""
    import json

    print(f"\nMerging {len(input_files)} files...")

    dfs = []
    for f in input_files:
        # Try parquet first, then JSON lines as fallback
        if f.exists():
            try:
                df = pd.read_parquet(f)
                print(f"  {f.name}: {len(df):,} commits (parquet)")
                dfs.append(df)
            except Exception as e:
                # Try JSON lines fallback
                jsonl_path = f.with_suffix('.jsonl')
                if jsonl_path.exists():
                    try:
                        commits = []
                        with open(jsonl_path, 'r', encoding='utf-8') as jf:
                            for line in jf:
                                if line.strip():
                                    commits.append(json.loads(line))
                        if commits:
                            df = pd.DataFrame(commits)
                            print(f"  {jsonl_path.name}: {len(df):,} commits (jsonl recovery)")
                            dfs.append(df)
                    except Exception as e2:
                        print(f"  {f.name}: FAILED to read - {e2}")
                else:
                    print(f"  {f.name}: FAILED - {e}")
        else:
            # Check for jsonl even if parquet doesn't exist
            jsonl_path = f.with_suffix('.jsonl')
            if jsonl_path.exists():
                try:
                    commits = []
                    with open(jsonl_path, 'r', encoding='utf-8') as jf:
                        for line in jf:
                            if line.strip():
                                commits.append(json.loads(line))
                    if commits:
                        df = pd.DataFrame(commits)
                        print(f"  {jsonl_path.name}: {len(df):,} commits (jsonl)")
                        dfs.append(df)
                except Exception as e:
                    print(f"  {jsonl_path.name}: FAILED - {e}")

    if not dfs:
        print("No data to merge!")
        return 0

    # Concatenate and remove duplicates by SHA
    merged = pd.concat(dfs, ignore_index=True)
    original_count = len(merged)

    merged = merged.drop_duplicates(subset=['sha'], keep='first')
    final_count = len(merged)

    if original_count != final_count:
        print(f"  Removed {original_count - final_count:,} duplicates")

    # Sort by committed_date
    merged = merged.sort_values('committed_date', ascending=False)

    # Save merged file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_file, index=False)

    print(f"\nMerged output: {output_file}")
    print(f"Total unique commits: {final_count:,}")

    return final_count


def main():
    parser = argparse.ArgumentParser(
        description="Parallel download of Claude Code commits using multiple tokens",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output file path for merged results (.parquet)"
    )
    parser.add_argument(
        "--start-date",
        type=parse_date,
        default=date(2025, 1, 1),
        help="Start date (default: 2025-01-01)"
    )
    parser.add_argument(
        "--end-date",
        type=parse_date,
        default=None,
        help="End date (default: today)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without executing"
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary worker files after merging"
    )

    args = parser.parse_args()

    # Get tokens
    tokens = get_tokens_from_env()

    if not tokens:
        print("Error: No GitHub tokens found in .env file")
        print("Expected: GITHUB_TOKEN_1, GITHUB_TOKEN_2, etc.")
        sys.exit(1)

    print(f"Found {len(tokens)} tokens")

    # Set end date
    end_date = args.end_date or date.today()

    # Calculate balanced date ranges
    ranges = calculate_balanced_ranges(args.start_date, end_date, len(tokens))

    # Match tokens to ranges
    worker_configs = []
    for i, ((start, end), (token_name, token)) in enumerate(zip(ranges, tokens)):
        worker_configs.append({
            'worker_id': i + 1,
            'token_name': token_name,
            'token': token,
            'start_date': start,
            'end_date': end,
        })

    # Print plan
    output_path = Path(args.output)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".parquet")

    temp_dir = output_path.parent / "temp_workers"

    print("\n" + "=" * 70)
    print("PARALLEL DOWNLOAD PLAN")
    print("=" * 70)
    print(f"Date range: {args.start_date} to {end_date}")
    print(f"Workers: {len(worker_configs)}")
    print(f"Output: {output_path}")
    print(f"Temp files: {temp_dir}")
    print("\nWorker assignments:")
    print("-" * 70)

    for cfg in worker_configs:
        print(f"  Worker {cfg['worker_id']}: {cfg['start_date']} to {cfg['end_date']}")
        print(f"           Token: {cfg['token_name']}")

    print("-" * 70)

    if args.dry_run:
        print("\n[DRY RUN] Would start parallel download with above configuration")
        return

    # Create temp directory
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Prepare worker arguments
    worker_args = []
    temp_files = []

    for cfg in worker_configs:
        temp_file = temp_dir / f"commits_worker_{cfg['worker_id']}.parquet"
        temp_files.append(temp_file)

        worker_args.append((
            cfg['worker_id'],
            cfg['token'],
            cfg['start_date'],
            cfg['end_date'],
            temp_file,
        ))

    print(f"\nStarting {len(worker_args)} parallel workers...")
    print("=" * 70)

    start_time = time.time()

    # Run workers in parallel using multiprocessing
    with multiprocessing.Pool(processes=len(worker_args)) as pool:
        results = pool.starmap(worker_process, worker_args)

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("WORKER RESULTS")
    print("=" * 70)

    total_commits = 0
    for worker_id, count, path in results:
        print(f"  Worker {worker_id}: {count:,} commits")
        total_commits += count

    print(f"\nTotal from workers: {total_commits:,} commits")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")

    # Merge results
    final_count = merge_parquet_files(temp_files, output_path)

    # Clean up temp files
    if not args.keep_temp:
        print("\nCleaning up temp files...")
        for f in temp_files:
            if f.exists():
                f.unlink()
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"Output file: {output_path}")
    print(f"Total commits: {final_count:,}")
    print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")


if __name__ == "__main__":
    # Required for Windows multiprocessing
    multiprocessing.freeze_support()
    main()
