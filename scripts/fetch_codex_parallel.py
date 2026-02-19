#!/usr/bin/env python3
"""Parallel downloader for OpenAI Codex PRs using multiple API tokens.

This script fetches all GitHub PRs with branches containing 'codex' and filters
for those starting with 'codex/' (the OpenAI Codex naming convention).

The search strategy:
1. Use GitHub's issue search API with `is:pr head:codex`
2. Split date ranges across workers to maximize throughput
3. Enrich PRs with full details to get branch names
4. Filter for `codex/` prefix locally

Usage:
    python scripts/fetch_codex_parallel.py -o data/output/codex_prs.parquet

    # Custom date range
    python scripts/fetch_codex_parallel.py --start-date 2025-05-01 -o output.parquet

    # Dry run (show plan without executing)
    python scripts/fetch_codex_parallel.py --dry-run -o output.parquet

    # Skip enrichment (faster but no branch filtering)
    python scripts/fetch_codex_parallel.py --skip-enrichment -o output.parquet
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

from github_fetcher.codex_fetcher import CodexPRFetcher


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


def get_monthly_pr_counts(token: str) -> Dict[str, int]:
    """Get PR counts by month to help with load balancing."""
    fetcher = CodexPRFetcher(token)
    query_base = "is:pr head:codex"

    # Define months from May 2025 onwards (Codex launch)
    months = [
        ("2025-01-01", "2025-04-30", "Jan-Apr 2025"),
        ("2025-05-01", "2025-05-31", "May 2025"),
        ("2025-06-01", "2025-06-30", "Jun 2025"),
        ("2025-07-01", "2025-07-31", "Jul 2025"),
        ("2025-08-01", "2025-08-31", "Aug 2025"),
        ("2025-09-01", "2025-09-30", "Sep 2025"),
        ("2025-10-01", "2025-10-31", "Oct 2025"),
        ("2025-11-01", "2025-11-30", "Nov 2025"),
        ("2025-12-01", "2025-12-31", "Dec 2025"),
        ("2026-01-01", "2026-01-31", "Jan 2026"),
        ("2026-02-01", "2026-02-28", "Feb 2026"),
    ]

    counts = {}
    for start, end, label in months:
        query = f'{query_base} created:{start}..{end}'
        try:
            count = fetcher.get_search_count(query)
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

    Based on PR volume distribution (from plan):
    - Jan-Apr 2025: ~109 PRs
    - May 2025: ~65K PRs
    - Jun 2025: ~428K PRs
    - Jul 2025: ~289K PRs
    - Aug 2025: ~414K PRs
    - Sep 2025: ~408K PRs
    - Oct 2025: ~492K PRs
    - Nov 2025: ~313K PRs
    - Dec 2025: ~271K PRs
    - Jan 2026: ~291K PRs
    - Feb 2026: ~114K PRs (partial)
    """
    if num_workers == 8:
        # Optimized for 8 workers based on volume estimates
        # Total ~3M PRs, target ~385K per worker
        ranges = [
            (date(2025, 1, 1), date(2025, 5, 31)),    # ~65K PRs
            (date(2025, 6, 1), date(2025, 6, 30)),    # ~428K PRs
            (date(2025, 7, 1), date(2025, 8, 15)),    # ~400K PRs
            (date(2025, 8, 16), date(2025, 9, 30)),   # ~400K PRs
            (date(2025, 10, 1), date(2025, 10, 31)),  # ~492K PRs
            (date(2025, 11, 1), date(2025, 11, 30)),  # ~313K PRs
            (date(2025, 12, 1), date(2025, 12, 31)),  # ~271K PRs
            (date(2026, 1, 1), end_date),             # ~405K PRs
        ]
    elif num_workers == 6:
        # Rebalanced for ~500K PRs per worker (~17h max)
        ranges = [
            (date(2025, 1, 1), date(2025, 6, 30)),    # ~493K PRs (Jan-Jun)
            (date(2025, 7, 1), date(2025, 8, 15)),    # ~496K PRs (Jul + half Aug)
            (date(2025, 8, 16), date(2025, 9, 30)),   # ~618K PRs (half Aug + Sep)
            (date(2025, 10, 1), date(2025, 10, 31)),  # ~492K PRs (Oct)
            (date(2025, 11, 1), date(2025, 12, 31)),  # ~584K PRs (Nov-Dec)
            (date(2026, 1, 1), end_date),             # ~416K PRs (Jan-Feb 2026)
        ]
    elif num_workers == 4:
        ranges = [
            (date(2025, 1, 1), date(2025, 8, 31)),    # ~1.2M PRs
            (date(2025, 9, 1), date(2025, 10, 31)),   # ~900K PRs
            (date(2025, 11, 1), date(2025, 12, 31)),  # ~584K PRs
            (date(2026, 1, 1), end_date),             # ~405K PRs
        ]
    elif num_workers == 2:
        ranges = [
            (date(2025, 1, 1), date(2025, 10, 31)),   # ~2.1M PRs
            (date(2025, 11, 1), end_date),            # ~1M PRs
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
    output_path: Path,
    skip_enrichment: bool = False
) -> Tuple[int, int, int, str]:
    """
    Worker process that fetches PRs for a specific date range.

    Returns:
        Tuple of (worker_id, total_prs, filtered_prs, output_file_path)
    """
    try:
        print(f"[Worker {worker_id}] Starting: {start_date} to {end_date}")
        print(f"[Worker {worker_id}] Output: {output_path}")
        print(f"[Worker {worker_id}] Skip enrichment: {skip_enrichment}")

        fetcher = CodexPRFetcher(token)

        if skip_enrichment:
            # Fast mode: fetch without branch filtering
            prs = fetcher.fetch_codex_prs(
                start_date=start_date,
                end_date=end_date,
                output_path=output_path,
                filter_prefix=False  # Can't filter without enrichment
            )
            total = len(prs)
            filtered = 0  # No filtering in fast mode
        else:
            # Full mode: fetch and enrich with branch info
            prs = fetcher.fetch_and_enrich_prs(
                start_date=start_date,
                end_date=end_date,
                output_path=output_path
            )
            total = len(prs)
            filtered = total  # All are filtered to codex/ prefix

        print(f"[Worker {worker_id}] Completed: {total:,} PRs (filtered: {filtered:,})")
        return (worker_id, total, filtered, str(output_path))

    except Exception as e:
        print(f"[Worker {worker_id}] Error: {e}")
        import traceback
        traceback.print_exc()
        return (worker_id, 0, 0, str(output_path))


def merge_files(input_files: List[Path], output_file: Path) -> Tuple[int, int]:
    """
    Merge multiple parquet/jsonl files into one, removing duplicates.

    Returns:
        Tuple of (total_prs, unique_repos)
    """
    import json

    print(f"\nMerging {len(input_files)} files...")

    dfs = []
    for f in input_files:
        # Try parquet first, then JSON lines as fallback
        if f.exists():
            try:
                df = pd.read_parquet(f)
                print(f"  {f.name}: {len(df):,} PRs (parquet)")
                dfs.append(df)
            except Exception as e:
                # Try JSON lines fallback
                jsonl_path = f.with_suffix('.jsonl')
                if jsonl_path.exists():
                    try:
                        prs = []
                        with open(jsonl_path, 'r', encoding='utf-8') as jf:
                            for line in jf:
                                if line.strip():
                                    prs.append(json.loads(line))
                        if prs:
                            df = pd.DataFrame(prs)
                            print(f"  {jsonl_path.name}: {len(df):,} PRs (jsonl recovery)")
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
                    prs = []
                    with open(jsonl_path, 'r', encoding='utf-8') as jf:
                        for line in jf:
                            if line.strip():
                                prs.append(json.loads(line))
                    if prs:
                        df = pd.DataFrame(prs)
                        print(f"  {jsonl_path.name}: {len(df):,} PRs (jsonl)")
                        dfs.append(df)
                except Exception as e:
                    print(f"  {jsonl_path.name}: FAILED - {e}")

    if not dfs:
        print("No data to merge!")
        return 0, 0

    # Concatenate and remove duplicates by pr_id
    merged = pd.concat(dfs, ignore_index=True)
    original_count = len(merged)

    merged = merged.drop_duplicates(subset=['pr_id'], keep='first')
    final_count = len(merged)

    if original_count != final_count:
        print(f"  Removed {original_count - final_count:,} duplicates")

    # Sort by created_at (newest first)
    if 'created_at' in merged.columns:
        merged = merged.sort_values('created_at', ascending=False)

    # Save merged file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_file, index=False)

    # Also save as JSONL
    jsonl_output = output_file.with_suffix('.jsonl')
    with open(jsonl_output, 'w', encoding='utf-8') as f:
        for _, row in merged.iterrows():
            f.write(json.dumps(row.to_dict(), default=str, ensure_ascii=False) + '\n')

    print(f"\nMerged output: {output_file}")
    print(f"Total unique PRs: {final_count:,}")

    # Count unique repos
    unique_repos = 0
    if 'repo_nwo' in merged.columns:
        unique_repos = merged['repo_nwo'].nunique()
        print(f"Unique repositories: {unique_repos:,}")

        # Save unique repos summary
        repos_output = output_file.parent / "codex_repos.csv"
        repo_stats = merged.groupby('repo_nwo').agg(
            codex_pr_count=('pr_id', 'count'),
            first_codex_pr=('created_at', 'min'),
            last_codex_pr=('created_at', 'max')
        ).reset_index()
        repo_stats = repo_stats.sort_values('codex_pr_count', ascending=False)
        repo_stats.to_csv(repos_output, index=False)
        print(f"Repos summary: {repos_output}")

    return final_count, unique_repos


def main():
    parser = argparse.ArgumentParser(
        description="Parallel download of OpenAI Codex PRs using multiple tokens",
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
    parser.add_argument(
        "--skip-enrichment",
        action="store_true",
        help="Skip PR enrichment (faster but no branch filtering)"
    )
    parser.add_argument(
        "--show-counts",
        action="store_true",
        help="Show monthly PR counts and exit"
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

    # Show counts if requested
    if args.show_counts:
        print("\nFetching monthly PR counts...")
        counts = get_monthly_pr_counts(tokens[0][1])
        total = 0
        for month, count in counts.items():
            print(f"  {month}: {count:,}")
            total += count
        print(f"  Total: {total:,}")
        return

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

    temp_dir = output_path.parent / "temp_codex_workers"

    print("\n" + "=" * 70)
    print("PARALLEL DOWNLOAD PLAN - CODEX PRs")
    print("=" * 70)
    print(f"Date range: {args.start_date} to {end_date}")
    print(f"Workers: {len(worker_configs)}")
    print(f"Output: {output_path}")
    print(f"Temp files: {temp_dir}")
    print(f"Skip enrichment: {args.skip_enrichment}")
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
        temp_file = temp_dir / f"codex_prs_worker_{cfg['worker_id']}.parquet"
        temp_files.append(temp_file)

        worker_args.append((
            cfg['worker_id'],
            cfg['token'],
            cfg['start_date'],
            cfg['end_date'],
            temp_file,
            args.skip_enrichment,
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

    total_prs = 0
    total_filtered = 0
    for worker_id, count, filtered, path in results:
        print(f"  Worker {worker_id}: {count:,} PRs (filtered: {filtered:,})")
        total_prs += count
        total_filtered += filtered

    print(f"\nTotal from workers: {total_prs:,} PRs")
    print(f"Total filtered (codex/): {total_filtered:,} PRs")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")

    # Merge results
    final_count, unique_repos = merge_files(temp_files, output_path)

    # Clean up temp files
    if not args.keep_temp:
        print("\nCleaning up temp files...")
        for f in temp_files:
            if f.exists():
                f.unlink()
            jsonl_f = f.with_suffix('.jsonl')
            if jsonl_f.exists():
                jsonl_f.unlink()
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"Output file: {output_path}")
    print(f"Total PRs: {final_count:,}")
    print(f"Unique repos: {unique_repos:,}")
    print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")


if __name__ == "__main__":
    # Required for Windows multiprocessing
    multiprocessing.freeze_support()
    main()
