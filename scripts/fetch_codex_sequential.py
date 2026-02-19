#!/usr/bin/env python3
"""Sequential downloader for OpenAI Codex PRs - runs one date range at a time.

This is a memory-efficient version that processes date ranges sequentially
instead of in parallel, making it suitable for systems with limited RAM.

Usage:
    python scripts/fetch_codex_sequential.py -o data/output/codex_prs.parquet

    # Resume from a specific worker (if previous run was interrupted)
    python scripts/fetch_codex_sequential.py -o data/output/codex_prs.parquet --resume-from 3
"""

import argparse
import json
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import List, Tuple

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
    for i in range(1, 20):
        token = os.environ.get(f"GITHUB_TOKEN_{i}")
        if token:
            tokens.append((f"GITHUB_TOKEN_{i}", token))
    if not tokens:
        default_token = os.environ.get("GITHUB_TOKEN")
        if default_token:
            tokens.append(("GITHUB_TOKEN", default_token))
    return tokens


def get_date_ranges(end_date: date) -> List[Tuple[int, date, date, str]]:
    """
    Get date ranges for sequential processing.
    Returns list of (worker_id, start_date, end_date, description)
    """
    return [
        (1, date(2025, 1, 1), date(2025, 5, 31), "Jan-May 2025 (~65K)"),
        (2, date(2025, 6, 1), date(2025, 6, 30), "Jun 2025 (~428K)"),
        (3, date(2025, 7, 1), date(2025, 7, 31), "Jul 2025 (~289K)"),
        (4, date(2025, 8, 1), date(2025, 8, 31), "Aug 2025 (~414K)"),
        (5, date(2025, 9, 1), date(2025, 9, 30), "Sep 2025 (~408K)"),
        (6, date(2025, 10, 1), date(2025, 10, 31), "Oct 2025 (~492K)"),
        (7, date(2025, 11, 1), date(2025, 11, 30), "Nov 2025 (~313K)"),
        (8, date(2025, 12, 1), date(2025, 12, 31), "Dec 2025 (~271K)"),
        (9, date(2026, 1, 1), date(2026, 1, 31), "Jan 2026 (~291K)"),
        (10, date(2026, 2, 1), min(end_date, date(2026, 2, 28)), "Feb 2026 (~114K)"),
    ]


def merge_files(temp_dir: Path, output_file: Path) -> Tuple[int, int]:
    """Merge all worker files into one, removing duplicates."""
    print(f"\nMerging files from {temp_dir}...")

    dfs = []
    for f in sorted(temp_dir.glob("codex_prs_worker_*.jsonl")):
        try:
            prs = []
            with open(f, 'r', encoding='utf-8') as jf:
                for line in jf:
                    if line.strip():
                        prs.append(json.loads(line))
            if prs:
                df = pd.DataFrame(prs)
                print(f"  {f.name}: {len(df):,} PRs")
                dfs.append(df)
        except Exception as e:
            print(f"  {f.name}: FAILED - {e}")

    if not dfs:
        print("No data to merge!")
        return 0, 0

    merged = pd.concat(dfs, ignore_index=True)
    original_count = len(merged)

    merged = merged.drop_duplicates(subset=['pr_id'], keep='first')
    final_count = len(merged)

    if original_count != final_count:
        print(f"  Removed {original_count - final_count:,} duplicates")

    if 'created_at' in merged.columns:
        merged = merged.sort_values('created_at', ascending=False)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_file, index=False)

    jsonl_output = output_file.with_suffix('.jsonl')
    with open(jsonl_output, 'w', encoding='utf-8') as f:
        for _, row in merged.iterrows():
            f.write(json.dumps(row.to_dict(), default=str, ensure_ascii=False) + '\n')

    print(f"\nMerged output: {output_file}")
    print(f"Total unique PRs: {final_count:,}")

    unique_repos = 0
    if 'repo_nwo' in merged.columns:
        unique_repos = merged['repo_nwo'].nunique()
        print(f"Unique repositories: {unique_repos:,}")

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
        description="Sequential download of OpenAI Codex PRs (memory efficient)",
    )
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output file path for merged results (.parquet)")
    parser.add_argument("--start-date", type=parse_date, default=date(2025, 1, 1),
                        help="Start date (default: 2025-01-01)")
    parser.add_argument("--end-date", type=parse_date, default=None,
                        help="End date (default: today)")
    parser.add_argument("--resume-from", type=int, default=1,
                        help="Resume from worker N (skip earlier ranges)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without executing")
    parser.add_argument("--merge-only", action="store_true",
                        help="Only merge existing temp files")

    args = parser.parse_args()

    tokens = get_tokens_from_env()
    if not tokens:
        print("Error: No GitHub tokens found in .env file")
        sys.exit(1)

    # Use first token for all requests (sequential = 1 token is fine)
    token_name, token = tokens[0]
    print(f"Using token: {token_name}")

    end_date = args.end_date or date.today()
    output_path = Path(args.output)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".parquet")

    temp_dir = output_path.parent / "temp_codex_workers"

    # Get date ranges
    ranges = get_date_ranges(end_date)

    print("\n" + "=" * 70)
    print("SEQUENTIAL DOWNLOAD PLAN - CODEX PRs")
    print("=" * 70)
    print(f"Date range: {args.start_date} to {end_date}")
    print(f"Output: {output_path}")
    print(f"Temp files: {temp_dir}")
    print(f"Resume from: Worker {args.resume_from}")
    print("\nDate ranges:")
    print("-" * 70)
    for worker_id, start, end, desc in ranges:
        status = "SKIP" if worker_id < args.resume_from else "TODO"
        print(f"  [{status}] Worker {worker_id}: {start} to {end} - {desc}")
    print("-" * 70)

    if args.dry_run:
        print("\n[DRY RUN] Would process ranges sequentially")
        return

    if args.merge_only:
        merge_files(temp_dir, output_path)
        return

    temp_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    total_prs = 0

    for worker_id, start, end, desc in ranges:
        if worker_id < args.resume_from:
            print(f"\n[Worker {worker_id}] Skipping (resume from {args.resume_from})")
            continue

        if start > end_date:
            continue

        end = min(end, end_date)
        temp_file = temp_dir / f"codex_prs_worker_{worker_id}.parquet"

        print(f"\n{'=' * 70}")
        print(f"[Worker {worker_id}] {desc}")
        print(f"[Worker {worker_id}] {start} to {end}")
        print(f"[Worker {worker_id}] Output: {temp_file}")
        print("=" * 70)

        fetcher = CodexPRFetcher(token)
        prs = fetcher.fetch_codex_prs(
            start_date=start,
            end_date=end,
            output_path=temp_file,
            filter_prefix=False
        )

        count = len(prs)
        total_prs += count
        print(f"[Worker {worker_id}] Completed: {count:,} PRs")
        print(f"[Worker {worker_id}] Total so far: {total_prs:,} PRs")

        # Clear memory
        del fetcher
        del prs

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE - MERGING FILES")
    print("=" * 70)

    final_count, unique_repos = merge_files(temp_dir, output_path)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Output file: {output_path}")
    print(f"Total PRs: {final_count:,}")
    print(f"Unique repos: {unique_repos:,}")
    print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")


if __name__ == "__main__":
    main()
