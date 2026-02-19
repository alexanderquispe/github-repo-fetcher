#!/usr/bin/env python3
"""Resume script for Codex PR download.

Continues the interrupted download from where each worker stopped.
Loads existing PRs from temp files and fetches the remaining date ranges.

Resume dates (1 day buffer before last downloaded):
- Worker 1: Jun 5, 2025 → Jun 30, 2025
- Worker 2: Jul 25, 2025 → Aug 15, 2025
- Worker 3: Aug 30, 2025 → Sep 30, 2025
- Worker 4: Oct 15, 2025 → Oct 31, 2025
- Worker 5: Nov 13, 2025 → Dec 31, 2025
- Worker 6: Jan 17, 2026 → Feb 9, 2026

Usage:
    python scripts/resume_codex_download.py
    python scripts/resume_codex_download.py --dry-run
"""

import argparse
import json
import multiprocessing
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
import pandas as pd

from github_fetcher.codex_fetcher import CodexPRFetcher


# Resume configuration: (worker_id, start_date, end_date)
RESUME_RANGES = [
    (1, date(2025, 6, 5), date(2025, 6, 30)),
    (2, date(2025, 7, 25), date(2025, 8, 15)),
    (3, date(2025, 8, 30), date(2025, 9, 30)),
    (4, date(2025, 10, 15), date(2025, 10, 31)),
    (5, date(2025, 11, 13), date(2025, 12, 31)),
    (6, date(2026, 1, 17), date(2026, 2, 9)),
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


def load_existing_prs(jsonl_path: Path) -> Tuple[List[Dict], Set[int]]:
    """Load existing PRs from a JSONL file.

    Returns:
        Tuple of (list of PR dicts, set of PR IDs)
    """
    prs = []
    seen_ids = set()

    if not jsonl_path.exists():
        print(f"  No existing file: {jsonl_path}")
        return prs, seen_ids

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    pr = json.loads(line)
                    pr_id = pr.get("pr_id")
                    if pr_id and pr_id not in seen_ids:
                        seen_ids.add(pr_id)
                        prs.append(pr)
                except json.JSONDecodeError:
                    continue

    print(f"  Loaded {len(prs):,} existing PRs from {jsonl_path.name}")
    return prs, seen_ids


def worker_process(
    worker_id: int,
    token: str,
    start_date: date,
    end_date: date,
    output_path: Path,
) -> Tuple[int, int, int, str]:
    """
    Worker process that resumes fetching PRs for a specific date range.

    Returns:
        Tuple of (worker_id, existing_prs, new_prs, output_file_path)
    """
    try:
        jsonl_path = output_path.with_suffix(".jsonl")

        print(f"[Worker {worker_id}] Resuming: {start_date} to {end_date}")
        print(f"[Worker {worker_id}] Output: {jsonl_path}")

        # Load existing PRs
        existing_prs, seen_ids = load_existing_prs(jsonl_path)
        existing_count = len(existing_prs)

        # Create fetcher and fetch remaining PRs
        fetcher = CodexPRFetcher(token)

        print(f"[Worker {worker_id}] Fetching new PRs...")
        new_prs = fetcher.fetch_codex_prs(
            start_date=start_date,
            end_date=end_date,
            output_path=None,  # We'll save manually
            filter_prefix=False  # No filtering since branch info not available
        )

        # Filter out duplicates
        unique_new_prs = []
        for pr in new_prs:
            pr_id = pr.get("pr_id")
            if pr_id and pr_id not in seen_ids:
                seen_ids.add(pr_id)
                unique_new_prs.append(pr)

        new_count = len(unique_new_prs)

        # Combine existing and new PRs
        all_prs = existing_prs + unique_new_prs

        # Save combined results
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for pr in all_prs:
                f.write(json.dumps(pr, ensure_ascii=False) + "\n")

        # Also save as parquet
        try:
            df = pd.DataFrame(all_prs)
            df.to_parquet(output_path, index=False)
        except Exception as e:
            print(f"[Worker {worker_id}] Warning: Could not save parquet: {e}")

        print(f"[Worker {worker_id}] Done: {existing_count:,} existing + {new_count:,} new = {len(all_prs):,} total")
        return (worker_id, existing_count, new_count, str(output_path))

    except Exception as e:
        print(f"[Worker {worker_id}] Error: {e}")
        import traceback
        traceback.print_exc()
        return (worker_id, 0, 0, str(output_path))


def merge_files(temp_dir: Path, output_file: Path) -> Tuple[int, int]:
    """Merge worker files into final output."""
    print(f"\nMerging worker files from {temp_dir}...")

    dfs = []
    for jsonl_file in sorted(temp_dir.glob("codex_prs_worker_*.jsonl")):
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
        repos_output = output_file.parent / "codex_repos.csv"
        repo_stats = merged.groupby("repo_nwo").agg(
            codex_pr_count=("pr_id", "count"),
            first_codex_pr=("created_at", "min"),
            last_codex_pr=("created_at", "max")
        ).reset_index()
        repo_stats = repo_stats.sort_values("codex_pr_count", ascending=False)
        repo_stats.to_csv(repos_output, index=False)
        print(f"Repos summary: {repos_output}")

    return final_count, unique_repos


def main():
    parser = argparse.ArgumentParser(description="Resume Codex PR download")
    parser.add_argument("--dry-run", action="store_true", help="Show plan only")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/output/codex_prs.parquet",
        help="Final merged output file"
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip final merge step"
    )
    args = parser.parse_args()

    # Get tokens
    tokens = get_tokens_from_env()
    if len(tokens) < 6:
        print(f"Error: Need at least 6 tokens, found {len(tokens)}")
        sys.exit(1)

    print(f"Found {len(tokens)} tokens (using first 6)")

    output_path = Path(args.output)
    temp_dir = output_path.parent / "temp_codex_workers"

    # Print plan
    print("\n" + "=" * 70)
    print("RESUME CODEX PR DOWNLOAD")
    print("=" * 70)
    print(f"Temp directory: {temp_dir}")
    print(f"Final output: {output_path}")
    print("\nResume ranges:")
    print("-" * 70)

    for worker_id, start_date, end_date in RESUME_RANGES:
        temp_file = temp_dir / f"codex_prs_worker_{worker_id}.jsonl"
        existing = 0
        if temp_file.exists():
            with open(temp_file, "r", encoding="utf-8") as f:
                existing = sum(1 for line in f if line.strip())
        print(f"  Worker {worker_id}: {start_date} to {end_date}")
        print(f"           Existing PRs: {existing:,}")
        print(f"           Token: {tokens[worker_id - 1][0]}")

    print("-" * 70)

    if args.dry_run:
        print("\n[DRY RUN] Would resume download with above configuration")
        return

    # Prepare worker arguments
    worker_args = []
    for worker_id, start_date, end_date in RESUME_RANGES:
        temp_file = temp_dir / f"codex_prs_worker_{worker_id}.parquet"
        worker_args.append((
            worker_id,
            tokens[worker_id - 1][1],  # Use token matching worker ID
            start_date,
            end_date,
            temp_file,
        ))

    print(f"\nStarting {len(worker_args)} parallel workers...")
    print("=" * 70)

    start_time = time.time()

    # Run workers in parallel
    with multiprocessing.Pool(processes=len(worker_args)) as pool:
        results = pool.starmap(worker_process, worker_args)

    elapsed = time.time() - start_time

    # Print results
    print("\n" + "=" * 70)
    print("WORKER RESULTS")
    print("=" * 70)

    total_existing = 0
    total_new = 0
    for worker_id, existing_count, new_count, path in results:
        print(f"  Worker {worker_id}: {existing_count:,} existing + {new_count:,} new")
        total_existing += existing_count
        total_new += new_count

    print(f"\nTotal: {total_existing:,} existing + {total_new:,} new = {total_existing + total_new:,}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")

    # Merge files
    if not args.skip_merge:
        final_count, unique_repos = merge_files(temp_dir, output_path)

        print("\n" + "=" * 70)
        print("DOWNLOAD COMPLETE")
        print("=" * 70)
        print(f"Output file: {output_path}")
        print(f"Total PRs: {final_count:,}")
        print(f"Unique repos: {unique_repos:,}")
        print(f"Total time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
