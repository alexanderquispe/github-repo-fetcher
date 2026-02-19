#!/usr/bin/env python3
"""
Fetch Claude Code co-authored commits by downloading GH Archive hourly files.

This is the FREE approach - downloads and processes hourly JSON archives.

Prerequisites:
    pip install requests pandas pyarrow tqdm

Usage:
    # Full download (Feb 2025 - today)
    python scripts/fetch_claude_gharchive.py -o data/output/claude_commits_gharchive.parquet

    # Custom date range
    python scripts/fetch_claude_gharchive.py --start-date 2025-06-01 --end-date 2025-06-30 -o output.parquet

    # Parallel downloads (faster)
    python scripts/fetch_claude_gharchive.py --workers 4 -o output.parquet

    # Resume interrupted download
    python scripts/fetch_claude_gharchive.py --resume -o output.parquet

Data Estimate:
    - ~8,600 hourly files for full year
    - ~841 GB compressed download
    - ~10-20 hours with parallel downloads
    - Cost: FREE
"""

import argparse
import gzip
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import requests
import pandas as pd
from tqdm import tqdm


# GH Archive URL pattern
GHARCHIVE_URL = "https://data.gharchive.org/{year}-{month:02d}-{day:02d}-{hour}.json.gz"

# Claude co-author pattern
CLAUDE_PATTERN = re.compile(
    r'Co-Authored-By:.*?noreply@anthropic\.com',
    re.IGNORECASE
)

# Extract Claude model from message
MODEL_PATTERN = re.compile(
    r'Co-Authored-By:\s*Claude\s*([^<]*)?<[^>]*@anthropic\.com>',
    re.IGNORECASE
)


def parse_date(date_str: str) -> date:
    """Parse a date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid date format: {date_str}. Use YYYY-MM-DD."
        )


def generate_hour_urls(start_date: date, end_date: date) -> Generator[Tuple[str, str], None, None]:
    """Generate URLs for each hour in the date range."""
    current = datetime.combine(start_date, datetime.min.time())
    end = datetime.combine(end_date, datetime.max.time())

    while current <= end:
        url = GHARCHIVE_URL.format(
            year=current.year,
            month=current.month,
            day=current.day,
            hour=current.hour
        )
        file_id = f"{current.year}-{current.month:02d}-{current.day:02d}-{current.hour}"
        yield url, file_id
        current += timedelta(hours=1)


def extract_claude_model(message: str) -> str:
    """Extract Claude model name from commit message."""
    match = MODEL_PATTERN.search(message)
    if match:
        model_part = match.group(1)
        if model_part:
            model_part = model_part.strip()
            if model_part:
                return model_part
        return "Claude"
    return "Claude"


def process_gharchive_file(url: str, file_id: str, temp_dir: Path) -> Tuple[str, List[Dict], Optional[str]]:
    """
    Download and process a single GH Archive file.

    Returns:
        Tuple of (file_id, list of claude commits, error message or None)
    """
    commits = []
    error = None

    try:
        # Download with retry
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=60, stream=True)
                if response.status_code == 404:
                    # File doesn't exist (future date or missing hour)
                    return file_id, [], None
                response.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

        # Process gzipped JSON lines
        with gzip.GzipFile(fileobj=response.raw) as gz:
            for line in gz:
                try:
                    event = json.loads(line.decode('utf-8'))

                    # Only process PushEvents
                    if event.get('type') != 'PushEvent':
                        continue

                    # Check commits in the push
                    payload = event.get('payload', {})
                    event_commits = payload.get('commits', [])

                    for commit in event_commits:
                        message = commit.get('message', '')

                        # Check for Claude co-author
                        if CLAUDE_PATTERN.search(message):
                            repo = event.get('repo', {})
                            actor = event.get('actor', {})
                            author = commit.get('author', {})

                            commits.append({
                                'sha': commit.get('sha', ''),
                                'repo_name': repo.get('name', ''),
                                'repo_id': repo.get('id', ''),
                                'actor_login': actor.get('login', ''),
                                'actor_id': actor.get('id', ''),
                                'author_name': author.get('name', ''),
                                'author_email': author.get('email', ''),
                                'message': message,
                                'event_date': event.get('created_at', ''),
                                'claude_model': extract_claude_model(message),
                            })

                except json.JSONDecodeError:
                    continue

    except Exception as e:
        error = str(e)

    return file_id, commits, error


def save_progress(commits: List[Dict], output_path: Path, processed_files: set, progress_file: Path):
    """Save current progress to disk."""
    if commits:
        df = pd.DataFrame(commits)
        df.to_parquet(output_path, index=False)

    # Save processed file IDs
    with open(progress_file, 'w') as f:
        f.write('\n'.join(sorted(processed_files)))


def load_progress(output_path: Path, progress_file: Path) -> Tuple[List[Dict], set]:
    """Load previous progress if resuming."""
    commits = []
    processed_files = set()

    if output_path.exists():
        df = pd.read_parquet(output_path)
        commits = df.to_dict('records')

    if progress_file.exists():
        with open(progress_file, 'r') as f:
            processed_files = set(f.read().strip().split('\n'))

    return commits, processed_files


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Claude commits from GH Archive (free download approach)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output file path (.parquet)"
    )
    parser.add_argument(
        "--start-date",
        type=parse_date,
        default=date(2025, 2, 1),
        help="Start date (default: 2025-02-01)"
    )
    parser.add_argument(
        "--end-date",
        type=parse_date,
        default=None,
        help="End date (default: today)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel download workers (default: 4)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous progress"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="Save progress every N files (default: 100)"
    )

    args = parser.parse_args()

    end_date = args.end_date or date.today()
    output_path = Path(args.output)
    if not output_path.suffix:
        output_path = output_path.with_suffix('.parquet')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_file = output_path.with_suffix('.progress')
    temp_dir = output_path.parent / "temp_gharchive"
    temp_dir.mkdir(exist_ok=True)

    # Calculate total files
    urls = list(generate_hour_urls(args.start_date, end_date))
    total_files = len(urls)

    # Estimate download size
    estimated_gb = (total_files * 100) / 1024  # ~100MB per file average

    print("=" * 60)
    print("GH ARCHIVE DOWNLOAD")
    print("=" * 60)
    print(f"Date range: {args.start_date} to {end_date}")
    print(f"Total hourly files: {total_files:,}")
    print(f"Estimated download: ~{estimated_gb:.0f} GB")
    print(f"Parallel workers: {args.workers}")
    print(f"Output: {output_path}")
    print("=" * 60)

    # Load previous progress if resuming
    all_commits = []
    processed_files = set()

    if args.resume:
        all_commits, processed_files = load_progress(output_path, progress_file)
        if processed_files:
            print(f"\nResuming: {len(processed_files):,} files already processed")
            print(f"Existing commits: {len(all_commits):,}")

    # Filter out already processed files
    urls_to_process = [(url, fid) for url, fid in urls if fid not in processed_files]

    if not urls_to_process:
        print("\nAll files already processed!")
        return

    print(f"\nFiles to process: {len(urls_to_process):,}")
    print("\nStarting download and processing...")

    errors = []
    files_since_save = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_gharchive_file, url, fid, temp_dir): fid
            for url, fid in urls_to_process
        }

        # Process results as they complete
        with tqdm(total=len(urls_to_process), desc="Processing", unit="file") as pbar:
            for future in as_completed(futures):
                file_id = futures[future]

                try:
                    fid, commits, error = future.result()

                    if error:
                        errors.append((fid, error))
                    else:
                        all_commits.extend(commits)
                        processed_files.add(fid)
                        files_since_save += 1

                        # Update progress bar
                        pbar.set_postfix({
                            'commits': len(all_commits),
                            'errors': len(errors)
                        })

                    # Save progress periodically
                    if files_since_save >= args.save_interval:
                        save_progress(all_commits, output_path, processed_files, progress_file)
                        files_since_save = 0

                except Exception as e:
                    errors.append((file_id, str(e)))

                pbar.update(1)

    # Final save
    save_progress(all_commits, output_path, processed_files, progress_file)

    # Remove duplicates by SHA
    if all_commits:
        df = pd.DataFrame(all_commits)
        original_count = len(df)
        df = df.drop_duplicates(subset=['sha'], keep='first')
        final_count = len(df)

        if original_count != final_count:
            print(f"\nRemoved {original_count - final_count:,} duplicate commits")
            df.to_parquet(output_path, index=False)

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"Total commits: {len(all_commits):,}")
    print(f"Files processed: {len(processed_files):,}")
    print(f"Errors: {len(errors)}")
    print(f"Output: {output_path}")

    if all_commits:
        df = pd.read_parquet(output_path)
        print(f"\nUnique repos: {df['repo_name'].nunique():,}")
        print(f"Unique authors: {df['actor_login'].nunique():,}")

    if errors and len(errors) <= 10:
        print("\nErrors encountered:")
        for fid, err in errors[:10]:
            print(f"  {fid}: {err}")

    # Clean up progress file on successful completion
    if not errors and progress_file.exists():
        progress_file.unlink()


if __name__ == "__main__":
    main()
