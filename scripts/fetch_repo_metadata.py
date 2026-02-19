#!/usr/bin/env python3
"""Parallel fetcher for repo metadata using batched GraphQL alias queries.

Reads the unique repo list from claude_commits_full.jsonl and fetches
full metadata (33 columns + README) for each repo using 6 parallel workers.

Usage:
    python scripts/fetch_repo_metadata.py
    python scripts/fetch_repo_metadata.py --no-readme   # skip README for speed
    python scripts/fetch_repo_metadata.py --dry-run      # show plan only
"""

import argparse
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv


def get_tokens_from_env() -> List[Tuple[str, str]]:
    """Get all available tokens from environment."""
    load_dotenv()
    tokens = []
    for i in range(1, 20):
        token = os.environ.get(f"GITHUB_TOKEN_{i}")
        if token:
            tokens.append((f"GITHUB_TOKEN_{i}", token))
    return tokens


def load_unique_repos(commits_file: Path) -> List[str]:
    """Extract unique repo names from commits JSONL file."""
    print(f"Loading unique repos from {commits_file}...")
    repos = []
    seen = set()
    with open(commits_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                commit = json.loads(line)
                nwo = commit.get("repo_nwo", "")
                if nwo and nwo not in seen:
                    seen.add(nwo)
                    repos.append(nwo)
    print(f"Found {len(repos):,} unique repos")
    return repos


def worker_process(
    worker_id: int,
    token: str,
    repo_list: List[str],
    output_path: Path,
    include_readme: bool,
) -> Tuple[int, int, str]:
    """Worker process that fetches repo metadata for a chunk of repos."""
    try:
        print(f"[Worker {worker_id}] Starting: {len(repo_list):,} repos -> {output_path.name}")

        from github_fetcher.batch_repo_fetcher import BatchRepoFetcher

        fetcher = BatchRepoFetcher(token, include_readme=include_readme)
        repos = fetcher.fetch_repos(repo_list, output_path)

        print(f"[Worker {worker_id}] Completed: {len(repos):,} repos")
        return (worker_id, len(repos), str(output_path))

    except Exception as e:
        print(f"[Worker {worker_id}] Error: {e}")
        import traceback
        traceback.print_exc()
        return (worker_id, 0, str(output_path))


def merge_results(temp_dir: Path, output_file: Path) -> int:
    """Merge worker JSONL files into one."""
    print(f"\nMerging results...")

    seen_nwo = set()
    total = 0

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as out:
        for worker_file in sorted(temp_dir.glob("repos_worker_*.jsonl")):
            count = 0
            with open(worker_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        repo = json.loads(line)
                        nwo = repo.get("nwo", "")
                        if nwo and nwo not in seen_nwo:
                            seen_nwo.add(nwo)
                            out.write(line)
                            count += 1
            print(f"  {worker_file.name}: {count:,} repos")
            total += count

    print(f"\nMerged output: {output_file}")
    print(f"Total unique repos: {total:,}")
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Fetch repo metadata for all Claude Code repos"
    )
    parser.add_argument(
        "--commits-file",
        type=str,
        default="data/output/claude_commits_full.jsonl",
        help="Input commits JSONL file",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/output/claude_repos_metadata.jsonl",
        help="Output file path",
    )
    parser.add_argument("--no-readme", action="store_true", help="Skip README content")
    parser.add_argument("--dry-run", action="store_true", help="Show plan only")
    args = parser.parse_args()

    # Get tokens
    tokens = get_tokens_from_env()
    if not tokens:
        print("Error: No GitHub tokens found in .env")
        sys.exit(1)
    print(f"Found {len(tokens)} tokens")

    # Load repos
    commits_file = Path(args.commits_file)
    repo_list = load_unique_repos(commits_file)

    # Split repos across workers
    num_workers = len(tokens)
    chunk_size = len(repo_list) // num_workers
    chunks = []
    for i in range(num_workers):
        start = i * chunk_size
        end = start + chunk_size if i < num_workers - 1 else len(repo_list)
        chunks.append(repo_list[start:end])

    output_path = Path(args.output)
    temp_dir = output_path.parent / "temp_repo_workers"

    print("\n" + "=" * 70)
    print("REPO METADATA FETCH PLAN")
    print("=" * 70)
    print(f"Total repos: {len(repo_list):,}")
    print(f"Workers: {num_workers}")
    print(f"Include README: {not args.no_readme}")
    print(f"Output: {output_path}")
    print(f"\nWorker assignments:")
    for i, chunk in enumerate(chunks):
        print(f"  Worker {i+1}: {len(chunk):,} repos (token: {tokens[i][0]})")
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN] Would start parallel fetch with above config")
        return

    # Create temp dir
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Prepare worker args
    worker_args = []
    for i, chunk in enumerate(chunks):
        temp_file = temp_dir / f"repos_worker_{i+1}.jsonl"
        worker_args.append((
            i + 1,
            tokens[i][1],
            chunk,
            temp_file,
            not args.no_readme,
        ))

    print(f"\nStarting {num_workers} parallel workers...")
    start_time = time.time()

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.starmap(worker_process, worker_args)

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("WORKER RESULTS")
    print("=" * 70)
    total_fetched = 0
    for worker_id, count, path in results:
        print(f"  Worker {worker_id}: {count:,} repos")
        total_fetched += count
    print(f"\nTotal from workers: {total_fetched:,}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")

    # Merge
    final_count = merge_results(temp_dir, output_path)

    print("\n" + "=" * 70)
    print("FETCH COMPLETE")
    print("=" * 70)
    print(f"Output: {output_path}")
    print(f"Total repos: {final_count:,}")
    print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
