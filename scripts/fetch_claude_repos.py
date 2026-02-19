#!/usr/bin/env python3
"""CLI entry point for fetching Claude Code co-authored commits and repositories.

This script finds all GitHub repositories and commits where Claude appears as a
commit co-author, enabling research on Claude Code adoption patterns.

Examples:
    # Phase 1: Find all commits
    python fetch_claude_repos.py --commits-only -o claude_commits.parquet

    # Phase 2: Enrich repos (after Phase 1)
    python fetch_claude_repos.py --enrich-from claude_commits.parquet -o claude_repos.parquet

    # Both phases together
    python fetch_claude_repos.py -o claude_repos.parquet

    # Custom date range
    python fetch_claude_repos.py --start-date 2025-01-20 -o recent_claude.parquet

    # Save both commits and repos
    python fetch_claude_repos.py --commits-output commits.parquet -o repos.parquet
"""

import argparse
import os
import sys
from datetime import date, datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

from github_fetcher.claude_fetcher import ClaudeCommitFetcher


def parse_date(date_str: str) -> date:
    """Parse a date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid date format: {date_str}. Use YYYY-MM-DD."
        )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch Claude Code co-authored commits and repositories from GitHub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Timeline Context:
  - February 2025: Claude Code released (preview/beta)
  - May 2025: Generally available with Claude 4
  - October 2025: Web version launched on claude.ai

Examples:
  # Phase 1 only: Find all Claude co-authored commits
  python fetch_claude_repos.py --commits-only -o data/claude_commits.parquet

  # Phase 2 only: Enrich repositories from existing commit data
  python fetch_claude_repos.py --enrich-from data/claude_commits.parquet -o data/claude_repos.parquet

  # Both phases in one run (saves both commits and repos)
  python fetch_claude_repos.py --commits-output data/commits.parquet -o data/repos.parquet

  # Quick test with recent data only
  python fetch_claude_repos.py --start-date 2025-01-20 --commits-only -o test_commits.parquet

  # Full historical search
  python fetch_claude_repos.py --start-date 2025-02-01 -o data/all_claude_repos.parquet

Output Schema:
  claude_commits.parquet:
    - sha, repo_nwo, message, committed_date
    - author_name, author_email, author_login
    - claude_model (extracted from Co-Authored-By line)

  claude_repos.parquet:
    - All standard repo fields (33 columns)
    - claude_commit_count, first_claude_commit, last_claude_commit
    - claude_collaborators (JSON list), claude_models_used (JSON list)
        """
    )

    # Mode selection
    mode_group = parser.add_argument_group("Mode (choose one)")
    mode_group.add_argument(
        "--commits-only",
        action="store_true",
        help="Phase 1 only: Fetch commits without enriching repos"
    )
    mode_group.add_argument(
        "--enrich-from",
        type=str,
        metavar="FILE",
        help="Phase 2 only: Enrich repos from existing commits parquet file"
    )

    # Date range options
    date_group = parser.add_argument_group("Date range")
    date_group.add_argument(
        "--start-date",
        type=parse_date,
        default=date(2025, 2, 1),
        help="Start date for commit search (default: 2025-02-01, Claude Code release)"
    )
    date_group.add_argument(
        "--end-date",
        type=parse_date,
        default=None,
        help="End date for commit search (default: today)"
    )

    # Output options
    output_group = parser.add_argument_group("Output options")
    output_group.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output file path for repos (.parquet)"
    )
    output_group.add_argument(
        "--commits-output",
        type=str,
        metavar="FILE",
        help="Optional: Also save commits to this file (for full runs)"
    )

    # Authentication
    auth_group = parser.add_argument_group("Authentication")
    auth_group.add_argument(
        "--token",
        type=str,
        help="GitHub Personal Access Token (or set GITHUB_TOKEN env var)"
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Get token
    token = args.token or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Error: GitHub token required. Set GITHUB_TOKEN env var or use --token")
        print("Get a token at: https://github.com/settings/tokens")
        print("Required scopes: public_repo (for public repo access)")
        sys.exit(1)

    # Validate arguments
    if args.commits_only and args.enrich_from:
        print("Error: Cannot use both --commits-only and --enrich-from")
        sys.exit(1)

    # Determine output paths
    output_path = Path(args.output)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".parquet")

    commits_output = None
    if args.commits_output:
        commits_output = Path(args.commits_output)
        if not commits_output.suffix:
            commits_output = commits_output.with_suffix(".parquet")

    # Initialize fetcher
    try:
        fetcher = ClaudeCommitFetcher(token)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    try:
        if args.enrich_from:
            # Phase 2 only: Enrich from existing commits file
            commits_file = Path(args.enrich_from)
            if not commits_file.exists():
                print(f"Error: Commits file not found: {commits_file}")
                sys.exit(1)

            repos = fetcher.enrich_repositories(
                commits_file=commits_file,
                output_path=output_path
            )

            print(f"\nDone! Enriched {len(repos):,} repositories")
            print(f"Output saved to: {output_path}")

        elif args.commits_only:
            # Phase 1 only: Fetch commits
            commits = fetcher.fetch_claude_commits(
                start_date=args.start_date,
                end_date=args.end_date,
                output_path=output_path  # Save commits to main output
            )

            print(f"\nDone! Fetched {len(commits):,} commits")
            print(f"Output saved to: {output_path}")

        else:
            # Both phases
            commits, repos = fetcher.fetch_all(
                start_date=args.start_date,
                end_date=args.end_date,
                commits_output=commits_output,
                repos_output=output_path
            )

            print(f"\nDone!")
            print(f"  Commits: {len(commits):,}")
            print(f"  Repositories: {len(repos):,}")
            if commits_output:
                print(f"  Commits saved to: {commits_output}")
            print(f"  Repos saved to: {output_path}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        print("Partial results may have been saved to the output file(s).")
        sys.exit(0)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\nPartial results may have been saved to the output file(s).")
        sys.exit(1)


if __name__ == "__main__":
    main()
