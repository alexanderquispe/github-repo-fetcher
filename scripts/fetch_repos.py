#!/usr/bin/env python3
"""CLI entry point for fetching GitHub repository data."""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

from github_fetcher import GitHubFetcher


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch GitHub repository data using GraphQL API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch repos from Peru (default: stars>=1, no forks)
  python fetch_repos.py --location "Peru" -o peru.parquet

  # Fetch repos with 5+ stars
  python fetch_repos.py --location "Peru" --min-stars 5 -o peru.parquet

  # Add custom filters (language, date, size, etc.)
  python fetch_repos.py --location "Peru" --min-stars 5 \\
      --filter "language:Python pushed:>=2022-01-01" -o peru_python.parquet

  # Include forks (disabled by default)
  python fetch_repos.py --location "Peru" --include-forks -o peru_with_forks.parquet

  # Limit to first 100 users (for testing)
  python fetch_repos.py --location "Peru" --max-users 100 -o peru_test.parquet

  # Fetch by custom query (direct search, not location-based)
  python fetch_repos.py --query "topic:machine-learning stars:>=100" -o ml_repos.parquet

  # Fetch a single repository
  python fetch_repos.py --repo "microsoft/vscode" -o vscode.parquet

Available filters for --filter:
  language:Python       - Filter by language
  pushed:>=2022-01-01   - Recent activity
  created:>=2020-01-01  - Creation date
  size:>=10             - Minimum size in KB
  license:mit           - License type
  archived:false        - Exclude archived
  topic:data-science    - By topic
        """
    )

    # Search mode (mutually exclusive conceptually, but we handle in code)
    search_group = parser.add_argument_group("Search mode (choose one)")
    search_group.add_argument(
        "--location",
        type=str,
        help="Fetch repos from users/orgs in this location (e.g., 'Peru', 'Brazil')"
    )
    search_group.add_argument(
        "--query",
        type=str,
        help="Custom GitHub search query (direct search, not location-based)"
    )
    search_group.add_argument(
        "--repo",
        type=str,
        help="Fetch a single repository by 'owner/name'"
    )

    # Filter options
    filter_group = parser.add_argument_group("Filters")
    filter_group.add_argument(
        "--min-stars",
        type=int,
        default=1,
        help="Minimum star count (default: 1)"
    )
    filter_group.add_argument(
        "--filter",
        type=str,
        default="",
        help="Additional GitHub search filters (e.g., 'language:Python pushed:>=2022-01-01')"
    )
    filter_group.add_argument(
        "--include-forks",
        action="store_true",
        help="Include forked repos (excluded by default)"
    )

    # Output options
    output_group = parser.add_argument_group("Output options")
    output_group.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output file path (.parquet or .csv)"
    )
    output_group.add_argument(
        "--max-users",
        type=int,
        default=None,
        help="Maximum users to process in --location mode (default: all)"
    )
    output_group.add_argument(
        "--max-repos",
        type=int,
        default=10000,
        help="Maximum repos to fetch in --query mode (default: 10000)"
    )
    output_group.add_argument(
        "--no-orgs",
        action="store_true",
        help="Exclude organizations, only fetch from user accounts"
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
        sys.exit(1)

    # Validate arguments
    if not args.repo and not args.location and not args.query:
        print("Error: Must specify --location, --query, or --repo")
        parser.print_help()
        sys.exit(1)

    # Determine output path
    output_path = Path(args.output)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".parquet")

    # Initialize fetcher
    try:
        fetcher = GitHubFetcher(token)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Fetch data
    try:
        if args.repo:
            # Single repository mode
            print(f"Fetching repository: {args.repo}")
            repo_data = fetcher.fetch_repo_details(args.repo)
            if repo_data:
                fetcher._repos = [repo_data]
            else:
                print(f"Repository not found: {args.repo}")
                sys.exit(1)

        elif args.query:
            # Custom query mode (direct search)
            fetcher.fetch_by_query(
                custom_query=args.query,
                min_stars=args.min_stars,
                max_repos=args.max_repos,
                output_path=output_path
            )

        else:
            # Location-based search (two-step approach)
            fetcher.fetch_by_location_two_step(
                location=args.location,
                include_orgs=not args.no_orgs,
                min_stars=args.min_stars,
                max_users=args.max_users,
                include_forks=args.include_forks,
                extra_filter=args.filter,
                output_path=output_path
            )

        # Save output
        if output_path.suffix == ".csv":
            fetcher.save_to_csv(output_path)
        else:
            fetcher.save_to_parquet(output_path)

        print(f"\nDone! Output saved to: {output_path}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving progress...")
        if output_path.suffix == ".csv":
            fetcher.save_to_csv(output_path)
        else:
            fetcher.save_to_parquet(output_path)
        sys.exit(0)

    except Exception as e:
        print(f"\nError: {e}")
        # Save partial results on error
        if fetcher._repos:
            print("Saving partial results...")
            if output_path.suffix == ".csv":
                fetcher.save_to_csv(output_path)
            else:
                fetcher.save_to_parquet(output_path)
        sys.exit(1)


if __name__ == "__main__":
    main()
