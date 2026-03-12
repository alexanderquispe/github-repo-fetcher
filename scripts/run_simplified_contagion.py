#!/usr/bin/env python3
"""
Simplified Contagion Analysis Pipeline for Claude Code Adoption.

This script runs a cleaner, focused pipeline:

1. Identify Claude users as of May 2025
2. Fetch ALL repos for each Claude user
3. Fetch contributors for those repos (collaborators)
4. Check: Did collaborators adopt Claude by June 2025?

Usage:
    # Full pipeline (fetch data from GitHub API)
    python scripts/run_simplified_contagion.py --fetch

    # Analysis only (use existing data)
    python scripts/run_simplified_contagion.py --analyze

    # Both
    python scripts/run_simplified_contagion.py --fetch --analyze
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd


# Key dates
SNAPSHOT_DATE = datetime(2025, 5, 31)  # Identify Claude users
OUTCOME_DATE = datetime(2025, 6, 30)   # Check collaborator adoption


def print_header(title: str) -> None:
    """Print a section header."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def step1_identify_claude_users(data_dir: Path) -> set:
    """
    Step 1: Identify Claude users as of May 2025.

    Returns:
        Set of user logins
    """
    print_header("STEP 1: IDENTIFY CLAUDE USERS (May 2025)")

    commits_path = data_dir / "claude_commits_full.parquet"

    if not commits_path.exists():
        print(f"ERROR: {commits_path} not found!")
        sys.exit(1)

    df = pd.read_parquet(commits_path)
    df['committed_date'] = pd.to_datetime(df['committed_date'], errors='coerce')

    print(f"Total commits: {len(df):,}")
    print(f"Date range: {df['committed_date'].min().date()} to {df['committed_date'].max().date()}")

    # Filter to May 2025
    mask = df['committed_date'] <= SNAPSHOT_DATE
    users = set(df[mask]['author_login'].dropna().unique())

    print(f"\nClaude users by {SNAPSHOT_DATE.date()}: {len(users):,}")

    # Save user list
    user_df = pd.DataFrame({'user_login': list(users)})
    user_df.to_parquet(data_dir / "claude_users_may2025.parquet", index=False)
    print(f"Saved to {data_dir / 'claude_users_may2025.parquet'}")

    return users


def step2_fetch_user_repos(data_dir: Path, users: set, token: str) -> dict:
    """
    Step 2: Fetch ALL repos for each Claude user.

    Returns:
        Dict mapping user -> list of repo NWOs
    """
    print_header("STEP 2: FETCH REPOS FOR CLAUDE USERS")

    from src.github_fetcher.user_repo_fetcher import UserRepoFetcher

    fetcher = UserRepoFetcher(token)

    output_path = data_dir / "user_repos.parquet"
    resume_path = output_path if output_path.exists() else None

    user_repos = fetcher.fetch_repos_for_users(
        users=list(users),
        output_path=output_path,
        resume_from=resume_path
    )

    # Summary
    total_repos = sum(len(repos) for repos in user_repos.values())
    unique_repos = fetcher.get_all_repos()

    print(f"\nTotal repo associations: {total_repos:,}")
    print(f"Unique repos: {len(unique_repos):,}")

    return user_repos


def step3_fetch_contributors(data_dir: Path, repos: set, token: str) -> None:
    """
    Step 3: Fetch contributors for all repos.
    """
    print_header("STEP 3: FETCH CONTRIBUTORS FOR REPOS")

    from src.github_fetcher.contributor_fetcher import ContributorFetcher

    fetcher = ContributorFetcher(token)

    output_path = data_dir / "contributors.parquet"
    resume_path = output_path if output_path.exists() else None

    fetcher.fetch_all_repos(
        repos=list(repos),
        output_path=output_path,
        resume_from=resume_path
    )


def step4_run_analysis(data_dir: Path, output_dir: Path) -> dict:
    """
    Step 4: Run contagion analysis.
    """
    print_header("STEP 4: CONTAGION ANALYSIS")

    from analysis.contagion_analysis import run_contagion_analysis

    results = run_contagion_analysis(data_dir, output_dir)

    return results


def generate_summary_report(results: dict, output_dir: Path) -> None:
    """Generate a human-readable summary report."""
    print_header("GENERATING SUMMARY REPORT")

    report_lines = [
        "# Claude Code Contagion Analysis Report",
        f"\nGenerated: {datetime.now().isoformat()}",
        f"\nSnapshot Date: May 31, 2025 (identify Claude users)",
        f"Outcome Date: June 30, 2025 (check collaborator adoption)",
        "\n---\n",
    ]

    if "adoption_rates" in results:
        rates = results["adoption_rates"]
        report_lines.extend([
            "## Key Findings\n",
            f"- **Claude users (May 2025)**: {rates.get('claude_users_may', 0):,}",
            f"- **Claude users (June 2025)**: {rates.get('claude_users_june', 0):,}",
            f"- **New adopters in June**: {rates.get('new_adopters_june', 0):,}",
            "",
            f"- **Total collaborators of Claude users**: {rates.get('total_collaborators', 0):,}",
            f"- **Exposed collaborators** (not already users): {rates.get('exposed_collaborators', 0):,}",
            f"- **Exposed who adopted Claude**: {rates.get('exposed_adopted', 0):,}",
            "",
            f"### Adoption Rate Among Exposed Collaborators: **{rates.get('exposed_adoption_rate', 0):.2%}**",
            "",
        ])

    if "exposure_analysis" in results:
        report_lines.extend([
            "## Adoption Rate by Exposure Level\n",
            "| # Claude Connections | Collaborators | Adopted | Rate |",
            "|---------------------|---------------|---------|------|",
        ])
        for row in results["exposure_analysis"][:10]:
            rate_pct = f"{row['adoption_rate']:.2%}"
            report_lines.append(
                f"| {row['exposure_level']} | {row['num_collaborators']:,} | {row['num_adopted']:,} | {rate_pct} |"
            )
        report_lines.append("")

    if "timing_analysis" in results and "error" not in results.get("timing_analysis", {}):
        timing = results["timing_analysis"]
        report_lines.extend([
            "## Time to Adoption\n",
            f"For collaborators who adopted Claude after their Claude-using connection:\n",
            f"- **Mean**: {timing.get('mean_gap_days', 0):.1f} days",
            f"- **Median**: {timing.get('median_gap_days', 0):.1f} days",
            f"- **Range**: {timing.get('min_gap_days', 0)} - {timing.get('max_gap_days', 0)} days",
            "",
        ])

    report_lines.extend([
        "## Interpretation\n",
        "If the adoption rate among collaborators of Claude users is significantly higher",
        "than the baseline adoption rate, this suggests **peer influence / contagion** in",
        "Claude Code adoption.",
        "",
        "Higher adoption rates at higher exposure levels (more Claude-using connections)",
        "would provide stronger evidence of a dose-response relationship.",
        "",
        "---",
        "*Generated by Claude Code Contagion Analysis Pipeline*",
    ])

    report_path = output_dir / "contagion_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"Saved report to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Claude Code Contagion Analysis Pipeline"
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch data from GitHub API (requires GITHUB_TOKEN)"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run contagion analysis"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/output"),
        help="Directory containing input data"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/output"),
        help="Directory for output files"
    )

    args = parser.parse_args()

    if not args.fetch and not args.analyze:
        print("Please specify --fetch, --analyze, or both")
        parser.print_help()
        sys.exit(1)

    data_dir = args.data_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print_header("CLAUDE CODE CONTAGION ANALYSIS PIPELINE")
    print(f"\nData directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    if args.fetch:
        # Check for token
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            print("\nERROR: GITHUB_TOKEN environment variable not set")
            print("Please set your GitHub Personal Access Token:")
            print("  export GITHUB_TOKEN=ghp_your_token_here")
            sys.exit(1)

        # Step 1: Identify Claude users
        claude_users = step1_identify_claude_users(data_dir)

        # Step 2: Fetch repos for each Claude user
        user_repos = step2_fetch_user_repos(data_dir, claude_users, token)

        # Get unique repos
        unique_repos = set()
        for repos in user_repos.values():
            unique_repos.update(repos)

        print(f"\nUnique repos to fetch contributors for: {len(unique_repos):,}")

        # Step 3: Fetch contributors
        step3_fetch_contributors(data_dir, unique_repos, token)

    if args.analyze:
        # Step 4: Run analysis
        results = step4_run_analysis(data_dir, output_dir)

        if "error" not in results:
            # Generate report
            generate_summary_report(results, output_dir)

    print_header("PIPELINE COMPLETE")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
