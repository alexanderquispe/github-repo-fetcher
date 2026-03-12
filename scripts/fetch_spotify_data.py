#!/usr/bin/env python3
"""
Spotify Case Study: Fetch organization data and analyze Claude Code adoption.

Usage:
    python scripts/fetch_spotify_data.py --token YOUR_TOKEN
    python scripts/fetch_spotify_data.py --analyze  # Run analysis only
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd
import requests
from tqdm import tqdm

# Spotify GitHub organization
ORG_NAME = "spotify"


class SpotifyDataFetcher:
    """Fetches data for Spotify organization from GitHub API."""

    API_BASE = "https://api.github.com"

    def __init__(self, token: str):
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        self._rate_remaining = None

    def _update_rate_limit(self, response):
        if 'X-RateLimit-Remaining' in response.headers:
            self._rate_remaining = int(response.headers['X-RateLimit-Remaining'])

    def fetch_org_repos(self, org: str = ORG_NAME) -> List[Dict]:
        """Fetch all repositories for an organization."""
        print(f"\nFetching repos for {org}...")

        repos = []
        page = 1

        while True:
            url = f"{self.API_BASE}/orgs/{org}/repos"
            params = {"per_page": 100, "page": page, "type": "all"}

            response = self.session.get(url, params=params)
            self._update_rate_limit(response)

            if response.status_code != 200:
                print(f"Error: {response.status_code} - {response.text}")
                break

            page_repos = response.json()
            if not page_repos:
                break

            for repo in page_repos:
                repos.append({
                    "name": repo["name"],
                    "full_name": repo["full_name"],
                    "description": repo.get("description", ""),
                    "language": repo.get("language", ""),
                    "stars": repo.get("stargazers_count", 0),
                    "forks": repo.get("forks_count", 0),
                    "created_at": repo.get("created_at"),
                    "updated_at": repo.get("updated_at"),
                    "topics": repo.get("topics", []),
                })

            print(f"  Page {page}: {len(page_repos)} repos (total: {len(repos)})")
            page += 1

        print(f"  Total repos: {len(repos)}")
        return repos

    def fetch_repo_contributors(self, repo_full_name: str) -> List[Dict]:
        """Fetch contributors for a single repo."""
        contributors = []
        page = 1

        while True:
            url = f"{self.API_BASE}/repos/{repo_full_name}/contributors"
            params = {"per_page": 100, "page": page, "anon": "false"}

            response = self.session.get(url, params=params)
            self._update_rate_limit(response)

            if response.status_code == 404:
                return []  # Repo not found or empty

            if response.status_code == 403:
                # Rate limited
                if 'Retry-After' in response.headers:
                    wait = int(response.headers['Retry-After'])
                    print(f"\n  Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                else:
                    time.sleep(60)
                    continue

            if response.status_code != 200:
                return []

            page_contribs = response.json()
            if not page_contribs:
                break

            for c in page_contribs:
                contributors.append({
                    "repo": repo_full_name,
                    "login": c.get("login", ""),
                    "contributions": c.get("contributions", 0),
                    "type": c.get("type", "User"),
                })

            if len(page_contribs) < 100:
                break

            page += 1

        return contributors

    def fetch_all_contributors(self, repos: List[Dict], output_path: Path = None) -> pd.DataFrame:
        """Fetch contributors for all repos."""
        print(f"\nFetching contributors for {len(repos)} repos...")

        all_contributors = []

        for repo in tqdm(repos, desc="Fetching contributors"):
            contributors = self.fetch_repo_contributors(repo["full_name"])
            all_contributors.extend(contributors)

            # Show rate limit
            if self._rate_remaining and self._rate_remaining < 100:
                tqdm.write(f"  Rate limit: {self._rate_remaining}")

        df = pd.DataFrame(all_contributors)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_path, index=False)
            print(f"\nSaved {len(df)} contributor records to {output_path}")

        return df


def match_to_claude_users(
    contributors_df: pd.DataFrame,
    claude_commits_path: Path
) -> pd.DataFrame:
    """Match Spotify contributors to Claude Code users."""
    print("\nMatching contributors to Claude Code users...")

    # Load Claude commits
    claude_df = pd.read_parquet(claude_commits_path)
    claude_df['committed_date'] = pd.to_datetime(claude_df['committed_date'], utc=True)

    # Get unique Claude users and their first use date
    claude_users = claude_df.groupby('author_login')['committed_date'].min().reset_index()
    claude_users.columns = ['login', 'first_claude_use']

    print(f"  Total Claude users: {len(claude_users):,}")

    # Get unique Spotify contributors
    spotify_users = contributors_df['login'].unique()
    print(f"  Spotify contributors: {len(spotify_users):,}")

    # Match
    spotify_claude = pd.merge(
        pd.DataFrame({'login': spotify_users}),
        claude_users,
        on='login',
        how='left'
    )

    spotify_claude['is_claude_user'] = spotify_claude['first_claude_use'].notna()

    matched = spotify_claude[spotify_claude['is_claude_user']]
    print(f"  Spotify + Claude intersection: {len(matched):,}")

    return spotify_claude


def build_collaboration_network(contributors_df: pd.DataFrame):
    """Build collaboration network from contributor data."""
    print("\nBuilding collaboration network...")

    # Group by repo to get collaborators
    repo_users = contributors_df.groupby('repo')['login'].apply(set).to_dict()

    # Build edges (users who share repos)
    from collections import defaultdict
    edges = defaultdict(int)

    for repo, users in repo_users.items():
        users_list = list(users)
        for i in range(len(users_list)):
            for j in range(i + 1, len(users_list)):
                edge = tuple(sorted([users_list[i], users_list[j]]))
                edges[edge] += 1

    print(f"  Nodes: {len(contributors_df['login'].unique()):,}")
    print(f"  Edges: {len(edges):,}")

    return edges


def analyze_spotify_adoption(
    contributors_df: pd.DataFrame,
    spotify_claude_df: pd.DataFrame,
    output_dir: Path
):
    """Analyze Claude Code adoption within Spotify."""
    print("\n" + "=" * 60)
    print("SPOTIFY CLAUDE CODE ADOPTION ANALYSIS")
    print("=" * 60)

    # Basic stats
    total_contributors = len(contributors_df['login'].unique())
    claude_users = spotify_claude_df[spotify_claude_df['is_claude_user']]
    n_claude = len(claude_users)

    print(f"\nTotal Spotify contributors: {total_contributors:,}")
    print(f"Claude Code users: {n_claude:,} ({100*n_claude/total_contributors:.1f}%)")

    if n_claude == 0:
        print("\nNo Spotify contributors found in Claude Code data.")
        return {}

    # Adoption timeline
    print("\n--- Adoption Timeline ---")
    claude_users_sorted = claude_users.sort_values('first_claude_use')

    # Monthly adoption
    claude_users_sorted['month'] = claude_users_sorted['first_claude_use'].dt.to_period('M')
    monthly = claude_users_sorted.groupby('month').size()
    print("\nMonthly new adopters:")
    for month, count in monthly.items():
        print(f"  {month}: {count}")

    # First adopters
    print("\n--- First 10 Adopters ---")
    first_10 = claude_users_sorted.head(10)[['login', 'first_claude_use']]
    for _, row in first_10.iterrows():
        print(f"  {row['login']}: {row['first_claude_use'].strftime('%Y-%m-%d')}")

    # Repos with most Claude users
    print("\n--- Repos with Most Claude Users ---")
    repo_claude_count = contributors_df[
        contributors_df['login'].isin(claude_users['login'])
    ].groupby('repo')['login'].nunique().sort_values(ascending=False)

    for repo, count in repo_claude_count.head(10).items():
        print(f"  {repo}: {count} Claude users")

    # Build results
    results = {
        "total_contributors": total_contributors,
        "claude_users": n_claude,
        "adoption_rate": n_claude / total_contributors,
        "monthly_adoption": {str(k): int(v) for k, v in monthly.items()},
        "first_adopters": claude_users_sorted.head(10)['login'].tolist(),
        "top_repos_by_claude_users": repo_claude_count.head(10).to_dict(),
    }

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "spotify_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved results to {output_dir / 'spotify_analysis.json'}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Spotify Case Study")
    parser.add_argument("--token", type=str, help="GitHub token")
    parser.add_argument("--fetch", action="store_true", help="Fetch data from GitHub")
    parser.add_argument("--analyze", action="store_true", help="Run analysis")
    parser.add_argument("--data-dir", type=Path, default=Path("data/output"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/output"))

    args = parser.parse_args()

    if not args.fetch and not args.analyze:
        args.fetch = True
        args.analyze = True

    if args.fetch:
        if not args.token:
            print("ERROR: --token required for fetching")
            return

        fetcher = SpotifyDataFetcher(args.token)

        # Fetch repos
        repos = fetcher.fetch_org_repos()
        repos_df = pd.DataFrame(repos)
        repos_df.to_parquet(args.data_dir / "spotify_repos.parquet", index=False)

        # Fetch contributors
        contributors_df = fetcher.fetch_all_contributors(
            repos,
            output_path=args.data_dir / "spotify_contributors.parquet"
        )

        # Match to Claude users
        spotify_claude_df = match_to_claude_users(
            contributors_df,
            args.data_dir / "claude_commits.parquet"
        )
        spotify_claude_df.to_parquet(
            args.data_dir / "spotify_claude_users.parquet",
            index=False
        )

    if args.analyze:
        # Load data
        contributors_df = pd.read_parquet(args.data_dir / "spotify_contributors.parquet")
        spotify_claude_df = pd.read_parquet(args.data_dir / "spotify_claude_users.parquet")

        # Run analysis
        results = analyze_spotify_adoption(
            contributors_df,
            spotify_claude_df,
            args.output_dir
        )


if __name__ == "__main__":
    main()
