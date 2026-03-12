"""Fetcher for repository contributors with rate limiting."""

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import requests
from tqdm import tqdm

from .utils import exponential_backoff


class ContributorFetcher:
    """Fetcher for repository contributors using GitHub REST API.

    Fetches contributor lists for repositories to build stronger collaboration
    network edges. Contributors are users who have committed to a repository.

    API: GET /repos/{owner}/{repo}/contributors
    Rate Limit: 5000 requests/hour (authenticated)
    """

    API_BASE = "https://api.github.com"

    # Save progress interval
    SAVE_INTERVAL = 500

    # Rate limit configuration
    RATE_LIMIT_THRESHOLD = 100  # Wait when remaining < this

    def __init__(self, token: str):
        """
        Initialize the contributor fetcher.

        Args:
            token: GitHub Personal Access Token
        """
        if not token:
            raise ValueError("GitHub token is required")

        self.token = token
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })

        # Rate limit tracking
        self._rate_limit_remaining: Optional[int] = None
        self._rate_limit_reset: Optional[datetime] = None

        # Results storage
        self._contributors: List[Dict[str, Any]] = []
        self._failed_repos: List[str] = []

    def _update_rate_limit(self, headers: Dict[str, str]) -> None:
        """Update rate limit tracking from response headers."""
        if 'X-RateLimit-Remaining' in headers:
            self._rate_limit_remaining = int(headers['X-RateLimit-Remaining'])
        if 'X-RateLimit-Reset' in headers:
            reset_timestamp = int(headers['X-RateLimit-Reset'])
            self._rate_limit_reset = datetime.fromtimestamp(reset_timestamp)

    def _wait_for_rate_limit(self) -> None:
        """Wait if rate limit is low."""
        if self._rate_limit_remaining is not None and self._rate_limit_remaining < self.RATE_LIMIT_THRESHOLD:
            if self._rate_limit_reset:
                wait_time = (self._rate_limit_reset - datetime.now()).total_seconds()
                if wait_time > 0:
                    print(f"\n  Rate limit low ({self._rate_limit_remaining}). Waiting {wait_time:.0f}s...")
                    time.sleep(wait_time + 5)

    @exponential_backoff(
        max_retries=3,
        base_delay=2.0,
        exceptions=(requests.exceptions.RequestException,)
    )
    def fetch_contributors(
        self,
        repo_nwo: str,
        per_page: int = 100,
        max_contributors: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Fetch contributors for a single repository.

        Args:
            repo_nwo: Repository in "owner/name" format
            per_page: Results per page (max 100)
            max_contributors: Maximum contributors to fetch per repo

        Returns:
            List of contributor dictionaries with login, contributions count
        """
        self._wait_for_rate_limit()

        contributors = []
        page = 1
        max_pages = max_contributors // per_page + 1

        while page <= max_pages:
            url = f"{self.API_BASE}/repos/{repo_nwo}/contributors"
            params = {
                "per_page": min(per_page, 100),
                "page": page,
                "anon": "false",  # Only return users with GitHub accounts
            }

            response = self.session.get(url, params=params)
            self._update_rate_limit(response.headers)

            # Handle rate limiting
            if response.status_code == 403:
                retry_after = response.headers.get('Retry-After')
                if retry_after:
                    wait_time = int(retry_after)
                    print(f"\n  Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                elif 'secondary rate limit' in response.text.lower():
                    print("\n  Secondary rate limit hit. Waiting 60s...")
                    time.sleep(60)
                    continue

            # Handle 404 (repo not found, deleted, or private)
            if response.status_code == 404:
                return []

            # Handle other errors
            if response.status_code != 200:
                if response.status_code == 204:  # No content (empty repo)
                    return []
                response.raise_for_status()

            items = response.json()
            if not items:
                break

            for item in items:
                contributors.append({
                    "repo_nwo": repo_nwo,
                    "user_login": item.get("login", ""),
                    "contributions": item.get("contributions", 0),
                    "user_id": item.get("id"),
                    "user_type": item.get("type", "User"),
                })

            # Check if there are more pages
            if len(items) < per_page:
                break

            page += 1

        return contributors

    def fetch_all_repos(
        self,
        repos: List[str],
        output_path: Optional[Path] = None,
        resume_from: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Fetch contributors for all repositories.

        Args:
            repos: List of repository names in "owner/name" format
            output_path: Path to save results (parquet)
            resume_from: Path to existing parquet to resume from

        Returns:
            DataFrame with all contributors
        """
        # Track already processed repos if resuming
        processed_repos: Set[str] = set()

        if resume_from and resume_from.exists():
            print(f"Resuming from {resume_from}...")
            existing_df = pd.read_parquet(resume_from)
            self._contributors = existing_df.to_dict('records')
            processed_repos = set(existing_df['repo_nwo'].unique())
            print(f"  Loaded {len(self._contributors):,} existing contributors from {len(processed_repos):,} repos")

        # Filter out already processed repos
        repos_to_process = [r for r in repos if r not in processed_repos]

        print(f"\nFetching contributors for {len(repos_to_process):,} repositories...")
        print(f"  (Skipping {len(processed_repos):,} already processed)")

        pbar = tqdm(repos_to_process, desc="Fetching contributors", unit="repo")

        for repo_nwo in pbar:
            try:
                contributors = self.fetch_contributors(repo_nwo)
                self._contributors.extend(contributors)

                # Update progress bar
                remaining = self._rate_limit_remaining if self._rate_limit_remaining else "?"
                pbar.set_postfix({
                    'contributors': len(self._contributors),
                    'rate_limit': remaining
                })

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    # Repo not found - skip silently
                    pass
                else:
                    self._failed_repos.append(repo_nwo)
                    pbar.set_postfix({'error': f'{repo_nwo}: {e}'})
            except Exception as e:
                self._failed_repos.append(repo_nwo)
                pbar.set_postfix({'error': str(e)})

            # Periodic save
            if output_path and len(self._contributors) % self.SAVE_INTERVAL == 0:
                self._save_contributors(output_path)

        pbar.close()

        # Final save
        if output_path:
            self._save_contributors(output_path)

        # Summary
        print(f"\nFetched {len(self._contributors):,} contributor records")
        if self._failed_repos:
            print(f"  Failed repos: {len(self._failed_repos):,}")

        return self.to_dataframe()

    def _save_contributors(self, output_path: Path) -> None:
        """Save contributors to parquet file."""
        if not self._contributors:
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self._contributors)
        df.to_parquet(output_path, index=False)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert contributors to DataFrame."""
        if not self._contributors:
            return pd.DataFrame()
        return pd.DataFrame(self._contributors)

    def get_rate_limit_info(self) -> str:
        """Get formatted rate limit info string."""
        remaining = self._rate_limit_remaining if self._rate_limit_remaining is not None else '?'
        reset = self._rate_limit_reset.strftime('%H:%M:%S') if self._rate_limit_reset else '?'
        return f"Rate limit remaining: {remaining}, resets at: {reset}"


def get_unique_repos_from_datasets(data_dir: Path) -> List[str]:
    """
    Extract unique repository names from all AI tool datasets.

    Args:
        data_dir: Directory containing parquet files

    Returns:
        List of unique repository NWO strings
    """
    repos: Set[str] = set()

    # Files and their repo column names
    datasets = [
        ("claude_commits_full.parquet", "repo_nwo"),
        ("codex_prs.parquet", "repo_nwo"),
        ("copilot_prs.parquet", "repo_nwo"),
        ("jules_commits.parquet", "repo_nwo"),
    ]

    for filename, col in datasets:
        filepath = data_dir / filename
        if filepath.exists():
            try:
                df = pd.read_parquet(filepath)
                if col in df.columns:
                    file_repos = df[col].dropna().unique()
                    repos.update(file_repos)
                    print(f"  {filename}: {len(file_repos):,} repos")
            except Exception as e:
                print(f"  {filename}: Error - {e}")

    return list(repos)


if __name__ == "__main__":
    import os
    from pathlib import Path

    # Load token
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("Please set GITHUB_TOKEN environment variable")
        exit(1)

    data_dir = Path("data/output")

    # Get unique repos
    print("Finding unique repositories from datasets...")
    repos = get_unique_repos_from_datasets(data_dir)
    print(f"\nTotal unique repositories: {len(repos):,}")

    # Fetch contributors
    fetcher = ContributorFetcher(token)
    df = fetcher.fetch_all_repos(
        repos=repos,
        output_path=data_dir / "contributors.parquet",
        resume_from=data_dir / "contributors.parquet"
    )

    print(f"\nFinal dataset: {len(df):,} contributor records")
    print(f"Unique users: {df['user_login'].nunique():,}")
    print(f"Unique repos: {df['repo_nwo'].nunique():,}")
