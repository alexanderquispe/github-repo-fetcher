"""Fetcher for all repositories a user has contributed to."""

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import requests
from tqdm import tqdm

from .utils import exponential_backoff


class UserRepoFetcher:
    """Fetcher for all repositories a user owns or has contributed to.

    Uses GitHub REST API to fetch:
    1. User's own repositories (repos they own)
    2. User's contributed repositories (repos they've pushed to)

    API Endpoints:
    - GET /users/{username}/repos (owned repos)
    - GET /users/{username}/events (contribution events - to find other repos)
    """

    API_BASE = "https://api.github.com"

    # Save progress interval
    SAVE_INTERVAL = 100

    # Rate limit configuration
    RATE_LIMIT_THRESHOLD = 100

    def __init__(self, token: str):
        """
        Initialize the user repo fetcher.

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
        self._user_repos: Dict[str, List[str]] = {}  # user -> list of repo_nwo
        self._failed_users: List[str] = []

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
    def fetch_user_repos(
        self,
        username: str,
        include_forks: bool = True,
        per_page: int = 100,
        max_repos: int = 500
    ) -> List[str]:
        """
        Fetch all repositories for a user.

        Gets both owned repos and repos the user has contributed to.

        Args:
            username: GitHub username
            include_forks: Whether to include forked repositories
            per_page: Results per page (max 100)
            max_repos: Maximum repos to fetch per user

        Returns:
            List of repository NWO strings (owner/name)
        """
        repos: Set[str] = set()

        # 1. Fetch owned/member repos
        owned = self._fetch_owned_repos(username, include_forks, per_page, max_repos)
        repos.update(owned)

        # 2. Fetch repos from recent events (contributions)
        # This catches repos they've pushed to but don't own
        contributed = self._fetch_contributed_repos(username)
        repos.update(contributed)

        return list(repos)

    def _fetch_owned_repos(
        self,
        username: str,
        include_forks: bool,
        per_page: int,
        max_repos: int
    ) -> List[str]:
        """Fetch repos owned by or associated with user."""
        self._wait_for_rate_limit()

        repos = []
        page = 1
        max_pages = max_repos // per_page + 1

        while page <= max_pages:
            url = f"{self.API_BASE}/users/{username}/repos"
            params = {
                "per_page": min(per_page, 100),
                "page": page,
                "type": "all",  # owner, collaborator, organization_member
                "sort": "pushed",
                "direction": "desc",
            }

            response = self.session.get(url, params=params)
            self._update_rate_limit(response.headers)

            if response.status_code == 404:
                return []  # User not found

            if response.status_code == 403:
                self._handle_rate_limit(response)
                continue

            if response.status_code != 200:
                response.raise_for_status()

            items = response.json()
            if not items:
                break

            for repo in items:
                if not include_forks and repo.get("fork", False):
                    continue
                repos.append(repo["full_name"])

            if len(items) < per_page:
                break

            page += 1

        return repos

    def _fetch_contributed_repos(self, username: str, max_events: int = 300) -> List[str]:
        """Fetch repos user has contributed to via events API."""
        self._wait_for_rate_limit()

        repos = set()
        page = 1
        max_pages = 3  # Events API only returns last 90 days, 100 per page

        while page <= max_pages:
            url = f"{self.API_BASE}/users/{username}/events/public"
            params = {"per_page": 100, "page": page}

            response = self.session.get(url, params=params)
            self._update_rate_limit(response.headers)

            if response.status_code != 200:
                break

            events = response.json()
            if not events:
                break

            for event in events:
                repo = event.get("repo", {})
                repo_name = repo.get("name")
                if repo_name:
                    repos.add(repo_name)

            if len(events) < 100:
                break

            page += 1

        return list(repos)

    def _handle_rate_limit(self, response: requests.Response) -> None:
        """Handle rate limit response."""
        retry_after = response.headers.get('Retry-After')
        if retry_after:
            wait_time = int(retry_after)
            print(f"\n  Rate limited. Waiting {wait_time}s...")
            time.sleep(wait_time)
        elif 'secondary rate limit' in response.text.lower():
            print("\n  Secondary rate limit hit. Waiting 60s...")
            time.sleep(60)
        else:
            time.sleep(60)

    def fetch_repos_for_users(
        self,
        users: List[str],
        output_path: Optional[Path] = None,
        resume_from: Optional[Path] = None
    ) -> Dict[str, List[str]]:
        """
        Fetch repos for all specified users.

        Args:
            users: List of GitHub usernames
            output_path: Path to save results (parquet)
            resume_from: Path to existing data to resume from

        Returns:
            Dict mapping username -> list of repo NWOs
        """
        # Resume from existing data if available
        processed_users: Set[str] = set()

        if resume_from and resume_from.exists():
            print(f"Resuming from {resume_from}...")
            existing_df = pd.read_parquet(resume_from)
            for _, row in existing_df.iterrows():
                user = row["user_login"]
                repos = row["repos"] if isinstance(row["repos"], list) else eval(row["repos"])
                self._user_repos[user] = repos
                processed_users.add(user)
            print(f"  Loaded {len(processed_users):,} users")

        # Filter out processed users
        users_to_process = [u for u in users if u not in processed_users]

        print(f"\nFetching repos for {len(users_to_process):,} users...")
        print(f"  (Skipping {len(processed_users):,} already processed)")

        pbar = tqdm(users_to_process, desc="Fetching user repos", unit="user")

        for username in pbar:
            try:
                repos = self.fetch_user_repos(username)
                self._user_repos[username] = repos

                remaining = self._rate_limit_remaining if self._rate_limit_remaining else "?"
                pbar.set_postfix({
                    'repos': len(repos),
                    'rate_limit': remaining
                })

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    self._user_repos[username] = []
                else:
                    self._failed_users.append(username)
            except Exception as e:
                self._failed_users.append(username)

            # Periodic save
            if output_path and len(self._user_repos) % self.SAVE_INTERVAL == 0:
                self._save_results(output_path)

        pbar.close()

        # Final save
        if output_path:
            self._save_results(output_path)

        print(f"\nFetched repos for {len(self._user_repos):,} users")
        if self._failed_users:
            print(f"  Failed: {len(self._failed_users):,}")

        return self._user_repos

    def _save_results(self, output_path: Path) -> None:
        """Save user repos to parquet file."""
        if not self._user_repos:
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)

        records = [
            {"user_login": user, "repos": repos, "repo_count": len(repos)}
            for user, repos in self._user_repos.items()
        ]

        df = pd.DataFrame(records)
        df.to_parquet(output_path, index=False)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        records = [
            {"user_login": user, "repos": repos, "repo_count": len(repos)}
            for user, repos in self._user_repos.items()
        ]
        return pd.DataFrame(records)

    def get_all_repos(self) -> Set[str]:
        """Get set of all unique repos across all users."""
        all_repos = set()
        for repos in self._user_repos.values():
            all_repos.update(repos)
        return all_repos

    def get_rate_limit_info(self) -> str:
        """Get formatted rate limit info string."""
        remaining = self._rate_limit_remaining if self._rate_limit_remaining is not None else '?'
        reset = self._rate_limit_reset.strftime('%H:%M:%S') if self._rate_limit_reset else '?'
        return f"Rate limit remaining: {remaining}, resets at: {reset}"
