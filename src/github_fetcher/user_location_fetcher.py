"""Fetcher for GitHub user location/country information."""

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import requests
from tqdm import tqdm

from .utils import exponential_backoff


class UserLocationFetcher:
    """Fetcher for GitHub user profile location data.

    Uses GitHub REST API to fetch user profiles which include location info.

    API: GET /users/{username}
    Rate Limit: 5000 requests/hour (authenticated)
    """

    API_BASE = "https://api.github.com"

    # Save progress interval
    SAVE_INTERVAL = 100

    # Rate limit configuration
    RATE_LIMIT_THRESHOLD = 100  # Wait when remaining < this

    def __init__(self, token: str):
        """
        Initialize the user location fetcher.

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
        self._user_locations: List[Dict[str, Any]] = []
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
    def fetch_user_location(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Fetch location info for a single user.

        Args:
            username: GitHub username

        Returns:
            Dictionary with user location data, or None if user not found
        """
        self._wait_for_rate_limit()

        url = f"{self.API_BASE}/users/{username}"
        response = self.session.get(url)
        self._update_rate_limit(response.headers)

        # Handle rate limiting
        if response.status_code == 403:
            retry_after = response.headers.get('Retry-After')
            if retry_after:
                wait_time = int(retry_after)
                print(f"\n  Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
                return self.fetch_user_location(username)
            elif 'secondary rate limit' in response.text.lower():
                print("\n  Secondary rate limit hit. Waiting 60s...")
                time.sleep(60)
                return self.fetch_user_location(username)

        # Handle 404 (user not found)
        if response.status_code == 404:
            return None

        # Handle other errors
        if response.status_code != 200:
            response.raise_for_status()

        data = response.json()

        return {
            "username": username,
            "user_id": data.get("id") or "",
            "type": data.get("type") or "",
            "location": data.get("location") or "",
            "company": data.get("company") or "",
            "bio": data.get("bio") or "",
            "name": data.get("name") or "",
            "email": data.get("email") or "",
            "twitter_username": data.get("twitter_username") or "",
            "blog": data.get("blog") or "",
            "hireable": data.get("hireable") or False,
            "public_repos": data.get("public_repos", 0),
            "public_gists": data.get("public_gists", 0),
            "followers": data.get("followers", 0),
            "following": data.get("following", 0),
            "created_at": data.get("created_at") or "",
            "updated_at": data.get("updated_at") or "",
            "html_url": data.get("html_url") or "",
        }

    def fetch_all_users(
        self,
        users: List[str],
        output_path: Optional[Path] = None,
        resume_from: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Fetch location data for all users.

        Args:
            users: List of GitHub usernames
            output_path: Path to save results (parquet)
            resume_from: Path to existing parquet to resume from

        Returns:
            DataFrame with all user location data
        """
        # Track already processed users if resuming
        processed_users: Set[str] = set()

        if resume_from and resume_from.exists():
            print(f"Resuming from {resume_from}...")
            existing_df = pd.read_parquet(resume_from)
            self._user_locations = existing_df.to_dict('records')
            processed_users = set(existing_df['username'].unique())
            print(f"  Loaded {len(self._user_locations):,} existing users")

        # Filter out already processed users
        users_to_process = [u for u in users if u not in processed_users]

        print(f"\nFetching locations for {len(users_to_process):,} users...")
        print(f"  (Skipping {len(processed_users):,} already processed)")

        pbar = tqdm(users_to_process, desc="Fetching user locations", unit="user")

        for username in pbar:
            try:
                user_data = self.fetch_user_location(username)
                if user_data:
                    self._user_locations.append(user_data)
                else:
                    # User not found - record with empty location
                    self._user_locations.append({
                        "username": username,
                        "user_id": 0,
                        "type": "NotFound",
                        "location": "",
                        "company": "",
                        "bio": "",
                        "name": "",
                        "email": "",
                        "twitter_username": "",
                        "blog": "",
                        "hireable": False,
                        "public_repos": 0,
                        "public_gists": 0,
                        "followers": 0,
                        "following": 0,
                        "created_at": "",
                        "updated_at": "",
                        "html_url": "",
                    })

                # Update progress bar
                remaining = self._rate_limit_remaining if self._rate_limit_remaining else "?"
                pbar.set_postfix({
                    'fetched': len(self._user_locations),
                    'rate_limit': remaining
                })

            except requests.exceptions.HTTPError as e:
                self._failed_users.append(username)
                pbar.set_postfix({'error': f'{username}: {e}'})
            except Exception as e:
                self._failed_users.append(username)
                pbar.set_postfix({'error': str(e)})

            # Periodic save
            if output_path and len(self._user_locations) % self.SAVE_INTERVAL == 0:
                self._save_results(output_path)

        pbar.close()

        # Final save
        if output_path:
            self._save_results(output_path)

        # Summary
        print(f"\nFetched {len(self._user_locations):,} user records")
        if self._failed_users:
            print(f"  Failed users: {len(self._failed_users):,}")

        return self.to_dataframe()

    def _save_results(self, output_path: Path) -> None:
        """Save user locations to parquet file."""
        if not self._user_locations:
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self._user_locations)
        df.to_parquet(output_path, index=False)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        if not self._user_locations:
            return pd.DataFrame()
        return pd.DataFrame(self._user_locations)

    def get_rate_limit_info(self) -> str:
        """Get formatted rate limit info string."""
        remaining = self._rate_limit_remaining if self._rate_limit_remaining is not None else '?'
        reset = self._rate_limit_reset.strftime('%H:%M:%S') if self._rate_limit_reset else '?'
        return f"Rate limit remaining: {remaining}, resets at: {reset}"


def get_unique_users_from_csv(csv_path: Path, username_column: str = "repo_owner") -> List[str]:
    """
    Extract unique usernames from a CSV file.

    Args:
        csv_path: Path to CSV file
        username_column: Column name containing usernames

    Returns:
        List of unique usernames
    """
    df = pd.read_csv(csv_path)
    users = df[username_column].dropna().unique().tolist()
    print(f"Found {len(users):,} unique users in {csv_path.name}")
    return users
