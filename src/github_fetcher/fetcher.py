"""Main fetcher class for extracting GitHub repository data."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from datetime import date, timedelta

from .client import GitHubGraphQLClient
from .queries import (
    SEARCH_REPOS_QUERY,
    SINGLE_REPO_QUERY,
    USER_COUNT_QUERY,
    USER_SEARCH_QUERY,
    USER_REPOS_QUERY,
)
from .utils import build_search_query, extract_repo_data


class GitHubFetcher:
    """Fetcher for GitHub repository data using GraphQL API."""

    # Maximum repos per search request (reduced due to README content complexity)
    BATCH_SIZE = 10

    # Save progress every N repos
    SAVE_INTERVAL = 100

    # Minimum rate limit before waiting
    MIN_RATE_LIMIT = 100

    def __init__(self, token: str):
        """
        Initialize the fetcher.

        Args:
            token: GitHub Personal Access Token
        """
        self.client = GitHubGraphQLClient(token)
        self._repos: List[Dict[str, Any]] = []

    def search_repositories(
        self,
        query: str,
        max_repos: int = 1000,
        output_path: Optional[Path] = None,
        location_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for repositories using GitHub search query.

        Args:
            query: GitHub search query string
            max_repos: Maximum number of repos to fetch
            output_path: Optional path for incremental saves
            location_filter: Optional location to filter by (post-fetch filter on owner_location)

        Returns:
            List of repository data dictionaries
        """
        self._repos = []
        cursor = None
        total_matched = 0
        total_fetched = 0

        # Get initial count
        print(f"Searching: {query}")
        if location_filter:
            print(f"Location filter: {location_filter}")
        print(f"Target: {max_repos} repositories")
        print(f"{self.client.get_rate_limit_info()}")
        print()

        # Create progress bar
        pbar = tqdm(total=max_repos, desc="Fetching repos", unit="repo")

        while total_matched < max_repos:
            # Check rate limit before each batch
            self.client.wait_for_rate_limit(self.MIN_RATE_LIMIT)

            # Execute search query
            variables = {
                "query": query,
                "first": self.BATCH_SIZE,
                "after": cursor
            }

            try:
                data = self.client.execute(SEARCH_REPOS_QUERY, variables)
            except Exception as e:
                print(f"\nError during search: {e}")
                break

            search_data = data.get('search', {})
            nodes = search_data.get('nodes', [])
            page_info = search_data.get('pageInfo', {})

            if not nodes:
                print("\nNo more results found.")
                break

            # Process each repository
            for node in nodes:
                if node:  # Skip null nodes
                    repo_data = extract_repo_data(node)
                    total_fetched += 1

                    # Apply location filter if specified
                    if location_filter:
                        owner_location = (repo_data.get('owner_location') or '').lower()
                        if location_filter.lower() not in owner_location:
                            continue

                    self._repos.append(repo_data)
                    total_matched += 1
                    pbar.update(1)

                    if total_matched >= max_repos:
                        break

            # Save progress incrementally
            if output_path and total_matched % self.SAVE_INTERVAL == 0:
                self._save_progress(output_path)

            # Check for more pages
            if not page_info.get('hasNextPage', False):
                print("\nReached end of search results.")
                break

            cursor = page_info.get('endCursor')

            # Update progress bar description with rate limit
            remaining_requests = self.client.rate_limit.get('remaining', '?') if self.client.rate_limit else '?'
            if location_filter:
                pbar.set_postfix({'rate_limit': remaining_requests, 'scanned': total_fetched})
            else:
                pbar.set_postfix({'rate_limit': remaining_requests})

        pbar.close()

        # Final save
        if output_path:
            self._save_progress(output_path)

        if location_filter:
            print(f"\nFetched {len(self._repos)} repositories (scanned {total_fetched} total)")
        else:
            print(f"\nFetched {len(self._repos)} repositories")
        print(f"{self.client.get_rate_limit_info()}")

        return self._repos

    def fetch_repo_details(self, nwo: str) -> Optional[Dict[str, Any]]:
        """
        Fetch details for a single repository.

        Args:
            nwo: Repository in "owner/name" format

        Returns:
            Repository data dictionary or None if not found
        """
        parts = nwo.split('/')
        if len(parts) != 2:
            raise ValueError(f"Invalid nwo format: {nwo}. Expected 'owner/name'")

        owner, name = parts

        self.client.wait_for_rate_limit(self.MIN_RATE_LIMIT)

        try:
            data = self.client.execute(SINGLE_REPO_QUERY, {
                "owner": owner,
                "name": name
            })
            repo_node = data.get('repository')
            if repo_node:
                return extract_repo_data(repo_node)
        except Exception as e:
            print(f"Error fetching {nwo}: {e}")

        return None

    def fetch_by_location(
        self,
        location: str,
        min_stars: int = 5,
        language: Optional[str] = None,
        max_repos: int = 1000,
        output_path: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch repositories by owner location.

        Note: GitHub search doesn't support location filter directly, so this
        fetches repos broadly and filters by owner_location post-fetch.

        Args:
            location: Location to filter by (e.g., "Peru", "Brazil")
            min_stars: Minimum star count
            language: Optional language filter
            max_repos: Maximum repos to fetch
            output_path: Optional output path for saving

        Returns:
            List of repository data dictionaries
        """
        query = build_search_query(
            language=language,
            min_stars=min_stars
        )
        return self.search_repositories(
            query,
            max_repos,
            output_path,
            location_filter=location
        )

    def fetch_by_query(
        self,
        custom_query: str,
        min_stars: int = 5,
        max_repos: int = 1000,
        output_path: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch repositories using a custom search query.

        Args:
            custom_query: Custom GitHub search query
            min_stars: Minimum star count (added if not in query)
            max_repos: Maximum repos to fetch
            output_path: Optional output path for saving

        Returns:
            List of repository data dictionaries
        """
        query = build_search_query(custom_query=custom_query, min_stars=min_stars)
        return self.search_repositories(query, max_repos, output_path)

    def fetch_users_by_location(
        self,
        location: str,
        include_orgs: bool = True,
        max_users: Optional[int] = None
    ) -> List[str]:
        """
        Fetch all users/orgs from a location using date splitting to handle >1000 limit.

        Args:
            location: Location to search (e.g., "Peru")
            include_orgs: Whether to include organizations
            max_users: Maximum users to fetch (None = all)

        Returns:
            List of usernames/logins
        """
        all_users = []

        # Fetch users
        print(f"Fetching users from {location}...")
        user_query = f'location:"{location}" type:user'
        users = self._fetch_users_with_date_split(user_query, max_users)
        all_users.extend(users)
        print(f"  Found {len(users)} users")

        # Fetch organizations
        if include_orgs:
            print(f"Fetching organizations from {location}...")
            org_query = f'location:"{location}" type:org'
            orgs = self._fetch_users_with_date_split(
                org_query,
                max_users - len(all_users) if max_users else None
            )
            all_users.extend(orgs)
            print(f"  Found {len(orgs)} organizations")

        print(f"Total accounts: {len(all_users)}")
        return all_users

    def _fetch_users_with_date_split(
        self,
        base_query: str,
        max_users: Optional[int] = None
    ) -> List[str]:
        """
        Fetch users using date range splitting to overcome 1000 result limit.
        """
        users = []
        start_date = date(2008, 1, 1)  # GitHub's founding year
        end_date = date.today()

        # Queue of date ranges to process
        ranges = [(start_date, end_date)]

        with tqdm(desc="Searching users", unit="user") as pbar:
            while ranges and (max_users is None or len(users) < max_users):
                s, e = ranges.pop(0)
                created_q = f"created:{s.isoformat()}..{e.isoformat()}"
                query = f"{base_query} {created_q}"

                self.client.wait_for_rate_limit(self.MIN_RATE_LIMIT)

                # Get count for this range
                try:
                    data = self.client.execute(USER_COUNT_QUERY, {"query": query})
                    count = data.get('search', {}).get('userCount', 0)
                except Exception as ex:
                    print(f"\n  Error counting users: {ex}")
                    continue

                if count == 0:
                    continue

                if count <= 1000:
                    # Can fetch all users in this range
                    remaining = max_users - len(users) if max_users else count
                    new_users = self._paginated_user_search(query, min(count, remaining))
                    for u in new_users:
                        if u not in users:
                            users.append(u)
                            pbar.update(1)
                            if max_users and len(users) >= max_users:
                                break
                else:
                    # Split the date range in half
                    days = (e - s).days
                    if days <= 1:
                        # Can't split further, just get first 1000
                        new_users = self._paginated_user_search(query, 1000)
                        for u in new_users:
                            if u not in users:
                                users.append(u)
                                pbar.update(1)
                    else:
                        mid = s + timedelta(days=days // 2)
                        ranges.insert(0, (mid + timedelta(days=1), e))
                        ranges.insert(0, (s, mid))

        return users

    def _paginated_user_search(self, query: str, max_count: int) -> List[str]:
        """Fetch users with pagination."""
        users = []
        cursor = None

        while len(users) < max_count:
            self.client.wait_for_rate_limit(self.MIN_RATE_LIMIT)

            variables = {
                "query": query,
                "first": min(100, max_count - len(users)),
                "after": cursor
            }

            try:
                data = self.client.execute(USER_SEARCH_QUERY, variables)
            except Exception as ex:
                print(f"\n  Error in paginated search: {ex}")
                break

            search_data = data.get('search', {})
            nodes = search_data.get('nodes', [])
            page_info = search_data.get('pageInfo', {})

            for node in nodes:
                if node and node.get('login'):
                    users.append(node['login'])
                    if len(users) >= max_count:
                        break

            if not page_info.get('hasNextPage', False):
                break
            cursor = page_info.get('endCursor')

        return users

    def fetch_repos_for_users(
        self,
        usernames: List[str],
        output_path: Optional[Path] = None,
        min_stars: int = 1,
        include_forks: bool = False,
        extra_filter: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Fetch all repositories for a list of users/organizations.

        Args:
            usernames: List of GitHub usernames/org names
            output_path: Optional path for incremental saves
            min_stars: Minimum stars filter (default: 1)
            include_forks: Include forked repos (default: False)
            extra_filter: Additional search filters

        Returns:
            List of repository data dictionaries
        """
        self._repos = []

        print(f"Fetching repos for {len(usernames)} users/orgs...")
        print(f"{self.client.get_rate_limit_info()}")

        pbar = tqdm(usernames, desc="Users processed", unit="user")

        for username in pbar:
            self.client.wait_for_rate_limit(self.MIN_RATE_LIMIT)

            user_repos = self._fetch_user_repos(username, min_stars, include_forks, extra_filter)
            self._repos.extend(user_repos)

            pbar.set_postfix({
                'repos': len(self._repos),
                'rate': self.client.rate_limit.get('remaining', '?') if self.client.rate_limit else '?'
            })

            # Save progress periodically
            if output_path and len(self._repos) % self.SAVE_INTERVAL == 0:
                self._save_progress(output_path)

        pbar.close()

        # Final save
        if output_path:
            self._save_progress(output_path)

        print(f"\nFetched {len(self._repos)} total repositories")
        print(f"{self.client.get_rate_limit_info()}")

        return self._repos

    def _fetch_user_repos(
        self,
        username: str,
        min_stars: int = 1,
        include_forks: bool = False,
        extra_filter: str = ""
    ) -> List[Dict[str, Any]]:
        """Fetch repos for a single user using search API with filters (Option B)."""
        repos = []
        cursor = None

        # Build search query with user and filters
        query_parts = [f"user:{username}"]

        # Add stars filter
        if min_stars > 0:
            query_parts.append(f"stars:>={min_stars}")

        # Exclude forks by default
        if not include_forks:
            query_parts.append("fork:false")

        # Add any extra filters
        if extra_filter:
            query_parts.append(extra_filter)

        query = " ".join(query_parts)

        while True:
            variables = {
                "query": query,
                "first": 100,
                "after": cursor
            }

            try:
                data = self.client.execute(SEARCH_REPOS_QUERY, variables)
            except Exception as ex:
                # User might not exist or be inaccessible
                break

            search_data = data.get('search', {})
            nodes = search_data.get('nodes', [])
            page_info = search_data.get('pageInfo', {})

            if not nodes:
                break

            for node in nodes:
                if not node:
                    continue
                # Use existing extract_repo_data which handles SEARCH_REPOS_QUERY format
                repo = extract_repo_data(node)
                repos.append(repo)

            if not page_info.get('hasNextPage', False):
                break
            cursor = page_info.get('endCursor')

        return repos

    def _extract_repo_from_user_query(
        self,
        node: Dict[str, Any],
        owner_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract repo data from USER_REPOS_QUERY response."""
        # Extract languages
        languages_data = node.get('languages', {})
        languages = [l.get('name') for l in languages_data.get('nodes', []) if l] if languages_data else []

        # Extract topics
        topics_data = node.get('repositoryTopics', {})
        topics = [
            t.get('topic', {}).get('name')
            for t in topics_data.get('nodes', [])
            if t and t.get('topic')
        ] if topics_data else []

        # Extract README
        readme_obj = node.get('object')
        readme_content = readme_obj.get('text', '') if readme_obj else ''

        # Extract license
        license_info = node.get('licenseInfo', {}) or {}

        # Extract primary language
        primary_lang = node.get('primaryLanguage', {}) or {}

        return {
            'nwo': node.get('nameWithOwner', ''),
            'name': node.get('name', ''),
            'description': node.get('description', ''),
            'url': node.get('url', ''),
            'homepage_url': node.get('homepageUrl', ''),
            'created_at': node.get('createdAt', ''),
            'updated_at': node.get('updatedAt', ''),
            'pushed_at': node.get('pushedAt', ''),
            'stars': node.get('stargazerCount', 0),
            'forks': node.get('forkCount', 0),
            'watchers': node.get('watchers', {}).get('totalCount', 0) if node.get('watchers') else 0,
            'open_issues': node.get('issues', {}).get('totalCount', 0) if node.get('issues') else 0,
            'disk_usage_kb': node.get('diskUsage', 0),
            'primary_language': primary_lang.get('name', ''),
            'languages': languages,
            'topics': topics,
            'is_fork': node.get('isFork', False),
            'is_archived': node.get('isArchived', False),
            'is_private': node.get('isPrivate', False),
            'is_template': node.get('isTemplate', False),
            'has_wiki': node.get('hasWikiEnabled', False),
            'has_issues': node.get('hasIssuesEnabled', False),
            'license_key': license_info.get('key', ''),
            'license_name': license_info.get('name', ''),
            'readme_content': readme_content,
            **owner_info  # Add owner info
        }

    def fetch_by_location_two_step(
        self,
        location: str,
        include_orgs: bool = True,
        min_stars: int = 1,
        max_users: Optional[int] = None,
        include_forks: bool = False,
        extra_filter: str = "",
        output_path: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch all repos from a location using two-step approach:
        1. Find all users/orgs in the location
        2. Fetch repos for each user/org

        Args:
            location: Location to search (e.g., "Peru")
            include_orgs: Include organizations
            min_stars: Minimum stars filter (default: 1)
            max_users: Max users to fetch (None = all)
            include_forks: Include forked repos (default: False)
            extra_filter: Additional GitHub search filters (e.g., "language:Python")
            output_path: Path for saving results

        Returns:
            List of repository data dictionaries
        """
        # Step 1: Get all users
        print(f"=== Step 1: Finding users in {location} ===")
        usernames = self.fetch_users_by_location(location, include_orgs, max_users)

        if not usernames:
            print("No users found!")
            return []

        # Step 2: Fetch repos for each user
        print(f"\n=== Step 2: Fetching repos for {len(usernames)} users ===")
        print(f"Filters: stars>={min_stars}, forks={'included' if include_forks else 'excluded'}")
        if extra_filter:
            print(f"Extra filters: {extra_filter}")

        return self.fetch_repos_for_users(
            usernames,
            output_path,
            min_stars,
            include_forks,
            extra_filter
        )

    def _save_progress(self, output_path: Path) -> None:
        """Save current progress to file."""
        if not self._repos:
            return

        df = pd.DataFrame(self._repos)

        # Convert list columns to JSON strings for parquet compatibility
        for col in ['languages', 'topics']:
            if col in df.columns:
                df[col] = df[col].apply(json.dumps)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to parquet
        df.to_parquet(output_path, index=False)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert fetched repos to DataFrame.

        Returns:
            DataFrame with all repository data
        """
        if not self._repos:
            return pd.DataFrame()

        df = pd.DataFrame(self._repos)

        # Convert list columns to JSON strings
        for col in ['languages', 'topics']:
            if col in df.columns:
                df[col] = df[col].apply(json.dumps)

        return df

    def save_to_parquet(self, output_path: Path) -> None:
        """
        Save fetched repos to parquet file.

        Args:
            output_path: Path to output parquet file
        """
        df = self.to_dataframe()
        if df.empty:
            print("No data to save.")
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"Saved {len(df)} repositories to {output_path}")

    def save_to_csv(self, output_path: Path) -> None:
        """
        Save fetched repos to CSV file.

        Args:
            output_path: Path to output CSV file
        """
        df = self.to_dataframe()
        if df.empty:
            print("No data to save.")
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} repositories to {output_path}")
