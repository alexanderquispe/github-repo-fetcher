"""Fetcher for finding Claude Code co-authored commits and repositories."""

import json
import time
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
from tqdm import tqdm

from .client import GitHubGraphQLClient
from .rest_client import GitHubRESTClient, extract_commit_data, extract_claude_model
from .queries import SEARCH_REPOS_QUERY
from .utils import extract_repo_data


class ClaudeCommitFetcher:
    """Fetcher for finding Claude Code co-authored commits across GitHub.

    This uses a two-phase approach:
    1. REST API commit search to find all co-authored commits
    2. GraphQL API to enrich repository metadata

    The search query finds commits with:
        "Co-Authored-By" "noreply@anthropic.com"

    This catches all Claude Code variants:
    - Co-Authored-By: Claude <noreply@anthropic.com>
    - Co-Authored-By: Claude Sonnet <noreply@anthropic.com>
    - Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
    """

    # Base search query for Claude co-authored commits
    CLAUDE_SEARCH_QUERY = '"Co-Authored-By" "noreply@anthropic.com"'

    # Save progress interval
    SAVE_INTERVAL = 100

    # Minimum GraphQL rate limit before waiting
    MIN_GRAPHQL_RATE_LIMIT = 100

    def __init__(self, token: str):
        """
        Initialize the Claude commit fetcher.

        Args:
            token: GitHub Personal Access Token
        """
        self.rest_client = GitHubRESTClient(token)
        self.graphql_client = GitHubGraphQLClient(token)
        self._commits: List[Dict[str, Any]] = []
        self._repos: List[Dict[str, Any]] = []

    def fetch_claude_commits(
        self,
        start_date: date = date(2025, 1, 1),
        end_date: Optional[date] = None,
        output_path: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch all Claude co-authored commits using REST API commit search.

        Uses date-splitting to overcome the 1000 result limit per query.

        Args:
            start_date: Start of date range (default: Feb 2025, Claude Code release)
            end_date: End of date range (default: today)
            output_path: Optional path for incremental saves

        Returns:
            List of commit data dictionaries
        """
        if end_date is None:
            end_date = date.today()

        self._commits = []

        print(f"Searching for Claude co-authored commits...")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Query: {self.CLAUDE_SEARCH_QUERY}")
        print()

        # Get initial estimate
        initial_query = f"{self.CLAUDE_SEARCH_QUERY} committer-date:{start_date}..{end_date}"
        try:
            total_estimate = self.rest_client.get_search_count(initial_query)
            print(f"Estimated total commits: {total_estimate:,}")
            if total_estimate == 0:
                print("No commits found.")
                return []
        except Exception as e:
            print(f"Error getting count: {e}")
            total_estimate = None

        print(f"{self.rest_client.get_rate_limit_info()}")
        print()

        # Fetch commits using date splitting (with periodic saving)
        commits = self._fetch_commits_with_date_split(start_date, end_date, output_path)
        self._commits = commits

        # Save final results
        if output_path:
            self._save_commits(output_path)
            print(f"  Final save: {len(commits):,} commits to {output_path}")

        print(f"\nFetched {len(self._commits):,} unique commits")
        print(f"{self.rest_client.get_rate_limit_info()}")

        return self._commits

    def _fetch_commits_with_date_split(
        self,
        start_date: date,
        end_date: date,
        output_path: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch commits using date range splitting to overcome 1000 result limit.

        Uses recursive time-splitting strategy:
        1. Split by days until single day
        2. Split single day by 6-hour intervals
        3. Split 6-hour intervals by 1-hour intervals
        4. Split hours by 10-minute intervals if needed

        This ensures 100% coverage even for very busy periods.
        Saves progress every 1000 commits to prevent data loss.

        Args:
            start_date: Start date
            end_date: End date
            output_path: Optional path for periodic saves

        Returns:
            List of commit dictionaries
        """
        from datetime import datetime as dt

        commits = []
        seen_shas: Set[str] = set()
        last_save_count = 0
        SAVE_EVERY = 1000  # Save every 1000 commits

        # Queue of ranges to process: (start_datetime, end_datetime, is_time_range)
        # is_time_range=False means date-only, is_time_range=True means datetime with hours/minutes
        start_dt = dt.combine(start_date, dt.min.time())
        end_dt = dt.combine(end_date, dt.max.time().replace(microsecond=0))
        ranges = [(start_dt, end_dt, False)]

        with tqdm(desc="Fetching commits", unit="commit") as pbar:
            while ranges:
                s, e, is_time_range = ranges.pop(0)

                # Build query with date/time range
                if is_time_range:
                    # Use full ISO datetime format for time-based queries
                    date_filter = f"committer-date:{s.isoformat()}..{e.isoformat()}"
                else:
                    # Use date-only format
                    date_filter = f"committer-date:{s.date()}..{e.date()}"

                query = f"{self.CLAUDE_SEARCH_QUERY} {date_filter}"

                # Get count for this range (with retry)
                count = None
                for retry in range(3):
                    try:
                        count = self.rest_client.get_search_count(query)
                        break
                    except Exception as ex:
                        if retry < 2:
                            print(f"\n  Error counting commits for {s}..{e}: {ex}. Retrying...")
                            time.sleep(5 * (retry + 1))
                        else:
                            print(f"\n  Error counting commits for {s}..{e}: {ex}. Adding back to queue.")
                            # Add back to queue to retry later
                            ranges.append((s, e, is_time_range))
                            count = None

                if count is None:
                    continue

                if count == 0:
                    continue

                if count <= 1000:
                    # Can fetch all commits in this range
                    if is_time_range:
                        pbar.set_description(f"Fetching {s.date()} {s.hour:02d}:{s.minute:02d} ({count:,})")
                    else:
                        pbar.set_description(f"Fetching {s.date()} to {e.date()} ({count:,})")

                    new_commits = self._paginated_commit_search(query)

                    for commit in new_commits:
                        sha = commit['sha']
                        if sha not in seen_shas:
                            seen_shas.add(sha)
                            commits.append(commit)
                            pbar.update(1)

                    # Periodic save to prevent data loss
                    if output_path and len(commits) - last_save_count >= SAVE_EVERY:
                        self._commits = commits
                        self._save_commits(output_path)
                        last_save_count = len(commits)
                        pbar.set_postfix({'saved': f'{len(commits):,}'})
                else:
                    # Need to split the range
                    if not is_time_range:
                        # Still in date-only mode
                        days = (e.date() - s.date()).days

                        if days > 0:
                            # Split days in half
                            mid_date = s.date() + timedelta(days=days // 2)
                            mid_dt = dt.combine(mid_date, dt.max.time().replace(microsecond=0))
                            next_dt = dt.combine(mid_date + timedelta(days=1), dt.min.time())
                            ranges.insert(0, (next_dt, e, False))
                            ranges.insert(0, (s, mid_dt, False))
                        else:
                            # Single day with >1000 commits - switch to time-based splitting
                            # Split into 4 x 6-hour intervals
                            day_start = dt.combine(s.date(), dt.min.time())
                            for hour_offset in [18, 12, 6, 0]:  # Add in reverse for LIFO
                                interval_start = day_start.replace(hour=hour_offset)
                                if hour_offset == 18:
                                    interval_end = day_start.replace(hour=23, minute=59, second=59)
                                else:
                                    interval_end = day_start.replace(hour=hour_offset + 5, minute=59, second=59)
                                ranges.insert(0, (interval_start, interval_end, True))
                    else:
                        # Already in time-based mode - split further
                        total_seconds = (e - s).total_seconds()

                        if total_seconds > 3600:  # More than 1 hour - split in half
                            mid = s + timedelta(seconds=total_seconds / 2)
                            # Round to nearest minute
                            mid = mid.replace(second=0, microsecond=0)
                            ranges.insert(0, (mid, e, True))
                            ranges.insert(0, (s, mid - timedelta(seconds=1), True))
                        elif total_seconds > 600:  # More than 10 minutes - split into 10-min chunks
                            # Split into ~6 intervals
                            interval_mins = max(1, int(total_seconds / 60 / 6))
                            intervals = []
                            current = s
                            while current < e:
                                interval_end = min(current + timedelta(minutes=interval_mins) - timedelta(seconds=1), e)
                                intervals.append((current, interval_end, True))
                                current = interval_end + timedelta(seconds=1)
                            for interval in reversed(intervals):
                                ranges.insert(0, interval)
                        elif total_seconds > 60:  # More than 1 minute - split by minute
                            # Split into 1-minute intervals
                            intervals = []
                            current = s
                            while current < e:
                                interval_end = min(current + timedelta(minutes=1) - timedelta(seconds=1), e)
                                intervals.append((current, interval_end, True))
                                current = interval_end + timedelta(seconds=1)
                            for interval in reversed(intervals):
                                ranges.insert(0, interval)
                        else:
                            # Less than 1 minute with >1000 commits - extremely rare
                            # Split into 10-second intervals as last resort
                            if total_seconds > 10:
                                intervals = []
                                current = s
                                while current < e:
                                    interval_end = min(current + timedelta(seconds=10) - timedelta(seconds=1), e)
                                    intervals.append((current, interval_end, True))
                                    current = interval_end + timedelta(seconds=1)
                                for interval in reversed(intervals):
                                    ranges.insert(0, interval)
                            else:
                                # Absolute minimum interval - fetch what we can (very rare edge case)
                                pbar.set_description(f"Fetching {s.date()} {s.hour:02d}:{s.minute:02d}:{s.second:02d} (>1000, max)")
                                new_commits = self._paginated_commit_search(query, max_pages=10)

                                for commit in new_commits:
                                    sha = commit['sha']
                                    if sha not in seen_shas:
                                        seen_shas.add(sha)
                                        commits.append(commit)
                                        pbar.update(1)

                                # Periodic save
                                if output_path and len(commits) - last_save_count >= SAVE_EVERY:
                                    self._commits = commits
                                    self._save_commits(output_path)
                                    last_save_count = len(commits)

        return commits

    def _paginated_commit_search(
        self,
        query: str,
        max_pages: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fetch commits with pagination (up to max_pages * 100 results).

        Args:
            query: Search query
            max_pages: Maximum pages to fetch (default 10 = 1000 results)

        Returns:
            List of commit dictionaries
        """
        commits = []
        page = 1

        while page <= max_pages:
            try:
                result = self.rest_client.search_commits(
                    query=query,
                    per_page=100,
                    page=page
                )
            except Exception as ex:
                print(f"\n  Error in paginated search (page {page}): {ex}")
                break

            items = result.get('items', [])
            if not items:
                break

            for item in items:
                commit_data = extract_commit_data(item)
                commits.append(commit_data)

            # Check if there are more results
            total_count = result.get('total_count', 0)
            if page * 100 >= total_count:
                break

            page += 1

        return commits

    def _save_commits(self, output_path: Path) -> None:
        """Save commits to file (JSON lines for temp, parquet for final)."""
        if not self._commits:
            return

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use JSON lines for incremental saves (survives interruptions)
        # Convert to parquet only at the end
        jsonl_path = output_path.with_suffix('.jsonl')

        # Write as JSON lines (append-safe format)
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for commit in self._commits:
                f.write(json.dumps(commit, ensure_ascii=False) + '\n')

        # Also save as parquet (but this may be corrupted on interrupt)
        try:
            df = pd.DataFrame(self._commits)
            df.to_parquet(output_path, index=False)
        except Exception as e:
            print(f"  Warning: Could not save parquet: {e}")

    def enrich_repositories(
        self,
        commits_file: Optional[Path] = None,
        output_path: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """
        Enrich repository data from commit data.

        Phase 2: For each unique repo found in commits, fetch full metadata
        and calculate Claude-specific metrics.

        Args:
            commits_file: Path to parquet file with commit data (from Phase 1)
            output_path: Path to save enriched repo data

        Returns:
            List of enriched repository dictionaries
        """
        # Load commits from file or use in-memory data
        if commits_file:
            print(f"Loading commits from {commits_file}...")
            df = pd.read_parquet(commits_file)
            commits = df.to_dict('records')
        else:
            commits = self._commits

        if not commits:
            print("No commits to process.")
            return []

        # Group commits by repository
        repo_commits: Dict[str, List[Dict]] = defaultdict(list)
        for commit in commits:
            nwo = commit.get('repo_nwo', '')
            if nwo:
                repo_commits[nwo].append(commit)

        unique_repos = list(repo_commits.keys())
        print(f"\nEnriching {len(unique_repos):,} unique repositories...")
        print(f"{self.graphql_client.get_rate_limit_info()}")
        print()

        self._repos = []

        pbar = tqdm(unique_repos, desc="Enriching repos", unit="repo")

        for nwo in pbar:
            # Check rate limit
            self.graphql_client.wait_for_rate_limit(self.MIN_GRAPHQL_RATE_LIMIT)

            # Get repository metadata
            repo_data = self._fetch_repo_metadata(nwo)

            if repo_data:
                # Add Claude-specific metrics
                repo_commits_list = repo_commits[nwo]
                claude_metrics = self._calculate_claude_metrics(repo_commits_list)
                repo_data.update(claude_metrics)

                self._repos.append(repo_data)

            # Update progress bar
            remaining = self.graphql_client.rate_limit.get('remaining', '?') if self.graphql_client.rate_limit else '?'
            pbar.set_postfix({'rate_limit': remaining})

            # Save progress periodically
            if output_path and len(self._repos) % self.SAVE_INTERVAL == 0:
                self._save_repos(output_path)

        pbar.close()

        # Final save
        if output_path:
            self._save_repos(output_path)

        print(f"\nEnriched {len(self._repos):,} repositories")
        print(f"{self.graphql_client.get_rate_limit_info()}")

        return self._repos

    def _fetch_repo_metadata(self, nwo: str) -> Optional[Dict[str, Any]]:
        """
        Fetch repository metadata using GraphQL API.

        Args:
            nwo: Repository in "owner/name" format

        Returns:
            Repository data dictionary or None if not found
        """
        parts = nwo.split('/')
        if len(parts) != 2:
            return None

        owner, name = parts

        try:
            # Use the search query to get full repo details
            query = f"repo:{nwo}"
            variables = {
                "query": query,
                "first": 1,
                "after": None
            }

            data = self.graphql_client.execute(SEARCH_REPOS_QUERY, variables)
            nodes = data.get('search', {}).get('nodes', [])

            if nodes and nodes[0]:
                return extract_repo_data(nodes[0])

        except Exception as e:
            # Repository might be deleted, private, or inaccessible
            pass

        return None

    def _calculate_claude_metrics(
        self,
        commits: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate Claude-specific metrics from commit data.

        Args:
            commits: List of commit dictionaries for a single repo

        Returns:
            Dictionary with Claude metrics
        """
        if not commits:
            return {}

        # Sort by date
        sorted_commits = sorted(
            commits,
            key=lambda c: c.get('committed_date', '')
        )

        # Get unique collaborators (authors who worked with Claude)
        collaborators = set()
        for c in commits:
            author = c.get('author_login') or c.get('author_name')
            if author:
                collaborators.add(author)

        # Get unique Claude models used
        models = set()
        for c in commits:
            model = c.get('claude_model', '')
            if model:
                models.add(model)

        first_commit = sorted_commits[0] if sorted_commits else {}
        last_commit = sorted_commits[-1] if sorted_commits else {}

        return {
            'claude_commit_count': len(commits),
            'first_claude_commit': first_commit.get('committed_date', ''),
            'last_claude_commit': last_commit.get('committed_date', ''),
            'claude_collaborators': json.dumps(list(collaborators)),
            'claude_models_used': json.dumps(list(models)),
        }

    def _save_repos(self, output_path: Path) -> None:
        """Save repositories to parquet file."""
        if not self._repos:
            return

        df = pd.DataFrame(self._repos)

        # Convert list columns to JSON strings for parquet compatibility
        for col in ['languages', 'topics']:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, list) else x
                )

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(output_path, index=False)

    def fetch_all(
        self,
        start_date: date = date(2025, 1, 1),
        end_date: Optional[date] = None,
        commits_output: Optional[Path] = None,
        repos_output: Optional[Path] = None
    ) -> tuple:
        """
        Run both phases: fetch commits and enrich repositories.

        Args:
            start_date: Start of date range
            end_date: End of date range (default: today)
            commits_output: Path to save commit data
            repos_output: Path to save repo data

        Returns:
            Tuple of (commits list, repos list)
        """
        # Phase 1: Fetch commits
        print("=" * 60)
        print("PHASE 1: Fetching Claude co-authored commits")
        print("=" * 60)
        commits = self.fetch_claude_commits(start_date, end_date, commits_output)

        if not commits:
            return commits, []

        # Phase 2: Enrich repositories
        print()
        print("=" * 60)
        print("PHASE 2: Enriching repository metadata")
        print("=" * 60)
        repos = self.enrich_repositories(output_path=repos_output)

        return commits, repos

    def to_commits_dataframe(self) -> pd.DataFrame:
        """Convert fetched commits to DataFrame."""
        if not self._commits:
            return pd.DataFrame()
        return pd.DataFrame(self._commits)

    def to_repos_dataframe(self) -> pd.DataFrame:
        """Convert enriched repos to DataFrame."""
        if not self._repos:
            return pd.DataFrame()

        df = pd.DataFrame(self._repos)

        # Convert list columns to JSON strings
        for col in ['languages', 'topics']:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, list) else x
                )

        return df
