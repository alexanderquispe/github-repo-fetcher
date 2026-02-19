"""Fetcher for finding GitHub Copilot PRs on GitHub."""

import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import requests
from tqdm import tqdm

from .utils import exponential_backoff


class CopilotPRFetcher:
    """Fetcher for finding GitHub Copilot PRs across GitHub.

    GitHub Copilot creates branches named `copilot/...` when making PRs.
    GitHub's search API doesn't support prefix matching (`head:copilot/`),
    so we fetch all PRs with `head:copilot` and filter locally.

    The strategy:
    1. Use REST API issue search with `is:pr head:copilot`
    2. Date-split to overcome 1000 result limit
    3. Filter locally for branches starting with `copilot/`
    """

    # Base search query for Copilot PRs
    SEARCH_QUERY = "is:pr head:copilot"

    # API endpoint for issue/PR search
    SEARCH_URL = "https://api.github.com/search/issues"

    # Search API rate limit: 30 requests per minute
    SEARCH_RATE_LIMIT = 30
    SEARCH_RATE_WINDOW = 60  # seconds

    # Save progress interval
    SAVE_EVERY = 1000

    def __init__(self, token: str):
        """
        Initialize the Copilot PR fetcher.

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

        # Track search rate limit
        self._search_requests: List[float] = []
        self._rate_limit_remaining: Optional[int] = None
        self._rate_limit_reset: Optional[datetime] = None

        # Results
        self._prs: List[Dict[str, Any]] = []

    def _update_rate_limit_from_headers(self, headers: Dict[str, str]) -> None:
        """Update rate limit tracking from response headers."""
        if 'X-RateLimit-Remaining' in headers:
            self._rate_limit_remaining = int(headers['X-RateLimit-Remaining'])
        if 'X-RateLimit-Reset' in headers:
            reset_timestamp = int(headers['X-RateLimit-Reset'])
            self._rate_limit_reset = datetime.fromtimestamp(reset_timestamp)

    def wait_for_search_rate_limit(self) -> None:
        """Wait if we've exceeded the search rate limit (30 req/min)."""
        now = time.time()
        window_start = now - self.SEARCH_RATE_WINDOW

        # Remove old requests outside the window
        self._search_requests = [t for t in self._search_requests if t > window_start]

        # Check if we need to wait
        if len(self._search_requests) >= self.SEARCH_RATE_LIMIT:
            oldest = min(self._search_requests)
            wait_time = oldest + self.SEARCH_RATE_WINDOW - now + 1
            if wait_time > 0:
                print(f"  Search rate limit reached ({self.SEARCH_RATE_LIMIT}/min). Waiting {wait_time:.0f}s...")
                time.sleep(wait_time)

        # Also check GitHub's rate limit headers
        if self._rate_limit_remaining is not None and self._rate_limit_remaining < 5:
            if self._rate_limit_reset:
                now_dt = datetime.now()
                wait_time = (self._rate_limit_reset - now_dt).total_seconds()
                if wait_time > 0:
                    print(f"  GitHub rate limit low ({self._rate_limit_remaining}). Waiting {wait_time:.0f}s...")
                    time.sleep(wait_time + 5)

    def _record_search_request(self) -> None:
        """Record a search request for rate limiting."""
        self._search_requests.append(time.time())

    def get_rate_limit_info(self) -> str:
        """Get formatted rate limit info string."""
        recent_requests = len([t for t in self._search_requests
                              if t > time.time() - self.SEARCH_RATE_WINDOW])
        remaining = self._rate_limit_remaining if self._rate_limit_remaining is not None else '?'
        return f"Search rate: {recent_requests}/{self.SEARCH_RATE_LIMIT}/min, API remaining: {remaining}"

    @exponential_backoff(
        max_retries=3,
        base_delay=2.0,
        exceptions=(requests.exceptions.RequestException,)
    )
    def search_prs(
        self,
        query: str,
        sort: str = "created",
        order: str = "desc",
        per_page: int = 100,
        page: int = 1
    ) -> Dict[str, Any]:
        """
        Search PRs using the GitHub REST API.

        Args:
            query: Search query string
            sort: Sort field (created, updated, comments)
            order: Sort order (asc, desc)
            per_page: Results per page (max 100)
            page: Page number (1-indexed)

        Returns:
            API response with total_count, incomplete_results, and items
        """
        self.wait_for_search_rate_limit()

        params = {
            "q": query,
            "sort": sort,
            "order": order,
            "per_page": min(per_page, 100),
            "page": page,
        }

        self._record_search_request()
        response = self.session.get(self.SEARCH_URL, params=params)
        self._update_rate_limit_from_headers(response.headers)

        # Handle rate limiting response
        if response.status_code == 403:
            retry_after = response.headers.get('Retry-After')
            if retry_after:
                wait_time = int(retry_after)
                print(f"  Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
                return self.search_prs(query, sort, order, per_page, page)
            if 'secondary rate limit' in response.text.lower():
                print("  Secondary rate limit hit. Waiting 60s...")
                time.sleep(60)
                return self.search_prs(query, sort, order, per_page, page)

        response.raise_for_status()
        return response.json()

    def get_search_count(self, query: str) -> int:
        """Get the total count of results for a search query."""
        result = self.search_prs(query, per_page=1, page=1)
        return result.get('total_count', 0)

    def fetch_copilot_prs(
        self,
        start_date: date = date(2025, 5, 1),
        end_date: Optional[date] = None,
        output_path: Optional[Path] = None,
        filter_prefix: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Fetch all PRs with 'copilot' in branch name.

        Uses date-splitting to overcome 1000 result limit.
        Optionally filters locally for branches starting with 'copilot/'.

        Args:
            start_date: Start of date range (default: May 2025, Copilot launch)
            end_date: End of date range (default: today)
            output_path: Optional path for incremental saves
            filter_prefix: If True, filter for 'copilot/' prefix (default: True)

        Returns:
            List of PR data dictionaries
        """
        if end_date is None:
            end_date = date.today()

        self._prs = []

        print(f"Searching for Copilot PRs...")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Query: {self.SEARCH_QUERY}")
        print(f"Filter for 'copilot/' prefix: {filter_prefix}")
        print()

        # Get initial estimate
        initial_query = f"{self.SEARCH_QUERY} created:{start_date}..{end_date}"
        try:
            total_estimate = self.get_search_count(initial_query)
            print(f"Estimated total PRs (with 'copilot' in branch): {total_estimate:,}")
            if total_estimate == 0:
                print("No PRs found.")
                return []
        except Exception as e:
            print(f"Error getting count: {e}")
            total_estimate = None

        print(f"{self.get_rate_limit_info()}")
        print()

        # Fetch PRs using date splitting
        prs = self._fetch_prs_with_date_split(start_date, end_date, output_path, filter_prefix)
        self._prs = prs

        # Save final results
        if output_path:
            self._save_prs(output_path)
            print(f"  Final save: {len(prs):,} PRs to {output_path}")

        print(f"\nFetched {len(self._prs):,} unique PRs")
        print(f"{self.get_rate_limit_info()}")

        return self._prs

    def _fetch_prs_with_date_split(
        self,
        start_date: date,
        end_date: date,
        output_path: Optional[Path] = None,
        filter_prefix: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Fetch PRs using date range splitting to overcome 1000 result limit.

        Uses recursive time-splitting strategy:
        1. Split by days until single day
        2. Split single day by 6-hour intervals
        3. Split 6-hour intervals by 1-hour intervals
        """
        prs = []
        seen_ids: Set[int] = set()
        last_save_count = 0
        filtered_count = 0  # Track how many were filtered out

        # Queue of ranges to process: (start_datetime, end_datetime, is_time_range)
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time().replace(microsecond=0))
        ranges = [(start_dt, end_dt, False)]

        with tqdm(desc="Fetching PRs", unit="pr") as pbar:
            while ranges:
                s, e, is_time_range = ranges.pop(0)

                # Build query with date/time range
                if is_time_range:
                    date_filter = f"created:{s.isoformat()}..{e.isoformat()}"
                else:
                    date_filter = f"created:{s.date()}..{e.date()}"

                query = f"{self.SEARCH_QUERY} {date_filter}"

                # Get count for this range (with retry)
                count = None
                for retry in range(3):
                    try:
                        count = self.get_search_count(query)
                        break
                    except Exception as ex:
                        if retry < 2:
                            print(f"\n  Error counting PRs for {s}..{e}: {ex}. Retrying...")
                            time.sleep(5 * (retry + 1))
                        else:
                            print(f"\n  Error counting PRs for {s}..{e}: {ex}. Adding back to queue.")
                            ranges.append((s, e, is_time_range))
                            count = None

                if count is None:
                    continue

                if count == 0:
                    continue

                if count <= 1000:
                    # Can fetch all PRs in this range
                    if is_time_range:
                        pbar.set_description(f"Fetching {s.date()} {s.hour:02d}:{s.minute:02d} ({count:,})")
                    else:
                        pbar.set_description(f"Fetching {s.date()} to {e.date()} ({count:,})")

                    new_prs = self._paginated_pr_search(query)

                    for pr_item in new_prs:
                        pr_data = self._extract_pr_data(pr_item)
                        pr_id = pr_data['pr_id']

                        if pr_id in seen_ids:
                            continue

                        seen_ids.add(pr_id)

                        # Apply prefix filter if requested
                        if filter_prefix:
                            if self._is_copilot_pr(pr_data):
                                prs.append(pr_data)
                                pbar.update(1)
                            else:
                                filtered_count += 1
                        else:
                            prs.append(pr_data)
                            pbar.update(1)

                    # Periodic save to prevent data loss
                    if output_path and len(prs) - last_save_count >= self.SAVE_EVERY:
                        self._prs = prs
                        self._save_prs(output_path)
                        last_save_count = len(prs)
                        pbar.set_postfix({
                            'saved': f'{len(prs):,}',
                            'filtered': f'{filtered_count:,}'
                        })
                else:
                    # Need to split the range
                    if not is_time_range:
                        days = (e.date() - s.date()).days

                        if days > 0:
                            mid_date = s.date() + timedelta(days=days // 2)
                            mid_dt = datetime.combine(mid_date, datetime.max.time().replace(microsecond=0))
                            next_dt = datetime.combine(mid_date + timedelta(days=1), datetime.min.time())
                            ranges.insert(0, (next_dt, e, False))
                            ranges.insert(0, (s, mid_dt, False))
                        else:
                            # Single day with >1000 PRs - switch to time-based splitting
                            day_start = datetime.combine(s.date(), datetime.min.time())
                            for hour_offset in [18, 12, 6, 0]:
                                interval_start = day_start.replace(hour=hour_offset)
                                if hour_offset == 18:
                                    interval_end = day_start.replace(hour=23, minute=59, second=59)
                                else:
                                    interval_end = day_start.replace(hour=hour_offset + 5, minute=59, second=59)
                                ranges.insert(0, (interval_start, interval_end, True))
                    else:
                        # Already in time-based mode - split further
                        total_seconds = (e - s).total_seconds()

                        if total_seconds > 3600:
                            mid = s + timedelta(seconds=total_seconds / 2)
                            mid = mid.replace(second=0, microsecond=0)
                            ranges.insert(0, (mid, e, True))
                            ranges.insert(0, (s, mid - timedelta(seconds=1), True))
                        elif total_seconds > 600:
                            interval_mins = max(1, int(total_seconds / 60 / 6))
                            intervals = []
                            current = s
                            while current < e:
                                interval_end = min(current + timedelta(minutes=interval_mins) - timedelta(seconds=1), e)
                                intervals.append((current, interval_end, True))
                                current = interval_end + timedelta(seconds=1)
                            for interval in reversed(intervals):
                                ranges.insert(0, interval)
                        elif total_seconds > 60:
                            intervals = []
                            current = s
                            while current < e:
                                interval_end = min(current + timedelta(minutes=1) - timedelta(seconds=1), e)
                                intervals.append((current, interval_end, True))
                                current = interval_end + timedelta(seconds=1)
                            for interval in reversed(intervals):
                                ranges.insert(0, interval)
                        else:
                            # Very small interval - just fetch what we can
                            if is_time_range:
                                pbar.set_description(f"Fetching {s.date()} {s.hour:02d}:{s.minute:02d}:{s.second:02d} (>1000, max)")
                            else:
                                pbar.set_description(f"Fetching {s.date()} (>1000, max)")

                            new_prs = self._paginated_pr_search(query, max_pages=10)

                            for pr_item in new_prs:
                                pr_data = self._extract_pr_data(pr_item)
                                pr_id = pr_data['pr_id']

                                if pr_id in seen_ids:
                                    continue

                                seen_ids.add(pr_id)

                                if filter_prefix:
                                    if self._is_copilot_pr(pr_data):
                                        prs.append(pr_data)
                                        pbar.update(1)
                                    else:
                                        filtered_count += 1
                                else:
                                    prs.append(pr_data)
                                    pbar.update(1)

                            if output_path and len(prs) - last_save_count >= self.SAVE_EVERY:
                                self._prs = prs
                                self._save_prs(output_path)
                                last_save_count = len(prs)

        if filter_prefix:
            print(f"\n  Filtered out {filtered_count:,} PRs without 'copilot/' prefix")

        return prs

    def _paginated_pr_search(
        self,
        query: str,
        max_pages: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fetch PRs with pagination (up to max_pages * 100 results).

        Args:
            query: Search query
            max_pages: Maximum pages to fetch (default 10 = 1000 results)

        Returns:
            List of PR items from the API
        """
        prs = []
        page = 1

        while page <= max_pages:
            try:
                result = self.search_prs(
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

            prs.extend(items)

            total_count = result.get('total_count', 0)
            if page * 100 >= total_count:
                break

            page += 1

        return prs

    def _extract_pr_data(self, pr_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract PR fields from API response.

        Note: The search/issues API doesn't return head/base ref directly.
        We need to parse it from the pull_request URL or make an extra API call.
        For efficiency, we extract what's available and note limitations.

        Args:
            pr_item: PR item from search results

        Returns:
            Dictionary with PR data
        """
        # Extract repo from repository_url
        repo_url = pr_item.get('repository_url', '')
        repo_nwo = repo_url.split('/repos/')[-1] if '/repos/' in repo_url else ''

        # The search/issues API returns pull_request.url but not head/base directly
        # We'll need to fetch the full PR details to get the branch names
        pull_request_info = pr_item.get('pull_request', {})

        return {
            'pr_id': pr_item.get('id'),
            'pr_number': pr_item.get('number'),
            'pr_url': pr_item.get('html_url', ''),
            'title': pr_item.get('title', ''),
            'state': pr_item.get('state', ''),
            'created_at': pr_item.get('created_at', ''),
            'updated_at': pr_item.get('updated_at', ''),
            'closed_at': pr_item.get('closed_at'),
            'user_login': pr_item.get('user', {}).get('login', ''),
            'repo_nwo': repo_nwo,
            'head_ref': '',  # Will be populated by _enrich_pr_with_branch
            'base_ref': '',  # Will be populated by _enrich_pr_with_branch
            'pr_api_url': pull_request_info.get('url', ''),
        }

    def _enrich_pr_with_branch(self, pr_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch full PR details to get head/base branch names.

        Args:
            pr_data: PR data with pr_api_url

        Returns:
            PR data enriched with head_ref and base_ref
        """
        pr_api_url = pr_data.get('pr_api_url', '')
        if not pr_api_url:
            return pr_data

        self.wait_for_search_rate_limit()
        self._record_search_request()

        try:
            response = self.session.get(pr_api_url)
            self._update_rate_limit_from_headers(response.headers)

            if response.status_code == 200:
                full_pr = response.json()
                pr_data['head_ref'] = full_pr.get('head', {}).get('ref', '')
                pr_data['base_ref'] = full_pr.get('base', {}).get('ref', '')
        except Exception as e:
            pass  # Keep empty refs if we can't fetch

        return pr_data

    def _is_copilot_pr(self, pr_data: Dict[str, Any]) -> bool:
        """
        Filter for branches starting with 'copilot/'.

        Args:
            pr_data: PR data dictionary

        Returns:
            True if head_ref starts with 'copilot/'
        """
        head_ref = pr_data.get('head_ref', '')
        return head_ref.startswith('copilot/')

    def _save_prs(self, output_path: Path) -> None:
        """Save PRs to file (JSONL and parquet)."""
        if not self._prs:
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write as JSON lines (append-safe format)
        jsonl_path = output_path.with_suffix('.jsonl')
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for pr in self._prs:
                f.write(json.dumps(pr, ensure_ascii=False) + '\n')

        # Also save as parquet
        try:
            df = pd.DataFrame(self._prs)
            df.to_parquet(output_path, index=False)
        except Exception as e:
            print(f"  Warning: Could not save parquet: {e}")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert fetched PRs to DataFrame."""
        if not self._prs:
            return pd.DataFrame()
        return pd.DataFrame(self._prs)

    def get_unique_repos(self) -> pd.DataFrame:
        """
        Get unique repos with Copilot PR counts.

        Returns:
            DataFrame with repo_nwo, copilot_pr_count, first_copilot_pr, last_copilot_pr
        """
        if not self._prs:
            return pd.DataFrame()

        df = pd.DataFrame(self._prs)

        # Group by repo
        repo_stats = df.groupby('repo_nwo').agg(
            copilot_pr_count=('pr_id', 'count'),
            first_copilot_pr=('created_at', 'min'),
            last_copilot_pr=('created_at', 'max')
        ).reset_index()

        return repo_stats.sort_values('copilot_pr_count', ascending=False)
