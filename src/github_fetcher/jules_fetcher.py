"""Fetcher for finding Google Jules commits on GitHub."""

import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import requests
from tqdm import tqdm

from .utils import exponential_backoff


class JulesCommitFetcher:
    """Fetcher for finding Google Jules commits across GitHub.

    Google Jules creates commits as `google-labs-jules[bot]`.
    We search for commits authored by this bot account.
    """

    # Bot author name
    JULES_AUTHOR = "google-labs-jules[bot]"

    # API endpoint for commit search
    SEARCH_URL = "https://api.github.com/search/commits"

    # Search API rate limit: 30 requests per minute
    SEARCH_RATE_LIMIT = 30
    SEARCH_RATE_WINDOW = 60  # seconds

    # Save progress interval
    SAVE_EVERY = 1000

    def __init__(self, token: str):
        if not token:
            raise ValueError("GitHub token is required")

        self.token = token
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })

        self._search_requests: List[float] = []
        self._rate_limit_remaining: Optional[int] = None
        self._rate_limit_reset: Optional[datetime] = None
        self._commits: List[Dict[str, Any]] = []

    def _update_rate_limit_from_headers(self, headers: Dict[str, str]) -> None:
        if 'X-RateLimit-Remaining' in headers:
            self._rate_limit_remaining = int(headers['X-RateLimit-Remaining'])
        if 'X-RateLimit-Reset' in headers:
            reset_timestamp = int(headers['X-RateLimit-Reset'])
            self._rate_limit_reset = datetime.fromtimestamp(reset_timestamp)

    def wait_for_search_rate_limit(self) -> None:
        now = time.time()
        window_start = now - self.SEARCH_RATE_WINDOW
        self._search_requests = [t for t in self._search_requests if t > window_start]

        if len(self._search_requests) >= self.SEARCH_RATE_LIMIT:
            oldest = min(self._search_requests)
            wait_time = oldest + self.SEARCH_RATE_WINDOW - now + 1
            if wait_time > 0:
                print(f"  Search rate limit reached. Waiting {wait_time:.0f}s...")
                time.sleep(wait_time)

        if self._rate_limit_remaining is not None and self._rate_limit_remaining < 5:
            if self._rate_limit_reset:
                now_dt = datetime.now()
                wait_time = (self._rate_limit_reset - now_dt).total_seconds()
                if wait_time > 0:
                    print(f"  GitHub rate limit low. Waiting {wait_time:.0f}s...")
                    time.sleep(wait_time + 5)

    def _record_search_request(self) -> None:
        self._search_requests.append(time.time())

    @exponential_backoff(max_retries=3, base_delay=2.0, exceptions=(requests.exceptions.RequestException,))
    def search_commits(self, query: str, sort: str = "committer-date", order: str = "desc",
                       per_page: int = 100, page: int = 1) -> Dict[str, Any]:
        self.wait_for_search_rate_limit()

        params = {"q": query, "sort": sort, "order": order, "per_page": min(per_page, 100), "page": page}
        self._record_search_request()
        response = self.session.get(self.SEARCH_URL, params=params)
        self._update_rate_limit_from_headers(response.headers)

        if response.status_code == 403:
            retry_after = response.headers.get('Retry-After')
            if retry_after:
                time.sleep(int(retry_after))
                return self.search_commits(query, sort, order, per_page, page)
            if 'secondary rate limit' in response.text.lower():
                time.sleep(60)
                return self.search_commits(query, sort, order, per_page, page)

        response.raise_for_status()
        return response.json()

    def get_search_count(self, query: str) -> int:
        result = self.search_commits(query, per_page=1, page=1)
        return result.get('total_count', 0)

    def fetch_jules_commits(self, start_date: date, end_date: Optional[date] = None,
                            output_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        if end_date is None:
            end_date = date.today()

        self._commits = []
        print(f"Searching for Jules commits...")
        print(f"Author: {self.JULES_AUTHOR}")
        print(f"Date range: {start_date} to {end_date}")

        initial_query = f"author:{self.JULES_AUTHOR} committer-date:{start_date}..{end_date}"
        try:
            total_estimate = self.get_search_count(initial_query)
            print(f"Estimated total commits: {total_estimate:,}")
            if total_estimate == 0:
                return []
        except Exception as e:
            print(f"Error getting count: {e}")

        commits = self._fetch_commits_with_date_split(start_date, end_date, output_path)
        self._commits = commits

        if output_path:
            self._save_commits(output_path)
            print(f"  Final save: {len(commits):,} commits")

        print(f"\nFetched {len(self._commits):,} unique commits")
        return self._commits

    def _fetch_commits_with_date_split(self, start_date: date, end_date: date,
                                        output_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        commits = []
        seen_shas: Set[str] = set()
        last_save_count = 0

        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time().replace(microsecond=0))
        ranges = [(start_dt, end_dt, False)]

        with tqdm(desc="Fetching commits", unit="commit") as pbar:
            while ranges:
                s, e, is_time_range = ranges.pop(0)

                if is_time_range:
                    date_filter = f"committer-date:{s.isoformat()}..{e.isoformat()}"
                else:
                    date_filter = f"committer-date:{s.date()}..{e.date()}"

                query = f"author:{self.JULES_AUTHOR} {date_filter}"

                count = None
                for retry in range(3):
                    try:
                        count = self.get_search_count(query)
                        break
                    except Exception as ex:
                        if retry < 2:
                            time.sleep(5 * (retry + 1))
                        else:
                            ranges.append((s, e, is_time_range))

                if count is None or count == 0:
                    continue

                if count <= 1000:
                    if is_time_range:
                        pbar.set_description(f"Fetching {s.date()} {s.hour:02d}:{s.minute:02d} ({count:,})")
                    else:
                        pbar.set_description(f"Fetching {s.date()} to {e.date()} ({count:,})")

                    new_commits = self._paginated_commit_search(query)

                    for commit_item in new_commits:
                        commit_data = self._extract_commit_data(commit_item)
                        sha = commit_data['sha']
                        if sha not in seen_shas:
                            seen_shas.add(sha)
                            commits.append(commit_data)
                            pbar.update(1)

                    if output_path and len(commits) - last_save_count >= self.SAVE_EVERY:
                        self._commits = commits
                        self._save_commits(output_path)
                        last_save_count = len(commits)
                else:
                    if not is_time_range:
                        days = (e.date() - s.date()).days
                        if days > 0:
                            mid_date = s.date() + timedelta(days=days // 2)
                            mid_dt = datetime.combine(mid_date, datetime.max.time().replace(microsecond=0))
                            next_dt = datetime.combine(mid_date + timedelta(days=1), datetime.min.time())
                            ranges.insert(0, (next_dt, e, False))
                            ranges.insert(0, (s, mid_dt, False))
                        else:
                            day_start = datetime.combine(s.date(), datetime.min.time())
                            for hour_offset in [18, 12, 6, 0]:
                                interval_start = day_start.replace(hour=hour_offset)
                                interval_end = day_start.replace(hour=hour_offset + 5, minute=59, second=59) if hour_offset < 18 else day_start.replace(hour=23, minute=59, second=59)
                                ranges.insert(0, (interval_start, interval_end, True))
                    else:
                        total_seconds = (e - s).total_seconds()
                        if total_seconds > 3600:
                            mid = s + timedelta(seconds=total_seconds / 2)
                            mid = mid.replace(second=0, microsecond=0)
                            ranges.insert(0, (mid, e, True))
                            ranges.insert(0, (s, mid - timedelta(seconds=1), True))
                        else:
                            pbar.set_description(f"Fetching {s.date()} {s.hour:02d}:{s.minute:02d} (>1000)")
                            new_commits = self._paginated_commit_search(query, max_pages=10)
                            for commit_item in new_commits:
                                commit_data = self._extract_commit_data(commit_item)
                                sha = commit_data['sha']
                                if sha not in seen_shas:
                                    seen_shas.add(sha)
                                    commits.append(commit_data)
                                    pbar.update(1)

        return commits

    def _paginated_commit_search(self, query: str, max_pages: int = 10) -> List[Dict[str, Any]]:
        commits = []
        page = 1
        while page <= max_pages:
            try:
                result = self.search_commits(query=query, per_page=100, page=page)
            except Exception:
                break
            items = result.get('items', [])
            if not items:
                break
            commits.extend(items)
            if page * 100 >= result.get('total_count', 0):
                break
            page += 1
        return commits

    def _extract_commit_data(self, commit_item: Dict[str, Any]) -> Dict[str, Any]:
        commit = commit_item.get('commit', {})
        author = commit.get('author', {})
        committer = commit.get('committer', {})
        repo = commit_item.get('repository', {})

        return {
            'sha': commit_item.get('sha', ''),
            'repo_nwo': repo.get('full_name', ''),
            'message': commit.get('message', ''),
            'author_name': author.get('name', ''),
            'author_email': author.get('email', ''),
            'author_date': author.get('date', ''),
            'committer_name': committer.get('name', ''),
            'committer_email': committer.get('email', ''),
            'committer_date': committer.get('date', ''),
            'html_url': commit_item.get('html_url', ''),
        }

    def _save_commits(self, output_path: Path) -> None:
        if not self._commits:
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_path.with_suffix('.jsonl')
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for commit in self._commits:
                f.write(json.dumps(commit, ensure_ascii=False) + '\n')
        try:
            df = pd.DataFrame(self._commits)
            df.to_parquet(output_path, index=False)
        except Exception as e:
            print(f"  Warning: Could not save parquet: {e}")

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._commits) if self._commits else pd.DataFrame()

    def get_unique_repos(self) -> pd.DataFrame:
        if not self._commits:
            return pd.DataFrame()
        df = pd.DataFrame(self._commits)
        repo_stats = df.groupby('repo_nwo').agg(
            jules_commit_count=('sha', 'count'),
            first_jules_commit=('committer_date', 'min'),
            last_jules_commit=('committer_date', 'max')
        ).reset_index()
        return repo_stats.sort_values('jules_commit_count', ascending=False)
