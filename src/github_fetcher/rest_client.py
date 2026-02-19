"""GitHub REST API client for commit search with rate limiting."""

import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from .utils import exponential_backoff


class GitHubRESTClient:
    """Client for GitHub REST API with commit search and rate limiting.

    The REST API search has different rate limits than the GraphQL API:
    - Search API: 30 requests/minute (authenticated)
    - Each request returns up to 100 results
    - Maximum 1000 results per search query (use date-splitting to overcome)
    """

    SEARCH_URL = "https://api.github.com/search/commits"
    API_BASE = "https://api.github.com"

    # Search API rate limit: 30 requests per minute
    SEARCH_RATE_LIMIT = 30
    SEARCH_RATE_WINDOW = 60  # seconds

    def __init__(self, token: str):
        """
        Initialize the GitHub REST client.

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

    def _update_rate_limit_from_headers(self, headers: Dict[str, str]) -> None:
        """Update rate limit tracking from response headers."""
        if 'X-RateLimit-Remaining' in headers:
            self._rate_limit_remaining = int(headers['X-RateLimit-Remaining'])
        if 'X-RateLimit-Reset' in headers:
            reset_timestamp = int(headers['X-RateLimit-Reset'])
            self._rate_limit_reset = datetime.fromtimestamp(reset_timestamp)

    def wait_for_search_rate_limit(self) -> None:
        """
        Wait if we've exceeded the search rate limit (30 req/min).

        Uses a sliding window approach to track requests.
        """
        now = time.time()
        window_start = now - self.SEARCH_RATE_WINDOW

        # Remove old requests outside the window
        self._search_requests = [t for t in self._search_requests if t > window_start]

        # Check if we need to wait
        if len(self._search_requests) >= self.SEARCH_RATE_LIMIT:
            # Wait until the oldest request falls outside the window
            oldest = min(self._search_requests)
            wait_time = oldest + self.SEARCH_RATE_WINDOW - now + 1  # +1 for buffer
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

    @exponential_backoff(
        max_retries=3,
        base_delay=2.0,
        exceptions=(requests.exceptions.RequestException,)
    )
    def search_commits(
        self,
        query: str,
        sort: str = "committer-date",
        order: str = "desc",
        per_page: int = 100,
        page: int = 1
    ) -> Dict[str, Any]:
        """
        Search commits using the GitHub REST API.

        Args:
            query: Search query string
            sort: Sort field (committer-date, author-date)
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
                return self.search_commits(query, sort, order, per_page, page)
            # Check if it's a secondary rate limit
            if 'secondary rate limit' in response.text.lower():
                print("  Secondary rate limit hit. Waiting 60s...")
                time.sleep(60)
                return self.search_commits(query, sort, order, per_page, page)

        response.raise_for_status()
        return response.json()

    def get_search_count(self, query: str) -> int:
        """
        Get the total count of results for a search query.

        Args:
            query: Search query string

        Returns:
            Total count of matching commits
        """
        result = self.search_commits(query, per_page=1, page=1)
        return result.get('total_count', 0)

    def get_rate_limit_info(self) -> str:
        """Get formatted rate limit info string."""
        recent_requests = len([t for t in self._search_requests
                              if t > time.time() - self.SEARCH_RATE_WINDOW])
        remaining = self._rate_limit_remaining if self._rate_limit_remaining is not None else '?'
        return f"Search rate: {recent_requests}/{self.SEARCH_RATE_LIMIT}/min, API remaining: {remaining}"


def extract_claude_model(commit_message: str) -> str:
    """
    Extract the Claude model name from a commit message.

    Args:
        commit_message: Full commit message

    Returns:
        Model name (e.g., "Sonnet", "Opus 4.5", "Claude") or empty string
    """
    # Look for Co-Authored-By line with Anthropic email
    pattern = r'Co-Authored-By:\s*Claude\s*([^<]*)?<[^>]*@anthropic\.com>'
    match = re.search(pattern, commit_message, re.IGNORECASE)

    if match:
        model_part = match.group(1)
        if model_part:
            model_part = model_part.strip()
            if model_part:
                return model_part
        return "Claude"  # Default if just "Claude <email>"

    return ""


def extract_commit_data(commit_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and flatten commit data from REST API response.

    Args:
        commit_item: Commit item from search results

    Returns:
        Flattened dictionary with commit data
    """
    commit_info = commit_item.get('commit', {})
    author_info = commit_info.get('author', {})
    committer_info = commit_info.get('committer', {})
    repo_info = commit_item.get('repository', {})

    # Get GitHub user info if available
    author_user = commit_item.get('author', {}) or {}

    message = commit_info.get('message', '')

    return {
        'sha': commit_item.get('sha', ''),
        'repo_nwo': repo_info.get('full_name', ''),
        'repo_url': repo_info.get('html_url', ''),
        'commit_url': commit_item.get('html_url', ''),
        'message': message,
        'committed_date': committer_info.get('date', ''),
        'authored_date': author_info.get('date', ''),
        'author_name': author_info.get('name', ''),
        'author_email': author_info.get('email', ''),
        'author_login': author_user.get('login', ''),
        'committer_name': committer_info.get('name', ''),
        'committer_email': committer_info.get('email', ''),
        'claude_model': extract_claude_model(message),
    }
