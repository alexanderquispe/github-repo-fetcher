"""GitHub GraphQL API client with authentication and rate limiting."""

import time
from typing import Any, Dict, Optional

import requests

from .queries import RATE_LIMIT_QUERY
from .utils import calculate_wait_time, exponential_backoff, format_rate_limit_info


class GitHubGraphQLClient:
    """Client for GitHub GraphQL API with authentication and rate limiting."""

    API_URL = "https://api.github.com/graphql"

    def __init__(self, token: str):
        """
        Initialize the GitHub GraphQL client.

        Args:
            token: GitHub Personal Access Token
        """
        if not token:
            raise ValueError("GitHub token is required")

        self.token = token
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        })

        # Track rate limit info
        self._rate_limit: Optional[Dict[str, Any]] = None

    @property
    def rate_limit(self) -> Optional[Dict[str, Any]]:
        """Get the current rate limit info."""
        return self._rate_limit

    @exponential_backoff(
        max_retries=3,
        base_delay=2.0,
        exceptions=(requests.exceptions.RequestException,)
    )
    def execute(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a GraphQL query.

        Args:
            query: GraphQL query string
            variables: Query variables

        Returns:
            GraphQL response data

        Raises:
            requests.exceptions.RequestException: On network errors
            ValueError: On GraphQL errors
        """
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        response = self.session.post(self.API_URL, json=payload)
        response.raise_for_status()

        result = response.json()

        # Update rate limit info if present
        if 'data' in result and 'rateLimit' in result['data']:
            self._rate_limit = result['data']['rateLimit']

        # Check for GraphQL errors
        if 'errors' in result:
            error_messages = [e.get('message', str(e)) for e in result['errors']]
            raise ValueError(f"GraphQL errors: {'; '.join(error_messages)}")

        return result.get('data', {})

    def check_rate_limit(self) -> Dict[str, Any]:
        """
        Check current rate limit status.

        Returns:
            Rate limit info dict with 'remaining', 'limit', 'resetAt'
        """
        data = self.execute(RATE_LIMIT_QUERY)
        return data.get('rateLimit', {})

    def wait_for_rate_limit(self, min_remaining: int = 100) -> None:
        """
        Wait if rate limit is below threshold.

        Args:
            min_remaining: Minimum remaining requests before waiting
        """
        if not self._rate_limit:
            self.check_rate_limit()

        remaining = self._rate_limit.get('remaining', 0)
        reset_at = self._rate_limit.get('resetAt', '')

        if remaining < min_remaining and reset_at:
            wait_time = calculate_wait_time(reset_at)
            if wait_time > 0:
                print(f"\n  Rate limit low ({remaining} remaining). Waiting {wait_time:.0f}s until reset...")
                time.sleep(wait_time)
                # Check again after waiting
                self.check_rate_limit()

    def get_rate_limit_info(self) -> str:
        """Get formatted rate limit info string."""
        if not self._rate_limit:
            self.check_rate_limit()
        return format_rate_limit_info(self._rate_limit)
