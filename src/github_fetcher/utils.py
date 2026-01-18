"""Utility functions for rate limiting, retry logic, and data processing."""

import time
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional


def parse_iso_datetime(dt_string: Optional[str]) -> Optional[str]:
    """Parse ISO datetime string and return in standard format."""
    if not dt_string:
        return None
    return dt_string


def exponential_backoff(
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorator for exponential backoff retry logic.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential calculation
        exceptions: Tuple of exceptions to catch and retry on
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        raise
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    print(f"  Retry {attempt + 1}/{max_retries} after {delay:.1f}s: {str(e)[:100]}")
                    time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator


def calculate_wait_time(reset_at: str) -> float:
    """
    Calculate seconds to wait until rate limit reset.

    Args:
        reset_at: ISO format datetime string of reset time

    Returns:
        Seconds to wait (minimum 0)
    """
    reset_time = datetime.fromisoformat(reset_at.replace('Z', '+00:00'))
    now = datetime.now(reset_time.tzinfo)
    wait_seconds = (reset_time - now).total_seconds()
    return max(0, wait_seconds + 5)  # Add 5 second buffer


def format_rate_limit_info(rate_limit: Dict[str, Any]) -> str:
    """Format rate limit info for display."""
    remaining = rate_limit.get('remaining', '?')
    limit = rate_limit.get('limit', '?')
    reset_at = rate_limit.get('resetAt', '')

    if reset_at:
        try:
            reset_time = datetime.fromisoformat(reset_at.replace('Z', '+00:00'))
            reset_str = reset_time.strftime('%H:%M:%S')
        except (ValueError, TypeError):
            reset_str = reset_at
    else:
        reset_str = '?'

    return f"Rate limit: {remaining}/{limit} (resets at {reset_str})"


def extract_repo_data(repo_node: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and flatten repository data from GraphQL response.

    Args:
        repo_node: Repository node from GraphQL response

    Returns:
        Flattened dictionary with all repository data
    """
    if not repo_node:
        return {}

    # Extract owner info
    owner = repo_node.get('owner', {}) or {}
    owner_typename = owner.get('__typename', '')

    # Extract languages list
    languages_nodes = repo_node.get('languages', {})
    if languages_nodes:
        languages = [lang.get('name') for lang in languages_nodes.get('nodes', []) if lang]
    else:
        languages = []

    # Extract topics list
    topics_data = repo_node.get('repositoryTopics', {})
    if topics_data:
        topics = [
            node.get('topic', {}).get('name')
            for node in topics_data.get('nodes', [])
            if node and node.get('topic')
        ]
    else:
        topics = []

    # Extract README content
    readme_obj = repo_node.get('object')
    readme_content = readme_obj.get('text', '') if readme_obj else ''

    # Extract license info
    license_info = repo_node.get('licenseInfo', {}) or {}

    # Extract primary language
    primary_lang = repo_node.get('primaryLanguage', {}) or {}

    return {
        # Repository metadata
        'nwo': repo_node.get('nameWithOwner', ''),
        'name': repo_node.get('name', ''),
        'description': repo_node.get('description', ''),
        'url': repo_node.get('url', ''),
        'homepage_url': repo_node.get('homepageUrl', ''),
        'created_at': repo_node.get('createdAt', ''),
        'updated_at': repo_node.get('updatedAt', ''),
        'pushed_at': repo_node.get('pushedAt', ''),

        # Metrics
        'stars': repo_node.get('stargazerCount', 0),
        'forks': repo_node.get('forkCount', 0),
        'watchers': repo_node.get('watchers', {}).get('totalCount', 0) if repo_node.get('watchers') else 0,
        'open_issues': repo_node.get('issues', {}).get('totalCount', 0) if repo_node.get('issues') else 0,
        'disk_usage_kb': repo_node.get('diskUsage', 0),

        # Languages & topics
        'primary_language': primary_lang.get('name', ''),
        'languages': languages,
        'topics': topics,

        # Flags
        'is_fork': repo_node.get('isFork', False),
        'is_archived': repo_node.get('isArchived', False),
        'is_private': repo_node.get('isPrivate', False),
        'is_template': repo_node.get('isTemplate', False),
        'has_wiki': repo_node.get('hasWikiEnabled', False),
        'has_issues': repo_node.get('hasIssuesEnabled', False),

        # License
        'license_key': license_info.get('key', ''),
        'license_name': license_info.get('name', ''),

        # Owner information
        'owner_login': owner.get('login', ''),
        'owner_type': owner_typename,
        'owner_location': owner.get('location', ''),
        'owner_company': owner.get('company', '') if owner_typename == 'User' else '',
        'owner_bio': owner.get('bio', '') if owner_typename == 'User' else owner.get('description', ''),
        'owner_email': owner.get('email', ''),
        'owner_followers': owner.get('followers', {}).get('totalCount', 0) if owner.get('followers') else 0,
        'owner_created_at': owner.get('createdAt', ''),

        # README content
        'readme_content': readme_content,
    }


def build_search_query(
    language: Optional[str] = None,
    min_stars: int = 5,
    custom_query: Optional[str] = None
) -> str:
    """
    Build a GitHub search query string.

    Note: GitHub repository search doesn't support location filtering directly.
    Location filtering is done post-fetch based on owner_location field.

    Args:
        language: Filter by programming language
        min_stars: Minimum star count
        custom_query: Custom query string (overrides language)

    Returns:
        GitHub search query string
    """
    if custom_query:
        # Append stars filter to custom query if not already present
        if 'stars:' not in custom_query:
            return f"{custom_query} stars:>={min_stars}"
        return custom_query

    parts = []

    # Add language filter
    if language:
        parts.append(f'language:{language}')

    # Add stars filter
    parts.append(f'stars:>={min_stars}')

    # Sort by stars for consistent results
    parts.append('sort:stars')

    return ' '.join(parts)
