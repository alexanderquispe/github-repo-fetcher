"""Batch repository metadata fetcher using GraphQL alias queries.

Fetches metadata for a list of known repo names (owner/name) in batches
of 50 per query using GraphQL aliases, achieving ~50x throughput vs
single-repo queries.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from tqdm import tqdm

from .utils import calculate_wait_time, extract_repo_data

# Fields to fetch per repo (same as SINGLE_REPO_QUERY)
REPO_FRAGMENT = """
  nameWithOwner name description url homepageUrl createdAt updatedAt pushedAt
  stargazerCount forkCount diskUsage
  primaryLanguage { name }
  languages(first: 10) { nodes { name } }
  repositoryTopics(first: 20) { nodes { topic { name } } }
  licenseInfo { key name }
  isFork isArchived isPrivate isTemplate hasWikiEnabled hasIssuesEnabled
  watchers { totalCount }
  issues(states: OPEN) { totalCount }
  owner {
    login __typename
    ... on User { location company bio email followers { totalCount } createdAt }
    ... on Organization { location email description }
  }
  object(expression: "HEAD:README.md") { ... on Blob { text } }
"""

REPO_FRAGMENT_NO_README = """
  nameWithOwner name description url homepageUrl createdAt updatedAt pushedAt
  stargazerCount forkCount diskUsage
  primaryLanguage { name }
  languages(first: 10) { nodes { name } }
  repositoryTopics(first: 20) { nodes { topic { name } } }
  licenseInfo { key name }
  isFork isArchived isPrivate isTemplate hasWikiEnabled hasIssuesEnabled
  watchers { totalCount }
  issues(states: OPEN) { totalCount }
  owner {
    login __typename
    ... on User { location company bio email followers { totalCount } createdAt }
    ... on Organization { location email description }
  }
"""


class BatchRepoFetcher:
    """Fetches repo metadata in batches using GraphQL aliases."""

    API_URL = "https://api.github.com/graphql"
    BATCH_SIZE = 50
    SAVE_INTERVAL = 500  # Save every N repos
    MIN_RATE_LIMIT = 50

    def __init__(self, token: str, include_readme: bool = True):
        self.token = token
        self.include_readme = include_readme
        self.fragment = REPO_FRAGMENT if include_readme else REPO_FRAGMENT_NO_README
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        })
        self._rate_limit: Optional[Dict[str, Any]] = None

    def _execute_raw(self, query: str) -> Dict[str, Any]:
        """Execute GraphQL query, returning full result (tolerates partial errors)."""
        for attempt in range(4):
            try:
                resp = self.session.post(self.API_URL, json={"query": query}, timeout=60)
                if resp.status_code == 502 or resp.status_code == 503:
                    time.sleep(5 * (attempt + 1))
                    continue
                if resp.status_code == 403:
                    # Rate limited - wait
                    self._wait_for_rate_limit()
                    continue
                resp.raise_for_status()
                result = resp.json()
                # Update rate limit
                data = result.get("data", {})
                if "rateLimit" in data:
                    self._rate_limit = data["rateLimit"]
                return result
            except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
                if attempt == 3:
                    raise
                print(f"  Retry {attempt+1}/3: {str(e)[:100]}")
                time.sleep(5 * (attempt + 1))
        return {}

    def _wait_for_rate_limit(self):
        """Wait if rate limit is low."""
        if not self._rate_limit:
            return
        remaining = self._rate_limit.get("remaining", 5000)
        reset_at = self._rate_limit.get("resetAt", "")
        if remaining < self.MIN_RATE_LIMIT and reset_at:
            wait_time = calculate_wait_time(reset_at)
            if wait_time > 0:
                print(f"\n  Rate limit low ({remaining}). Waiting {wait_time:.0f}s...")
                time.sleep(wait_time)

    def _build_batch_query(self, repos: List[str]) -> str:
        """Build a GraphQL query with aliases for multiple repos."""
        parts = []
        for i, nwo in enumerate(repos):
            if "/" not in nwo:
                continue
            owner, name = nwo.split("/", 1)
            # Escape quotes
            owner = owner.replace('"', '\\"')
            name = name.replace('"', '\\"')
            parts.append(
                f'repo{i}: repository(owner: "{owner}", name: "{name}") {{ {self.fragment} }}'
            )
        query = "query { " + " ".join(parts) + " rateLimit { remaining resetAt limit cost } }"
        return query

    def _parse_batch_result(self, result: Dict, batch: List[str]) -> List[Dict[str, Any]]:
        """Parse batch query result, handling null/missing repos."""
        data = result.get("data", {})
        repos = []
        for i, nwo in enumerate(batch):
            key = f"repo{i}"
            repo_node = data.get(key)
            if repo_node:
                repo_data = extract_repo_data(repo_node)
                repos.append(repo_data)
        return repos

    def _save_progress(self, repos: List[Dict], output_path: Path):
        """Save repos to JSONL file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for repo in repos:
                # Convert lists to JSON strings for consistency
                row = dict(repo)
                for key in ["languages", "topics"]:
                    if key in row and isinstance(row[key], list):
                        row[key] = json.dumps(row[key])
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def fetch_repos(
        self,
        repo_list: List[str],
        output_path: Path,
    ) -> List[Dict[str, Any]]:
        """
        Fetch metadata for a list of repos.

        Args:
            repo_list: List of "owner/name" strings
            output_path: Path to save JSONL output

        Returns:
            List of repo metadata dicts
        """
        all_repos = []
        failed_repos = []
        total = len(repo_list)

        print(f"Fetching metadata for {total:,} repos (batch size: {self.BATCH_SIZE})")
        print(f"Output: {output_path}")

        pbar = tqdm(total=total, desc="Fetching repos", unit="repo")

        for start in range(0, total, self.BATCH_SIZE):
            batch = repo_list[start:start + self.BATCH_SIZE]

            # Check rate limit
            self._wait_for_rate_limit()

            # Build and execute query
            query = self._build_batch_query(batch)
            try:
                result = self._execute_raw(query)
                repos = self._parse_batch_result(result, batch)
                all_repos.extend(repos)
            except Exception as e:
                print(f"\n  Batch failed at index {start}: {e}")
                failed_repos.extend(batch)

            pbar.update(len(batch))

            # Periodic save
            if len(all_repos) % self.SAVE_INTERVAL < self.BATCH_SIZE and len(all_repos) > 0:
                self._save_progress(all_repos, output_path)

        pbar.close()

        # Final save
        self._save_progress(all_repos, output_path)

        print(f"\nDone! Fetched {len(all_repos):,} repos ({total - len(all_repos) - len(failed_repos):,} deleted/private, {len(failed_repos):,} failed)")
        print(f"Saved to: {output_path}")

        return all_repos
