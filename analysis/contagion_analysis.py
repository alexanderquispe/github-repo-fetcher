"""
Simplified Contagion Analysis for AI Tool Adoption.

This module implements a cleaner approach to studying contagion:

1. Identify Claude users as of May 2025 (early adopters)
2. Find ALL repos for each Claude user
3. Find ALL collaborators from those repos
4. Check: Did collaborators adopt Claude by June 2025?

Key Question: Are collaborators of Claude users more likely to adopt Claude?
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# Key dates for the analysis (timezone-aware for comparison with commit dates)
CLAUDE_RELEASE_DATE = pd.Timestamp('2025-02-01', tz='UTC')  # February 2025
SNAPSHOT_DATE = pd.Timestamp('2025-05-31 23:59:59', tz='UTC')  # May 2025 - identify Claude users
OUTCOME_DATE = pd.Timestamp('2025-06-30 23:59:59', tz='UTC')   # June 2025 - check collaborator adoption


class ClaudeUserIdentifier:
    """Identifies Claude Code users from commit data."""

    def __init__(self, commits_path: Path):
        """
        Initialize with path to Claude commits data.

        Args:
            commits_path: Path to claude_commits_full.parquet
        """
        self.commits_path = Path(commits_path)
        self.commits_df: Optional[pd.DataFrame] = None
        self.users_by_date: Dict[datetime, Set[str]] = {}

    def load_commits(self) -> pd.DataFrame:
        """Load and parse Claude commits data."""
        print(f"Loading commits from {self.commits_path}...")

        self.commits_df = pd.read_parquet(self.commits_path)
        self.commits_df['committed_date'] = pd.to_datetime(
            self.commits_df['committed_date'], errors='coerce', utc=True
        )

        print(f"  Loaded {len(self.commits_df):,} commits")
        print(f"  Date range: {self.commits_df['committed_date'].min()} to {self.commits_df['committed_date'].max()}")

        return self.commits_df

    def get_users_as_of(self, as_of_date: datetime) -> Set[str]:
        """
        Get all users who had used Claude by a given date.

        Args:
            as_of_date: Cutoff date

        Returns:
            Set of user logins
        """
        if self.commits_df is None:
            self.load_commits()

        mask = self.commits_df['committed_date'] <= as_of_date
        users = set(self.commits_df[mask]['author_login'].dropna().unique())

        print(f"  Claude users as of {as_of_date.date()}: {len(users):,}")

        return users

    def get_user_first_use(self) -> Dict[str, datetime]:
        """
        Get first Claude use date for each user.

        Returns:
            Dict mapping user_login -> first commit datetime
        """
        if self.commits_df is None:
            self.load_commits()

        first_use = self.commits_df.groupby('author_login')['committed_date'].min()

        return first_use.to_dict()

    def get_users_between(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Set[str]:
        """
        Get users who first used Claude between two dates.

        Args:
            start_date: Start of period
            end_date: End of period

        Returns:
            Set of user logins who adopted in this period
        """
        first_use = self.get_user_first_use()

        users = {
            user for user, date in first_use.items()
            if start_date <= date <= end_date
        }

        return users


class CollaboratorNetwork:
    """Builds and analyzes the collaborator network."""

    def __init__(self):
        self.user_repos: Dict[str, List[str]] = {}      # user -> their repos
        self.repo_contributors: Dict[str, Set[str]] = {}  # repo -> contributors
        self.user_collaborators: Dict[str, Set[str]] = {}  # user -> their collaborators

    def load_user_repos(self, user_repos_path: Path) -> None:
        """
        Load user repos data.

        Args:
            user_repos_path: Path to user_repos.parquet
        """
        print(f"Loading user repos from {user_repos_path}...")

        df = pd.read_parquet(user_repos_path)

        for _, row in df.iterrows():
            user = row['user_login']
            repos = row['repos']
            if isinstance(repos, str):
                repos = eval(repos)  # Convert string repr to list
            self.user_repos[user] = repos

        print(f"  Loaded repos for {len(self.user_repos):,} users")

    def load_contributors(self, contributors_path: Path) -> None:
        """
        Load contributors data.

        Args:
            contributors_path: Path to contributors.parquet
        """
        print(f"Loading contributors from {contributors_path}...")

        df = pd.read_parquet(contributors_path)

        for _, row in df.iterrows():
            repo = row['repo_nwo']
            user = row['user_login']
            if pd.notna(repo) and pd.notna(user):
                if repo not in self.repo_contributors:
                    self.repo_contributors[repo] = set()
                self.repo_contributors[repo].add(user)

        print(f"  Loaded contributors for {len(self.repo_contributors):,} repos")

    def build_collaborator_map(self, focal_users: Set[str]) -> Dict[str, Set[str]]:
        """
        Build map of collaborators for focal users.

        For each focal user, find all other users who contributed to the same repos.

        Args:
            focal_users: Set of users to find collaborators for

        Returns:
            Dict mapping user -> set of collaborators
        """
        print(f"Building collaborator map for {len(focal_users):,} focal users...")

        self.user_collaborators = {}

        for user in focal_users:
            collaborators = set()

            # Get all repos this user contributed to
            user_repo_list = self.user_repos.get(user, [])

            for repo in user_repo_list:
                # Get all contributors to this repo
                repo_contribs = self.repo_contributors.get(repo, set())
                collaborators.update(repo_contribs)

            # Remove self
            collaborators.discard(user)

            self.user_collaborators[user] = collaborators

        total_collaborators = sum(len(c) for c in self.user_collaborators.values())
        unique_collaborators = set()
        for collabs in self.user_collaborators.values():
            unique_collaborators.update(collabs)

        print(f"  Total collaborator relationships: {total_collaborators:,}")
        print(f"  Unique collaborators: {len(unique_collaborators):,}")

        return self.user_collaborators

    def get_all_collaborators(self) -> Set[str]:
        """Get all unique collaborators across all focal users."""
        all_collabs = set()
        for collabs in self.user_collaborators.values():
            all_collabs.update(collabs)
        return all_collabs


class ContagionAnalyzer:
    """
    Main analyzer for the contagion study.

    Compares adoption rates:
    - Among collaborators of Claude users (exposed)
    - Among non-collaborators (unexposed, baseline)
    """

    def __init__(
        self,
        claude_users_may: Set[str],
        claude_users_june: Set[str],
        collaborators: Set[str],
        user_collaborator_map: Dict[str, Set[str]]
    ):
        """
        Initialize the analyzer.

        Args:
            claude_users_may: Users who adopted Claude by May 2025
            claude_users_june: Users who adopted Claude by June 2025
            collaborators: All collaborators of May Claude users
            user_collaborator_map: Map of Claude user -> their collaborators
        """
        self.claude_users_may = claude_users_may
        self.claude_users_june = claude_users_june
        self.collaborators = collaborators
        self.user_collaborator_map = user_collaborator_map

        # Derived sets
        self.new_adopters_june = claude_users_june - claude_users_may
        self.exposed_collaborators = collaborators - claude_users_may  # Not already users

    def compute_adoption_rates(self) -> Dict[str, Any]:
        """
        Compute adoption rates for exposed vs unexposed groups.

        Returns:
            Dict with adoption rate statistics
        """
        # Exposed: collaborators of May Claude users (who weren't already users)
        exposed = self.exposed_collaborators

        # How many exposed adopted by June?
        exposed_adopted = exposed & self.claude_users_june

        # Adoption rate among exposed
        exposed_rate = len(exposed_adopted) / len(exposed) if exposed else 0

        results = {
            "claude_users_may": len(self.claude_users_may),
            "claude_users_june": len(self.claude_users_june),
            "new_adopters_june": len(self.new_adopters_june),
            "total_collaborators": len(self.collaborators),
            "exposed_collaborators": len(exposed),
            "exposed_adopted": len(exposed_adopted),
            "exposed_adoption_rate": exposed_rate,
        }

        return results

    def compute_influence_by_exposure_level(self) -> pd.DataFrame:
        """
        Analyze adoption rate by number of Claude-using collaborators.

        Returns:
            DataFrame with exposure level analysis
        """
        # For each collaborator, count how many Claude users they're connected to
        collaborator_exposure: Dict[str, int] = defaultdict(int)

        for claude_user, collabs in self.user_collaborator_map.items():
            for collab in collabs:
                if collab not in self.claude_users_may:  # Don't count existing users
                    collaborator_exposure[collab] += 1

        # Group by exposure level
        exposure_groups: Dict[int, List[str]] = defaultdict(list)
        for collab, exposure in collaborator_exposure.items():
            exposure_groups[exposure].append(collab)

        # Compute adoption rate for each exposure level
        results = []
        for exposure_level in sorted(exposure_groups.keys()):
            group = set(exposure_groups[exposure_level])
            adopted = group & self.claude_users_june
            rate = len(adopted) / len(group) if group else 0

            results.append({
                "exposure_level": exposure_level,
                "num_collaborators": len(group),
                "num_adopted": len(adopted),
                "adoption_rate": rate,
            })

        return pd.DataFrame(results)

    def compute_time_to_adoption(
        self,
        first_use_dates: Dict[str, datetime]
    ) -> Dict[str, Any]:
        """
        Analyze time between collaborator's Claude user connection and their adoption.

        Args:
            first_use_dates: Dict mapping user -> first Claude use datetime

        Returns:
            Dict with timing statistics
        """
        # For collaborators who adopted, compute:
        # - When did their Claude-using collaborator first use Claude?
        # - When did they adopt?
        # - Time gap

        time_gaps = []

        for collab in self.exposed_collaborators & self.claude_users_june:
            collab_adoption = first_use_dates.get(collab)
            if not collab_adoption:
                continue

            # Find earliest Claude-using collaborator
            earliest_influencer_date = None

            for claude_user, collabs in self.user_collaborator_map.items():
                if collab in collabs:
                    influencer_date = first_use_dates.get(claude_user)
                    if influencer_date and influencer_date < collab_adoption:
                        if earliest_influencer_date is None or influencer_date < earliest_influencer_date:
                            earliest_influencer_date = influencer_date

            if earliest_influencer_date:
                gap_days = (collab_adoption - earliest_influencer_date).days
                time_gaps.append(gap_days)

        if not time_gaps:
            return {"error": "No time gap data available"}

        return {
            "num_influenced_adopters": len(time_gaps),
            "mean_gap_days": np.mean(time_gaps),
            "median_gap_days": np.median(time_gaps),
            "min_gap_days": min(time_gaps),
            "max_gap_days": max(time_gaps),
            "std_gap_days": np.std(time_gaps),
        }

    def generate_report(self, first_use_dates: Dict[str, datetime]) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report.

        Args:
            first_use_dates: Dict mapping user -> first Claude use datetime

        Returns:
            Complete analysis results
        """
        report = {}

        # Basic adoption rates
        report["adoption_rates"] = self.compute_adoption_rates()

        # Exposure level analysis
        exposure_df = self.compute_influence_by_exposure_level()
        report["exposure_analysis"] = exposure_df.to_dict('records')

        # Time to adoption
        report["timing_analysis"] = self.compute_time_to_adoption(first_use_dates)

        # Summary statistics
        rates = report["adoption_rates"]
        report["summary"] = {
            "finding": f"Collaborators of Claude users adopted at {rates['exposed_adoption_rate']:.2%} rate",
            "exposed_group_size": rates['exposed_collaborators'],
            "exposed_adopters": rates['exposed_adopted'],
        }

        return report


def run_contagion_analysis(
    data_dir: Path,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Run the complete contagion analysis pipeline.

    Args:
        data_dir: Directory containing input data files
        output_dir: Directory for output files

    Returns:
        Analysis results
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CLAUDE CODE CONTAGION ANALYSIS")
    print("=" * 70)
    print(f"\nSnapshot date: {SNAPSHOT_DATE.date()} (identify Claude users)")
    print(f"Outcome date:  {OUTCOME_DATE.date()} (check collaborator adoption)")

    # Step 1: Identify Claude users
    print("\n" + "-" * 50)
    print("Step 1: Identifying Claude Users")
    print("-" * 50)

    identifier = ClaudeUserIdentifier(data_dir / "claude_commits.parquet")
    identifier.load_commits()

    claude_users_may = identifier.get_users_as_of(SNAPSHOT_DATE)
    claude_users_june = identifier.get_users_as_of(OUTCOME_DATE)
    first_use_dates = identifier.get_user_first_use()

    print(f"\n  Claude users by May 2025: {len(claude_users_may):,}")
    print(f"  Claude users by June 2025: {len(claude_users_june):,}")
    print(f"  New adopters in June: {len(claude_users_june - claude_users_may):,}")

    # Step 2: Load collaborator network
    print("\n" + "-" * 50)
    print("Step 2: Loading Collaborator Network")
    print("-" * 50)

    network = CollaboratorNetwork()

    user_repos_path = data_dir / "user_repos.parquet"
    contributors_path = data_dir / "contributors.parquet"

    if not user_repos_path.exists():
        print(f"\n  ERROR: {user_repos_path} not found!")
        print("  Please run the data fetching pipeline first.")
        return {"error": "Missing user_repos.parquet"}

    if not contributors_path.exists():
        print(f"\n  ERROR: {contributors_path} not found!")
        print("  Please run the data fetching pipeline first.")
        return {"error": "Missing contributors.parquet"}

    network.load_user_repos(user_repos_path)
    network.load_contributors(contributors_path)

    # Step 3: Build collaborator map for May Claude users
    print("\n" + "-" * 50)
    print("Step 3: Building Collaborator Map")
    print("-" * 50)

    collaborator_map = network.build_collaborator_map(claude_users_may)
    all_collaborators = network.get_all_collaborators()

    # Step 4: Analyze contagion
    print("\n" + "-" * 50)
    print("Step 4: Analyzing Contagion")
    print("-" * 50)

    analyzer = ContagionAnalyzer(
        claude_users_may=claude_users_may,
        claude_users_june=claude_users_june,
        collaborators=all_collaborators,
        user_collaborator_map=collaborator_map
    )

    report = analyzer.generate_report(first_use_dates)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    rates = report["adoption_rates"]
    print(f"\nClaude Users (May 2025):     {rates['claude_users_may']:,}")
    print(f"Claude Users (June 2025):    {rates['claude_users_june']:,}")
    print(f"New Adopters (June):         {rates['new_adopters_june']:,}")
    print(f"\nTotal Collaborators:         {rates['total_collaborators']:,}")
    print(f"Exposed (not already users): {rates['exposed_collaborators']:,}")
    print(f"Exposed who Adopted:         {rates['exposed_adopted']:,}")
    print(f"\nAdoption Rate (Exposed):     {rates['exposed_adoption_rate']:.2%}")

    print("\n" + "-" * 50)
    print("Adoption Rate by Exposure Level")
    print("-" * 50)
    print(f"{'Exposure':<10} {'Collaborators':<15} {'Adopted':<10} {'Rate':<10}")
    print("-" * 45)
    for row in report["exposure_analysis"][:10]:  # Top 10
        print(f"{row['exposure_level']:<10} {row['num_collaborators']:<15,} {row['num_adopted']:<10,} {row['adoption_rate']:.2%}")

    if "timing_analysis" in report and "error" not in report["timing_analysis"]:
        timing = report["timing_analysis"]
        print("\n" + "-" * 50)
        print("Time to Adoption (days after first Claude-using collaborator)")
        print("-" * 50)
        print(f"Mean:   {timing['mean_gap_days']:.1f} days")
        print(f"Median: {timing['median_gap_days']:.1f} days")
        print(f"Min:    {timing['min_gap_days']} days")
        print(f"Max:    {timing['max_gap_days']} days")

    # Save results
    print("\n" + "-" * 50)
    print("Saving Results")
    print("-" * 50)

    with open(output_dir / "contagion_analysis.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Saved to {output_dir / 'contagion_analysis.json'}")

    # Save exposure analysis as CSV
    exposure_df = pd.DataFrame(report["exposure_analysis"])
    exposure_df.to_csv(output_dir / "exposure_analysis.csv", index=False)
    print(f"  Saved to {output_dir / 'exposure_analysis.csv'}")

    return report


if __name__ == "__main__":
    data_dir = Path("data/output")
    output_dir = Path("analysis/output")

    results = run_contagion_analysis(data_dir, output_dir)
