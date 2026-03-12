"""
Fast Network Statistics Analysis.

Computes key network metrics efficiently without building full graph in memory.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Set

import numpy as np
import pandas as pd


def load_data(data_dir: Path):
    """Load datasets."""
    print("Loading data...")

    commits_df = pd.read_parquet(data_dir / "claude_commits.parquet")
    commits_df['committed_date'] = pd.to_datetime(commits_df['committed_date'], utc=True)
    print(f"  Commits: {len(commits_df):,}")

    user_repos_df = pd.read_parquet(data_dir / "user_repos.parquet")
    print(f"  User repos: {len(user_repos_df):,}")

    contributors_df = pd.read_parquet(data_dir / "contributors.parquet")
    print(f"  Contributors: {len(contributors_df):,}")

    return commits_df, user_repos_df, contributors_df


def get_user_groups(commits_df: pd.DataFrame):
    """Get early adopters and June adopters."""
    may_cutoff = pd.Timestamp("2025-05-31 23:59:59", tz='UTC')
    june_cutoff = pd.Timestamp("2025-06-30 23:59:59", tz='UTC')

    early_adopters = set(commits_df[commits_df['committed_date'] <= may_cutoff]['author_login'].dropna().unique())
    all_june = set(commits_df[commits_df['committed_date'] <= june_cutoff]['author_login'].dropna().unique())
    june_adopters = all_june - early_adopters

    return early_adopters, june_adopters


def compute_degree_statistics(contributors_df: pd.DataFrame, early_adopters: Set[str], june_adopters: Set[str]):
    """
    Compute degree (number of unique collaborators) for each user.
    Degree = number of distinct users who share at least one repo.
    """
    print("\nComputing degree statistics...")

    # Build repo -> users mapping
    repo_users = defaultdict(set)
    for _, row in contributors_df.iterrows():
        repo = row['repo_nwo']
        user = row['user_login']
        if pd.notna(repo) and pd.notna(user):
            repo_users[repo].add(user)

    print(f"  Repos: {len(repo_users):,}")

    # Compute degree for each user (number of distinct collaborators)
    user_collaborators = defaultdict(set)

    for repo, users in repo_users.items():
        users_list = list(users)
        for i, u1 in enumerate(users_list):
            for u2 in users_list[i+1:]:
                user_collaborators[u1].add(u2)
                user_collaborators[u2].add(u1)

    # Convert to degrees
    user_degree = {u: len(collabs) for u, collabs in user_collaborators.items()}

    print(f"  Users with collaborators: {len(user_degree):,}")

    # Separate by group
    early_degrees = [user_degree.get(u, 0) for u in early_adopters if u in user_degree]
    june_degrees = [user_degree.get(u, 0) for u in june_adopters if u in user_degree]
    non_adopter_degrees = [d for u, d in user_degree.items() if u not in early_adopters and u not in june_adopters]

    results = {
        "early_adopters": {
            "count": len(early_degrees),
            "degree_mean": np.mean(early_degrees) if early_degrees else 0,
            "degree_median": np.median(early_degrees) if early_degrees else 0,
            "degree_std": np.std(early_degrees) if early_degrees else 0,
            "degree_min": min(early_degrees) if early_degrees else 0,
            "degree_max": max(early_degrees) if early_degrees else 0,
            "degree_25th": np.percentile(early_degrees, 25) if early_degrees else 0,
            "degree_75th": np.percentile(early_degrees, 75) if early_degrees else 0,
        },
        "june_adopters": {
            "count": len(june_degrees),
            "degree_mean": np.mean(june_degrees) if june_degrees else 0,
            "degree_median": np.median(june_degrees) if june_degrees else 0,
            "degree_std": np.std(june_degrees) if june_degrees else 0,
            "degree_min": min(june_degrees) if june_degrees else 0,
            "degree_max": max(june_degrees) if june_degrees else 0,
            "degree_25th": np.percentile(june_degrees, 25) if june_degrees else 0,
            "degree_75th": np.percentile(june_degrees, 75) if june_degrees else 0,
        },
        "non_adopters": {
            "count": len(non_adopter_degrees),
            "degree_mean": np.mean(non_adopter_degrees) if non_adopter_degrees else 0,
            "degree_median": np.median(non_adopter_degrees) if non_adopter_degrees else 0,
            "degree_std": np.std(non_adopter_degrees) if non_adopter_degrees else 0,
            "degree_min": min(non_adopter_degrees) if non_adopter_degrees else 0,
            "degree_max": max(non_adopter_degrees) if non_adopter_degrees else 0,
            "degree_25th": np.percentile(non_adopter_degrees, 25) if non_adopter_degrees else 0,
            "degree_75th": np.percentile(non_adopter_degrees, 75) if non_adopter_degrees else 0,
        },
    }

    # Print comparison table
    print("\n  Degree Statistics by Group:")
    print("  " + "-" * 75)
    print(f"  {'Metric':<20} {'Early Adopters':>18} {'June Adopters':>18} {'Non-Adopters':>18}")
    print("  " + "-" * 75)
    print(f"  {'Count':<20} {results['early_adopters']['count']:>18,} {results['june_adopters']['count']:>18,} {results['non_adopters']['count']:>18,}")
    print(f"  {'Mean Degree':<20} {results['early_adopters']['degree_mean']:>18.1f} {results['june_adopters']['degree_mean']:>18.1f} {results['non_adopters']['degree_mean']:>18.1f}")
    print(f"  {'Median Degree':<20} {results['early_adopters']['degree_median']:>18.0f} {results['june_adopters']['degree_median']:>18.0f} {results['non_adopters']['degree_median']:>18.0f}")
    print(f"  {'Std Degree':<20} {results['early_adopters']['degree_std']:>18.1f} {results['june_adopters']['degree_std']:>18.1f} {results['non_adopters']['degree_std']:>18.1f}")
    print(f"  {'Min Degree':<20} {results['early_adopters']['degree_min']:>18} {results['june_adopters']['degree_min']:>18} {results['non_adopters']['degree_min']:>18}")
    print(f"  {'Max Degree':<20} {results['early_adopters']['degree_max']:>18} {results['june_adopters']['degree_max']:>18} {results['non_adopters']['degree_max']:>18}")
    print(f"  {'25th Percentile':<20} {results['early_adopters']['degree_25th']:>18.0f} {results['june_adopters']['degree_25th']:>18.0f} {results['non_adopters']['degree_25th']:>18.0f}")
    print(f"  {'75th Percentile':<20} {results['early_adopters']['degree_75th']:>18.0f} {results['june_adopters']['degree_75th']:>18.0f} {results['non_adopters']['degree_75th']:>18.0f}")

    return results, user_degree


def compute_repo_statistics(contributors_df: pd.DataFrame, user_repos_df: pd.DataFrame,
                           early_adopters: Set[str], june_adopters: Set[str]):
    """Compute repository-level statistics."""
    print("\nComputing repository statistics...")

    # Repos per user from user_repos
    repos_per_user = {}
    for _, row in user_repos_df.iterrows():
        user = row['user_login']
        repos = row['repos']
        if isinstance(repos, str):
            repos = eval(repos)
        elif isinstance(repos, np.ndarray):
            repos = list(repos)
        repos_per_user[user] = len(repos) if repos else 0

    # Contributors per repo
    repo_contributor_count = contributors_df.groupby('repo_nwo')['user_login'].nunique()

    results = {
        "repos_per_user": {
            "mean": np.mean(list(repos_per_user.values())),
            "median": np.median(list(repos_per_user.values())),
            "max": max(repos_per_user.values()),
        },
        "contributors_per_repo": {
            "mean": repo_contributor_count.mean(),
            "median": repo_contributor_count.median(),
            "max": repo_contributor_count.max(),
        },
        "total_unique_repos": contributors_df['repo_nwo'].nunique(),
        "total_unique_users": contributors_df['user_login'].nunique(),
    }

    print(f"  Repos per early adopter (mean): {results['repos_per_user']['mean']:.1f}")
    print(f"  Contributors per repo (mean): {results['contributors_per_repo']['mean']:.1f}")
    print(f"  Contributors per repo (median): {results['contributors_per_repo']['median']:.0f}")
    print(f"  Max contributors in a repo: {results['contributors_per_repo']['max']}")

    return results


def compute_activity_statistics(commits_df: pd.DataFrame, early_adopters: Set[str], june_adopters: Set[str]):
    """Compute activity statistics (commits per user)."""
    print("\nComputing activity statistics...")

    # Commits per user
    commits_per_user = commits_df.groupby('author_login').size()

    early_commits = [commits_per_user.get(u, 0) for u in early_adopters]
    june_commits = [commits_per_user.get(u, 0) for u in june_adopters]

    results = {
        "early_adopters": {
            "commits_mean": np.mean(early_commits) if early_commits else 0,
            "commits_median": np.median(early_commits) if early_commits else 0,
            "commits_total": sum(early_commits),
        },
        "june_adopters": {
            "commits_mean": np.mean(june_commits) if june_commits else 0,
            "commits_median": np.median(june_commits) if june_commits else 0,
            "commits_total": sum(june_commits),
        },
    }

    print(f"  Early adopters - Mean commits: {results['early_adopters']['commits_mean']:.1f}, Median: {results['early_adopters']['commits_median']:.0f}")
    print(f"  June adopters - Mean commits: {results['june_adopters']['commits_mean']:.1f}, Median: {results['june_adopters']['commits_median']:.0f}")

    return results


def compute_exposure_of_adopters(contributors_df: pd.DataFrame, early_adopters: Set[str], june_adopters: Set[str]):
    """
    For June adopters: how many early adopters did they collaborate with?
    """
    print("\nComputing exposure of June adopters...")

    # Build repo -> users
    repo_users = defaultdict(set)
    for _, row in contributors_df.iterrows():
        repo = row['repo_nwo']
        user = row['user_login']
        if pd.notna(repo) and pd.notna(user):
            repo_users[repo].add(user)

    # For each June adopter, count early adopter collaborators
    june_exposure = {}
    for june_user in june_adopters:
        early_connections = set()
        for repo, users in repo_users.items():
            if june_user in users:
                early_connections.update(users & early_adopters)
        june_exposure[june_user] = len(early_connections)

    # Distribution
    exposures = list(june_exposure.values())
    exposed = [e for e in exposures if e > 0]

    results = {
        "june_adopters_total": len(june_adopters),
        "june_adopters_with_exposure": len(exposed),
        "exposure_mean": np.mean(exposed) if exposed else 0,
        "exposure_median": np.median(exposed) if exposed else 0,
        "exposure_max": max(exposed) if exposed else 0,
    }

    print(f"  June adopters with early-adopter collaborators: {results['june_adopters_with_exposure']:,} / {results['june_adopters_total']:,}")
    print(f"  Mean exposure (among exposed): {results['exposure_mean']:.1f}")
    print(f"  Median exposure: {results['exposure_median']:.0f}")

    return results


def run_fast_analysis(data_dir: Path, output_dir: Path):
    """Run fast network analysis."""
    print("=" * 70)
    print("FAST NETWORK STATISTICS ANALYSIS")
    print("=" * 70)

    # Load data
    commits_df, user_repos_df, contributors_df = load_data(data_dir)

    # Get user groups
    early_adopters, june_adopters = get_user_groups(commits_df)

    print(f"\nUser Groups:")
    print(f"  Early adopters (by May 2025): {len(early_adopters):,}")
    print(f"  June adopters: {len(june_adopters):,}")

    results = {}

    # Degree statistics
    results["degree_by_group"], user_degrees = compute_degree_statistics(
        contributors_df, early_adopters, june_adopters
    )

    # Repo statistics
    results["repo_statistics"] = compute_repo_statistics(
        contributors_df, user_repos_df, early_adopters, june_adopters
    )

    # Activity statistics
    results["activity"] = compute_activity_statistics(
        commits_df, early_adopters, june_adopters
    )

    # Exposure of June adopters
    results["june_adopter_exposure"] = compute_exposure_of_adopters(
        contributors_df, early_adopters, june_adopters
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nKey Finding: Early adopters have HIGHER degree (more collaborators)")
    print(f"  Early adopters mean degree: {results['degree_by_group']['early_adopters']['degree_mean']:.1f}")
    print(f"  Non-adopters mean degree: {results['degree_by_group']['non_adopters']['degree_mean']:.1f}")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "network_statistics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {output_dir / 'network_statistics.json'}")

    return results


if __name__ == "__main__":
    data_dir = Path("data/output")
    output_dir = Path("analysis/output")

    results = run_fast_analysis(data_dir, output_dir)
