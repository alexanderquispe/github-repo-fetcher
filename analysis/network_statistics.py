"""
Network Statistics for Collaboration Network Analysis.

Computes descriptive statistics about the collaboration network
and characteristics of early adopters vs. non-adopters.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

# Optional: networkx for advanced metrics
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not installed. Some metrics will be unavailable.")
    print("Install with: pip install networkx")


def load_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all required datasets."""
    print("Loading data...")

    # Claude commits
    commits_df = pd.read_parquet(data_dir / "claude_commits.parquet")
    commits_df['committed_date'] = pd.to_datetime(commits_df['committed_date'], utc=True)
    print(f"  Commits: {len(commits_df):,}")

    # User repos
    user_repos_df = pd.read_parquet(data_dir / "user_repos.parquet")
    print(f"  User repos: {len(user_repos_df):,}")

    # Contributors
    contributors_df = pd.read_parquet(data_dir / "contributors.parquet")
    print(f"  Contributors: {len(contributors_df):,}")

    return commits_df, user_repos_df, contributors_df


def get_claude_users_by_date(commits_df: pd.DataFrame, cutoff_date: str) -> Set[str]:
    """Get Claude users as of a specific date."""
    cutoff = pd.Timestamp(cutoff_date, tz='UTC')
    mask = commits_df['committed_date'] <= cutoff
    return set(commits_df[mask]['author_login'].dropna().unique())


def build_collaboration_graph(
    user_repos_df: pd.DataFrame,
    contributors_df: pd.DataFrame,
    focal_users: Set[str]
) -> 'nx.Graph':
    """
    Build collaboration network as undirected graph.

    Nodes: All users (focal + collaborators)
    Edges: Two users share at least one repository
    Edge weight: Number of shared repositories
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx required for graph analysis")

    print("\nBuilding collaboration graph...")

    # Build repo -> contributors mapping
    repo_contributors = {}
    for _, row in contributors_df.iterrows():
        repo = row['repo_nwo']
        user = row['user_login']
        if pd.notna(repo) and pd.notna(user):
            if repo not in repo_contributors:
                repo_contributors[repo] = set()
            repo_contributors[repo].add(user)

    print(f"  Repos with contributors: {len(repo_contributors):,}")

    # Build user -> repos mapping for focal users
    user_repos = {}
    for _, row in user_repos_df.iterrows():
        user = row['user_login']
        repos = row['repos']
        if isinstance(repos, str):
            repos = eval(repos)
        elif isinstance(repos, np.ndarray):
            repos = list(repos)
        user_repos[user] = set(repos) if repos else set()

    # Create graph
    G = nx.Graph()

    # Add focal users as nodes with attribute
    for user in focal_users:
        G.add_node(user, is_focal=True, is_early_adopter=True)

    # For each focal user, add edges to collaborators
    edge_weights = Counter()

    for focal_user in focal_users:
        user_repo_list = user_repos.get(focal_user, set())

        for repo in user_repo_list:
            repo_contribs = repo_contributors.get(repo, set())

            for collaborator in repo_contribs:
                if collaborator != focal_user:
                    # Add collaborator node if not exists
                    if collaborator not in G:
                        G.add_node(collaborator, is_focal=False, is_early_adopter=False)

                    # Count shared repos
                    edge_key = tuple(sorted([focal_user, collaborator]))
                    edge_weights[edge_key] += 1

    # Add weighted edges
    for (u, v), weight in edge_weights.items():
        G.add_edge(u, v, weight=weight, shared_repos=weight)

    print(f"  Nodes: {G.number_of_nodes():,}")
    print(f"  Edges: {G.number_of_edges():,}")

    return G


def compute_basic_network_stats(G: 'nx.Graph') -> Dict[str, Any]:
    """Compute basic network statistics."""
    print("\nComputing basic network statistics...")

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    stats = {
        "num_nodes": n_nodes,
        "num_edges": n_edges,
        "density": nx.density(G),
        "is_connected": nx.is_connected(G),
    }

    # Connected components
    components = list(nx.connected_components(G))
    stats["num_components"] = len(components)
    stats["largest_component_size"] = len(max(components, key=len))
    stats["largest_component_fraction"] = stats["largest_component_size"] / n_nodes

    print(f"  Nodes: {n_nodes:,}")
    print(f"  Edges: {n_edges:,}")
    print(f"  Density: {stats['density']:.6f}")
    print(f"  Connected: {stats['is_connected']}")
    print(f"  Components: {stats['num_components']:,}")
    print(f"  Largest component: {stats['largest_component_size']:,} ({stats['largest_component_fraction']:.1%})")

    return stats


def compute_degree_statistics(G: 'nx.Graph') -> Dict[str, Any]:
    """Compute degree distribution statistics."""
    print("\nComputing degree statistics...")

    degrees = [d for n, d in G.degree()]

    stats = {
        "degree_mean": np.mean(degrees),
        "degree_median": np.median(degrees),
        "degree_std": np.std(degrees),
        "degree_min": min(degrees),
        "degree_max": max(degrees),
        "degree_25th": np.percentile(degrees, 25),
        "degree_75th": np.percentile(degrees, 75),
        "degree_90th": np.percentile(degrees, 90),
        "degree_99th": np.percentile(degrees, 99),
    }

    print(f"  Mean degree: {stats['degree_mean']:.2f}")
    print(f"  Median degree: {stats['degree_median']:.0f}")
    print(f"  Std degree: {stats['degree_std']:.2f}")
    print(f"  Min/Max: {stats['degree_min']} / {stats['degree_max']}")
    print(f"  25th/75th percentile: {stats['degree_25th']:.0f} / {stats['degree_75th']:.0f}")

    return stats


def compute_centrality_for_groups(
    G: 'nx.Graph',
    early_adopters: Set[str],
    june_adopters: Set[str]
) -> Dict[str, Any]:
    """
    Compare centrality measures between early adopters,
    June adopters (influenced), and non-adopters.
    """
    print("\nComputing centrality measures...")

    # Compute centralities (on largest component for efficiency)
    largest_cc = max(nx.connected_components(G), key=len)
    G_largest = G.subgraph(largest_cc).copy()

    print(f"  Computing on largest component ({len(G_largest):,} nodes)...")

    # Degree centrality (fast)
    degree_cent = nx.degree_centrality(G_largest)

    # Betweenness (slow for large graphs - sample if needed)
    if len(G_largest) > 10000:
        print("  Using sampled betweenness centrality (large graph)...")
        betweenness_cent = nx.betweenness_centrality(G_largest, k=min(1000, len(G_largest)))
    else:
        betweenness_cent = nx.betweenness_centrality(G_largest)

    # Eigenvector centrality (may not converge)
    try:
        eigenvector_cent = nx.eigenvector_centrality(G_largest, max_iter=1000)
    except:
        print("  Eigenvector centrality did not converge, skipping...")
        eigenvector_cent = None

    # Clustering coefficient
    clustering = nx.clustering(G_largest)

    # Categorize nodes
    results = {
        "early_adopters": {"count": 0, "degree": [], "betweenness": [], "clustering": []},
        "june_adopters": {"count": 0, "degree": [], "betweenness": [], "clustering": []},
        "non_adopters": {"count": 0, "degree": [], "betweenness": [], "clustering": []},
    }

    if eigenvector_cent:
        for key in results:
            results[key]["eigenvector"] = []

    for node in G_largest.nodes():
        if node in early_adopters:
            group = "early_adopters"
        elif node in june_adopters:
            group = "june_adopters"
        else:
            group = "non_adopters"

        results[group]["count"] += 1
        results[group]["degree"].append(degree_cent[node])
        results[group]["betweenness"].append(betweenness_cent[node])
        results[group]["clustering"].append(clustering[node])

        if eigenvector_cent:
            results[group]["eigenvector"].append(eigenvector_cent[node])

    # Compute summary statistics for each group
    summary = {}
    for group, data in results.items():
        summary[group] = {
            "count": data["count"],
            "degree_centrality_mean": np.mean(data["degree"]) if data["degree"] else 0,
            "degree_centrality_median": np.median(data["degree"]) if data["degree"] else 0,
            "betweenness_centrality_mean": np.mean(data["betweenness"]) if data["betweenness"] else 0,
            "betweenness_centrality_median": np.median(data["betweenness"]) if data["betweenness"] else 0,
            "clustering_coef_mean": np.mean(data["clustering"]) if data["clustering"] else 0,
            "clustering_coef_median": np.median(data["clustering"]) if data["clustering"] else 0,
        }

        if eigenvector_cent and data.get("eigenvector"):
            summary[group]["eigenvector_centrality_mean"] = np.mean(data["eigenvector"])
            summary[group]["eigenvector_centrality_median"] = np.median(data["eigenvector"])

    # Print comparison
    print("\n  Centrality Comparison by Group:")
    print("  " + "-" * 70)
    print(f"  {'Metric':<30} {'Early Adopters':>15} {'June Adopters':>15} {'Non-Adopters':>15}")
    print("  " + "-" * 70)
    print(f"  {'Count':<30} {summary['early_adopters']['count']:>15,} {summary['june_adopters']['count']:>15,} {summary['non_adopters']['count']:>15,}")
    print(f"  {'Degree Centrality (mean)':<30} {summary['early_adopters']['degree_centrality_mean']:>15.6f} {summary['june_adopters']['degree_centrality_mean']:>15.6f} {summary['non_adopters']['degree_centrality_mean']:>15.6f}")
    print(f"  {'Betweenness Centrality (mean)':<30} {summary['early_adopters']['betweenness_centrality_mean']:>15.6f} {summary['june_adopters']['betweenness_centrality_mean']:>15.6f} {summary['non_adopters']['betweenness_centrality_mean']:>15.6f}")
    print(f"  {'Clustering Coef (mean)':<30} {summary['early_adopters']['clustering_coef_mean']:>15.4f} {summary['june_adopters']['clustering_coef_mean']:>15.4f} {summary['non_adopters']['clustering_coef_mean']:>15.4f}")

    return summary


def compute_adopter_degree_analysis(
    G: 'nx.Graph',
    early_adopters: Set[str],
    june_adopters: Set[str]
) -> pd.DataFrame:
    """
    Analyze degree distribution for different groups.
    """
    print("\nAnalyzing degree by adoption status...")

    records = []
    for node in G.nodes():
        degree = G.degree(node)

        if node in early_adopters:
            status = "early_adopter"
        elif node in june_adopters:
            status = "june_adopter"
        else:
            status = "non_adopter"

        records.append({
            "user": node,
            "degree": degree,
            "status": status
        })

    df = pd.DataFrame(records)

    # Summary by status
    summary = df.groupby('status')['degree'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
    print("\n  Degree by Adoption Status:")
    print(summary.to_string())

    return df


def compute_edge_weight_statistics(G: 'nx.Graph') -> Dict[str, Any]:
    """Analyze edge weights (shared repos)."""
    print("\nAnalyzing edge weights (shared repos)...")

    weights = [d.get('weight', 1) for u, v, d in G.edges(data=True)]

    stats = {
        "edge_weight_mean": np.mean(weights),
        "edge_weight_median": np.median(weights),
        "edge_weight_max": max(weights),
        "edges_with_1_shared": sum(1 for w in weights if w == 1),
        "edges_with_2plus_shared": sum(1 for w in weights if w >= 2),
        "edges_with_5plus_shared": sum(1 for w in weights if w >= 5),
    }

    print(f"  Mean shared repos per edge: {stats['edge_weight_mean']:.2f}")
    print(f"  Median shared repos: {stats['edge_weight_median']:.0f}")
    print(f"  Max shared repos: {stats['edge_weight_max']}")
    print(f"  Edges with 1 shared repo: {stats['edges_with_1_shared']:,}")
    print(f"  Edges with 2+ shared repos: {stats['edges_with_2plus_shared']:,}")
    print(f"  Edges with 5+ shared repos: {stats['edges_with_5plus_shared']:,}")

    return stats


def run_network_analysis(data_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """Run complete network analysis."""

    print("=" * 70)
    print("NETWORK STATISTICS ANALYSIS")
    print("=" * 70)

    # Load data
    commits_df, user_repos_df, contributors_df = load_data(data_dir)

    # Get user groups
    early_adopters = get_claude_users_by_date(commits_df, "2025-05-31 23:59:59")
    june_adopters = get_claude_users_by_date(commits_df, "2025-06-30 23:59:59") - early_adopters

    print(f"\nUser Groups:")
    print(f"  Early adopters (by May 2025): {len(early_adopters):,}")
    print(f"  June adopters (Jun 2025): {len(june_adopters):,}")

    if not HAS_NETWORKX:
        print("\nERROR: networkx required for network analysis")
        print("Install with: pip install networkx")
        return {"error": "networkx not installed"}

    # Build graph
    G = build_collaboration_graph(user_repos_df, contributors_df, early_adopters)

    # Compute statistics
    results = {}

    # Basic stats
    results["basic"] = compute_basic_network_stats(G)

    # Degree stats
    results["degree"] = compute_degree_statistics(G)

    # Edge weight stats
    results["edge_weights"] = compute_edge_weight_statistics(G)

    # Centrality by group
    results["centrality_by_group"] = compute_centrality_for_groups(G, early_adopters, june_adopters)

    # Degree analysis
    degree_df = compute_adopter_degree_analysis(G, early_adopters, june_adopters)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "network_statistics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {output_dir / 'network_statistics.json'}")

    degree_df.to_csv(output_dir / "degree_by_status.csv", index=False)
    print(f"Saved to {output_dir / 'degree_by_status.csv'}")

    return results


if __name__ == "__main__":
    data_dir = Path("data/output")
    output_dir = Path("analysis/output")

    results = run_network_analysis(data_dir, output_dir)
