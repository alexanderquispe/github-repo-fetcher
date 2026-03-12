"""
Enhanced Collaboration Network Builder for AI Tool Adoption Analysis.

This module provides advanced network construction utilities that combine
multiple data sources (commits, PRs, contributors) to build weighted
collaboration networks.
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not installed. Run: pip install networkx")


class CollaborationNetworkBuilder:
    """
    Builds weighted developer collaboration networks from multiple data sources.

    Edge Types (by priority):
    1. Shared Repository: Both contributed to same repo
    2. Co-authorship: Explicit co-author relationships
    3. Contributors: Both appear in repo's contributor list
    """

    def __init__(self):
        if not HAS_NETWORKX:
            raise ImportError("networkx is required. Install with: pip install networkx")

        self.G = nx.Graph()
        self.repo_contributors: Dict[str, Set[str]] = defaultdict(set)
        self.edge_sources: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def build_from_contributors(
        self,
        contributors_df: pd.DataFrame,
        repo_col: str = "repo_nwo",
        user_col: str = "user_login",
        weight_col: Optional[str] = "contributions"
    ) -> "CollaborationNetworkBuilder":
        """
        Build network edges from contributor data.

        Args:
            contributors_df: DataFrame with contributor data
            repo_col: Column for repository identifier
            user_col: Column for user login
            weight_col: Optional column for contribution weight

        Returns:
            self for method chaining
        """
        print(f"Building from {len(contributors_df):,} contributor records...")

        # Group contributors by repo
        repo_users: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

        for _, row in contributors_df.iterrows():
            user = row[user_col]
            repo = row[repo_col]
            weight = row.get(weight_col, 1) if weight_col else 1

            if pd.notna(user) and pd.notna(repo):
                repo_users[repo].append((user, weight))
                self.repo_contributors[repo].add(user)

        # Create edges between all contributors of each repo
        for repo, contributors in repo_users.items():
            for i, (user1, w1) in enumerate(contributors):
                for user2, w2 in contributors[i+1:]:
                    edge = tuple(sorted([user1, user2]))
                    # Weight by minimum contributions (both had to contribute)
                    min_contrib = min(w1, w2)
                    self.edge_sources[edge]["contributors"] += min_contrib

        print(f"  Repos with contributors: {len(repo_users):,}")

        return self

    def build_from_shared_repos(
        self,
        df: pd.DataFrame,
        user_col: str = "author_login",
        repo_col: str = "repo_nwo"
    ) -> "CollaborationNetworkBuilder":
        """
        Build network edges from shared repository contributions.

        Args:
            df: DataFrame with commit or PR data
            user_col: Column for user login
            repo_col: Column for repository identifier

        Returns:
            self for method chaining
        """
        print(f"Building from {len(df):,} records...")

        # Group by repo
        for _, row in df.iterrows():
            user = row[user_col]
            repo = row[repo_col]
            if pd.notna(user) and pd.notna(repo):
                self.repo_contributors[repo].add(user)

        # Create edges
        for repo, users in self.repo_contributors.items():
            users = list(users)
            for i, user1 in enumerate(users):
                for user2 in users[i+1:]:
                    edge = tuple(sorted([user1, user2]))
                    self.edge_sources[edge]["shared_repo"] += 1

        print(f"  Repos: {len(self.repo_contributors):,}")

        return self

    def build_from_coauthorship(
        self,
        df: pd.DataFrame,
        author_col: str = "author_login",
        coauthor_col: str = "coauthor_login"
    ) -> "CollaborationNetworkBuilder":
        """
        Build network edges from explicit co-authorship relationships.

        Args:
            df: DataFrame with co-authorship data
            author_col: Column for primary author
            coauthor_col: Column for co-author

        Returns:
            self for method chaining
        """
        print(f"Building from {len(df):,} co-authorship records...")

        count = 0
        for _, row in df.iterrows():
            author = row[author_col]
            coauthor = row[coauthor_col]

            if pd.notna(author) and pd.notna(coauthor) and author != coauthor:
                edge = tuple(sorted([author, coauthor]))
                self.edge_sources[edge]["coauthor"] += 1
                count += 1

        print(f"  Co-author edges: {count:,}")

        return self

    def combine_networks(
        self,
        weights: Optional[Dict[str, float]] = None
    ) -> nx.Graph:
        """
        Combine all edge sources into final weighted network.

        Args:
            weights: Optional dict mapping edge source -> weight multiplier
                    Default: {"coauthor": 3.0, "shared_repo": 1.0, "contributors": 0.5}

        Returns:
            Combined weighted NetworkX graph
        """
        if weights is None:
            weights = {
                "coauthor": 3.0,      # Strongest signal
                "shared_repo": 1.0,    # Direct collaboration
                "contributors": 0.5,   # Weaker signal (could be large repos)
            }

        self.G = nx.Graph()

        for edge, sources in self.edge_sources.items():
            user1, user2 = edge

            # Calculate total weighted edge weight
            total_weight = 0
            for source, count in sources.items():
                multiplier = weights.get(source, 1.0)
                total_weight += count * multiplier

            if total_weight > 0:
                self.G.add_edge(
                    user1, user2,
                    weight=total_weight,
                    sources=dict(sources)
                )

        print(f"Combined network: {self.G.number_of_nodes():,} nodes, "
              f"{self.G.number_of_edges():,} edges")

        return self.G

    def add_tool_adoption_attributes(
        self,
        adoption_data: Dict[str, Dict[str, datetime]]
    ) -> nx.Graph:
        """
        Add tool adoption attributes to nodes.

        Args:
            adoption_data: Dict mapping tool -> {user -> first_use_datetime}

        Returns:
            Graph with updated node attributes
        """
        for tool, user_times in adoption_data.items():
            for user, adopt_time in user_times.items():
                if user in self.G:
                    self.G.nodes[user][f"{tool}_adopted"] = True
                    self.G.nodes[user][f"{tool}_first_use"] = adopt_time.isoformat() if adopt_time else None

        # Add summary attributes
        for node in self.G.nodes():
            tools = []
            for tool in adoption_data.keys():
                if self.G.nodes[node].get(f"{tool}_adopted"):
                    tools.append(tool)
            self.G.nodes[node]["tools"] = tools
            self.G.nodes[node]["num_tools"] = len(tools)

        return self.G

    def filter_by_min_weight(self, min_weight: float = 1.0) -> nx.Graph:
        """
        Filter edges to keep only those above minimum weight.

        Args:
            min_weight: Minimum edge weight to keep

        Returns:
            Filtered graph
        """
        edges_to_remove = [
            (u, v) for u, v, w in self.G.edges(data="weight")
            if w < min_weight
        ]
        self.G.remove_edges_from(edges_to_remove)

        # Remove isolated nodes
        isolates = list(nx.isolates(self.G))
        self.G.remove_nodes_from(isolates)

        print(f"After filtering (min_weight={min_weight}): "
              f"{self.G.number_of_nodes():,} nodes, {self.G.number_of_edges():,} edges")

        return self.G

    def get_largest_component(self) -> nx.Graph:
        """
        Get the largest connected component.

        Returns:
            Subgraph of largest component
        """
        if self.G.number_of_nodes() == 0:
            return self.G

        largest_cc = max(nx.connected_components(self.G), key=len)
        return self.G.subgraph(largest_cc).copy()

    def compute_network_statistics(self) -> Dict[str, Any]:
        """
        Compute comprehensive network statistics.

        Returns:
            Dict with network metrics
        """
        if self.G.number_of_nodes() == 0:
            return {"error": "Empty graph"}

        stats = {
            "nodes": self.G.number_of_nodes(),
            "edges": self.G.number_of_edges(),
            "density": nx.density(self.G),
        }

        # Degree statistics
        degrees = [d for n, d in self.G.degree()]
        stats["avg_degree"] = np.mean(degrees)
        stats["median_degree"] = np.median(degrees)
        stats["max_degree"] = max(degrees)
        stats["min_degree"] = min(degrees)

        # Connected components
        components = list(nx.connected_components(self.G))
        stats["num_components"] = len(components)
        stats["largest_component_size"] = len(max(components, key=len))
        stats["largest_component_fraction"] = stats["largest_component_size"] / stats["nodes"]

        # Clustering coefficient (sample if large)
        if self.G.number_of_nodes() < 10000:
            stats["avg_clustering"] = nx.average_clustering(self.G)
        else:
            sample_nodes = list(self.G.nodes())[:5000]
            stats["avg_clustering_sample"] = nx.average_clustering(
                self.G, nodes=sample_nodes
            )

        # Edge weight statistics
        weights = [d["weight"] for _, _, d in self.G.edges(data=True)]
        if weights:
            stats["avg_edge_weight"] = np.mean(weights)
            stats["median_edge_weight"] = np.median(weights)
            stats["max_edge_weight"] = max(weights)

        # Diameter of largest component (expensive, only for small graphs)
        if stats["largest_component_size"] < 5000:
            largest = self.get_largest_component()
            try:
                stats["diameter"] = nx.diameter(largest)
                stats["avg_path_length"] = nx.average_shortest_path_length(largest)
            except nx.NetworkXError:
                pass

        return stats

    def export_to_gexf(self, output_path: Path) -> None:
        """
        Export network to GEXF format for visualization.

        Args:
            output_path: Path for output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert complex attributes to strings for GEXF
        G_export = self.G.copy()
        for node in G_export.nodes():
            for key, value in list(G_export.nodes[node].items()):
                if isinstance(value, (list, dict)):
                    G_export.nodes[node][key] = json.dumps(value)
                elif isinstance(value, datetime):
                    G_export.nodes[node][key] = value.isoformat()

        for u, v in G_export.edges():
            for key, value in list(G_export.edges[u, v].items()):
                if isinstance(value, dict):
                    G_export.edges[u, v][key] = json.dumps(value)

        nx.write_gexf(G_export, str(output_path))
        print(f"Exported network to {output_path}")


def build_full_collaboration_network(data_dir: Path) -> Tuple[nx.Graph, Dict]:
    """
    Build full collaboration network from all available data sources.

    Args:
        data_dir: Directory containing data files

    Returns:
        Tuple of (NetworkX graph, statistics dict)
    """
    data_dir = Path(data_dir)
    builder = CollaborationNetworkBuilder()

    # Load and process each data source
    print("\n" + "=" * 60)
    print("BUILDING COLLABORATION NETWORK")
    print("=" * 60)

    # 1. Contributors data (if available)
    contributors_path = data_dir / "contributors.parquet"
    if contributors_path.exists():
        print("\n1. Loading contributors data...")
        contributors_df = pd.read_parquet(contributors_path)
        builder.build_from_contributors(contributors_df)
    else:
        print("\n1. Contributors data not found, skipping...")

    # 2. Claude commits
    claude_path = data_dir / "claude_commits_full.parquet"
    if claude_path.exists():
        print("\n2. Loading Claude commits...")
        claude_df = pd.read_parquet(claude_path)
        builder.build_from_shared_repos(claude_df, user_col="author_login")
    else:
        print("\n2. Claude commits not found, skipping...")

    # 3. Codex PRs
    codex_path = data_dir / "codex_prs.parquet"
    if codex_path.exists():
        print("\n3. Loading Codex PRs...")
        codex_df = pd.read_parquet(codex_path)
        builder.build_from_shared_repos(codex_df, user_col="user_login")
    else:
        print("\n3. Codex PRs not found, skipping...")

    # 4. Copilot PRs
    copilot_path = data_dir / "copilot_prs.parquet"
    if copilot_path.exists():
        print("\n4. Loading Copilot PRs...")
        copilot_df = pd.read_parquet(copilot_path)
        builder.build_from_shared_repos(copilot_df, user_col="user_login")
    else:
        print("\n4. Copilot PRs not found, skipping...")

    # 5. Jules commits
    jules_path = data_dir / "jules_commits.parquet"
    if jules_path.exists():
        print("\n5. Loading Jules commits...")
        jules_df = pd.read_parquet(jules_path)
        # Jules uses committer_login for the human developer
        if "committer_login" in jules_df.columns:
            builder.build_from_shared_repos(jules_df, user_col="committer_login")
        elif "author_login" in jules_df.columns:
            builder.build_from_shared_repos(jules_df, user_col="author_login")
    else:
        print("\n5. Jules commits not found, skipping...")

    # Combine all sources
    print("\n" + "-" * 40)
    print("Combining all edge sources...")
    G = builder.combine_networks()

    # Compute statistics
    print("\n" + "-" * 40)
    print("Computing network statistics...")
    stats = builder.compute_network_statistics()

    return G, stats


if __name__ == "__main__":
    data_dir = Path("data/output")

    # Build network
    G, stats = build_full_collaboration_network(data_dir)

    # Print statistics
    print("\n" + "=" * 60)
    print("NETWORK STATISTICS")
    print("=" * 60)

    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value:,}")

    # Export
    output_dir = Path("analysis/output")
    output_dir.mkdir(exist_ok=True)

    builder = CollaborationNetworkBuilder()
    builder.G = G
    builder.export_to_gexf(output_dir / "collaboration_network.gexf")

    # Save statistics
    import json
    with open(output_dir / "network_statistics.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"\nSaved statistics to {output_dir / 'network_statistics.json'}")
