"""
Network Analysis Starter Code for AI Coding Tool Adoption Study.

This module provides utilities for building developer collaboration networks
from the AI coding tool dataset and analyzing cascade/diffusion dynamics.

Usage:
    from network_builder import DeveloperNetworkBuilder, CascadeAnalyzer

    # Build network from Claude commits
    builder = DeveloperNetworkBuilder()
    G = builder.build_from_commits("data/output/claude_commits.parquet")

    # Analyze cascades
    analyzer = CascadeAnalyzer(G)
    cascades = analyzer.detect_cascades(adoption_times)
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path

# Try to import network libraries
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not installed. Run: pip install networkx")


class DeveloperNetworkBuilder:
    """
    Builds developer collaboration networks from GitHub contribution data.

    A collaboration edge exists between two developers if they have both
    contributed to the same repository.
    """

    def __init__(self):
        if not HAS_NETWORKX:
            raise ImportError("networkx is required. Install with: pip install networkx")
        self.G = nx.Graph()
        self.repo_contributors: Dict[str, Set[str]] = defaultdict(set)

    def build_from_commits(
        self,
        filepath: str,
        user_col: str = "author_login",
        repo_col: str = "repo_nwo",
        min_shared_repos: int = 1
    ) -> nx.Graph:
        """
        Build collaboration network from commit data.

        Args:
            filepath: Path to parquet/csv file with commits
            user_col: Column name for developer username
            repo_col: Column name for repository identifier
            min_shared_repos: Minimum shared repos to create edge

        Returns:
            NetworkX graph with developers as nodes
        """
        # Load data
        if filepath.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        else:
            df = pd.read_csv(filepath)

        print(f"Loaded {len(df):,} records from {filepath}")

        # Build repo -> contributors mapping
        for _, row in df.iterrows():
            user = row[user_col]
            repo = row[repo_col]
            if pd.notna(user) and pd.notna(repo):
                self.repo_contributors[repo].add(user)

        print(f"Found {len(self.repo_contributors):,} repositories")

        # Build collaboration edges
        edge_weights = defaultdict(int)

        for repo, contributors in self.repo_contributors.items():
            contributors = list(contributors)
            for i, dev1 in enumerate(contributors):
                for dev2 in contributors[i+1:]:
                    edge = tuple(sorted([dev1, dev2]))
                    edge_weights[edge] += 1

        # Create graph
        self.G = nx.Graph()
        for (dev1, dev2), weight in edge_weights.items():
            if weight >= min_shared_repos:
                self.G.add_edge(dev1, dev2, weight=weight)

        print(f"Built network with {self.G.number_of_nodes():,} nodes "
              f"and {self.G.number_of_edges():,} edges")

        return self.G

    def build_from_prs(
        self,
        filepath: str,
        user_col: str = "user_login",
        repo_col: str = "repo_nwo",
        min_shared_repos: int = 1
    ) -> nx.Graph:
        """Build collaboration network from PR data (Codex/Copilot)."""
        return self.build_from_commits(filepath, user_col, repo_col, min_shared_repos)

    def add_node_attributes(self, attributes: Dict[str, Dict]):
        """
        Add attributes to nodes.

        Args:
            attributes: Dict mapping node -> {attr_name: value}
        """
        for node, attrs in attributes.items():
            if node in self.G:
                for key, value in attrs.items():
                    self.G.nodes[node][key] = value

    def get_network_stats(self) -> Dict:
        """Compute basic network statistics."""
        if self.G.number_of_nodes() == 0:
            return {}

        stats = {
            "nodes": self.G.number_of_nodes(),
            "edges": self.G.number_of_edges(),
            "density": nx.density(self.G),
            "avg_degree": sum(dict(self.G.degree()).values()) / self.G.number_of_nodes(),
        }

        # Connected components
        components = list(nx.connected_components(self.G))
        stats["num_components"] = len(components)
        stats["largest_component_size"] = len(max(components, key=len))
        stats["largest_component_fraction"] = stats["largest_component_size"] / stats["nodes"]

        # Clustering (sample if large)
        if self.G.number_of_nodes() < 10000:
            stats["avg_clustering"] = nx.average_clustering(self.G)
        else:
            sample_nodes = list(self.G.nodes())[:5000]
            stats["avg_clustering_sample"] = nx.average_clustering(self.G, nodes=sample_nodes)

        return stats


class AdoptionTimeExtractor:
    """
    Extracts first adoption time for each developer from contribution data.
    """

    @staticmethod
    def from_commits(
        filepath: str,
        user_col: str = "author_login",
        time_col: str = "committed_date"
    ) -> Dict[str, datetime]:
        """
        Extract first commit time (adoption time) for each developer.

        Returns:
            Dict mapping username -> first commit datetime
        """
        if filepath.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        else:
            df = pd.read_csv(filepath)

        # Parse dates
        df[time_col] = pd.to_datetime(df[time_col])

        # Get first commit per user
        adoption_times = df.groupby(user_col)[time_col].min().to_dict()

        print(f"Extracted adoption times for {len(adoption_times):,} developers")
        return adoption_times

    @staticmethod
    def from_prs(
        filepath: str,
        user_col: str = "user_login",
        time_col: str = "created_at"
    ) -> Dict[str, datetime]:
        """Extract first PR time (adoption time) for each developer."""
        return AdoptionTimeExtractor.from_commits(filepath, user_col, time_col)


class CascadeAnalyzer:
    """
    Analyzes adoption cascades through the developer network.

    Implements cascade detection and analysis for studying how AI tool
    adoption spreads through collaboration networks.
    """

    def __init__(self, G: nx.Graph):
        """
        Initialize with a collaboration network.

        Args:
            G: NetworkX graph representing developer collaborations
        """
        self.G = G
        self.cascades: Dict[str, Optional[str]] = {}  # adopter -> parent

    def detect_cascades(
        self,
        adoption_times: Dict[str, datetime],
        max_time_window_days: int = 30
    ) -> Dict[str, Optional[str]]:
        """
        Detect likely cascade relationships.

        For each adopter, identifies the most likely "parent" - the neighbor
        who adopted most recently before them (within time window).

        Args:
            adoption_times: Dict mapping username -> adoption datetime
            max_time_window_days: Max days between parent and child adoption

        Returns:
            Dict mapping each adopter -> their likely parent (or None if seed)
        """
        # Filter to users in the network
        network_users = set(self.G.nodes())
        valid_adoptions = {
            user: time for user, time in adoption_times.items()
            if user in network_users
        }

        print(f"Analyzing {len(valid_adoptions):,} adoptions in network")

        # Sort by adoption time
        sorted_adoptions = sorted(valid_adoptions.items(), key=lambda x: x[1])

        adopted = set()
        self.cascades = {}

        for user, adopt_time in sorted_adoptions:
            # Find neighbors who adopted before this user
            neighbors = set(self.G.neighbors(user))
            prior_adopters = neighbors & adopted

            if prior_adopters:
                # Find most recent prior adopter within time window
                best_parent = None
                best_time = None

                for neighbor in prior_adopters:
                    neighbor_time = valid_adoptions[neighbor]
                    time_diff = (adopt_time - neighbor_time).days

                    if time_diff <= max_time_window_days:
                        if best_time is None or neighbor_time > best_time:
                            best_parent = neighbor
                            best_time = neighbor_time

                self.cascades[user] = best_parent
            else:
                # Seed node (no prior adopting neighbors)
                self.cascades[user] = None

            adopted.add(user)

        # Statistics
        seeds = sum(1 for p in self.cascades.values() if p is None)
        influenced = len(self.cascades) - seeds

        print(f"Detected {seeds:,} seed nodes and {influenced:,} influenced adoptions")

        return self.cascades

    def get_cascade_trees(self) -> List[nx.DiGraph]:
        """
        Build cascade trees from detected cascades.

        Returns:
            List of directed graphs, each representing a cascade tree
        """
        # Build directed graph of influence
        influence_graph = nx.DiGraph()
        for child, parent in self.cascades.items():
            influence_graph.add_node(child)
            if parent is not None:
                influence_graph.add_edge(parent, child)

        # Find connected components (each is a cascade tree)
        trees = []
        for component in nx.weakly_connected_components(influence_graph):
            tree = influence_graph.subgraph(component).copy()
            trees.append(tree)

        # Sort by size
        trees.sort(key=lambda t: t.number_of_nodes(), reverse=True)

        print(f"Found {len(trees):,} cascade trees")
        print(f"Largest tree: {trees[0].number_of_nodes():,} nodes")

        return trees

    def compute_cascade_statistics(self) -> Dict:
        """Compute statistics about the detected cascades."""
        trees = self.get_cascade_trees()

        tree_sizes = [t.number_of_nodes() for t in trees]

        stats = {
            "num_cascades": len(trees),
            "total_adoptions": sum(tree_sizes),
            "mean_cascade_size": np.mean(tree_sizes),
            "median_cascade_size": np.median(tree_sizes),
            "max_cascade_size": max(tree_sizes),
            "num_seeds": sum(1 for p in self.cascades.values() if p is None),
            "influenced_fraction": sum(1 for p in self.cascades.values() if p is not None) / len(self.cascades)
        }

        # Cascade depth analysis
        depths = []
        for tree in trees:
            if tree.number_of_nodes() > 0:
                # Find root (node with no incoming edges)
                roots = [n for n in tree.nodes() if tree.in_degree(n) == 0]
                if roots:
                    for root in roots:
                        max_depth = max(nx.single_source_shortest_path_length(tree, root).values())
                        depths.append(max_depth)

        if depths:
            stats["mean_cascade_depth"] = np.mean(depths)
            stats["max_cascade_depth"] = max(depths)

        return stats


class NetworkVisualizer:
    """
    Visualization utilities for developer networks and cascades.
    """

    @staticmethod
    def prepare_for_gephi(G: nx.Graph, output_path: str):
        """Export network to GEXF format for Gephi visualization."""
        nx.write_gexf(G, output_path)
        print(f"Exported network to {output_path}")

    @staticmethod
    def export_cascade_tree(tree: nx.DiGraph, output_path: str):
        """Export cascade tree to GEXF format."""
        nx.write_gexf(tree, output_path)
        print(f"Exported cascade tree to {output_path}")


# =============================================================================
# Example usage
# =============================================================================

def example_claude_analysis():
    """
    Example: Analyze Claude Code adoption cascades.
    """
    data_dir = Path("data/output")

    # 1. Build collaboration network
    print("=" * 60)
    print("Building Developer Collaboration Network")
    print("=" * 60)

    builder = DeveloperNetworkBuilder()
    G = builder.build_from_commits(
        str(data_dir / "claude_commits.parquet"),
        user_col="author_login",
        repo_col="repo_nwo"
    )

    # Print network stats
    stats = builder.get_network_stats()
    print("\nNetwork Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value:,}")

    # 2. Extract adoption times
    print("\n" + "=" * 60)
    print("Extracting Adoption Times")
    print("=" * 60)

    adoption_times = AdoptionTimeExtractor.from_commits(
        str(data_dir / "claude_commits.parquet"),
        user_col="author_login",
        time_col="committed_date"
    )

    # 3. Detect cascades
    print("\n" + "=" * 60)
    print("Detecting Adoption Cascades")
    print("=" * 60)

    analyzer = CascadeAnalyzer(G)
    cascades = analyzer.detect_cascades(adoption_times)

    # Print cascade stats
    cascade_stats = analyzer.compute_cascade_statistics()
    print("\nCascade Statistics:")
    for key, value in cascade_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value:,}")

    # 4. Export for visualization
    print("\n" + "=" * 60)
    print("Exporting for Visualization")
    print("=" * 60)

    output_dir = Path("analysis/output")
    output_dir.mkdir(exist_ok=True)

    NetworkVisualizer.prepare_for_gephi(G, str(output_dir / "claude_network.gexf"))

    trees = analyzer.get_cascade_trees()
    if trees:
        NetworkVisualizer.export_cascade_tree(
            trees[0],
            str(output_dir / "largest_cascade.gexf")
        )

    return G, analyzer


def compare_tools():
    """
    Compare cascade dynamics across different AI coding tools.
    """
    data_dir = Path("data/output")
    results = {}

    tools = [
        ("Claude", "claude_commits.parquet", "author_login", "committed_date"),
        ("Codex", "codex_prs.parquet", "user_login", "created_at"),
        ("Copilot", "copilot_prs.parquet", "user_login", "created_at"),
    ]

    for tool_name, filename, user_col, time_col in tools:
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"Skipping {tool_name}: {filename} not found")
            continue

        print(f"\n{'=' * 60}")
        print(f"Analyzing {tool_name}")
        print("=" * 60)

        # Build network
        builder = DeveloperNetworkBuilder()
        G = builder.build_from_commits(str(filepath), user_col=user_col)

        # Extract adoption times
        adoption_times = AdoptionTimeExtractor.from_commits(
            str(filepath), user_col=user_col, time_col=time_col
        )

        # Detect cascades
        analyzer = CascadeAnalyzer(G)
        analyzer.detect_cascades(adoption_times)

        # Store results
        results[tool_name] = {
            "network_stats": builder.get_network_stats(),
            "cascade_stats": analyzer.compute_cascade_statistics()
        }

    # Print comparison
    print("\n" + "=" * 60)
    print("TOOL COMPARISON")
    print("=" * 60)

    print("\n{:<12} {:>12} {:>12} {:>12} {:>12}".format(
        "Tool", "Nodes", "Edges", "Cascades", "Influenced%"
    ))
    print("-" * 60)

    for tool, data in results.items():
        ns = data["network_stats"]
        cs = data["cascade_stats"]
        print("{:<12} {:>12,} {:>12,} {:>12,} {:>11.1%}".format(
            tool,
            ns.get("nodes", 0),
            ns.get("edges", 0),
            cs.get("num_cascades", 0),
            cs.get("influenced_fraction", 0)
        ))

    return results


if __name__ == "__main__":
    print("AI Coding Tool Network Analysis")
    print("================================\n")

    # Check if data exists
    data_dir = Path("data/output")
    if not data_dir.exists():
        print("Data directory not found. Please run the data collection scripts first.")
        print("Expected: data/output/claude_commits.parquet")
    else:
        # Run example analysis
        try:
            G, analyzer = example_claude_analysis()
        except FileNotFoundError as e:
            print(f"Data file not found: {e}")
            print("Please ensure you have collected data using the fetcher scripts.")
