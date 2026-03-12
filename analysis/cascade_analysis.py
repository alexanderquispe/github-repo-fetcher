"""
Cascade Detection and Analysis for AI Tool Adoption.

This module provides utilities for detecting, analyzing, and visualizing
adoption cascades through developer collaboration networks.
"""

import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class CascadeDetector:
    """
    Detects and analyzes adoption cascades in collaboration networks.

    A cascade is a spreading pattern where adoption of a tool spreads
    from one developer to connected developers over time.
    """

    def __init__(
        self,
        G: "nx.Graph",
        adoption_times: Dict[str, datetime]
    ):
        """
        Initialize the cascade detector.

        Args:
            G: NetworkX graph representing collaboration network
            adoption_times: Dict mapping user -> first adoption datetime
        """
        if not HAS_NETWORKX:
            raise ImportError("networkx required")

        self.G = G
        self.adoption_times = adoption_times
        self.network_users = set(G.nodes())
        self.adopters = set(adoption_times.keys()) & self.network_users

        # Cascade results
        self.parent_map: Dict[str, Optional[str]] = {}
        self.cascade_trees: List[nx.DiGraph] = []

    def detect_cascades(
        self,
        max_window_days: int = 30,
        weight_by_edge: bool = True
    ) -> Dict[str, Optional[str]]:
        """
        Detect cascade relationships by finding likely "parents" for each adopter.

        For each adopter, identifies the most likely parent - the neighbor who
        adopted most recently before them (within the time window).

        Args:
            max_window_days: Maximum days between parent and child adoption
            weight_by_edge: If True, prefer stronger collaboration edges

        Returns:
            Dict mapping each adopter -> their likely parent (or None if seed)
        """
        # Sort adopters by time
        sorted_adopters = sorted(
            [(u, t) for u, t in self.adoption_times.items() if u in self.network_users],
            key=lambda x: x[1]
        )

        adopted = set()
        self.parent_map = {}

        for user, adopt_time in sorted_adopters:
            # Find neighbors who adopted before this user
            neighbors = set(self.G.neighbors(user))
            prior_adopters = neighbors & adopted

            if prior_adopters:
                # Find best parent within time window
                best_parent = None
                best_score = -1

                for neighbor in prior_adopters:
                    neighbor_time = self.adoption_times[neighbor]
                    time_diff = (adopt_time - neighbor_time).days

                    if 0 < time_diff <= max_window_days:
                        # Score: prefer recent adoption and strong edges
                        recency_score = 1 / (time_diff + 1)

                        if weight_by_edge:
                            edge_weight = self.G[user][neighbor].get("weight", 1)
                            score = recency_score * np.log1p(edge_weight)
                        else:
                            score = recency_score

                        if score > best_score:
                            best_score = score
                            best_parent = neighbor

                self.parent_map[user] = best_parent
            else:
                # Seed node (no prior adopting neighbors)
                self.parent_map[user] = None

            adopted.add(user)

        # Statistics
        seeds = sum(1 for p in self.parent_map.values() if p is None)
        influenced = len(self.parent_map) - seeds

        print(f"Detected {seeds:,} seed nodes and {influenced:,} influenced adoptions")

        return self.parent_map

    def build_cascade_trees(self) -> List[nx.DiGraph]:
        """
        Build directed cascade trees from detected parent relationships.

        Returns:
            List of directed graphs, each representing a cascade tree
        """
        if not self.parent_map:
            self.detect_cascades()

        # Build directed graph of influence
        influence_graph = nx.DiGraph()

        for child, parent in self.parent_map.items():
            influence_graph.add_node(child)
            if parent is not None:
                influence_graph.add_edge(parent, child)

        # Find weakly connected components (each is a cascade tree)
        self.cascade_trees = []

        for component in nx.weakly_connected_components(influence_graph):
            tree = influence_graph.subgraph(component).copy()
            self.cascade_trees.append(tree)

        # Sort by size (largest first)
        self.cascade_trees.sort(key=lambda t: t.number_of_nodes(), reverse=True)

        print(f"Found {len(self.cascade_trees):,} cascade trees")
        if self.cascade_trees:
            print(f"Largest tree: {self.cascade_trees[0].number_of_nodes():,} nodes")

        return self.cascade_trees

    def compute_cascade_statistics(self) -> Dict[str, Any]:
        """
        Compute comprehensive statistics about detected cascades.

        Returns:
            Dict with cascade metrics
        """
        if not self.cascade_trees:
            self.build_cascade_trees()

        if not self.cascade_trees:
            return {"error": "No cascades detected"}

        # Basic counts
        tree_sizes = [t.number_of_nodes() for t in self.cascade_trees]

        stats = {
            "num_cascades": len(self.cascade_trees),
            "total_adopters": sum(tree_sizes),
            "mean_cascade_size": np.mean(tree_sizes),
            "median_cascade_size": np.median(tree_sizes),
            "std_cascade_size": np.std(tree_sizes),
            "max_cascade_size": max(tree_sizes),
            "min_cascade_size": min(tree_sizes),
        }

        # Seed statistics
        num_seeds = sum(1 for p in self.parent_map.values() if p is None)
        stats["num_seeds"] = num_seeds
        stats["seed_fraction"] = num_seeds / len(self.parent_map)
        stats["influenced_fraction"] = 1 - stats["seed_fraction"]

        # Cascade depth analysis
        depths = []
        for tree in self.cascade_trees:
            if tree.number_of_nodes() > 0:
                # Find roots (nodes with no incoming edges)
                roots = [n for n in tree.nodes() if tree.in_degree(n) == 0]
                for root in roots:
                    try:
                        path_lengths = nx.single_source_shortest_path_length(tree, root)
                        max_depth = max(path_lengths.values())
                        depths.append(max_depth)
                    except nx.NetworkXError:
                        pass

        if depths:
            stats["mean_cascade_depth"] = np.mean(depths)
            stats["median_cascade_depth"] = np.median(depths)
            stats["max_cascade_depth"] = max(depths)

        # Size distribution (categorized)
        size_distribution = {
            "1 (isolated)": sum(1 for s in tree_sizes if s == 1),
            "2-5": sum(1 for s in tree_sizes if 2 <= s <= 5),
            "6-10": sum(1 for s in tree_sizes if 6 <= s <= 10),
            "11-50": sum(1 for s in tree_sizes if 11 <= s <= 50),
            "51-100": sum(1 for s in tree_sizes if 51 <= s <= 100),
            "100+": sum(1 for s in tree_sizes if s > 100),
        }
        stats["size_distribution"] = size_distribution

        # Branching statistics
        branching_factors = []
        for tree in self.cascade_trees:
            for node in tree.nodes():
                out_degree = tree.out_degree(node)
                if out_degree > 0:
                    branching_factors.append(out_degree)

        if branching_factors:
            stats["mean_branching_factor"] = np.mean(branching_factors)
            stats["max_branching_factor"] = max(branching_factors)

        return stats

    def get_largest_cascades(
        self,
        n: int = 10
    ) -> List[Tuple[int, nx.DiGraph]]:
        """
        Get the N largest cascade trees.

        Args:
            n: Number of cascades to return

        Returns:
            List of (size, tree) tuples
        """
        if not self.cascade_trees:
            self.build_cascade_trees()

        return [
            (tree.number_of_nodes(), tree)
            for tree in self.cascade_trees[:n]
        ]

    def analyze_cascade_timing(self) -> Dict[str, Any]:
        """
        Analyze timing patterns within cascades.

        Returns:
            Dict with timing statistics
        """
        if not self.cascade_trees:
            self.build_cascade_trees()

        # Parent-child time differences
        time_diffs = []

        for child, parent in self.parent_map.items():
            if parent is not None:
                child_time = self.adoption_times[child]
                parent_time = self.adoption_times[parent]
                diff_days = (child_time - parent_time).days
                time_diffs.append(diff_days)

        if not time_diffs:
            return {"error": "No parent-child pairs found"}

        return {
            "mean_adoption_delay_days": np.mean(time_diffs),
            "median_adoption_delay_days": np.median(time_diffs),
            "std_adoption_delay_days": np.std(time_diffs),
            "min_adoption_delay_days": min(time_diffs),
            "max_adoption_delay_days": max(time_diffs),
            "quartile_25": np.percentile(time_diffs, 25),
            "quartile_75": np.percentile(time_diffs, 75),
        }

    def compute_transmission_metrics(self) -> Dict[str, float]:
        """
        Compute disease-model style transmission metrics.

        Returns:
            Dict with transmission metrics
        """
        if not self.parent_map:
            self.detect_cascades()

        # Count successful vs. unsuccessful transmissions
        successful = 0
        opportunities = 0

        sorted_adopters = sorted(
            [(u, t) for u, t in self.adoption_times.items() if u in self.network_users],
            key=lambda x: x[1]
        )

        adopted = set()

        for user, adopt_time in sorted_adopters:
            neighbors = set(self.G.neighbors(user))
            prior_adopter_neighbors = neighbors & adopted

            # Each prior adopter had an "opportunity" to influence
            opportunities += len(prior_adopter_neighbors)

            # One of them succeeded (or zero if this is a seed)
            if self.parent_map.get(user) is not None:
                successful += 1

            adopted.add(user)

        transmission_rate = successful / opportunities if opportunities > 0 else 0

        # Secondary attack rate (average infections per infected)
        children_counts = defaultdict(int)
        for child, parent in self.parent_map.items():
            if parent is not None:
                children_counts[parent] += 1

        if children_counts:
            mean_secondary = np.mean(list(children_counts.values()))
        else:
            mean_secondary = 0

        return {
            "transmission_rate": transmission_rate,
            "successful_transmissions": successful,
            "transmission_opportunities": opportunities,
            "mean_secondary_infections": mean_secondary,
            "max_secondary_infections": max(children_counts.values()) if children_counts else 0,
        }

    def export_cascades(self, output_dir: Path) -> Dict[str, Path]:
        """
        Export cascade data to files.

        Args:
            output_dir: Directory for output files

        Returns:
            Dict mapping output type -> file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved = {}

        # Save cascade trees as JSON
        cascade_data = []
        for i, tree in enumerate(self.cascade_trees):
            cascade_info = {
                "cascade_id": i,
                "size": tree.number_of_nodes(),
                "nodes": list(tree.nodes()),
                "edges": list(tree.edges()),
            }

            # Find root
            roots = [n for n in tree.nodes() if tree.in_degree(n) == 0]
            cascade_info["roots"] = roots

            # Get depth
            if roots:
                try:
                    paths = nx.single_source_shortest_path_length(tree, roots[0])
                    cascade_info["depth"] = max(paths.values())
                except nx.NetworkXError:
                    cascade_info["depth"] = 0

            cascade_data.append(cascade_info)

        path = output_dir / "cascade_trees.json"
        with open(path, "w") as f:
            json.dump(cascade_data, f, indent=2)
        saved["cascade_trees"] = path
        print(f"Saved cascade trees to {path}")

        # Save parent map
        path = output_dir / "cascade_parents.json"
        with open(path, "w") as f:
            json.dump(self.parent_map, f, indent=2)
        saved["cascade_parents"] = path

        # Save statistics
        stats = self.compute_cascade_statistics()
        timing = self.analyze_cascade_timing()
        transmission = self.compute_transmission_metrics()

        all_stats = {
            "cascade_statistics": stats,
            "timing_statistics": timing,
            "transmission_metrics": transmission,
        }

        path = output_dir / "cascade_statistics.json"
        with open(path, "w") as f:
            json.dump(all_stats, f, indent=2, default=str)
        saved["cascade_statistics"] = path
        print(f"Saved statistics to {path}")

        # Export largest cascade as GEXF
        if self.cascade_trees:
            largest = self.cascade_trees[0]
            path = output_dir / "largest_cascade.gexf"
            nx.write_gexf(largest, str(path))
            saved["largest_cascade_gexf"] = path
            print(f"Saved largest cascade to {path}")

        return saved


def analyze_cascades(
    G: "nx.Graph",
    adoption_times: Dict[str, datetime],
    output_dir: Optional[Path] = None,
    max_window_days: int = 30
) -> Dict[str, Any]:
    """
    Run comprehensive cascade analysis.

    Args:
        G: Collaboration network
        adoption_times: User adoption times
        output_dir: Optional directory to save results
        max_window_days: Time window for parent detection

    Returns:
        Dict with all cascade analysis results
    """
    results = {}

    print("\n" + "=" * 60)
    print("CASCADE ANALYSIS")
    print("=" * 60)

    # Create detector
    detector = CascadeDetector(G, adoption_times)

    # Detect cascades
    print("\n1. Detecting cascades...")
    parent_map = detector.detect_cascades(max_window_days=max_window_days)

    # Build trees
    print("\n2. Building cascade trees...")
    trees = detector.build_cascade_trees()

    # Compute statistics
    print("\n3. Computing statistics...")
    stats = detector.compute_cascade_statistics()
    results["cascade_statistics"] = stats

    print("\nCascade Statistics:")
    print(f"  Number of cascades: {stats['num_cascades']:,}")
    print(f"  Mean cascade size: {stats['mean_cascade_size']:.2f}")
    print(f"  Max cascade size: {stats['max_cascade_size']:,}")
    print(f"  Seed fraction: {stats['seed_fraction']:.2%}")
    print(f"  Influenced fraction: {stats['influenced_fraction']:.2%}")
    if "mean_cascade_depth" in stats:
        print(f"  Mean cascade depth: {stats['mean_cascade_depth']:.2f}")

    # Timing analysis
    print("\n4. Analyzing timing...")
    timing = detector.analyze_cascade_timing()
    results["timing_statistics"] = timing

    if "error" not in timing:
        print("\nTiming Statistics:")
        print(f"  Mean adoption delay: {timing['mean_adoption_delay_days']:.1f} days")
        print(f"  Median adoption delay: {timing['median_adoption_delay_days']:.1f} days")

    # Transmission metrics
    print("\n5. Computing transmission metrics...")
    transmission = detector.compute_transmission_metrics()
    results["transmission_metrics"] = transmission

    print("\nTransmission Metrics:")
    print(f"  Transmission rate: {transmission['transmission_rate']:.4f}")
    print(f"  Mean secondary infections: {transmission['mean_secondary_infections']:.2f}")
    print(f"  Max secondary infections: {transmission['max_secondary_infections']}")

    # Largest cascades
    print("\n6. Largest cascades:")
    largest = detector.get_largest_cascades(5)
    for i, (size, tree) in enumerate(largest):
        roots = [n for n in tree.nodes() if tree.in_degree(n) == 0]
        print(f"  {i+1}. Size: {size:,}, Root: {roots[0] if roots else 'N/A'}")

    # Export
    if output_dir:
        print("\n7. Exporting results...")
        saved = detector.export_cascades(output_dir)
        results["saved_files"] = {k: str(v) for k, v in saved.items()}

    return results


if __name__ == "__main__":
    from pathlib import Path
    from .collaboration_network import build_full_collaboration_network
    from .adoption_timeline import AdoptionTimelineBuilder

    data_dir = Path("data/output")
    output_dir = Path("analysis/output")

    # Build network
    G, stats = build_full_collaboration_network(data_dir)

    # Get adoption times
    builder = AdoptionTimelineBuilder(data_dir)
    builder.load_data()
    builder.build_user_timelines()
    adoption_times = builder.get_first_adoption_times("claude")

    print(f"\nNetwork: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f"Claude adopters: {len(adoption_times):,}")

    # Run cascade analysis
    results = analyze_cascades(G, adoption_times, output_dir=output_dir)
