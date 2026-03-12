#!/usr/bin/env python3
"""
Spotify Collaboration Network Analysis.

Builds the collaboration network and analyzes contagion patterns.
"""

import json
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


def main():
    print("=" * 70)
    print("SPOTIFY COLLABORATION NETWORK ANALYSIS")
    print("=" * 70)

    # Load data
    print("\n1. LOADING DATA...")
    contributors_df = pd.read_parquet("data/output/spotify_contributors.parquet")
    spotify_claude_df = pd.read_parquet("data/output/spotify_claude_users.parquet")

    print(f"   Contributors: {len(contributors_df):,} records")
    print(f"   Unique users: {contributors_df['login'].nunique():,}")
    print(f"   Unique repos: {contributors_df['repo'].nunique():,}")

    # Get Claude users info
    claude_users = (
        spotify_claude_df[spotify_claude_df["is_claude_user"]][["login", "first_claude_use"]]
        .set_index("login")["first_claude_use"]
        .to_dict()
    )
    print(f"   Claude users: {len(claude_users)}")

    # Step 1: Build repo -> users mapping
    print("\n2. BUILDING NETWORK...")
    repo_users = contributors_df.groupby("repo")["login"].apply(set).to_dict()
    print(f"   Repos: {len(repo_users)}")

    # Count contributors per repo
    collab_repos = {r: users for r, users in repo_users.items() if len(users) >= 2}
    print(f"   Collaborative repos (2+ contributors): {len(collab_repos)}")

    # Step 2: Build edges
    print("   Building edges...")
    edges = defaultdict(int)
    for repo, users in collab_repos.items():
        users_list = list(users)
        for i in range(len(users_list)):
            for j in range(i + 1, len(users_list)):
                edge = tuple(sorted([users_list[i], users_list[j]]))
                edges[edge] += 1

    print(f"   Total edges: {len(edges):,}")

    # Step 3: Create NetworkX graph
    print("   Creating graph...")
    G = nx.Graph()

    # Add all users as nodes
    all_users = contributors_df["login"].unique()
    for user in all_users:
        G.add_node(
            user,
            is_claude_user=user in claude_users,
            first_claude_use=str(claude_users.get(user, "")) if user in claude_users else None,
        )

    # Add edges with weights
    for (u1, u2), weight in edges.items():
        G.add_edge(u1, u2, weight=weight, shared_repos=weight)

    print(f"   Nodes: {G.number_of_nodes():,}")
    print(f"   Edges: {G.number_of_edges():,}")
    print(f"   Density: {nx.density(G):.6f}")

    # Step 4: Analyze components
    print("\n3. NETWORK STRUCTURE...")
    components = list(nx.connected_components(G))
    print(f"   Connected components: {len(components)}")

    largest_cc = max(components, key=len)
    G_lcc = G.subgraph(largest_cc).copy()
    print(f"   Largest component: {len(G_lcc):,} nodes ({100*len(G_lcc)/G.number_of_nodes():.1f}%)")

    # Claude users in largest component
    claude_in_lcc = [n for n in G_lcc.nodes() if n in claude_users]
    print(f"   Claude users in largest component: {len(claude_in_lcc)}")

    # Step 5: Compute centrality metrics
    print("\n4. COMPUTING CENTRALITY METRICS...")

    # Degree centrality
    degree_cent = nx.degree_centrality(G_lcc)
    print("   Degree centrality computed")

    # Betweenness centrality (sampled for speed)
    if len(G_lcc) > 1000:
        betweenness = nx.betweenness_centrality(G_lcc, k=min(500, len(G_lcc)))
    else:
        betweenness = nx.betweenness_centrality(G_lcc)
    print("   Betweenness centrality computed")

    # Clustering coefficient
    clustering = nx.clustering(G_lcc)
    print("   Clustering coefficient computed")

    # Raw degrees
    degrees = dict(G_lcc.degree())

    # Step 6: Compare Claude users vs non-users
    print("\n5. CENTRALITY COMPARISON...")
    print("=" * 70)

    claude_nodes = [n for n in G_lcc.nodes() if n in claude_users]
    non_claude_nodes = [n for n in G_lcc.nodes() if n not in claude_users]

    def get_stats(nodes, metric_dict):
        vals = [metric_dict[n] for n in nodes]
        return {
            "mean": np.mean(vals),
            "median": np.median(vals),
            "std": np.std(vals),
            "min": min(vals),
            "max": max(vals),
        }

    print(f"\n   Nodes in largest component:")
    print(f"   - Claude users: {len(claude_nodes)}")
    print(f"   - Non-Claude users: {len(non_claude_nodes)}")

    print(f"\n   DEGREE (number of collaborators):")
    claude_deg = get_stats(claude_nodes, degrees)
    non_deg = get_stats(non_claude_nodes, degrees)
    print(f"   - Claude users:     mean={claude_deg['mean']:.1f}, median={claude_deg['median']:.0f}, max={claude_deg['max']}")
    print(f"   - Non-Claude users: mean={non_deg['mean']:.1f}, median={non_deg['median']:.0f}, max={non_deg['max']}")

    print(f"\n   DEGREE CENTRALITY:")
    claude_dc = get_stats(claude_nodes, degree_cent)
    non_dc = get_stats(non_claude_nodes, degree_cent)
    print(f"   - Claude users:     mean={claude_dc['mean']:.5f}, median={claude_dc['median']:.5f}")
    print(f"   - Non-Claude users: mean={non_dc['mean']:.5f}, median={non_dc['median']:.5f}")

    print(f"\n   BETWEENNESS CENTRALITY:")
    claude_bc = get_stats(claude_nodes, betweenness)
    non_bc = get_stats(non_claude_nodes, betweenness)
    print(f"   - Claude users:     mean={claude_bc['mean']:.6f}, median={claude_bc['median']:.6f}")
    print(f"   - Non-Claude users: mean={non_bc['mean']:.6f}, median={non_bc['median']:.6f}")

    print(f"\n   CLUSTERING COEFFICIENT:")
    claude_cc = get_stats(claude_nodes, clustering)
    non_cc = get_stats(non_claude_nodes, clustering)
    print(f"   - Claude users:     mean={claude_cc['mean']:.4f}, median={claude_cc['median']:.4f}")
    print(f"   - Non-Claude users: mean={non_cc['mean']:.4f}, median={non_cc['median']:.4f}")

    # Step 7: Contagion analysis
    print("\n6. CONTAGION ANALYSIS...")
    print("=" * 70)

    # Sort Claude users by adoption date
    claude_sorted = sorted([(u, claude_users[u]) for u in claude_nodes], key=lambda x: x[1])

    print(f"\n   Adoption sequence (first 15):")
    for i, (user, date) in enumerate(claude_sorted[:15]):
        # Count how many Claude users this person is connected to who adopted BEFORE them
        predecessors = sum(
            1
            for neighbor in G_lcc.neighbors(user)
            if neighbor in claude_users and claude_users[neighbor] < date
        )
        print(f"   {i+1:2}. {user}: {str(date)[:10]} (connected to {predecessors} earlier adopters)")

    # Analyze: Were later adopters connected to earlier adopters?
    print(f"\n   CONNECTION TO EARLIER ADOPTERS:")
    connection_counts = []
    for i, (user, date) in enumerate(claude_sorted):
        if i == 0:
            continue  # Skip first adopter
        predecessors = sum(
            1
            for neighbor in G_lcc.neighbors(user)
            if neighbor in claude_users and claude_users[neighbor] < date
        )
        connection_counts.append(predecessors)

    connected_to_one = sum(1 for c in connection_counts if c >= 1)
    connected_to_two = sum(1 for c in connection_counts if c >= 2)

    print(f"   - Adopters connected to >=1 earlier adopter: {connected_to_one} / {len(connection_counts)} ({100*connected_to_one/len(connection_counts):.1f}%)")
    print(f"   - Adopters connected to >=2 earlier adopters: {connected_to_two} / {len(connection_counts)}")
    print(f"   - Mean connections to earlier adopters: {np.mean(connection_counts):.2f}")
    print(f"   - Max connections to earlier adopters: {max(connection_counts)}")

    # Step 8: Find potential cascade chains
    print(f"\n   POTENTIAL CASCADE CHAINS:")
    print("   (User A adopts -> Collaborator B adopts within 30 days)")

    cascades = []
    for i, (user_b, date_b) in enumerate(claude_sorted):
        for neighbor in G_lcc.neighbors(user_b):
            if neighbor in claude_users:
                date_a = claude_users[neighbor]
                if date_a < date_b:
                    gap = (date_b - date_a).days
                    if gap <= 30:
                        cascades.append((neighbor, user_b, gap))

    print(f"   Found {len(cascades)} potential cascades (adoption within 30 days of collaborator)")
    for a, b, gap in cascades[:10]:
        print(f"   - {a} -> {b} ({gap} days)")

    # Save results
    print("\n7. SAVING RESULTS...")
    results = {
        "network_stats": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "density": nx.density(G),
            "components": len(components),
            "largest_component": len(G_lcc),
            "claude_users_in_lcc": len(claude_nodes),
        },
        "centrality_comparison": {
            "claude_users": {
                "count": len(claude_nodes),
                "degree_mean": claude_deg["mean"],
                "degree_median": claude_deg["median"],
                "betweenness_mean": claude_bc["mean"],
                "clustering_mean": claude_cc["mean"],
            },
            "non_claude_users": {
                "count": len(non_claude_nodes),
                "degree_mean": non_deg["mean"],
                "degree_median": non_deg["median"],
                "betweenness_mean": non_bc["mean"],
                "clustering_mean": non_cc["mean"],
            },
        },
        "contagion": {
            "adopters_connected_to_earlier": connected_to_one,
            "total_adopters_after_first": len(connection_counts),
            "pct_connected_to_earlier": 100 * connected_to_one / len(connection_counts),
            "mean_connections_to_earlier": np.mean(connection_counts),
            "cascades_within_30_days": len(cascades),
        },
    }

    output_dir = Path("analysis/output")
    with open(output_dir / "spotify_network_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save graph
    nx.write_graphml(G_lcc, output_dir / "spotify_network.graphml")

    print(f"   Saved network analysis to {output_dir}/spotify_network_analysis.json")
    print(f"   Saved network graph to {output_dir}/spotify_network.graphml")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    ratio = claude_deg["mean"] / non_deg["mean"] if non_deg["mean"] > 0 else 0
    print(f"Claude users have {ratio:.1f}x more collaborators than non-users")
    print(f"{connected_to_one}/{len(connection_counts)} ({100*connected_to_one/len(connection_counts):.0f}%) adopters were connected to earlier adopters")
    print(f"{len(cascades)} potential contagion events (adoption within 30 days of collaborator)")


if __name__ == "__main__":
    main()
