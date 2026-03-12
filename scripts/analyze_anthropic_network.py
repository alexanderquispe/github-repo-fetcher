"""
Analyze Anthropic collaboration network and Claude Code adoption patterns.
Same methodology as Spotify analysis.
"""

import pandas as pd
import networkx as nx
from pathlib import Path
from collections import defaultdict
import json

output_dir = Path("analysis/output")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("ANTHROPIC COLLABORATION NETWORK ANALYSIS")
print("=" * 70)

# 1. Load data
print("\n1. LOADING DATA...")
contributors_df = pd.read_parquet("data/output/anthropic_contributors.parquet")
all_users_df = pd.read_parquet("data/output/anthropic_claude_users.parquet")

# Filter to Claude users only
claude_users_df = all_users_df[all_users_df['is_claude_user'] == True].copy()

print(f"   Contributors: {len(contributors_df):,} records")
print(f"   Unique users: {contributors_df['login'].nunique():,}")
print(f"   Unique repos: {contributors_df['repo'].nunique():,}")
print(f"   Claude users: {len(claude_users_df):,}")

# Filter out bots
bot_patterns = ['[bot]', '-bot', 'dependabot', 'renovate', 'github-actions', 'codecov']
def is_bot(login):
    login_lower = login.lower()
    return any(pattern in login_lower for pattern in bot_patterns)

contributors_df = contributors_df[~contributors_df['login'].apply(is_bot)]
claude_users_df = claude_users_df[~claude_users_df['login'].apply(is_bot)]

print(f"\n   After filtering bots:")
print(f"   Contributors: {len(contributors_df):,} records")
print(f"   Unique users: {contributors_df['login'].nunique():,}")
print(f"   Claude users: {len(claude_users_df):,}")

# Get unique users
unique_users = contributors_df['login'].unique()

# Create adoption dates dictionary
claude_users_df['first_claude_use'] = pd.to_datetime(claude_users_df['first_claude_use'], utc=True)
adoption_dates = dict(zip(claude_users_df['login'], claude_users_df['first_claude_use']))

# 2. Build network
print("\n2. BUILDING NETWORK...")
repo_contributors = defaultdict(set)
for _, row in contributors_df.iterrows():
    repo_contributors[row['repo']].add(row['login'])

# Count collaborative repos
collab_repos = {r: c for r, c in repo_contributors.items() if len(c) >= 2}
print(f"   Repos: {len(repo_contributors)}")
print(f"   Collaborative repos (2+ contributors): {len(collab_repos)}")

# Create edges
print("   Building edges...")
edges = defaultdict(int)
for repo, contribs in collab_repos.items():
    contribs = list(contribs)
    for i in range(len(contribs)):
        for j in range(i + 1, len(contribs)):
            u, v = sorted([contribs[i], contribs[j]])
            edges[(u, v)] += 1

print(f"   Total edges: {len(edges):,}")

# Create graph
print("   Creating graph...")
G = nx.Graph()
G.add_nodes_from(unique_users)
for (u, v), weight in edges.items():
    G.add_edge(u, v, weight=weight)

print(f"   Nodes: {G.number_of_nodes():,}")
print(f"   Edges: {G.number_of_edges():,}")
density = 2 * G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1)) if G.number_of_nodes() > 1 else 0
print(f"   Density: {density:.6f}")

# 3. Network structure
print("\n3. NETWORK STRUCTURE...")
components = list(nx.connected_components(G))
print(f"   Connected components: {len(components)}")

largest_cc = max(components, key=len)
print(f"   Largest component: {len(largest_cc)} nodes ({100*len(largest_cc)/G.number_of_nodes():.1f}%)")

G_lcc = G.subgraph(largest_cc).copy()
claude_in_lcc = [u for u in adoption_dates.keys() if u in G_lcc]
print(f"   Claude users in largest component: {len(claude_in_lcc)}")

# 4. Centrality metrics
print("\n4. COMPUTING CENTRALITY METRICS...")
print("   Degree centrality...")
degree_centrality = nx.degree_centrality(G_lcc)
print("   Betweenness centrality...")
betweenness = nx.betweenness_centrality(G_lcc)
print("   Clustering coefficient...")
clustering = nx.clustering(G_lcc)

# 5. Compare Claude users vs non-users
print("\n5. CENTRALITY COMPARISON...")
print("=" * 70)

claude_set = set(adoption_dates.keys()) & set(G_lcc.nodes())
non_claude_set = set(G_lcc.nodes()) - claude_set

claude_degrees = [G_lcc.degree(u) for u in claude_set]
non_claude_degrees = [G_lcc.degree(u) for u in non_claude_set]

claude_betweenness = [betweenness[u] for u in claude_set]
non_claude_betweenness = [betweenness[u] for u in non_claude_set]

claude_clustering = [clustering[u] for u in claude_set]
non_claude_clustering = [clustering[u] for u in non_claude_set]

import statistics

print(f"\n   Nodes in largest component:")
print(f"   - Claude users: {len(claude_set)}")
print(f"   - Non-Claude users: {len(non_claude_set)}")

print(f"\n   DEGREE (number of collaborators):")
print(f"   - Claude users:     mean={statistics.mean(claude_degrees):.1f}, median={statistics.median(claude_degrees):.0f}, max={max(claude_degrees)}")
print(f"   - Non-Claude users: mean={statistics.mean(non_claude_degrees):.1f}, median={statistics.median(non_claude_degrees):.0f}, max={max(non_claude_degrees)}")

print(f"\n   BETWEENNESS CENTRALITY:")
print(f"   - Claude users:     mean={statistics.mean(claude_betweenness):.6f}, median={statistics.median(claude_betweenness):.6f}")
print(f"   - Non-Claude users: mean={statistics.mean(non_claude_betweenness):.6f}, median={statistics.median(non_claude_betweenness):.6f}")

print(f"\n   CLUSTERING COEFFICIENT:")
print(f"   - Claude users:     mean={statistics.mean(claude_clustering):.4f}, median={statistics.median(claude_clustering):.4f}")
print(f"   - Non-Claude users: mean={statistics.mean(non_claude_clustering):.4f}, median={statistics.median(non_claude_clustering):.4f}")

# 6. Contagion analysis
print("\n6. CONTAGION ANALYSIS...")
print("=" * 70)

# Sort adopters by date
adopters = [(u, d) for u, d in adoption_dates.items() if u in G_lcc and pd.notna(d)]
adopters.sort(key=lambda x: x[1])

print(f"\n   Adoption sequence (first 15):")
for i, (user, date) in enumerate(adopters[:15]):
    # Count connections to earlier adopters
    earlier_adopters = {u for u, d in adopters[:i]}
    connections = len(set(G_lcc.neighbors(user)) & earlier_adopters)
    print(f"   {i+1:2}. {user}: {date.strftime('%Y-%m-%d')} (connected to {connections} earlier adopters)")

# Connection to earlier adopters
print(f"\n   CONNECTION TO EARLIER ADOPTERS:")
connections_to_earlier = []
for i, (user, date) in enumerate(adopters[1:], 1):  # Skip first adopter
    earlier_adopters = {u for u, d in adopters[:i]}
    connections = len(set(G_lcc.neighbors(user)) & earlier_adopters)
    connections_to_earlier.append(connections)

connected_count = sum(1 for c in connections_to_earlier if c >= 1)
print(f"   - Adopters connected to >=1 earlier adopter: {connected_count} / {len(connections_to_earlier)} ({100*connected_count/len(connections_to_earlier):.1f}%)")
connected_2plus = sum(1 for c in connections_to_earlier if c >= 2)
print(f"   - Adopters connected to >=2 earlier adopters: {connected_2plus} / {len(connections_to_earlier)}")
print(f"   - Mean connections to earlier adopters: {statistics.mean(connections_to_earlier):.2f}")
print(f"   - Max connections to earlier adopters: {max(connections_to_earlier)}")

# Cascade detection
print(f"\n   POTENTIAL CASCADE CHAINS:")
print(f"   (User A adopts -> Collaborator B adopts within 30 days)")
cascades = []
for i, (user, date) in enumerate(adopters[1:], 1):
    neighbors = set(G_lcc.neighbors(user))
    for j, (earlier_user, earlier_date) in enumerate(adopters[:i]):
        if earlier_user in neighbors:
            days_gap = (date - earlier_date).days
            if 0 <= days_gap <= 30:
                cascades.append((earlier_user, user, days_gap))

print(f"   Found {len(cascades)} potential cascades (adoption within 30 days of collaborator)")
for influencer, adopter, days in cascades[:10]:
    print(f"   - {influencer} -> {adopter} ({days} days)")

# 7. Monthly adoption
print("\n7. MONTHLY ADOPTION:")
print("=" * 70)
months = pd.date_range('2025-02-01', '2026-02-01', freq='MS', tz='UTC')
monthly_counts = []
for month_start in months:
    month_end = month_start + pd.offsets.MonthEnd(0)
    month_str = month_start.strftime('%Y-%m')

    count = sum(1 for u, d in adopters if d <= month_end)
    new_count = sum(1 for u, d in adopters if month_start <= d <= month_end)
    monthly_counts.append({'month': month_str, 'cumulative': count, 'new': new_count})
    if new_count > 0:
        print(f"   {month_str}: +{new_count:2} new, {count:2} total")

# 8. Top repos by Claude users
print("\n8. REPOS WITH MOST CLAUDE USERS:")
print("=" * 70)
repo_claude_counts = defaultdict(set)
for _, row in contributors_df.iterrows():
    if row['login'] in claude_set:
        repo_claude_counts[row['repo']].add(row['login'])

top_repos = sorted(repo_claude_counts.items(), key=lambda x: len(x[1]), reverse=True)[:15]
for repo, users in top_repos:
    print(f"   {repo}: {len(users)} Claude users")

# 9. Save results
print("\n9. SAVING RESULTS...")
results = {
    "network_stats": {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": round(density, 6),
        "components": len(components),
        "largest_component": len(largest_cc),
        "claude_users_in_lcc": len(claude_in_lcc)
    },
    "centrality_comparison": {
        "claude_users": {
            "count": len(claude_set),
            "degree_mean": round(statistics.mean(claude_degrees), 1),
            "degree_median": statistics.median(claude_degrees),
            "betweenness_mean": round(statistics.mean(claude_betweenness), 6),
            "clustering_mean": round(statistics.mean(claude_clustering), 4)
        },
        "non_claude_users": {
            "count": len(non_claude_set),
            "degree_mean": round(statistics.mean(non_claude_degrees), 1),
            "degree_median": statistics.median(non_claude_degrees),
            "betweenness_mean": round(statistics.mean(non_claude_betweenness), 6),
            "clustering_mean": round(statistics.mean(non_claude_clustering), 4)
        }
    },
    "contagion": {
        "adopters_connected_to_earlier": connected_count,
        "total_adopters_after_first": len(connections_to_earlier),
        "pct_connected_to_earlier": round(100 * connected_count / len(connections_to_earlier), 1),
        "mean_connections_to_earlier": round(statistics.mean(connections_to_earlier), 2),
        "cascades_within_30_days": len(cascades)
    }
}

with open(output_dir / "anthropic_network_analysis.json", 'w') as f:
    json.dump(results, f, indent=2)
print(f"   Saved: anthropic_network_analysis.json")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
