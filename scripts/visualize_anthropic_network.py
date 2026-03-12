"""
Visualize Anthropic collaboration network with Claude Code adoption over time.
Creates an interactive HTML visualization showing adoption spreading month by month.
"""

import pandas as pd
import networkx as nx
from pathlib import Path
from collections import defaultdict
import json

# Output directory
output_dir = Path("analysis/output/visualization")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("ANTHROPIC NETWORK VISUALIZATION")
print("=" * 70)

# Load data
print("\n1. Loading data...")
contributors_df = pd.read_parquet("data/output/anthropic_contributors.parquet")
all_users_df = pd.read_parquet("data/output/anthropic_claude_users.parquet")

# Filter bots
bot_patterns = ['[bot]', '-bot', 'dependabot', 'renovate', 'github-actions', 'codecov']
def is_bot(login):
    login_lower = login.lower()
    return any(pattern in login_lower for pattern in bot_patterns)

contributors_df = contributors_df[~contributors_df['login'].apply(is_bot)]

# Filter to Claude users only
claude_users_df = all_users_df[(all_users_df['is_claude_user'] == True) & (~all_users_df['login'].apply(is_bot))].copy()

print(f"   Contributors: {len(contributors_df):,}")
print(f"   Claude users: {len(claude_users_df):,}")

# Get unique users
unique_users = contributors_df['login'].unique()
print(f"   Unique contributors: {len(unique_users):,}")

# Create adoption dates dictionary
claude_users_df['first_claude_use'] = pd.to_datetime(claude_users_df['first_claude_use'], utc=True)
adoption_dates = dict(zip(claude_users_df['login'], claude_users_df['first_claude_use']))

print(f"\n2. Building collaboration network...")

# Build edges from shared repositories
repo_contributors = defaultdict(set)
for _, row in contributors_df.iterrows():
    repo_contributors[row['repo']].add(row['login'])

# Create edges
edges = defaultdict(int)
for repo, contribs in repo_contributors.items():
    contribs = list(contribs)
    if len(contribs) >= 2:
        for i in range(len(contribs)):
            for j in range(i + 1, len(contribs)):
                u, v = sorted([contribs[i], contribs[j]])
                edges[(u, v)] += 1

print(f"   Total edges: {len(edges):,}")

# Create graph
G = nx.Graph()
G.add_nodes_from(unique_users)
for (u, v), weight in edges.items():
    G.add_edge(u, v, weight=weight)

print(f"   Nodes: {G.number_of_nodes():,}")
print(f"   Edges: {G.number_of_edges():,}")

# Get largest connected component
largest_cc = max(nx.connected_components(G), key=len)
G_lcc = G.subgraph(largest_cc).copy()
print(f"   Largest component: {G_lcc.number_of_nodes():,} nodes")

# Filter to Claude users in LCC for visualization
claude_users_in_lcc = [u for u in adoption_dates.keys() if u in G_lcc]
print(f"   Claude users in LCC: {len(claude_users_in_lcc)}")

print("\n3. Computing layout for visualization...")

# For Anthropic, the network is smaller so we can visualize more of it
# Get all Claude users + their direct neighbors
claude_set = set(claude_users_in_lcc)
neighbors = set()
for user in claude_set:
    if user in G_lcc:
        neighbors.update(G_lcc.neighbors(user))

# For Anthropic, include more neighbors since network is smaller
neighbor_degrees = [(n, G_lcc.degree(n)) for n in neighbors if n not in claude_set]
neighbor_degrees.sort(key=lambda x: x[1], reverse=True)
top_neighbors = set([n for n, d in neighbor_degrees[:300]])  # Top 300 neighbors

viz_nodes = claude_set | top_neighbors
G_viz = G_lcc.subgraph(viz_nodes).copy()

print(f"   Visualization subgraph: {G_viz.number_of_nodes()} nodes, {G_viz.number_of_edges()} edges")

# Compute layout
print("   Computing spring layout...")
pos = nx.spring_layout(G_viz, k=2, iterations=50, seed=42)

print("\n4. Preparing monthly snapshots...")

# Get monthly adoption data
months = pd.date_range('2025-02-01', '2026-02-01', freq='MS', tz='UTC')
monthly_data = []

for month_start in months:
    month_end = month_start + pd.offsets.MonthEnd(0)
    month_str = month_start.strftime('%Y-%m')

    # Users who adopted by this month
    adopted_by_month = set()
    for user, date in adoption_dates.items():
        if pd.notna(date) and date <= month_end and user in viz_nodes:
            adopted_by_month.add(user)

    monthly_data.append({
        'month': month_str,
        'adopted_count': len(adopted_by_month),
        'adopted_users': list(adopted_by_month)
    })
    print(f"   {month_str}: {len(adopted_by_month)} Claude users")

print("\n5. Creating interactive HTML visualization...")

# Create HTML with D3.js for animation
html_content = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Anthropic Network - Claude Code Adoption</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #0f0f1a;
            color: #eee;
        }
        h1 {
            text-align: center;
            color: #d4a574;
            margin-bottom: 5px;
        }
        .subtitle {
            text-align: center;
            color: #888;
            margin-bottom: 20px;
        }
        #container {
            display: flex;
            justify-content: center;
            gap: 20px;
        }
        #network {
            background: #1a1a2e;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        #controls {
            background: #1a1a2e;
            padding: 20px;
            border-radius: 10px;
            width: 280px;
        }
        #month-display {
            font-size: 24px;
            font-weight: bold;
            color: #d4a574;
            margin-bottom: 15px;
        }
        #stats {
            margin-top: 20px;
            line-height: 1.8;
        }
        .stat-value {
            color: #d4a574;
            font-weight: bold;
        }
        button {
            background: #d4a574;
            color: #1a1a2e;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            margin: 5px;
        }
        button:hover {
            background: #e5b585;
        }
        button:disabled {
            background: #555;
            cursor: not-allowed;
        }
        #slider {
            width: 100%;
            margin: 15px 0;
        }
        .legend {
            margin-top: 20px;
            padding-top: 15px;
            border-top: 1px solid #333;
        }
        .legend-item {
            display: flex;
            align-items: center;
            margin: 8px 0;
        }
        .legend-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 10px;
        }
        .tooltip {
            position: absolute;
            background: rgba(0,0,0,0.9);
            color: white;
            padding: 8px 12px;
            border-radius: 5px;
            font-size: 12px;
            pointer-events: none;
            border: 1px solid #d4a574;
        }
        .key-finding {
            background: #2a2a3e;
            padding: 10px;
            border-radius: 5px;
            margin-top: 15px;
            font-size: 13px;
            border-left: 3px solid #d4a574;
        }
        .key-finding strong {
            color: #d4a574;
        }
    </style>
</head>
<body>
    <h1>Anthropic Network: Claude Code Adoption</h1>
    <p class="subtitle">Tracking how Claude Code spread through Anthropic's open-source collaboration network</p>

    <div id="container">
        <svg id="network" width="900" height="700"></svg>
        <div id="controls">
            <div id="month-display">Feb 2025</div>
            <div>
                <button id="play-btn">Play</button>
                <button id="reset-btn">Reset</button>
            </div>
            <input type="range" id="slider" min="0" max="11" value="0">

            <div id="stats">
                <div>Claude Users: <span class="stat-value" id="adopted-count">0</span> / 91</div>
                <div>Network Nodes: <span class="stat-value" id="total-nodes">0</span></div>
                <div>Connections: <span class="stat-value" id="total-edges">0</span></div>
            </div>

            <div class="legend">
                <div class="legend-item">
                    <div class="legend-dot" style="background: #d4a574;"></div>
                    <span>Claude Code User</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot" style="background: #444;"></div>
                    <span>Non-User (Collaborator)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot" style="background: #ff6b6b; width: 16px; height: 16px;"></div>
                    <span>New Adopter (this month)</span>
                </div>
            </div>

            <div class="key-finding">
                <strong>Key Finding:</strong><br>
                85.6% of adopters were connected to earlier adopters. 87 cascade events detected.
            </div>
        </div>
    </div>

    <div class="tooltip" id="tooltip" style="display: none;"></div>

    <script>
    // Network data
    const nodes = NODES_DATA;
    const links = LINKS_DATA;
    const monthlyAdoption = MONTHLY_DATA;
    const adoptionDates = ADOPTION_DATES;

    const months = ['2025-02', '2025-03', '2025-04', '2025-05', '2025-06', '2025-07',
                    '2025-08', '2025-09', '2025-10', '2025-11', '2025-12', '2026-01'];
    const monthNames = ['Feb 2025', 'Mar 2025', 'Apr 2025', 'May 2025', 'Jun 2025', 'Jul 2025',
                        'Aug 2025', 'Sep 2025', 'Oct 2025', 'Nov 2025', 'Dec 2025', 'Jan 2026'];

    let currentMonth = 0;
    let isPlaying = false;
    let playInterval;

    // Set up SVG
    const svg = d3.select("#network");
    const width = 900;
    const height = 700;

    // Create groups for links and nodes
    const linkGroup = svg.append("g").attr("class", "links");
    const nodeGroup = svg.append("g").attr("class", "nodes");

    // Create tooltip
    const tooltip = d3.select("#tooltip");

    // Scale positions to fit SVG
    const xExtent = d3.extent(nodes, d => d.x);
    const yExtent = d3.extent(nodes, d => d.y);
    const xScale = d3.scaleLinear().domain(xExtent).range([50, width - 50]);
    const yScale = d3.scaleLinear().domain(yExtent).range([50, height - 50]);

    // Draw links
    const linkElements = linkGroup.selectAll("line")
        .data(links)
        .enter()
        .append("line")
        .attr("x1", d => xScale(nodes.find(n => n.id === d.source).x))
        .attr("y1", d => yScale(nodes.find(n => n.id === d.source).y))
        .attr("x2", d => xScale(nodes.find(n => n.id === d.target).x))
        .attr("y2", d => yScale(nodes.find(n => n.id === d.target).y))
        .attr("stroke", "#333")
        .attr("stroke-width", d => Math.min(d.weight * 0.3, 2))
        .attr("stroke-opacity", 0.2);

    // Draw nodes
    const nodeElements = nodeGroup.selectAll("circle")
        .data(nodes)
        .enter()
        .append("circle")
        .attr("cx", d => xScale(d.x))
        .attr("cy", d => yScale(d.y))
        .attr("r", d => d.isClaude ? 8 : 4)
        .attr("fill", "#444")
        .attr("stroke", "#222")
        .attr("stroke-width", 1)
        .on("mouseover", function(event, d) {
            tooltip.style("display", "block")
                   .html(`<strong>${d.id}</strong><br>Claude User: ${d.isClaude ? 'Yes' : 'No'}${d.adoptionDate ? '<br>Adopted: ' + d.adoptionDate : ''}`);
        })
        .on("mousemove", function(event) {
            tooltip.style("left", (event.pageX + 10) + "px")
                   .style("top", (event.pageY - 10) + "px");
        })
        .on("mouseout", function() {
            tooltip.style("display", "none");
        });

    // Update function
    function updateMonth(monthIndex) {
        currentMonth = monthIndex;
        const monthStr = months[monthIndex];
        const prevMonthStr = monthIndex > 0 ? months[monthIndex - 1] : null;

        document.getElementById("month-display").textContent = monthNames[monthIndex];
        document.getElementById("slider").value = monthIndex;

        // Get adopted users up to this month
        const adoptedUsers = new Set(monthlyAdoption[monthStr] || []);
        const prevAdoptedUsers = prevMonthStr ? new Set(monthlyAdoption[prevMonthStr] || []) : new Set();

        // Update node colors
        nodeElements
            .transition()
            .duration(300)
            .attr("fill", d => {
                if (adoptedUsers.has(d.id)) {
                    // Check if new this month
                    if (!prevAdoptedUsers.has(d.id)) {
                        return "#ff6b6b";  // New adopter
                    }
                    return "#d4a574";  // Existing Claude user
                }
                return "#444";  // Non-user
            })
            .attr("r", d => {
                if (adoptedUsers.has(d.id) && !prevAdoptedUsers.has(d.id)) {
                    return 14;  // Bigger for new adopters
                }
                return d.isClaude ? 8 : 4;
            });

        // Update stats
        document.getElementById("adopted-count").textContent = adoptedUsers.size;
        document.getElementById("total-nodes").textContent = nodes.length;
        document.getElementById("total-edges").textContent = links.length;

        // Reset new adopter size after animation
        setTimeout(() => {
            nodeElements
                .transition()
                .duration(500)
                .attr("r", d => adoptedUsers.has(d.id) ? 8 : 4);
        }, 500);
    }

    // Play/pause functionality
    document.getElementById("play-btn").addEventListener("click", function() {
        if (isPlaying) {
            isPlaying = false;
            clearInterval(playInterval);
            this.textContent = "Play";
        } else {
            isPlaying = true;
            this.textContent = "Pause";
            playInterval = setInterval(() => {
                if (currentMonth < 11) {
                    updateMonth(currentMonth + 1);
                } else {
                    isPlaying = false;
                    clearInterval(playInterval);
                    document.getElementById("play-btn").textContent = "Play";
                }
            }, 1500);
        }
    });

    // Reset button
    document.getElementById("reset-btn").addEventListener("click", function() {
        isPlaying = false;
        clearInterval(playInterval);
        document.getElementById("play-btn").textContent = "Play";
        updateMonth(0);
    });

    // Slider
    document.getElementById("slider").addEventListener("input", function() {
        isPlaying = false;
        clearInterval(playInterval);
        document.getElementById("play-btn").textContent = "Play";
        updateMonth(parseInt(this.value));
    });

    // Initial render
    updateMonth(0);
    </script>
</body>
</html>
'''

# Prepare data for JavaScript
nodes_data = []
for node in G_viz.nodes():
    x, y = pos[node]
    is_claude = node in adoption_dates
    adoption_date = adoption_dates.get(node)
    nodes_data.append({
        'id': node,
        'x': float(x),
        'y': float(y),
        'isClaude': is_claude,
        'adoptionDate': adoption_date.strftime('%Y-%m-%d') if pd.notna(adoption_date) else None
    })

links_data = []
for u, v, data in G_viz.edges(data=True):
    links_data.append({
        'source': u,
        'target': v,
        'weight': data.get('weight', 1)
    })

# Monthly adoption lookup
monthly_adoption_dict = {}
for item in monthly_data:
    monthly_adoption_dict[item['month']] = item['adopted_users']

# Adoption dates for JS
adoption_dates_str = {k: v.strftime('%Y-%m-%d') for k, v in adoption_dates.items() if k in viz_nodes and pd.notna(v)}

# Replace placeholders in HTML
html_content = html_content.replace('NODES_DATA', json.dumps(nodes_data))
html_content = html_content.replace('LINKS_DATA', json.dumps(links_data))
html_content = html_content.replace('MONTHLY_DATA', json.dumps(monthly_adoption_dict))
html_content = html_content.replace('ADOPTION_DATES', json.dumps(adoption_dates_str))

# Save HTML
html_path = output_dir / "anthropic_network_animation.html"
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"   Saved: {html_path}")

# Also export for Gephi (GEXF format with time attributes)
print("\n6. Exporting for Gephi (GEXF format)...")

# Add time attributes to nodes
for node in G_viz.nodes():
    if node in adoption_dates and pd.notna(adoption_dates[node]):
        G_viz.nodes[node]['claude_user'] = True
        G_viz.nodes[node]['adoption_date'] = adoption_dates[node].strftime('%Y-%m-%d')
        G_viz.nodes[node]['adoption_month'] = adoption_dates[node].strftime('%Y-%m')
    else:
        G_viz.nodes[node]['claude_user'] = False
        G_viz.nodes[node]['adoption_date'] = ''
        G_viz.nodes[node]['adoption_month'] = ''

# Save GEXF
gexf_path = output_dir / "anthropic_network.gexf"
nx.write_gexf(G_viz, gexf_path)
print(f"   Saved: {gexf_path}")

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE!")
print("=" * 70)
print(f"\nInteractive HTML: {html_path.absolute()}")
print(f"Gephi file: {gexf_path.absolute()}")
