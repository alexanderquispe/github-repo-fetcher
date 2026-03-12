# Spotify Case Study: Claude Code Adoption Contagion

## Motivation

Spotify publicly announced their transition to Claude Code, making them an ideal case study for tracking how AI tool adoption spreads through an organization's internal collaboration network.

## Research Questions

1. **When did Claude Code adoption start at Spotify?**
2. **How did it spread through the organization?**
3. **Which teams/repos adopted first?**
4. **Is there evidence of peer influence within Spotify?**

---

## Data Collection Plan

### Step 1: Fetch Spotify Organization Repos

```
GET /orgs/spotify/repos
```

Spotify has 100+ public repositories on GitHub covering:
- Backend services
- Data pipelines
- Open source tools (e.g., Luigi, Backstage)
- Mobile SDKs

### Step 2: Fetch All Contributors

For each Spotify repo:
```
GET /repos/spotify/{repo}/contributors
```

Expected: 1,000+ unique contributors across all repos

### Step 3: Match to Claude Code Users

Cross-reference Spotify contributors with our existing `claude_commits.parquet` data:
- Which Spotify contributors used Claude Code?
- When did they first use it?

### Step 4: Build Internal Collaboration Network

```
Nodes: Spotify contributors
Edges: Shared repo contributions (with weight = # shared repos)
Node attributes:
  - first_claude_use_date (if any)
  - is_adopter
  - repos_contributed_to
```

### Step 5: Analyze Adoption Timeline

1. **Adoption curve**: Cumulative Spotify Claude users over time
2. **First adopters**: Which repos/teams started using Claude first?
3. **Cascade analysis**: Did adoption spread through the network?
4. **Centrality of early adopters**: Were early adopters central in the network?

---

## API Requirements

| Endpoint | Estimated Calls | Rate Impact |
|----------|----------------|-------------|
| List org repos | 2-3 pages | ~3 calls |
| List contributors | 100+ repos × ~2 pages | ~200-300 calls |
| **Total** | | ~300 calls |

This is very manageable with 1 token (~1 minute of fetching).

---

## Implementation

### New Script: `scripts/fetch_spotify_data.py`

```python
# 1. Fetch all Spotify repos
# 2. Fetch contributors for each repo
# 3. Match to Claude commits
# 4. Build network
# 5. Compute metrics
```

### Output Files

- `data/output/spotify_repos.parquet` - All Spotify repos
- `data/output/spotify_contributors.parquet` - All contributors
- `data/output/spotify_claude_users.parquet` - Spotify + Claude intersection
- `data/output/spotify_network.graphml` - Network file for visualization
- `analysis/output/spotify_analysis.json` - Results

---

## Expected Insights

### If contagion exists:
- Early adopters will be central (high degree, betweenness)
- Adoption will cluster by repo/team
- Time-lagged correlation: user A adopts → user B (collaborator) adopts soon after

### Visualization ideas:
- Network graph colored by adoption date
- Adoption timeline with key events marked
- Cascade trees showing influence chains

---

## Timeline

1. **Data fetching**: ~10 minutes (small org)
2. **Analysis**: ~30 minutes
3. **Visualization**: ~1 hour

---

## Token to Use

New token provided:
```
[REDACTED-TOKEN]
```
