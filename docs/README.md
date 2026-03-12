# CS 145 Project Proposals: Network Analysis of AI Coding Tool Adoption

This folder contains project proposal materials for analyzing the adoption and diffusion of AI coding assistants through GitHub's developer collaboration network.

## Files

### LaTeX Documents

| File | Description | Pages |
|------|-------------|-------|
| `project_proposal_submission.tex` | **Ready-to-submit** 2-page proposal with 2 ideas | 2 |
| `project_proposals.tex` | Full detailed document with 6 proposals | ~15 |
| `project_ideas_summary.tex` | Complete catalog of all ideas + technical notes | ~10 |

### To Compile

```bash
# Compile the submission version
pdflatex project_proposal_submission.tex

# Or compile full proposals
pdflatex project_proposals.tex
pdflatex project_ideas_summary.tex
```

## Project Ideas Summary

### Recommended for Submission

1. **Cascade Effects in AI Tool Adoption** (Primary)
   - Model adoption as information cascade through collaboration network
   - Fit Independent Cascade / Linear Threshold models
   - Predict future adoptions based on network position

2. **Influence vs. Homophily in Tool Selection** (Secondary)
   - Disentangle peer influence from selection effects
   - Temporal regression + matched pair analysis
   - Network randomization tests

### Alternative Ideas (in full documents)

3. Multi-tool ecosystem as multiplex network
4. Geographic diffusion analysis
5. Influence maximization for developer targeting
6. Community detection and tool preference clustering

## Data Available

| Tool | Events | Repositories |
|------|--------|--------------|
| Claude Code | 281K commits | 439K |
| GitHub Copilot | 1.08M PRs | 247K |
| OpenAI Codex | 3.09M PRs | 249K |
| Google Jules | 622K commits | 70K |
| **Total** | **5.08M** | **~600K** |

## Analysis Code

See `../analysis/` folder for starter code:

- `network_builder.py` - Build developer collaboration networks
- `influence_analysis.py` - Influence vs. homophily analysis

### Quick Start

```python
from analysis.network_builder import DeveloperNetworkBuilder, CascadeAnalyzer

# Build network
builder = DeveloperNetworkBuilder()
G = builder.build_from_commits("data/output/claude_commits.parquet")

# Analyze cascades
analyzer = CascadeAnalyzer(G)
cascades = analyzer.detect_cascades(adoption_times)
stats = analyzer.compute_cascade_statistics()
```

## Key Network Science Concepts

- Information cascades (Independent Cascade Model)
- Social influence vs. homophily
- Bipartite network projections
- Community detection (Louvain, Infomap)
- Temporal network dynamics
- Influence maximization

## References

1. Kempe, Kleinberg, & Tardos (2003). "Maximizing the spread of influence." KDD.
2. Aral, Muchnik, & Sundararajan (2009). "Distinguishing influence from homophily." PNAS.
3. Centola (2010). "Spread of behavior in online social networks." Science.
