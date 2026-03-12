# Plan: Refining the Contagion Study for Publication

## Current Findings Summary

| Metric | Value |
|--------|-------|
| Claude users (May 2025) | 3,575 |
| New adopters (June 2025) | 2,838 |
| Exposed collaborators | 776,336 |
| Exposed who adopted | 877 (0.11%) |
| Dose-response | 0.06% (1 connection) → 0.23% (10+ connections) |

**Preliminary insight:** Higher exposure correlates with higher adoption, but we need stronger evidence.

---

## Issues with Current Analysis

### 1. No Control Group (Baseline)
- We don't know the adoption rate among **unexposed** developers
- Without this, we can't claim exposure matters

### 2. No Statistical Significance Tests
- Is the 0.06% → 0.23% increase statistically significant?
- Need p-values, confidence intervals

### 3. Confounding Variables Not Controlled
- Maybe active developers are both more likely to collaborate AND adopt new tools
- Maybe certain languages/domains drive both collaboration and adoption

### 4. Selection Bias
- We only looked at collaborators of Claude users
- These might be different from general GitHub population

### 5. Temporal Causality Not Established
- Did exposure happen BEFORE adoption?
- Need to verify timeline: collaborator relationship → Claude user adopts → collaborator adopts

---

## Refinement Plan

### Phase 1: Establish Baseline (Control Group)

**Goal:** Compare adoption rate of exposed vs. unexposed developers

**Option A: Random Sample Control**
```
1. Sample 100,000 random GitHub users (not collaborators of Claude users)
2. Check how many adopted Claude by June 2025
3. Compare: Exposed adoption rate vs. Random adoption rate
```

**Option B: Matched Control**
```
1. For each exposed collaborator, find a "matched" developer with:
   - Similar activity level (commits/month)
   - Similar account age
   - Similar language profile
   - NOT a collaborator of Claude users
2. Compare adoption rates
```

**Data needed:**
- Random GitHub user sample OR
- Attributes for matching (activity, languages, account age)

### Phase 2: Statistical Testing

**Tests to run:**

| Test | Purpose |
|------|---------|
| Chi-square test | Is exposed vs. unexposed adoption difference significant? |
| Logistic regression | Does exposure predict adoption, controlling for confounders? |
| Dose-response regression | Is the exposure-adoption gradient significant? |
| Permutation test | Is the observed pattern unlikely under null hypothesis? |

**Key regression model:**
```
Adopted ~ Exposure_Level + Activity + Account_Age + Language + ...
```

### Phase 3: Control for Confounders

**Variables to collect and control:**

| Variable | Why it matters |
|----------|----------------|
| **Activity level** | Active developers more likely to adopt anything |
| **Account age** | Newer accounts might be more experimental |
| **Primary language** | Some language communities adopt faster |
| **Company/org** | Enterprise vs. individual adoption patterns |
| **Open source involvement** | OSS contributors may be early adopters |
| **Prior AI tool use** | Copilot/Codex users may switch faster |

**Data sources:**
- GitHub API: user profile, contribution history
- Existing data: Copilot/Codex PR data for prior AI tool use

### Phase 4: Establish Temporal Causality

**Current gap:** We know collaborators adopted, but did exposure precede adoption?

**Refined timeline analysis:**
```
For each adopter who was exposed:
1. When did they first collaborate with a Claude user? (T_collab)
2. When did that Claude user first use Claude? (T_claude_user)
3. When did the adopter first use Claude? (T_adoption)

Valid influence: T_collab < T_claude_user < T_adoption
```

**New metric:** % of adoptions where exposure clearly preceded adoption

### Phase 5: Network Analysis Enhancements

**Current:** Simple count of Claude-using connections

**Enhancements:**

| Analysis | What it shows |
|----------|---------------|
| **Cascade trees** | Visualize chains of influence |
| **Network position** | Do central developers spread adoption faster? |
| **Clustering coefficient** | Do tight-knit groups adopt together? |
| **Time-lagged correlation** | Does a user's adoption predict neighbor adoption in next period? |

### Phase 6: Robustness Checks

**Alternative specifications:**

1. **Different time windows**
   - May → June (current)
   - April → May
   - May → July (longer window)

2. **Different exposure definitions**
   - Binary (any connection vs. none)
   - Continuous (number of connections)
   - Weighted (by collaboration intensity)

3. **Different outcome definitions**
   - Any Claude use vs. sustained use (3+ commits)
   - Claude as primary tool vs. occasional use

4. **Placebo tests**
   - Do collaborators of Copilot users adopt Copilot at higher rates? (sanity check)
   - Do collaborators of Claude users adopt unrelated tools at higher rates? (should be no)

---

## Data Collection Needed

### Must Have (for basic paper)

| Data | Source | Effort |
|------|--------|--------|
| Random user sample (control) | GitHub API | Medium |
| User activity metrics | GitHub API (events) | Medium |
| Prior AI tool use | Existing Copilot/Codex data | Low |

### Nice to Have (for stronger paper)

| Data | Source | Effort |
|------|--------|--------|
| User profile details (company, location) | GitHub API | Medium |
| Repository metadata (language, stars) | GitHub API | Medium |
| Collaboration timeline | Commit/PR timestamps | Low (have it) |

---

## Paper Structure

### Title Options
- "Peer Influence in AI Coding Tool Adoption: Evidence from GitHub"
- "Contagion Effects in Developer Tool Adoption: A Network Analysis"
- "Does Collaboration Drive AI Tool Adoption? Evidence from Claude Code"

### Sections

1. **Introduction**
   - AI coding tools are rapidly spreading
   - Understanding adoption patterns matters for tool developers, researchers
   - Research question: Does peer influence drive adoption?

2. **Related Work**
   - Technology diffusion literature
   - Social contagion in software (e.g., package adoption)
   - AI tool adoption studies

3. **Data**
   - GitHub commit/PR data identifying AI tool users
   - Collaboration network construction
   - Sample statistics

4. **Methodology**
   - Contagion model specification
   - Control group construction
   - Identification strategy (how we establish causality)

5. **Results**
   - Main finding: Exposure increases adoption
   - Dose-response relationship
   - Heterogeneity (by language, experience, etc.)
   - Robustness checks

6. **Discussion**
   - Implications for tool developers
   - Limitations
   - Future work

7. **Conclusion**

---

## Implementation Priority

### Week 1-2: Core Improvements
- [ ] Create control group (random sample)
- [ ] Run chi-square test for exposed vs. unexposed
- [ ] Add logistic regression with basic controls

### Week 3-4: Confounders
- [ ] Fetch user activity data
- [ ] Add activity/experience controls to regression
- [ ] Check if results hold

### Week 5-6: Robustness
- [ ] Run different time windows
- [ ] Run placebo tests
- [ ] Document results

### Week 7-8: Writing
- [ ] Draft paper
- [ ] Create visualizations
- [ ] Internal review

---

## Key Visualizations Needed

1. **Network graph** - Sample of collaboration network with Claude users highlighted
2. **Adoption over time** - Cumulative adoption curve
3. **Dose-response curve** - Adoption rate vs. exposure level with confidence intervals
4. **Cascade examples** - Tree diagrams showing influence chains
5. **Regression coefficients** - Forest plot of exposure effect with controls

---

## Potential Venues

| Venue | Type | Fit |
|-------|------|-----|
| ICSE | Conference | Software engineering focus |
| FSE | Conference | Empirical software engineering |
| MSR | Conference | Mining software repositories |
| CSCW | Conference | Social/collaborative aspects |
| TSE | Journal | Longer, more thorough |
| EMSE | Journal | Empirical methods |

---

## Questions to Resolve

1. **Ethical considerations** - Are we analyzing public data appropriately?
2. **Reproducibility** - Can we share data/code?
3. **Generalizability** - Does this apply to other tools, not just Claude?
4. **Mechanism** - WHY does peer influence work? (word of mouth, observation, shared projects)
