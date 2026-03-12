# GitHub Peru Analytics: Developer Ecosystem Dashboard

## Assignment Instructions - Detailed Guide

---

# Table of Contents

1. [Project Overview](#1-project-overview)
2. [Learning Objectives](#2-learning-objectives)
3. [Project Scope and Data Collection](#3-project-scope-and-data-collection)
4. [Technical Stack](#4-technical-stack)
5. [System Architecture](#5-system-architecture)
6. [GitHub API Implementation](#6-github-api-implementation)
7. [Industry Classification with GPT-4](#7-industry-classification-with-gpt-4)
8. [User-Level Metrics](#8-user-level-metrics)
9. [Dashboard Implementation](#9-dashboard-implementation)
10. [AI Agents Integration](#10-ai-agents-integration)
11. [Repository Structure](#11-repository-structure)
12. [Video Deliverable](#12-video-deliverable)
13. [Deliverables Checklist](#13-deliverables-checklist)
14. [Evaluation Rubric](#14-evaluation-rubric)
15. [Appendix: Resources and Examples](#15-appendix-resources-and-examples)

---

# 1. Project Overview

## 1.1 Description

**GitHub Peru Analytics** is a data analytics platform that extracts, processes, and visualizes information about the Peruvian developer ecosystem using the GitHub API. You will build a system that can:

- Extract data from 1,000+ repositories associated with Peru
- Classify repositories by industry using AI (GPT-4)
- Calculate user-level metrics for developers
- Present insights through an interactive dashboard
- Use AI agents for automated data processing tasks

## 1.2 Why This Project Matters

In your professional career, you will need to:
- Work with REST APIs and handle rate limiting
- Process and analyze large datasets
- Use AI for classification and automation tasks
- Build data-driven dashboards for decision making
- Understand developer ecosystems and tech trends

This project gives you hands-on experience with real-world data engineering and analytics.

## 1.3 The Antigravity Easter Egg

Before starting, run this in Python to discover a fun Easter egg:

```python
import antigravity
```

This opens a classic XKCD comic about Python. It's a reminder that programming should be fun! Include a screenshot of this in your submission as proof you ran it.

---

# 2. Learning Objectives

By completing this project, you will:

| Objective | Skills Developed |
|-----------|-----------------|
| **API Integration** | REST APIs, authentication, rate limiting, pagination |
| **Data Engineering** | ETL pipelines, data cleaning, storage strategies |
| **AI Classification** | Prompt engineering, GPT-4 API, batch processing |
| **Analytics** | Metric design, aggregations, statistical analysis |
| **Visualization** | Dashboard design, interactive charts, UX |
| **Agent Development** | AI agents, task automation, workflow orchestration |

---

# 3. Project Scope and Data Collection

## 3.1 Data Requirements

Each student must collect data on **at least 1,000 repositories** from Peru.

### Repository Selection Strategy

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **By Stars** (Recommended) | Top 1000 repos by star count | High quality, active projects | May miss new projects |
| **By Recent Activity** | Most recently updated repos | Current activity | May include forks |
| **By User Location** | Users with location "Peru" | Direct association | Many users don't set location |
| **Combined** | Mix of strategies | Balanced coverage | More complex |

### Search Queries to Use

```python
# Option 1: Search by location in user profile
search_queries = [
    "location:Peru",
    "location:Lima",
    "location:Arequipa",
    "location:Trujillo",
    "location:Cusco"
]

# Option 2: Search repos with Peru-related topics
repo_queries = [
    "topic:peru",
    "topic:lima",
    "topic:peruvian"
]

# Option 3: Search users, then get their repos
user_search = "location:Peru type:user"
```

## 3.2 GitHub Token Setup

### REQUIRED: Create a Personal Access Token

1. Go to GitHub → Settings → Developer Settings → Personal Access Tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Configure:
   - **Note**: "GitHub Peru Analytics - [Your Name]"
   - **Expiration**: No expiration (for this project only)
   - **Scopes**: Select `public_repo`, `read:user`
4. Copy and save your token securely

### Security Warning

```
⚠️ NEVER commit your token to GitHub
⚠️ NEVER share your token with others
⚠️ Add .env to your .gitignore file
```

## 3.3 Rate Limiting Considerations

| Authentication | Rate Limit | Requests/Hour |
|---------------|------------|---------------|
| No token | 60 requests/hour | Very limited |
| With token | 5,000 requests/hour | Sufficient |
| GitHub App | 15,000 requests/hour | Not needed |

**Implementation Required**: Your code must handle rate limits gracefully with exponential backoff.

---

# 4. Technical Stack

## 4.1 Required Technologies

| Category | Technology | Purpose |
|----------|------------|---------|
| **Language** | Python 3.10+ | Primary programming language |
| **GitHub API** | REST API v3 or GraphQL v4 | Data extraction |
| **LLM API** | OpenAI GPT-4 API | Industry classification |
| **Database** | SQLite or PostgreSQL | Data storage |
| **Dashboard** | Streamlit or Plotly Dash | Visualization |

## 4.2 Recommended Packages

```txt
# Core
requests>=2.31.0
httpx>=0.25.0           # Async HTTP client (optional)
python-dotenv>=1.0.0

# OpenAI
openai>=1.12.0
tiktoken>=0.5.0

# Data Processing
pandas>=2.0.0
numpy>=1.24.0

# Database
sqlalchemy>=2.0.0
sqlite3                 # Built-in

# Visualization
streamlit>=1.31.0
plotly>=5.18.0
altair>=5.2.0

# Utilities
tqdm>=4.66.0
loguru>=0.7.0
tenacity>=8.2.0         # For retry logic

# Development
pytest>=7.4.0
black>=24.0.0
```

## 4.3 Environment Setup

Create a `.env` file (DO NOT commit):

```env
GITHUB_TOKEN=ghp_your_token_here
OPENAI_API_KEY=sk-your-api-key-here
```

Create a `.env.example` file (commit this):

```env
GITHUB_TOKEN=ghp_your_token_here
OPENAI_API_KEY=sk-your-api-key-here
```

---

# 5. System Architecture

## 5.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     GITHUB PERU ANALYTICS                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────────┐    │
│  │  GITHUB API  │───▶│  DATA EXTRACTION │───▶│     DATABASE       │    │
│  │              │    │     PIPELINE     │    │   (SQLite/Postgres)│    │
│  │ - Users      │    │                  │    │                    │    │
│  │ - Repos      │    │ - Rate Limiting  │    │ - Users table      │    │
│  │ - Commits    │    │ - Pagination     │    │ - Repos table      │    │
│  │ - Languages  │    │ - Error Handling │    │ - Languages table  │    │
│  └──────────────┘    └──────────────────┘    └─────────┬──────────┘    │
│                                                         │               │
│  ┌──────────────────────────────────────────────────────┼───────────┐  │
│  │                 AI CLASSIFICATION PIPELINE            │           │  │
│  │                                                       ▼           │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐   │  │
│  │  │   REPO      │───▶│   GPT-4     │───▶│   CLASSIFIED DATA   │   │  │
│  │  │   CONTEXT   │    │   AGENT     │    │   (21 Industries)   │   │  │
│  │  │             │    │             │    │                     │   │  │
│  │  │ - Name      │    │ - Classify  │    │ - Agriculture       │   │  │
│  │  │ - Desc      │    │ - Validate  │    │ - Mining            │   │  │
│  │  │ - README    │    │ - Batch     │    │ - Tech              │   │  │
│  │  │ - Topics    │    │             │    │ - Finance...        │   │  │
│  │  └─────────────┘    └─────────────┘    └──────────┬──────────┘   │  │
│  └───────────────────────────────────────────────────┼──────────────┘  │
│                                                       │                 │
│  ┌────────────────────────────────────────────────────┼─────────────┐  │
│  │                    ANALYTICS ENGINE                 │             │  │
│  │                                                     ▼             │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐   │  │
│  │  │ USER-LEVEL  │  │ REPO-LEVEL  │  │   ECOSYSTEM METRICS     │   │  │
│  │  │  METRICS    │  │  METRICS    │  │                         │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                      DASHBOARD (Streamlit)                        │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │  │
│  │  │ OVERVIEW │  │ USERS    │  │ REPOS    │  │ INDUSTRY         │  │  │
│  │  │ STATS    │  │ EXPLORER │  │ BROWSER  │  │ CLASSIFICATION   │  │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 5.2 Component Breakdown

| Component | Responsibility | Key Files |
|-----------|---------------|-----------|
| **Data Extraction** | GitHub API calls, rate limiting | `src/extraction/` |
| **Data Storage** | Database schema, CRUD operations | `src/database/` |
| **Classification** | GPT-4 industry classification | `src/classification/` |
| **Metrics** | User and repo level calculations | `src/metrics/` |
| **Agents** | AI-powered automation | `src/agents/` |
| **Dashboard** | Web interface and charts | `app/` |

---

# 6. GitHub API Implementation

## 6.1 Authentication and Setup

```python
# src/extraction/github_client.py
import os
import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

class GitHubClient:
    def __init__(self):
        self.token = os.getenv("GITHUB_TOKEN")
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json"
        }

    def check_rate_limit(self) -> dict:
        """Check current rate limit status."""
        response = requests.get(
            f"{self.base_url}/rate_limit",
            headers=self.headers
        )
        return response.json()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def make_request(self, endpoint: str, params: dict = None) -> dict:
        """Make a rate-limit-aware request to GitHub API."""
        response = requests.get(
            f"{self.base_url}/{endpoint}",
            headers=self.headers,
            params=params
        )

        # Check rate limit
        remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
        if remaining < 10:
            reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
            # Handle waiting logic here

        response.raise_for_status()
        return response.json()
```

## 6.2 User Search and Extraction

```python
# src/extraction/user_extractor.py

class UserExtractor:
    def __init__(self, client: GitHubClient):
        self.client = client

    def search_users_by_location(
        self,
        location: str,
        max_users: int = 1000
    ) -> list[dict]:
        """
        Search for users by location.

        Args:
            location: Location string (e.g., "Peru", "Lima")
            max_users: Maximum number of users to retrieve

        Returns:
            List of user dictionaries
        """
        users = []
        page = 1
        per_page = 100  # Max allowed by GitHub

        while len(users) < max_users:
            result = self.client.make_request(
                "search/users",
                params={
                    "q": f"location:{location}",
                    "per_page": per_page,
                    "page": page,
                    "sort": "followers",  # Get most influential first
                    "order": "desc"
                }
            )

            if not result.get("items"):
                break

            users.extend(result["items"])
            page += 1

            # GitHub search API limits to 1000 results
            if page * per_page >= 1000:
                break

        return users[:max_users]

    def get_user_details(self, username: str) -> dict:
        """Get detailed information for a specific user."""
        return self.client.make_request(f"users/{username}")

    def get_user_repos(self, username: str) -> list[dict]:
        """Get all repositories for a user."""
        repos = []
        page = 1

        while True:
            result = self.client.make_request(
                f"users/{username}/repos",
                params={
                    "per_page": 100,
                    "page": page,
                    "type": "owner"  # Only owned repos, not forks
                }
            )

            if not result:
                break

            repos.extend(result)
            page += 1

        return repos
```

## 6.3 Repository Data Extraction

```python
# src/extraction/repo_extractor.py

class RepoExtractor:
    def __init__(self, client: GitHubClient):
        self.client = client

    def search_repos_by_stars(
        self,
        location_users: list[str],
        min_stars: int = 1
    ) -> list[dict]:
        """
        Search for top repositories from users in a location.
        """
        repos = []

        for username in location_users:
            user_repos = self.client.make_request(
                f"users/{username}/repos",
                params={"sort": "stars", "direction": "desc"}
            )

            for repo in user_repos:
                if repo["stargazers_count"] >= min_stars:
                    repos.append(repo)

        # Sort all repos by stars and take top 1000
        repos.sort(key=lambda x: x["stargazers_count"], reverse=True)
        return repos[:1000]

    def get_repo_readme(self, owner: str, repo: str) -> str:
        """
        Get the README content of a repository.
        Returns empty string if not found.
        """
        try:
            result = self.client.make_request(
                f"repos/{owner}/{repo}/readme"
            )

            import base64
            content = base64.b64decode(result["content"]).decode("utf-8")
            return content[:5000]  # Limit to 5000 chars for API costs

        except Exception:
            return ""

    def get_repo_languages(self, owner: str, repo: str) -> dict:
        """Get the language breakdown of a repository."""
        return self.client.make_request(f"repos/{owner}/{repo}/languages")

    def get_repo_contributors(self, owner: str, repo: str) -> list[dict]:
        """Get the contributors of a repository."""
        return self.client.make_request(
            f"repos/{owner}/{repo}/contributors",
            params={"per_page": 100}
        )
```

## 6.4 Data to Extract

### User Data

| Field | API Endpoint | Description |
|-------|--------------|-------------|
| `id` | users/{username} | Unique GitHub ID |
| `login` | users/{username} | Username |
| `name` | users/{username} | Display name |
| `location` | users/{username} | User's location |
| `bio` | users/{username} | User biography |
| `company` | users/{username} | Company/organization |
| `blog` | users/{username} | Website URL |
| `email` | users/{username} | Public email |
| `followers` | users/{username} | Follower count |
| `following` | users/{username} | Following count |
| `public_repos` | users/{username} | Number of public repos |
| `created_at` | users/{username} | Account creation date |
| `updated_at` | users/{username} | Last update date |

### Repository Data

| Field | API Endpoint | Description |
|-------|--------------|-------------|
| `id` | repos/{owner}/{repo} | Unique repo ID |
| `name` | repos/{owner}/{repo} | Repository name |
| `full_name` | repos/{owner}/{repo} | owner/repo format |
| `description` | repos/{owner}/{repo} | Repository description |
| `topics` | repos/{owner}/{repo} | Repository topics/tags |
| `language` | repos/{owner}/{repo} | Primary language |
| `languages` | repos/{owner}/{repo}/languages | All languages used |
| `stargazers_count` | repos/{owner}/{repo} | Number of stars |
| `forks_count` | repos/{owner}/{repo} | Number of forks |
| `watchers_count` | repos/{owner}/{repo} | Number of watchers |
| `open_issues_count` | repos/{owner}/{repo} | Open issues count |
| `created_at` | repos/{owner}/{repo} | Creation date |
| `updated_at` | repos/{owner}/{repo} | Last update |
| `pushed_at` | repos/{owner}/{repo} | Last push |
| `license` | repos/{owner}/{repo} | License type |
| `readme` | repos/{owner}/{repo}/readme | README content |

---

# 7. Industry Classification with GPT-4

## 7.1 CIIU Industry Categories (Peru)

You will classify repositories into these 21 industry categories based on Peru's CIIU (Clasificación Industrial Internacional Uniforme):

| Code | Industry (Spanish) | Industry (English) |
|------|-------------------|-------------------|
| A | Agricultura, ganadería, silvicultura y pesca | Agriculture, forestry and fishing |
| B | Explotación de minas y canteras | Mining and quarrying |
| C | Industrias manufactureras | Manufacturing |
| D | Suministro de electricidad, gas, vapor | Electricity, gas, steam supply |
| E | Suministro de agua; alcantarillado | Water supply; sewerage |
| F | Construcción | Construction |
| G | Comercio al por mayor y menor | Wholesale and retail trade |
| H | Transporte y almacenamiento | Transportation and storage |
| I | Alojamiento y servicios de comida | Accommodation and food services |
| J | Información y comunicaciones | Information and communication |
| K | Actividades financieras y de seguros | Financial and insurance activities |
| L | Actividades inmobiliarias | Real estate activities |
| M | Actividades profesionales, científicas | Professional, scientific activities |
| N | Actividades administrativas y de apoyo | Administrative and support activities |
| O | Administración pública y defensa | Public administration and defense |
| P | Enseñanza | Education |
| Q | Salud humana y asistencia social | Human health and social work |
| R | Artes, entretenimiento y recreación | Arts, entertainment and recreation |
| S | Otras actividades de servicios | Other service activities |
| T | Actividades de hogares | Activities of households |
| U | Actividades de organizaciones extraterritoriales | Extraterritorial organizations |

## 7.2 Classification Implementation

```python
# src/classification/industry_classifier.py
from openai import OpenAI
import json

class IndustryClassifier:
    def __init__(self):
        self.client = OpenAI()
        self.industries = {
            "A": "Agriculture, forestry and fishing",
            "B": "Mining and quarrying",
            "C": "Manufacturing",
            "D": "Electricity, gas, steam supply",
            "E": "Water supply; sewerage",
            "F": "Construction",
            "G": "Wholesale and retail trade",
            "H": "Transportation and storage",
            "I": "Accommodation and food services",
            "J": "Information and communication",
            "K": "Financial and insurance activities",
            "L": "Real estate activities",
            "M": "Professional, scientific activities",
            "N": "Administrative and support activities",
            "O": "Public administration and defense",
            "P": "Education",
            "Q": "Human health and social work",
            "R": "Arts, entertainment and recreation",
            "S": "Other service activities",
            "T": "Activities of households",
            "U": "Extraterritorial organizations"
        }

    def classify_repository(
        self,
        name: str,
        description: str,
        readme: str,
        topics: list[str],
        language: str
    ) -> dict:
        """
        Classify a repository into an industry category.

        Returns:
            dict with keys: industry_code, industry_name, confidence, reasoning
        """

        prompt = f"""Analyze this GitHub repository and classify it into ONE of the following industry categories based on its potential application or the industry it serves.

REPOSITORY INFORMATION:
- Name: {name}
- Description: {description or 'No description'}
- Primary Language: {language or 'Not specified'}
- Topics: {', '.join(topics) if topics else 'None'}
- README (first 2000 chars): {readme[:2000] if readme else 'No README'}

INDUSTRY CATEGORIES:
{json.dumps(self.industries, indent=2)}

INSTRUCTIONS:
1. Analyze the repository's purpose, functionality, and potential use cases
2. Consider what industry would most benefit from or use this software
3. If it's a general-purpose tool (e.g., utility library), classify based on the most likely industry application
4. If truly generic (e.g., "hello world"), use "J" (Information and communication)

Respond in JSON format:
{{
    "industry_code": "X",
    "industry_name": "Full industry name",
    "confidence": "high|medium|low",
    "reasoning": "Brief explanation of why this classification was chosen"
}}
"""

        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at classifying software projects by industry. Always respond with valid JSON."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)

    def batch_classify(
        self,
        repositories: list[dict],
        batch_size: int = 10
    ) -> list[dict]:
        """
        Classify multiple repositories efficiently.
        """
        results = []

        for i in range(0, len(repositories), batch_size):
            batch = repositories[i:i+batch_size]

            for repo in batch:
                classification = self.classify_repository(
                    name=repo.get("name", ""),
                    description=repo.get("description", ""),
                    readme=repo.get("readme", ""),
                    topics=repo.get("topics", []),
                    language=repo.get("language", "")
                )

                results.append({
                    "repo_id": repo["id"],
                    "repo_name": repo["name"],
                    **classification
                })

        return results
```

## 7.3 Classification Guidelines

### High Confidence Classifications

| Repository Type | Industry Code | Reasoning |
|----------------|---------------|-----------|
| Mobile banking app | K | Direct financial service |
| Hospital management system | Q | Healthcare application |
| E-commerce platform | G | Retail/wholesale |
| School LMS | P | Education software |
| Mining data analysis | B | Mining industry tool |

### Low Confidence / Edge Cases

| Repository Type | Suggested Code | Alternative |
|----------------|----------------|-------------|
| Generic REST API | J | M (if consulting focused) |
| Machine learning library | J or M | Depends on domain |
| DevOps tools | J | N (if administrative) |
| Game projects | R | J (if engine/framework) |
| Personal portfolios | J | S (other services) |

---

# 8. User-Level Metrics

## 8.1 Required Metrics

You must calculate ALL of the following metrics for each user:

### Activity Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| `total_repos` | Count of owned repos | Number of public repositories |
| `total_stars_received` | Sum of stars across repos | Total stargazers |
| `total_forks_received` | Sum of forks across repos | Total forks |
| `avg_stars_per_repo` | total_stars / total_repos | Average popularity |
| `account_age_days` | today - created_at | Days since account creation |
| `repos_per_year` | total_repos / (age_years) | Repository creation rate |

### Influence Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| `followers` | Direct from API | Number of followers |
| `following` | Direct from API | Number following |
| `follower_ratio` | followers / following | Influence ratio |
| `h_index` | GitHub h-index | h repos with >= h stars |
| `impact_score` | stars + (forks * 2) + followers | Composite impact |

### Technical Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| `primary_languages` | Top 3 by repo count | Main programming languages |
| `language_diversity` | Unique languages count | Technical breadth |
| `industries_served` | Unique industry codes | Industry diversity |
| `has_readme_pct` | repos_with_readme / total | Documentation quality |
| `has_license_pct` | repos_with_license / total | Professionalism |

### Engagement Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| `total_open_issues` | Sum across repos | Engagement level |
| `days_since_last_push` | today - last_pushed_at | Activity recency |
| `is_active` | last_push < 90 days | Active status |
| `contribution_consistency` | Custom calculation | Regular activity |

## 8.2 Metrics Implementation

```python
# src/metrics/user_metrics.py
from datetime import datetime, timedelta
from collections import Counter

class UserMetricsCalculator:
    def __init__(self):
        self.today = datetime.now()

    def calculate_all_metrics(
        self,
        user: dict,
        repos: list[dict],
        classifications: list[dict]
    ) -> dict:
        """
        Calculate all user-level metrics.

        Args:
            user: User data from GitHub API
            repos: List of user's repositories
            classifications: Industry classifications for repos

        Returns:
            Dictionary with all calculated metrics
        """
        metrics = {}

        # Activity Metrics
        metrics["total_repos"] = len(repos)
        metrics["total_stars_received"] = sum(r["stargazers_count"] for r in repos)
        metrics["total_forks_received"] = sum(r["forks_count"] for r in repos)
        metrics["avg_stars_per_repo"] = (
            metrics["total_stars_received"] / metrics["total_repos"]
            if metrics["total_repos"] > 0 else 0
        )

        created_at = datetime.fromisoformat(user["created_at"].replace("Z", ""))
        metrics["account_age_days"] = (self.today - created_at).days
        metrics["repos_per_year"] = (
            metrics["total_repos"] / (metrics["account_age_days"] / 365)
            if metrics["account_age_days"] > 0 else 0
        )

        # Influence Metrics
        metrics["followers"] = user["followers"]
        metrics["following"] = user["following"]
        metrics["follower_ratio"] = (
            metrics["followers"] / metrics["following"]
            if metrics["following"] > 0 else metrics["followers"]
        )
        metrics["h_index"] = self._calculate_h_index(repos)
        metrics["impact_score"] = (
            metrics["total_stars_received"] +
            (metrics["total_forks_received"] * 2) +
            metrics["followers"]
        )

        # Technical Metrics
        languages = [r["language"] for r in repos if r["language"]]
        lang_counts = Counter(languages)
        metrics["primary_languages"] = [l for l, _ in lang_counts.most_common(3)]
        metrics["language_diversity"] = len(set(languages))

        industry_codes = [c["industry_code"] for c in classifications]
        metrics["industries_served"] = len(set(industry_codes))
        metrics["primary_industry"] = Counter(industry_codes).most_common(1)[0][0] if industry_codes else None

        # Documentation quality
        repos_with_readme = sum(1 for r in repos if r.get("has_readme", False))
        repos_with_license = sum(1 for r in repos if r.get("license"))
        metrics["has_readme_pct"] = repos_with_readme / len(repos) if repos else 0
        metrics["has_license_pct"] = repos_with_license / len(repos) if repos else 0

        # Engagement Metrics
        metrics["total_open_issues"] = sum(r["open_issues_count"] for r in repos)

        if repos:
            last_push = max(
                datetime.fromisoformat(r["pushed_at"].replace("Z", ""))
                for r in repos
            )
            metrics["days_since_last_push"] = (self.today - last_push).days
            metrics["is_active"] = metrics["days_since_last_push"] < 90
        else:
            metrics["days_since_last_push"] = None
            metrics["is_active"] = False

        return metrics

    def _calculate_h_index(self, repos: list[dict]) -> int:
        """
        Calculate h-index based on repository stars.
        h-index = h if h repos have at least h stars each.
        """
        stars = sorted([r["stargazers_count"] for r in repos], reverse=True)
        h = 0
        for i, s in enumerate(stars):
            if s >= i + 1:
                h = i + 1
            else:
                break
        return h
```

## 8.3 Aggregated Ecosystem Metrics

Calculate these metrics for the entire Peru ecosystem:

| Metric | Description |
|--------|-------------|
| `total_developers` | Total unique developers |
| `total_repositories` | Total repositories |
| `total_stars` | Sum of all stars |
| `avg_repos_per_user` | Average repos per user |
| `most_popular_languages` | Top 10 languages |
| `industry_distribution` | Repos per industry |
| `active_developer_pct` | % active in last 90 days |
| `avg_account_age` | Average developer tenure |

---

# 9. Dashboard Implementation

## 9.1 Required Pages/Views

### Page 1: Overview Dashboard

Display key ecosystem statistics:
- Total developers, repositories, stars
- Trend charts (if historical data available)
- Top 10 developers by impact score
- Top 10 repositories by stars

### Page 2: Developer Explorer

- Searchable/filterable table of all developers
- Sort by any metric
- Click to see developer detail
- Export to CSV

### Page 3: Repository Browser

- All 1000+ repositories
- Filter by industry, language, stars
- Search by name/description
- View classification details

### Page 4: Industry Analysis

- Distribution chart (pie/bar) of industries
- Top repos per industry
- Industry trends
- Developer specialization by industry

### Page 5: Language Analytics

- Programming language distribution
- Language trends
- Top developers per language
- Language by industry correlation

## 9.2 Streamlit Implementation

```python
# app/main.py
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="GitHub Peru Analytics",
    page_icon="🇵🇪",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    users_df = pd.read_csv("data/users.csv")
    repos_df = pd.read_csv("data/repositories.csv")
    classifications_df = pd.read_csv("data/classifications.csv")
    return users_df, repos_df, classifications_df

users_df, repos_df, classifications_df = load_data()

# Sidebar
st.sidebar.title("🇵🇪 GitHub Peru Analytics")
st.sidebar.markdown("Analyzing the Peruvian developer ecosystem")

page = st.sidebar.selectbox(
    "Navigate to:",
    ["Overview", "Developers", "Repositories", "Industries", "Languages"]
)

if page == "Overview":
    st.title("Peru Developer Ecosystem Overview")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Developers", len(users_df))
    with col2:
        st.metric("Total Repositories", len(repos_df))
    with col3:
        st.metric("Total Stars", repos_df["stargazers_count"].sum())
    with col4:
        active_pct = (users_df["is_active"].sum() / len(users_df)) * 100
        st.metric("Active Developers", f"{active_pct:.1f}%")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 Developers by Impact")
        top_devs = users_df.nlargest(10, "impact_score")[["login", "impact_score"]]
        fig = px.bar(top_devs, x="login", y="impact_score")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Industry Distribution")
        industry_counts = classifications_df["industry_name"].value_counts()
        fig = px.pie(values=industry_counts.values, names=industry_counts.index)
        st.plotly_chart(fig, use_container_width=True)

elif page == "Developers":
    st.title("Developer Explorer")

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        min_stars = st.slider("Minimum Stars", 0, 1000, 0)
    with col2:
        language_filter = st.multiselect(
            "Primary Language",
            options=users_df["primary_language_1"].unique()
        )
    with col3:
        active_only = st.checkbox("Active developers only")

    # Filter data
    filtered_df = users_df.copy()
    if min_stars > 0:
        filtered_df = filtered_df[filtered_df["total_stars_received"] >= min_stars]
    if language_filter:
        filtered_df = filtered_df[filtered_df["primary_language_1"].isin(language_filter)]
    if active_only:
        filtered_df = filtered_df[filtered_df["is_active"] == True]

    # Display table
    st.dataframe(
        filtered_df[["login", "name", "total_repos", "total_stars_received",
                     "followers", "impact_score", "primary_language_1"]],
        use_container_width=True
    )

# ... Continue for other pages
```

## 9.3 Required Visualizations

| Visualization | Type | Library |
|---------------|------|---------|
| Industry distribution | Pie/Donut chart | Plotly |
| Language distribution | Bar chart | Plotly/Altair |
| Developer impact ranking | Horizontal bar | Plotly |
| Activity timeline | Line chart | Plotly |
| Stars vs Forks scatter | Scatter plot | Plotly |
| Industry by language | Heatmap | Plotly |
| Geographic map (if location data) | Choropleth | Plotly |

---

# 10. AI Agents Integration

## 10.1 Agent Architecture

Build at least ONE of the following agents:

### Option A: Data Collection Agent

An autonomous agent that:
- Searches for Peru-related users and repos
- Handles rate limiting automatically
- Decides when to dig deeper vs. move on
- Logs its decisions and actions

### Option B: Classification Agent

An autonomous agent that:
- Reads repository information
- Decides if it needs more context (README, commits)
- Classifies into industries
- Handles edge cases intelligently

### Option C: Insights Agent

An autonomous agent that:
- Analyzes the collected data
- Generates insights about the ecosystem
- Answers natural language questions
- Creates summary reports

## 10.2 Agent Implementation Example

```python
# src/agents/classification_agent.py
from openai import OpenAI
import json

class ClassificationAgent:
    def __init__(self):
        self.client = OpenAI()
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_readme",
                    "description": "Fetch the README content of a repository",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "owner": {"type": "string"},
                            "repo": {"type": "string"}
                        },
                        "required": ["owner", "repo"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_languages",
                    "description": "Get the programming languages used in a repository",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "owner": {"type": "string"},
                            "repo": {"type": "string"}
                        },
                        "required": ["owner", "repo"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "classify_industry",
                    "description": "Classify a repository into an industry category",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "repo_name": {"type": "string"},
                            "industry_code": {"type": "string"},
                            "confidence": {"type": "string"},
                            "reasoning": {"type": "string"}
                        },
                        "required": ["repo_name", "industry_code", "confidence", "reasoning"]
                    }
                }
            }
        ]

    def run(self, repository: dict) -> dict:
        """
        Run the classification agent on a repository.
        The agent decides what information it needs and classifies the repo.
        """
        messages = [
            {
                "role": "system",
                "content": """You are an AI agent that classifies GitHub repositories into industry categories.

Your task is to analyze a repository and determine which industry it serves.
You have access to tools to get more information if needed.

Industry codes: A (Agriculture), B (Mining), C (Manufacturing), D (Utilities),
E (Water), F (Construction), G (Trade), H (Transport), I (Hospitality),
J (Information/Tech), K (Finance), L (Real Estate), M (Professional Services),
N (Administrative), O (Public Admin), P (Education), Q (Health), R (Arts),
S (Other Services), T (Households), U (International Orgs)

Steps:
1. Review the basic repository information
2. If unclear, fetch the README for more context
3. Make your classification using the classify_industry tool
"""
            },
            {
                "role": "user",
                "content": f"""Please classify this repository:

Name: {repository['name']}
Description: {repository.get('description', 'No description')}
Primary Language: {repository.get('language', 'Unknown')}
Topics: {', '.join(repository.get('topics', []))}
Stars: {repository.get('stargazers_count', 0)}
"""
            }
        ]

        while True:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )

            message = response.choices[0].message

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)

                    # Execute tool
                    if function_name == "get_readme":
                        result = self._get_readme(arguments["owner"], arguments["repo"])
                    elif function_name == "get_languages":
                        result = self._get_languages(arguments["owner"], arguments["repo"])
                    elif function_name == "classify_industry":
                        # Final classification - return result
                        return arguments

                    messages.append(message)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result)
                    })
            else:
                # No tool calls - agent is done (shouldn't happen with proper prompting)
                break

        return {"error": "Agent did not produce classification"}

    def _get_readme(self, owner: str, repo: str) -> str:
        # Implement GitHub API call
        pass

    def _get_languages(self, owner: str, repo: str) -> dict:
        # Implement GitHub API call
        pass
```

## 10.3 Agent Requirements

Your agent implementation must include:

| Requirement | Description |
|-------------|-------------|
| **Autonomy** | Agent makes decisions without human intervention |
| **Tool use** | Agent uses at least 2 different tools |
| **Reasoning** | Agent explains its decisions |
| **Error handling** | Agent handles failures gracefully |
| **Logging** | Agent logs its actions for debugging |

---

# 11. Repository Structure

## 11.1 Required Structure

```
github-peru-analytics/
│
├── README.md                    # Main documentation (REQUIRED)
├── requirements.txt             # Python dependencies (REQUIRED)
├── .env.example                # Environment template (REQUIRED)
├── .gitignore                  # Git ignore file (REQUIRED)
│
├── data/                        # Extracted and processed data
│   ├── raw/                    # Raw API responses
│   │   ├── users/
│   │   └── repos/
│   ├── processed/              # Cleaned data
│   │   ├── users.csv
│   │   ├── repositories.csv
│   │   └── classifications.csv
│   └── metrics/                # Calculated metrics
│       ├── user_metrics.csv
│       └── ecosystem_metrics.json
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── extraction/             # GitHub API extraction
│   │   ├── __init__.py
│   │   ├── github_client.py
│   │   ├── user_extractor.py
│   │   └── repo_extractor.py
│   ├── database/               # Data storage
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── crud.py
│   ├── classification/         # Industry classification
│   │   ├── __init__.py
│   │   └── industry_classifier.py
│   ├── metrics/                # Metrics calculation
│   │   ├── __init__.py
│   │   ├── user_metrics.py
│   │   └── ecosystem_metrics.py
│   └── agents/                 # AI agents
│       ├── __init__.py
│       └── classification_agent.py
│
├── app/                         # Dashboard application
│   ├── main.py                 # Main Streamlit app
│   ├── pages/
│   │   ├── 1_Overview.py
│   │   ├── 2_Developers.py
│   │   ├── 3_Repositories.py
│   │   ├── 4_Industries.py
│   │   └── 5_Languages.py
│   └── components/
│       └── charts.py
│
├── scripts/                     # Utility scripts
│   ├── extract_data.py         # Run data extraction
│   ├── classify_repos.py       # Run classification
│   └── calculate_metrics.py    # Calculate all metrics
│
├── notebooks/                   # Jupyter notebooks (optional)
│   └── exploration.ipynb
│
├── demo/                        # Demo materials
│   ├── screenshots/
│   │   ├── overview.png
│   │   ├── developers.png
│   │   └── industries.png
│   ├── video_link.md
│   └── antigravity_screenshot.png  # REQUIRED: Screenshot of antigravity easter egg
│
└── tests/                       # Unit tests
    ├── test_extraction.py
    ├── test_classification.py
    └── test_metrics.py
```

## 11.2 README.md Requirements

Your README must include:

### Section 1: Project Title and Description
- Clear project name with Peru focus
- One-paragraph description
- Screenshot of antigravity Easter egg

### Section 2: Key Findings
- Top 5 insights about Peru's developer ecosystem
- Most popular languages
- Industry distribution highlights

### Section 3: Data Collection
- How many users/repos collected
- Time period of data
- Rate limiting approach

### Section 4: Features
- List of all dashboard features
- Screenshots of each page

### Section 5: Installation
- Step-by-step setup instructions
- How to set up GitHub token
- How to set up OpenAI key

### Section 6: Usage
- How to run data extraction
- How to run classification
- How to start dashboard

### Section 7: Metrics Documentation
- All user-level metrics explained
- All ecosystem metrics explained

### Section 8: AI Agent Documentation
- Agent architecture
- Tool descriptions
- Example agent runs

### Section 9: Limitations
- At least 3 documented limitations
- Data collection biases
- Classification limitations

### Section 10: Author Information
- Your name
- Course information
- Date

---

# 12. Video Deliverable

## 12.1 Video Requirements

| Aspect | Requirement |
|--------|-------------|
| **Duration** | 3-5 minutes maximum |
| **Format** | MP4 or YouTube (unlisted) |
| **Resolution** | Minimum 720p |
| **Audio** | Clear narration required |

## 12.2 Video Structure

### Part 1: Introduction (0:00 - 0:30)
- Show antigravity Easter egg running
- Your name and project overview
- Key statistics about data collected

### Part 2: Data Pipeline (0:30 - 1:30)
- Show GitHub API extraction code
- Explain rate limiting handling
- Show classification with GPT-4

### Part 3: Dashboard Demo (1:30 - 3:30)
- Navigate through all 5 pages
- Show filters and interactivity
- Highlight key visualizations
- Point out interesting findings

### Part 4: AI Agent Demo (3:30 - 4:30)
- Show agent classifying a repository
- Explain agent decision-making
- Show tool usage

### Part 5: Conclusions (4:30 - 5:00)
- 3 key insights about Peru developers
- Limitations of your analysis
- Potential improvements

---

# 13. Deliverables Checklist

## Use this checklist before submission:

### Data Collection
- [ ] 1,000+ repositories collected
- [ ] User data for all repo owners
- [ ] README content extracted
- [ ] Languages data extracted
- [ ] Data saved in CSV/JSON format

### Classification
- [ ] All repos classified into 21 industries
- [ ] Classification includes confidence scores
- [ ] Classification reasoning documented

### Metrics
- [ ] All user-level metrics calculated
- [ ] Ecosystem-level metrics calculated
- [ ] Metrics saved to files

### Dashboard
- [ ] Overview page with key stats
- [ ] Developer explorer with filters
- [ ] Repository browser
- [ ] Industry analysis visualizations
- [ ] Language analytics

### AI Agent
- [ ] At least 1 agent implemented
- [ ] Agent uses multiple tools
- [ ] Agent reasoning logged

### Documentation
- [ ] README complete
- [ ] Antigravity screenshot included
- [ ] All metrics documented
- [ ] Limitations listed

### Code Quality
- [ ] Code runs with documented commands
- [ ] No hardcoded API keys
- [ ] Error handling implemented
- [ ] .env.example provided

### Video
- [ ] 3-5 minutes duration
- [ ] All sections covered
- [ ] Demo shows working system
- [ ] Link included in repository

---

# 14. Evaluation Rubric

## Total Points: 100

### 1. Data Collection (25 points)

| Criteria | Points | Description |
|----------|--------|-------------|
| API Implementation | 8 | Proper authentication, rate limiting |
| Data Volume | 7 | 1000+ repos, complete user data |
| Data Quality | 5 | Clean, no duplicates, proper format |
| README/Language extraction | 5 | Additional context data |

### 2. Industry Classification (20 points)

| Criteria | Points | Description |
|----------|--------|-------------|
| GPT-4 Integration | 8 | Proper API usage, prompting |
| Classification Quality | 7 | Reasonable categorization |
| Confidence Scores | 5 | Proper uncertainty handling |

### 3. User Metrics (15 points)

| Criteria | Points | Description |
|----------|--------|-------------|
| Activity Metrics | 5 | Repos, stars, forks |
| Influence Metrics | 5 | Followers, h-index |
| Technical Metrics | 5 | Languages, industries |

### 4. Dashboard (20 points)

| Criteria | Points | Description |
|----------|--------|-------------|
| Overview Page | 4 | Key statistics, charts |
| Developer Explorer | 4 | Filtering, sorting |
| Repository Browser | 4 | Search, details |
| Industry Analysis | 4 | Distribution, insights |
| Visual Design | 4 | Professional appearance |

### 5. AI Agent (10 points)

| Criteria | Points | Description |
|----------|--------|-------------|
| Agent Architecture | 4 | Proper tool usage |
| Autonomy | 3 | Makes decisions |
| Documentation | 3 | Reasoning logged |

### 6. Code & Documentation (10 points)

| Criteria | Points | Description |
|----------|--------|-------------|
| Repository Structure | 3 | Follows required structure |
| README Completeness | 4 | All sections present |
| Code Quality | 3 | Clean, documented |

---

# 15. Appendix: Resources and Examples

## 15.1 Useful Documentation

### GitHub API
- [GitHub REST API Documentation](https://docs.github.com/en/rest)
- [Rate Limiting](https://docs.github.com/en/rest/overview/resources-in-the-rest-api#rate-limiting)
- [Search API](https://docs.github.com/en/rest/search)

### OpenAI
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Function Calling](https://platform.openai.com/docs/guides/function-calling)

### Visualization
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Python](https://plotly.com/python/)

## 15.2 Cost Estimation

| Operation | Service | Estimated Cost |
|-----------|---------|----------------|
| GitHub API | Free | $0 |
| Classify 1000 repos (GPT-4) | OpenAI | ~$5-15 |
| Total project | - | ~$5-15 |

## 15.3 Sample Questions for Analysis

Use these questions to guide your insights:

1. What are the most popular programming languages in Peru?
2. Which industries are most represented in Peru's GitHub ecosystem?
3. What is the typical profile of a highly influential Peruvian developer?
4. How active is the Peruvian developer community?
5. Are there any emerging technology trends visible in the data?
6. How does Peru compare to global trends (based on your expectations)?

---

# Submission Instructions

1. Push your complete repository to GitHub
2. Ensure the repository is public
3. Upload your video to YouTube (unlisted) or Google Drive
4. Submit the following:
   - GitHub repository URL
   - Video link
   - Brief summary (100 words) of your key findings

## Deadline
[INSERT DATE HERE]

---

**Good luck! Remember to run `import antigravity` and include the screenshot!** 🇵🇪
