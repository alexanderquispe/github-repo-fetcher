# GitHub Peru Industry Dashboard

## Homework Assignment

---

# Table of Contents

1. [Project Overview](#1-project-overview)
2. [Learning Objectives](#2-learning-objectives)
3. [Setup Requirements](#3-setup-requirements)
4. [Step 1: Data Extraction](#4-step-1-data-extraction)
5. [Step 2: Industry Classification with CrewAI](#5-step-2-industry-classification-with-crewai)
6. [Step 3: Streamlit Dashboard](#6-step-3-streamlit-dashboard)
7. [Cost Estimation](#7-cost-estimation)
8. [Deliverables](#8-deliverables)
9. [Evaluation Rubric](#9-evaluation-rubric)

---

# 1. Project Overview

## Goal

Build a **Streamlit dashboard** that displays:
1. **Industry classification** of 1,000 Peruvian GitHub repositories (using CrewAI + GPT-4o-mini)
2. **Statistics and visualizations** from GitHub API data

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CREW: Peru GitHub Analysts                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │  RESEARCHER  │───▶│  CLASSIFIER  │───▶│   ANALYST    │               │
│  │    AGENT     │    │    AGENT     │    │    AGENT     │               │
│  │              │    │              │    │              │               │
│  │ Extracts     │    │ Classifies   │    │ Analyzes     │               │
│  │ 1000 repos   │    │ into 21      │    │ patterns &   │               │
│  │ from GitHub  │    │ industries   │    │ insights     │               │
│  │              │    │              │    │              │               │
│  │ Tools:       │    │ Tools:       │    │ Tools:       │               │
│  │ - GitHub API │    │ - GPT-4o-mini│    │ - Query data │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│                                                                          │
│                              │                                           │
│                              ▼                                           │
│                    ┌──────────────────┐                                 │
│                    │    STREAMLIT     │                                 │
│                    │    DASHBOARD     │                                 │
│                    │                  │                                 │
│                    │ - Industry charts│                                 │
│                    │ - Repo stats     │                                 │
│                    │ - Browser        │                                 │
│                    └──────────────────┘                                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Important: Model Selection

**USE GPT-4o-mini FOR ALL TASKS**

| Model | Cost per 1M Input | Cost per 1M Output | Use? |
|-------|-------------------|-------------------|------|
| GPT-4 | $30.00 | $60.00 | NO |
| GPT-4 Turbo | $10.00 | $30.00 | NO |
| GPT-4o | $2.50 | $10.00 | NO |
| **GPT-4o-mini** | **$0.15** | **$0.60** | **YES** |

GPT-4o-mini is **200x cheaper** than GPT-4 and sufficient for classification tasks.

---

# 2. Learning Objectives

| Skill | Description |
|-------|-------------|
| **API Integration** | Work with GitHub REST API |
| **CrewAI Agents** | Build multi-agent systems with roles and tools |
| **AI Classification** | Use GPT-4o-mini for text classification |
| **Data Visualization** | Build dashboards with Streamlit + Plotly |

---

# 3. Setup Requirements

## 3.1 The Antigravity Easter Egg

Before starting, run this in Python:

```python
import antigravity
```

**Take a screenshot and include it in your submission.**

## 3.2 GitHub Personal Access Token

1. Go to: GitHub → Settings → Developer Settings → Personal Access Tokens → Tokens (classic)
2. Click **"Generate new token (classic)"**
3. Configure:
   - **Note**: `GitHub Peru Dashboard - [Your Name]`
   - **Expiration**: **No expiration**
   - **Scopes**: Select `public_repo` and `read:user`
4. Copy and save your token

## 3.3 OpenAI API Key

You will receive a GPT-4o-mini API key from your instructor.

## 3.4 Environment Setup

Create a `.env` file (DO NOT commit to GitHub):

```env
GITHUB_TOKEN=ghp_your_token_here
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL_NAME=gpt-4o-mini
```

## 3.5 Required Packages

```txt
# requirements.txt

# CrewAI
crewai>=0.28.0
crewai-tools>=0.1.0

# OpenAI
openai>=1.12.0

# GitHub API
requests>=2.31.0

# Data Processing
pandas>=2.0.0

# Dashboard
streamlit>=1.31.0
plotly>=5.18.0

# Utilities
python-dotenv>=1.0.0
tqdm>=4.66.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

# 4. Step 1: Data Extraction

## 4.1 What to Extract

Extract data for **1,000 repositories** from Peru, sorted by stars.

| Field | Description |
|-------|-------------|
| `id` | Repository ID |
| `name` | Repository name |
| `full_name` | owner/repo format |
| `description` | Repo description |
| `owner_login` | Owner username |
| `stars` | Star count |
| `forks` | Fork count |
| `language` | Primary language |
| `topics` | Topic tags |
| `created_at` | Creation date |
| `updated_at` | Last update |
| `readme` | README content (first 2000 chars) |

## 4.2 GitHub Extraction Code

```python
# src/extraction.py
import os
import requests
import time
import base64
from dotenv import load_dotenv

load_dotenv()

class GitHubExtractor:
    def __init__(self):
        self.token = os.getenv("GITHUB_TOKEN")
        self.headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json"
        }
        self.base_url = "https://api.github.com"

    def search_peru_users(self, max_users=500):
        """Search for users located in Peru."""
        users = []
        page = 1

        while len(users) < max_users and page <= 10:
            print(f"  Fetching users page {page}...")
            response = requests.get(
                f"{self.base_url}/search/users",
                headers=self.headers,
                params={
                    "q": "location:Peru",
                    "sort": "followers",
                    "order": "desc",
                    "per_page": 100,
                    "page": page
                }
            )

            if response.status_code == 403:
                print("  Rate limited. Waiting 60 seconds...")
                time.sleep(60)
                continue

            data = response.json()
            users.extend(data.get("items", []))
            page += 1
            time.sleep(2)

        return users[:max_users]

    def get_user_repos(self, username):
        """Get repositories for a user."""
        response = requests.get(
            f"{self.base_url}/users/{username}/repos",
            headers=self.headers,
            params={"sort": "stars", "direction": "desc", "per_page": 100}
        )
        if response.status_code == 200:
            return response.json()
        return []

    def get_readme(self, owner, repo):
        """Get README content."""
        try:
            response = requests.get(
                f"{self.base_url}/repos/{owner}/{repo}/readme",
                headers=self.headers
            )
            if response.status_code == 200:
                content = response.json().get("content", "")
                return base64.b64decode(content).decode("utf-8")[:2000]
        except:
            pass
        return ""

    def extract_top_repos(self, num_repos=1000):
        """Extract top Peru repositories."""
        print("Step 1: Finding Peru users...")
        users = self.search_peru_users(max_users=500)
        print(f"  Found {len(users)} users")

        print("Step 2: Extracting repositories...")
        all_repos = []

        for i, user in enumerate(users):
            username = user["login"]
            print(f"  [{i+1}/{len(users)}] {username}")

            repos = self.get_user_repos(username)
            for repo in repos:
                if not repo.get("fork"):
                    all_repos.append({
                        "id": repo["id"],
                        "name": repo["name"],
                        "full_name": repo["full_name"],
                        "description": repo.get("description", ""),
                        "owner_login": username,
                        "stars": repo["stargazers_count"],
                        "forks": repo["forks_count"],
                        "language": repo.get("language", ""),
                        "topics": repo.get("topics", []),
                        "created_at": repo["created_at"],
                        "updated_at": repo["updated_at"],
                    })
            time.sleep(1)

        # Sort by stars, take top N
        all_repos.sort(key=lambda x: x["stars"], reverse=True)
        top_repos = all_repos[:num_repos]

        print(f"Step 3: Fetching READMEs for {len(top_repos)} repos...")
        for i, repo in enumerate(top_repos):
            if i % 50 == 0:
                print(f"  [{i}/{len(top_repos)}] Fetching READMEs...")
            owner, name = repo["full_name"].split("/")
            repo["readme"] = self.get_readme(owner, name)
            time.sleep(0.3)

        return top_repos


# Run extraction
if __name__ == "__main__":
    import pandas as pd

    extractor = GitHubExtractor()
    repos = extractor.extract_top_repos(1000)

    df = pd.DataFrame(repos)
    df.to_csv("data/repos.csv", index=False)
    print(f"Saved {len(df)} repositories to data/repos.csv")
```

---

# 5. Step 2: Industry Classification with CrewAI

## 5.1 The 21 Industry Categories (CIIU Peru)

| Code | Industry |
|------|----------|
| A | Agriculture, forestry, fishing |
| B | Mining and quarrying |
| C | Manufacturing |
| D | Electricity, gas supply |
| E | Water supply, sewerage |
| F | Construction |
| G | Wholesale and retail trade |
| H | Transportation and storage |
| I | Hotels and restaurants |
| J | Information and technology |
| K | Finance and insurance |
| L | Real estate |
| M | Professional services |
| N | Administrative services |
| O | Public administration |
| P | Education |
| Q | Healthcare |
| R | Arts and entertainment |
| S | Other services |
| T | Household activities |
| U | International organizations |

## 5.2 CrewAI Implementation

```python
# src/agents/crew.py
import os
import json
import pandas as pd
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

load_dotenv()

# ============================================
# CONFIGURE GPT-4o-mini
# ============================================

llm = LLM(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

# ============================================
# INDUSTRY CATEGORIES
# ============================================

INDUSTRIES = {
    "A": "Agriculture, forestry, fishing",
    "B": "Mining and quarrying",
    "C": "Manufacturing",
    "D": "Electricity, gas supply",
    "E": "Water supply, sewerage",
    "F": "Construction",
    "G": "Wholesale and retail trade",
    "H": "Transportation and storage",
    "I": "Hotels and restaurants",
    "J": "Information and technology",
    "K": "Finance and insurance",
    "L": "Real estate",
    "M": "Professional services",
    "N": "Administrative services",
    "O": "Public administration",
    "P": "Education",
    "Q": "Healthcare",
    "R": "Arts and entertainment",
    "S": "Other services",
    "T": "Household activities",
    "U": "International organizations"
}

# ============================================
# TOOLS
# ============================================

@tool("Load Repository Data")
def load_repo_data() -> str:
    """Load the extracted repository data from CSV."""
    df = pd.read_csv("data/repos.csv")
    return f"Loaded {len(df)} repositories. Columns: {list(df.columns)}"


@tool("Get Repository Details")
def get_repo_details(repo_index: int) -> str:
    """Get details of a specific repository by index."""
    df = pd.read_csv("data/repos.csv")
    if repo_index >= len(df):
        return "Invalid index"

    repo = df.iloc[repo_index]
    return f"""
Repository #{repo_index}:
- Name: {repo['name']}
- Full Name: {repo['full_name']}
- Description: {repo['description'] or 'No description'}
- Language: {repo['language'] or 'Unknown'}
- Stars: {repo['stars']}
- README: {str(repo['readme'])[:1500] if pd.notna(repo['readme']) else 'No README'}
"""


@tool("Classify Single Repository")
def classify_repository(repo_name: str, description: str, readme: str, language: str) -> str:
    """
    Classify a repository into one of 21 industries.
    Returns the industry code and name.
    """
    from openai import OpenAI

    client = OpenAI()

    prompt = f"""Classify this GitHub repository into ONE industry category.

REPOSITORY:
- Name: {repo_name}
- Description: {description or 'No description'}
- Language: {language or 'Unknown'}
- README excerpt: {readme[:1500] if readme else 'No README'}

INDUSTRIES:
{json.dumps(INDUSTRIES, indent=2)}

RULES:
1. Choose the industry that would MOST BENEFIT from this software
2. General programming tools/libraries = "J" (Information and technology)
3. Money/payments/banking = "K" (Finance)
4. Learning/tutorials/courses = "P" (Education)
5. If unclear = "S" (Other services)

Respond with ONLY this JSON:
{{"code": "X", "name": "Industry name", "confidence": "high/medium/low", "reason": "One sentence"}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # IMPORTANT: Use gpt-4o-mini
        messages=[
            {"role": "system", "content": "You classify software by industry. Respond only with valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


@tool("Save Classification Results")
def save_classification(repo_id: int, repo_name: str, industry_code: str, industry_name: str, confidence: str, reason: str) -> str:
    """Save a classification result to the results file."""
    result = {
        "repo_id": repo_id,
        "repo_name": repo_name,
        "industry_code": industry_code,
        "industry_name": industry_name,
        "confidence": confidence,
        "reason": reason
    }

    # Append to file
    import csv
    file_exists = os.path.exists("data/classifications.csv")

    with open("data/classifications.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

    return f"Saved classification for {repo_name}: {industry_code} - {industry_name}"


@tool("Query Classification Data")
def query_classifications(query_type: str) -> str:
    """
    Query the classification results.
    query_type: 'summary', 'top_industries', 'top_languages', 'industry_X' (where X is code)
    """
    df_repos = pd.read_csv("data/repos.csv")
    df_class = pd.read_csv("data/classifications.csv")

    merged = df_repos.merge(df_class, left_on="id", right_on="repo_id", how="inner")

    if query_type == "summary":
        return f"""
Total repositories classified: {len(merged)}
Total stars: {merged['stars'].sum()}
Unique industries: {merged['industry_code'].nunique()}
Unique languages: {merged['language'].nunique()}
"""

    elif query_type == "top_industries":
        counts = merged["industry_name"].value_counts().head(10)
        return f"Top 10 Industries:\n{counts.to_string()}"

    elif query_type == "top_languages":
        counts = merged["language"].value_counts().head(10)
        return f"Top 10 Languages:\n{counts.to_string()}"

    elif query_type.startswith("industry_"):
        code = query_type.split("_")[1].upper()
        filtered = merged[merged["industry_code"] == code]
        return f"Industry {code}: {len(filtered)} repos, {filtered['stars'].sum()} total stars"

    return "Unknown query type"


# ============================================
# AGENTS
# ============================================

researcher = Agent(
    role="GitHub Data Researcher",
    goal="Extract and prepare repository data from the Peru GitHub ecosystem",
    backstory="""You are an expert at working with GitHub data. You understand
    how to extract meaningful information from repositories including names,
    descriptions, README files, and metadata. You prepare data for classification.""",
    tools=[load_repo_data, get_repo_details],
    llm=llm,
    verbose=True
)

classifier = Agent(
    role="Industry Classification Specialist",
    goal="Classify each repository into one of 21 Peruvian industry categories (CIIU)",
    backstory="""You are an expert at understanding software projects and determining
    which industry they serve. You know the CIIU classification system used in Peru
    and can accurately categorize software projects based on their purpose, description,
    and code. You use GPT-4o-mini for cost-effective classification.""",
    tools=[get_repo_details, classify_repository, save_classification],
    llm=llm,
    verbose=True
)

analyst = Agent(
    role="Data Analyst",
    goal="Analyze classification results and provide insights about Peru's tech ecosystem",
    backstory="""You are a data analyst who specializes in understanding developer
    ecosystems. You find trends, patterns, and insights that help understand the
    technology landscape in Peru. You provide actionable insights for the dashboard.""",
    tools=[query_classifications],
    llm=llm,
    verbose=True
)


# ============================================
# TASKS
# ============================================

task_prepare = Task(
    description="""Load the repository data and verify it's ready for classification.
    Report how many repositories are available and their basic statistics.""",
    expected_output="A summary of the repository data ready for classification",
    agent=researcher
)

task_classify = Task(
    description="""Classify repositories into the 21 CIIU industry categories.

    For each repository:
    1. Get its details (name, description, README)
    2. Use the classify_repository tool to determine the industry
    3. Save the result using save_classification tool

    IMPORTANT: Process repositories in batches. Start with the first 100 repositories.
    Use GPT-4o-mini for all classifications to minimize cost.
    """,
    expected_output="Classification results saved for all processed repositories",
    agent=classifier
)

task_analyze = Task(
    description="""Analyze the classification results and provide insights:

    1. What are the top 5 industries in Peru's GitHub ecosystem?
    2. What are the most popular programming languages?
    3. Which industries have the highest average stars?
    4. What patterns or trends do you observe?
    5. What recommendations would you make based on this data?
    """,
    expected_output="A detailed analysis report with insights about Peru's tech ecosystem",
    agent=analyst
)


# ============================================
# CREW
# ============================================

def run_crew():
    """Run the full crew pipeline."""
    crew = Crew(
        agents=[researcher, classifier, analyst],
        tasks=[task_prepare, task_classify, task_analyze],
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()
    return result


if __name__ == "__main__":
    result = run_crew()
    print("\n" + "="*50)
    print("CREW ANALYSIS COMPLETE")
    print("="*50)
    print(result)
```

## 5.3 Simplified Classification Script (Alternative)

If you want to run classification without the full crew, use this simpler script:

```python
# src/classify_simple.py
import os
import json
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

INDUSTRIES = {
    "A": "Agriculture, forestry, fishing",
    "B": "Mining and quarrying",
    "C": "Manufacturing",
    "D": "Electricity, gas supply",
    "E": "Water supply, sewerage",
    "F": "Construction",
    "G": "Wholesale and retail trade",
    "H": "Transportation and storage",
    "I": "Hotels and restaurants",
    "J": "Information and technology",
    "K": "Finance and insurance",
    "L": "Real estate",
    "M": "Professional services",
    "N": "Administrative services",
    "O": "Public administration",
    "P": "Education",
    "Q": "Healthcare",
    "R": "Arts and entertainment",
    "S": "Other services",
    "T": "Household activities",
    "U": "International organizations"
}


def classify_repo(client, name, description, readme, language):
    """Classify a single repository using GPT-4o-mini."""

    prompt = f"""Classify this GitHub repository into ONE industry.

REPOSITORY:
- Name: {name}
- Description: {description or 'No description'}
- Language: {language or 'Unknown'}
- README: {readme[:1500] if readme else 'No README'}

INDUSTRIES:
{json.dumps(INDUSTRIES, indent=2)}

RULES:
- General programming tools = "J"
- Finance/payments = "K"
- Education/tutorials = "P"
- If unclear = "S"

Respond ONLY with JSON:
{{"code": "X", "name": "Industry name", "confidence": "high/medium/low", "reason": "Brief reason"}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # IMPORTANT: gpt-4o-mini
        messages=[
            {"role": "system", "content": "Classify software by industry. JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    try:
        return json.loads(response.choices[0].message.content)
    except:
        return {"code": "J", "name": "Information and technology", "confidence": "low", "reason": "Parse error"}


def main():
    # Load repos
    df = pd.read_csv("data/repos.csv")
    print(f"Classifying {len(df)} repositories...")

    client = OpenAI()
    results = []

    for _, repo in tqdm(df.iterrows(), total=len(df)):
        classification = classify_repo(
            client,
            name=repo["name"],
            description=repo.get("description", ""),
            readme=str(repo.get("readme", "")) if pd.notna(repo.get("readme")) else "",
            language=repo.get("language", "")
        )

        results.append({
            "repo_id": repo["id"],
            "repo_name": repo["name"],
            "industry_code": classification.get("code", "J"),
            "industry_name": classification.get("name", "Information and technology"),
            "confidence": classification.get("confidence", "low"),
            "reason": classification.get("reason", "")
        })

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("data/classifications.csv", index=False)
    print(f"Saved {len(results_df)} classifications to data/classifications.csv")

    # Print summary
    print("\nTop Industries:")
    print(results_df["industry_name"].value_counts().head(10))


if __name__ == "__main__":
    main()
```

---

# 6. Step 3: Streamlit Dashboard

## 6.1 Dashboard Requirements

| Page | Content |
|------|---------|
| **Industry Overview** | Pie chart, bar chart, top repos per industry |
| **Repository Stats** | Stars distribution, languages, trends |
| **Repository Browser** | Filter by industry, search, view details |

## 6.2 Implementation

```python
# app/dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="GitHub Peru Dashboard",
    page_icon="🇵🇪",
    layout="wide"
)

# ============================================
# LOAD DATA
# ============================================

@st.cache_data
def load_data():
    repos = pd.read_csv("data/repos.csv")
    classifications = pd.read_csv("data/classifications.csv")
    df = repos.merge(classifications, left_on="id", right_on="repo_id", how="left")
    return df

df = load_data()

# ============================================
# SIDEBAR
# ============================================

st.sidebar.title("🇵🇪 GitHub Peru")
st.sidebar.markdown("**Industry Classification Dashboard**")
st.sidebar.markdown("---")
st.sidebar.markdown("*Powered by CrewAI + GPT-4o-mini*")

page = st.sidebar.radio(
    "Navigate",
    ["🏭 Industry Overview", "📊 Repository Stats", "🔍 Repository Browser"]
)

# ============================================
# PAGE 1: INDUSTRY OVERVIEW
# ============================================

if page == "🏭 Industry Overview":
    st.title("🏭 Industry Classification Overview")
    st.markdown("Classification of 1,000 Peruvian GitHub repositories into 21 industries (CIIU)")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Repos", f"{len(df):,}")
    col2.metric("Total Stars", f"{df['stars'].sum():,}")
    col3.metric("Industries", df["industry_code"].nunique())
    col4.metric("Languages", df["language"].nunique())

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Repositories by Industry")
        industry_counts = df["industry_name"].value_counts().reset_index()
        industry_counts.columns = ["Industry", "Count"]

        fig = px.pie(
            industry_counts.head(10),
            values="Count",
            names="Industry",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📈 Top 10 Industries")
        fig = px.bar(
            industry_counts.head(10),
            x="Count",
            y="Industry",
            orientation="h",
            color="Count",
            color_continuous_scale="Blues"
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    # Table: Top repo per industry
    st.subheader("🏆 Top Repository per Industry")
    top_per_industry = df.loc[df.groupby("industry_name")["stars"].idxmax()]
    top_per_industry = top_per_industry.sort_values("stars", ascending=False)

    st.dataframe(
        top_per_industry[["industry_name", "full_name", "stars", "language", "reason"]].head(15),
        use_container_width=True,
        column_config={
            "industry_name": "Industry",
            "full_name": "Repository",
            "stars": st.column_config.NumberColumn("Stars", format="%d ⭐"),
            "language": "Language",
            "reason": "Classification Reason"
        }
    )

# ============================================
# PAGE 2: REPOSITORY STATS
# ============================================

elif page == "📊 Repository Stats":
    st.title("📊 Repository Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⭐ Stars Distribution")
        fig = px.histogram(
            df,
            x="stars",
            nbins=50,
            color_discrete_sequence=["#FF6B6B"]
        )
        fig.update_layout(xaxis_title="Stars", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("💻 Top 10 Programming Languages")
        lang_counts = df["language"].value_counts().head(10).reset_index()
        lang_counts.columns = ["Language", "Count"]

        fig = px.bar(
            lang_counts,
            x="Language",
            y="Count",
            color="Count",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⭐ Stars vs 🍴 Forks")
        fig = px.scatter(
            df,
            x="stars",
            y="forks",
            color="industry_name",
            hover_data=["full_name"],
            opacity=0.6
        )
        fig.update_layout(xaxis_title="Stars", yaxis_title="Forks")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📅 Repositories Created Over Time")
        df["year"] = pd.to_datetime(df["created_at"]).dt.year
        yearly = df.groupby("year").size().reset_index(name="count")

        fig = px.line(
            yearly,
            x="year",
            y="count",
            markers=True,
            color_discrete_sequence=["#4ECDC4"]
        )
        fig.update_layout(xaxis_title="Year", yaxis_title="Repositories Created")
        st.plotly_chart(fig, use_container_width=True)

    # Top repos table
    st.subheader("🏆 Top 20 Repositories by Stars")
    top_repos = df.nlargest(20, "stars")[["full_name", "stars", "forks", "language", "industry_name"]]
    st.dataframe(
        top_repos,
        use_container_width=True,
        column_config={
            "full_name": "Repository",
            "stars": st.column_config.NumberColumn("Stars", format="%d ⭐"),
            "forks": st.column_config.NumberColumn("Forks", format="%d 🍴"),
            "language": "Language",
            "industry_name": "Industry"
        }
    )

# ============================================
# PAGE 3: REPOSITORY BROWSER
# ============================================

elif page == "🔍 Repository Browser":
    st.title("🔍 Repository Browser")

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        industries = ["All"] + sorted(df["industry_name"].dropna().unique().tolist())
        selected_industry = st.selectbox("🏭 Filter by Industry", industries)

    with col2:
        languages = ["All"] + sorted(df["language"].dropna().unique().tolist())
        selected_language = st.selectbox("💻 Filter by Language", languages)

    with col3:
        search = st.text_input("🔎 Search by name")

    # Min stars filter
    min_stars = st.slider("⭐ Minimum Stars", 0, int(df["stars"].max()), 0)

    # Apply filters
    filtered = df.copy()

    if selected_industry != "All":
        filtered = filtered[filtered["industry_name"] == selected_industry]

    if selected_language != "All":
        filtered = filtered[filtered["language"] == selected_language]

    if search:
        filtered = filtered[
            filtered["name"].str.contains(search, case=False, na=False) |
            filtered["description"].str.contains(search, case=False, na=False)
        ]

    filtered = filtered[filtered["stars"] >= min_stars]
    filtered = filtered.sort_values("stars", ascending=False)

    st.markdown(f"**Showing {len(filtered)} repositories**")
    st.markdown("---")

    # Display results
    for _, repo in filtered.head(50).iterrows():
        with st.expander(f"⭐ {repo['stars']} | {repo['full_name']} | {repo['industry_name']}"):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**Description:** {repo['description'] or 'No description'}")
                st.markdown(f"**Language:** `{repo['language'] or 'Unknown'}`")
                st.markdown(f"**Industry:** {repo['industry_name']} ({repo['industry_code']})")
                st.markdown(f"**Classification Reason:** {repo.get('reason', 'N/A')}")
                st.markdown(f"**Confidence:** {repo.get('confidence', 'N/A')}")
                st.markdown(f"[View on GitHub](https://github.com/{repo['full_name']})")

            with col2:
                st.metric("Stars", f"{repo['stars']:,}")
                st.metric("Forks", f"{repo['forks']:,}")
```

## 6.3 Run the Dashboard

```bash
streamlit run app/dashboard.py
```

---

# 7. Cost Estimation

## 7.1 GPT-4o-mini Costs

**IMPORTANT:** This project uses **GPT-4o-mini** exclusively.

| Component | Tokens per Request | Total Tokens (1000 repos) |
|-----------|-------------------|---------------------------|
| Input (prompt + context) | ~800 | ~800,000 |
| Output (JSON response) | ~50 | ~50,000 |

**Cost Calculation:**

| | Tokens | Rate | Cost |
|---|--------|------|------|
| Input | 800,000 | $0.15 / 1M | $0.12 |
| Output | 50,000 | $0.60 / 1M | $0.03 |
| **Total** | | | **~$0.15** |

**CrewAI adds some overhead for agent reasoning:**

| Approach | Estimated Cost |
|----------|---------------|
| Simple script | ~$0.15 - $0.20 |
| CrewAI agents | ~$0.30 - $0.50 |

**Total project cost: Less than $1.00**

---

# 8. Deliverables

## 8.1 Checklist

| # | Item | Description |
|---|------|-------------|
| 1 | **Antigravity Screenshot** | Screenshot of `import antigravity` |
| 2 | **GitHub Token** | Created with no expiration |
| 3 | **`data/repos.csv`** | 1,000 repositories |
| 4 | **`data/classifications.csv`** | Industry classifications |
| 5 | **CrewAI Agents** | Working agent implementation |
| 6 | **Streamlit Dashboard** | 3 pages working |
| 7 | **Video Demo** | 3-5 minutes |
| 8 | **GitHub Repository** | All code |

## 8.2 Repository Structure

```
github-peru-dashboard/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── data/
│   ├── repos.csv              # 1000 repos from GitHub
│   └── classifications.csv    # GPT-4o-mini classifications
│
├── src/
│   ├── extraction.py          # GitHub API extraction
│   ├── classify_simple.py     # Simple classification script
│   └── agents/
│       └── crew.py            # CrewAI implementation
│
├── app/
│   └── dashboard.py           # Streamlit dashboard
│
└── screenshots/
    ├── antigravity.png
    ├── industry_overview.png
    ├── repo_stats.png
    └── repo_browser.png
```

## 8.3 Video Requirements

| Aspect | Requirement |
|--------|-------------|
| Duration | 3-5 minutes |
| Content | Show antigravity, explain CrewAI agents, demo all 3 dashboard pages |
| Format | MP4 or YouTube link |

---

# 9. Evaluation Rubric

## Total: 100 points

| Category | Points | Criteria |
|----------|--------|----------|
| **Data Extraction** | 20 | 1000 repos with all required fields |
| **CrewAI Agents** | 25 | Working agents with proper roles and tools |
| **Industry Classification** | 15 | All repos classified using GPT-4o-mini |
| **Dashboard - Industry Page** | 15 | Pie chart, bar chart, table |
| **Dashboard - Stats Page** | 10 | All visualizations working |
| **Dashboard - Browser Page** | 10 | Filters and search working |
| **Video Demo** | 5 | Complete demo |

---

# Submission

1. Push code to a **public GitHub repository**
2. Include video link in README
3. Submit:
   - GitHub repository URL
   - Video link

## Deadline

[INSERT DATE]

---

**Model Reminder: USE `gpt-4o-mini` FOR ALL CLASSIFICATIONS**

**Cost: Less than $1.00 for the entire project**

**Remember to run `import antigravity`!** 🇵🇪
