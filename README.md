# GitHub Repository Fetcher

A Python tool to efficiently fetch GitHub repository data using the GraphQL API. Designed for researchers and developers who need to extract comprehensive repository metadata by geographic location.

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Getting a GitHub Token](#getting-a-github-token)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Available Filters](#available-filters)
- [Output Format](#output-format)
- [Rate Limits](#rate-limits)
- [Examples](#examples)
- [Programmatic Usage](#programmatic-usage)
- [Troubleshooting](#troubleshooting)

---

## Features

- **Location-based fetching**: Get all repositories from users/organizations in a specific country or city
- **Efficient**: Uses GraphQL API to fetch ~30 data points per repo in a single call
- **Smart filtering**: Filter by stars, language, date, license, and more at the API level
- **Handles GitHub limits**: Automatically waits when rate limit is reached, then continues
- **Incremental saving**: Progress is saved every 100 repos, so you don't lose data if interrupted
- **Fork exclusion**: Excludes forked repositories by default (forks are copies, not original work)

---

## How It Works

The tool uses a **two-step approach** to fetch repositories by location:

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Find all users/organizations in the location        │
│                                                             │
│   Query: search(type: USER, query: "location:Peru")         │
│   Result: ["user1", "user2", "org1", ...]                   │
│                                                             │
│   Note: Uses date-range splitting to overcome GitHub's      │
│         1000 result limit per query                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: For each user, fetch their repositories             │
│                                                             │
│   Query: search(type: REPOSITORY,                           │
│                 query: "user:X stars:>=1 fork:false")       │
│   Result: Full repo data + README content                   │
│                                                             │
│   Note: Filters (stars, language, etc.) are applied at      │
│         the API level for efficiency                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Output: location_repos.parquet                              │
│                                                             │
│   33 columns including: nwo, description, stars, topics,    │
│   readme_content, owner_location, license, languages, etc.  │
└─────────────────────────────────────────────────────────────┘
```

### Why this approach?

GitHub's repository search API does **not** support filtering by user location directly. The only way to get repos from a specific location is to:
1. First find users in that location
2. Then fetch their repositories

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/github-repo-fetcher.git
cd github-repo-fetcher
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies:
- `requests` - HTTP client for API calls
- `pandas` - Data manipulation
- `pyarrow` - Parquet file support
- `python-dotenv` - Environment variable management
- `tqdm` - Progress bars

### 3. Set up your GitHub token

See [Getting a GitHub Token](#getting-a-github-token) below.

---

## Getting a GitHub Token

You need a GitHub Personal Access Token to use this tool.

### Step 1: Go to GitHub Token Settings

Visit: https://github.com/settings/tokens

### Step 2: Generate a new token

1. Click **"Generate new token"** → **"Generate new token (classic)"**
2. Give it a name (e.g., "repo-fetcher")
3. Select expiration (recommend: 90 days)
4. Select scopes:
   - `public_repo` - For fetching public repositories (recommended)
   - `repo` - For fetching private repositories (if needed)
5. Click **"Generate token"**
6. **Copy the token immediately** (you won't see it again!)

### Step 3: Configure the token

**Option A: Environment variable (recommended)**

```bash
# Linux/Mac
export GITHUB_TOKEN=ghp_your_token_here

# Windows (Command Prompt)
set GITHUB_TOKEN=ghp_your_token_here

# Windows (PowerShell)
$env:GITHUB_TOKEN="ghp_your_token_here"
```

**Option B: Create a .env file**

```bash
cp .env.example .env
```

Edit `.env` and add your token:
```
GITHUB_TOKEN=ghp_your_token_here
```

**Option C: Pass directly via CLI**

```bash
python scripts/fetch_repos.py --token ghp_your_token_here --location "Peru" -o peru.parquet
```

---

## Quick Start

### Fetch all repos from Peru

```bash
python scripts/fetch_repos.py --location "Peru" -o data/output/peru.parquet
```

### Fetch repos with 5+ stars

```bash
python scripts/fetch_repos.py --location "Peru" --min-stars 5 -o data/output/peru.parquet
```

### Fetch Python repos only

```bash
python scripts/fetch_repos.py --location "Peru" --min-stars 5 \
    --filter "language:Python" -o data/output/peru_python.parquet
```

### Test with a small batch first

```bash
python scripts/fetch_repos.py --location "Peru" --max-users 10 -o data/output/peru_test.parquet
```

---

## CLI Reference

```
python scripts/fetch_repos.py [OPTIONS]
```

### Search Mode (choose one)

| Option | Description |
|--------|-------------|
| `--location LOCATION` | Fetch repos from users/orgs in this location (e.g., "Peru", "Brazil", "Lima") |
| `--query QUERY` | Custom GitHub search query (direct search, not location-based) |
| `--repo OWNER/NAME` | Fetch a single repository |

### Filters

| Option | Default | Description |
|--------|---------|-------------|
| `--min-stars N` | 1 | Minimum star count |
| `--filter "..."` | (none) | Additional GitHub search filters |
| `--include-forks` | False | Include forked repositories (excluded by default) |

### Output Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output FILE`, `-o FILE` | (required) | Output file path (.parquet or .csv) |
| `--max-users N` | all | Maximum users to process (for testing) |
| `--max-repos N` | 10000 | Maximum repos (for --query mode) |
| `--no-orgs` | False | Exclude organizations, only fetch from user accounts |

### Authentication

| Option | Description |
|--------|-------------|
| `--token TOKEN` | GitHub Personal Access Token (or set GITHUB_TOKEN env var) |

---

## Available Filters

Use these with the `--filter` option:

### Language

```bash
--filter "language:Python"
--filter "language:JavaScript"
--filter "language:R"
```

### Date Filters

```bash
--filter "pushed:>=2022-01-01"      # Active in last 2 years
--filter "created:>=2020-01-01"     # Created after 2020
--filter "created:2020-01-01..2022-12-31"  # Date range
```

### Size

```bash
--filter "size:>=100"    # At least 100 KB
--filter "size:>=1000"   # At least 1 MB
```

### License

```bash
--filter "license:mit"
--filter "license:apache-2.0"
--filter "license:gpl-3.0"
```

### Other

```bash
--filter "archived:false"        # Exclude archived repos
--filter "topic:machine-learning"  # By topic
--filter "mirror:false"          # Exclude mirrors
```

### Combining Filters

```bash
--filter "language:Python pushed:>=2022-01-01 size:>=100 license:mit"
```

---

## Output Format

The output is a Parquet file (or CSV) with **33 columns**:

### Repository Metadata

| Column | Type | Description |
|--------|------|-------------|
| `nwo` | string | Full name: "owner/repo" |
| `name` | string | Repository name |
| `description` | string | Repository description |
| `url` | string | GitHub URL |
| `homepage_url` | string | Project website (if set) |
| `created_at` | string | Creation timestamp (ISO 8601) |
| `updated_at` | string | Last update timestamp |
| `pushed_at` | string | Last push timestamp |

### Metrics

| Column | Type | Description |
|--------|------|-------------|
| `stars` | int | Star count |
| `forks` | int | Fork count |
| `watchers` | int | Watcher count |
| `open_issues` | int | Open issue count |
| `disk_usage_kb` | int | Repository size in KB |

### Languages & Topics

| Column | Type | Description |
|--------|------|-------------|
| `primary_language` | string | Main programming language |
| `languages` | JSON list | All languages used (e.g., `["Python", "JavaScript"]`) |
| `topics` | JSON list | Topic tags (e.g., `["machine-learning", "api"]`) |

### Flags

| Column | Type | Description |
|--------|------|-------------|
| `is_fork` | bool | Is this a fork? |
| `is_archived` | bool | Is this archived? |
| `is_private` | bool | Is this private? |
| `is_template` | bool | Is this a template repo? |
| `has_wiki` | bool | Wiki enabled? |
| `has_issues` | bool | Issues enabled? |

### License

| Column | Type | Description |
|--------|------|-------------|
| `license_key` | string | SPDX license ID (e.g., "mit") |
| `license_name` | string | Full license name (e.g., "MIT License") |

### Owner Information

| Column | Type | Description |
|--------|------|-------------|
| `owner_login` | string | Username |
| `owner_type` | string | "User" or "Organization" |
| `owner_location` | string | Location (e.g., "Lima, Peru") |
| `owner_company` | string | Company name |
| `owner_bio` | string | User bio or org description |
| `owner_email` | string | Public email |
| `owner_followers` | int | Follower count |
| `owner_created_at` | string | Account creation date |

### Content

| Column | Type | Description |
|--------|------|-------------|
| `readme_content` | string | Full README.md text |

---

## Rate Limits

### GitHub API Limits

- **5,000 requests per hour** for authenticated users
- Each user query = 1 request
- Each repo batch = 1 request

### How the tool handles rate limits

1. **Checks before each request**: Monitors remaining quota
2. **Automatic waiting**: When quota < 100, waits until reset
3. **Progress saving**: Saves every 100 repos (no data loss if interrupted)

### Example output when rate limit is hit:

```
Users processed: 4500/12781 [2:15:00]
Rate limit: 98/5000 remaining

  Rate limit low (98 remaining). Waiting 2705s until reset...

[Waits ~45 minutes]

Rate limit: 5000/5000 (reset!)
Users processed: 4501/12781 [3:00:00]
```

### Estimated times

| Location | Users | API Calls | Time |
|----------|-------|-----------|------|
| Peru | ~12,800 | ~13,000 | ~3-4 hours |
| Brazil | ~100,000 | ~100,000 | ~24 hours |
| USA | ~3,000,000 | ~3,000,000 | ~25 days |

For large countries, use `--max-users` to limit scope or add filters.

---

## Examples

### Basic: Fetch all repos from a country

```bash
python scripts/fetch_repos.py \
    --location "Peru" \
    -o data/output/peru_all.parquet
```

### Filter by stars

```bash
python scripts/fetch_repos.py \
    --location "Peru" \
    --min-stars 10 \
    -o data/output/peru_10stars.parquet
```

### Filter by language and activity

```bash
python scripts/fetch_repos.py \
    --location "Peru" \
    --min-stars 5 \
    --filter "language:Python pushed:>=2023-01-01" \
    -o data/output/peru_python_active.parquet
```

### Exclude organizations (users only)

```bash
python scripts/fetch_repos.py \
    --location "Peru" \
    --no-orgs \
    -o data/output/peru_users_only.parquet
```

### Include forks

```bash
python scripts/fetch_repos.py \
    --location "Peru" \
    --include-forks \
    -o data/output/peru_with_forks.parquet
```

### Test with limited users

```bash
python scripts/fetch_repos.py \
    --location "Peru" \
    --max-users 100 \
    -o data/output/peru_test.parquet
```

### Fetch by topic (direct search, not location-based)

```bash
python scripts/fetch_repos.py \
    --query "topic:machine-learning stars:>=100" \
    -o data/output/ml_repos.parquet
```

### Fetch a single repository

```bash
python scripts/fetch_repos.py \
    --repo "microsoft/vscode" \
    -o data/output/vscode.parquet
```

### Save as CSV instead of Parquet

```bash
python scripts/fetch_repos.py \
    --location "Peru" \
    -o data/output/peru.csv
```

---

## Programmatic Usage

You can also use the library directly in Python:

```python
from pathlib import Path
from github_fetcher import GitHubFetcher

# Initialize with your token
fetcher = GitHubFetcher(token="ghp_your_token_here")

# Option 1: Fetch by location (two-step approach)
repos = fetcher.fetch_by_location_two_step(
    location="Peru",
    include_orgs=True,
    min_stars=5,
    include_forks=False,
    extra_filter="language:Python",
    max_users=100,  # Limit for testing
    output_path=Path("peru_python.parquet")
)

# Option 2: Fetch by custom query
repos = fetcher.fetch_by_query(
    custom_query="topic:data-science language:Python",
    min_stars=50,
    max_repos=1000
)

# Option 3: Fetch a single repo
repo = fetcher.fetch_repo_details("microsoft/vscode")

# Convert to DataFrame
df = fetcher.to_dataframe()

# Save to file
fetcher.save_to_parquet(Path("output.parquet"))
fetcher.save_to_csv(Path("output.csv"))
```

### Reading the output

```python
import pandas as pd
import json

# Read parquet file
df = pd.read_parquet("peru.parquet")

# Languages and topics are stored as JSON strings
# Convert back to lists if needed:
df['languages_list'] = df['languages'].apply(json.loads)
df['topics_list'] = df['topics'].apply(json.loads)

# Filter by owner location
lima_repos = df[df['owner_location'].str.contains('Lima', case=False, na=False)]

# Get Python repos
python_repos = df[df['primary_language'] == 'Python']

# Sort by stars
top_repos = df.sort_values('stars', ascending=False).head(100)
```

---

## Troubleshooting

### "Error: GitHub token required"

Set your token via environment variable or `.env` file. See [Getting a GitHub Token](#getting-a-github-token).

### "Rate limit low... Waiting..."

This is normal. The tool will automatically wait and continue. Don't interrupt it.

### "GraphQL errors: Resource limits exceeded"

The query is too complex. This shouldn't happen with the current implementation, but if it does:
- Try with `--max-users 10` first
- Report the issue with your exact command

### "502 Server Error" or "504 Gateway Timeout"

GitHub's servers are temporarily overloaded. The tool will retry automatically (up to 3 times with exponential backoff).

### Empty results

- Check if the location spelling is correct
- Try a broader location (e.g., "Peru" instead of "Lima, Peru")
- Lower the `--min-stars` threshold
- Remove restrictive filters

### Interrupted? Data lost?

No! Progress is saved every 100 repos. Check the output file - your partial results are there.

---

## License

MIT License
