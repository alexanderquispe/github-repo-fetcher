"""
Fetch Anthropic GitHub organization data and analyze Claude Code adoption.
Same analysis as Spotify case study.
"""

import requests
import pandas as pd
from pathlib import Path
from collections import defaultdict
import time

# GitHub API setup
HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

output_dir = Path("data/output")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("ANTHROPIC GITHUB DATA COLLECTION")
print("=" * 70)

# 1. Fetch all repos from Anthropic org
print("\n1. Fetching Anthropic repositories...")
repos = []
page = 1
while True:
    url = f"https://api.github.com/orgs/anthropics/repos?per_page=100&page={page}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        print(f"   Error: {response.status_code} - {response.text}")
        break
    data = response.json()
    if not data:
        break
    repos.extend(data)
    print(f"   Page {page}: {len(data)} repos")
    page += 1
    time.sleep(0.5)

print(f"   Total repos: {len(repos)}")

# Save repos
repos_df = pd.DataFrame([{
    'name': r['name'],
    'full_name': r['full_name'],
    'description': r.get('description', ''),
    'stargazers_count': r['stargazers_count'],
    'forks_count': r['forks_count'],
    'language': r.get('language', ''),
    'created_at': r['created_at'],
    'updated_at': r['updated_at']
} for r in repos])
repos_df.to_parquet(output_dir / "anthropic_repos.parquet", index=False)
print(f"   Saved: anthropic_repos.parquet")

# 2. Fetch contributors for each repo
print("\n2. Fetching contributors for each repo...")
all_contributors = []
for i, repo in enumerate(repos):
    repo_name = repo['full_name']
    page = 1
    repo_contributors = []
    while True:
        url = f"https://api.github.com/repos/{repo_name}/contributors?per_page=100&page={page}"
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 204:  # No content
            break
        if response.status_code != 200:
            print(f"   Error for {repo_name}: {response.status_code}")
            break
        data = response.json()
        if not data:
            break
        repo_contributors.extend(data)
        page += 1
        time.sleep(0.3)

    for c in repo_contributors:
        if c.get('login'):  # Skip if no login (anonymous)
            all_contributors.append({
                'repo': repo_name,
                'login': c['login'],
                'contributions': c['contributions'],
                'type': c.get('type', 'User')
            })

    if (i + 1) % 10 == 0:
        print(f"   Processed {i + 1}/{len(repos)} repos, {len(all_contributors)} contributors")

print(f"   Total contributor records: {len(all_contributors)}")

# Save contributors
contributors_df = pd.DataFrame(all_contributors)
contributors_df.to_parquet(output_dir / "anthropic_contributors.parquet", index=False)
print(f"   Saved: anthropic_contributors.parquet")

# Get unique contributors
unique_contributors = contributors_df['login'].unique()
print(f"   Unique contributors: {len(unique_contributors)}")

# 3. Match against Claude Code users
print("\n3. Matching against Claude Code users...")
claude_commits_df = pd.read_parquet("data/output/claude_commits.parquet")
claude_users = set(claude_commits_df['author_login'].unique())
print(f"   Total Claude Code users in dataset: {len(claude_users)}")

# Find matches
matched_users = []
for user in unique_contributors:
    is_claude = user in claude_users
    first_use = None
    if is_claude:
        user_commits = claude_commits_df[claude_commits_df['author_login'] == user]
        first_use = user_commits['committed_date'].min()
    matched_users.append({
        'login': user,
        'is_claude_user': is_claude,
        'first_claude_use': first_use
    })

matched_df = pd.DataFrame(matched_users)
matched_df.to_parquet(output_dir / "anthropic_claude_users.parquet", index=False)

claude_count = matched_df['is_claude_user'].sum()
print(f"   Claude Code users at Anthropic: {claude_count}")
print(f"   Adoption rate: {100 * claude_count / len(unique_contributors):.1f}%")

# Filter out bots
bot_patterns = ['[bot]', '-bot', 'dependabot', 'renovate', 'github-actions']
def is_bot(login):
    login_lower = login.lower()
    return any(pattern in login_lower for pattern in bot_patterns)

human_users = matched_df[~matched_df['login'].apply(is_bot)]
human_claude = human_users[human_users['is_claude_user'] == True]
print(f"\n   After filtering bots:")
print(f"   Human contributors: {len(human_users)}")
print(f"   Human Claude users: {len(human_claude)}")
print(f"   Human adoption rate: {100 * len(human_claude) / len(human_users):.1f}%")

# 4. Show Claude users
print("\n4. Claude Code users at Anthropic:")
claude_df = matched_df[matched_df['is_claude_user'] == True].copy()
claude_df['first_claude_use'] = pd.to_datetime(claude_df['first_claude_use'])
claude_df = claude_df.sort_values('first_claude_use')

# Filter bots for display
claude_humans = claude_df[~claude_df['login'].apply(is_bot)]
print(f"\n   First 20 adopters (excluding bots):")
for i, row in claude_humans.head(20).iterrows():
    date_str = row['first_claude_use'].strftime('%Y-%m-%d') if pd.notna(row['first_claude_use']) else 'Unknown'
    print(f"   {date_str}: {row['login']}")

print("\n" + "=" * 70)
print("DATA COLLECTION COMPLETE")
print("=" * 70)
