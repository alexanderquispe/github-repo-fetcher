"""GraphQL query templates for GitHub API."""

# Main search query that fetches all repository data in one call
SEARCH_REPOS_QUERY = """
query SearchRepos($query: String!, $first: Int!, $after: String) {
  search(query: $query, type: REPOSITORY, first: $first, after: $after) {
    pageInfo {
      hasNextPage
      endCursor
    }
    repositoryCount
    nodes {
      ... on Repository {
        nameWithOwner
        name
        description
        url
        homepageUrl
        createdAt
        updatedAt
        pushedAt
        stargazerCount
        forkCount
        diskUsage
        primaryLanguage {
          name
        }
        languages(first: 10) {
          nodes {
            name
          }
        }
        repositoryTopics(first: 20) {
          nodes {
            topic {
              name
            }
          }
        }
        licenseInfo {
          key
          name
        }
        isFork
        isArchived
        isPrivate
        isTemplate
        hasWikiEnabled
        hasIssuesEnabled
        watchers {
          totalCount
        }
        issues(states: OPEN) {
          totalCount
        }
        owner {
          login
          __typename
          ... on User {
            location
            company
            bio
            email
            followers {
              totalCount
            }
            createdAt
          }
          ... on Organization {
            location
            email
            description
          }
        }
        object(expression: "HEAD:README.md") {
          ... on Blob {
            text
          }
        }
      }
    }
  }
  rateLimit {
    remaining
    resetAt
    limit
    cost
  }
}
"""

# Query to count users/orgs by location
USER_COUNT_QUERY = """
query($query: String!) {
  search(query: $query, type: USER, first: 1) {
    userCount
  }
  rateLimit {
    remaining
    resetAt
  }
}
"""

# Query to search users/orgs with pagination
USER_SEARCH_QUERY = """
query($query: String!, $first: Int!, $after: String) {
  search(query: $query, type: USER, first: $first, after: $after) {
    userCount
    pageInfo {
      hasNextPage
      endCursor
    }
    nodes {
      ... on User {
        login
        __typename
      }
      ... on Organization {
        login
        __typename
      }
    }
  }
  rateLimit {
    remaining
    resetAt
  }
}
"""

# Query to fetch repositories for a specific user/org (with full details)
USER_REPOS_QUERY = """
query($login: String!, $first: Int!, $after: String) {
  repositoryOwner(login: $login) {
    login
    ... on User {
      location
      company
      bio
      email
      followers { totalCount }
      createdAt
    }
    ... on Organization {
      location
      email
      description
    }
    repositories(first: $first, after: $after, ownerAffiliations: OWNER) {
      totalCount
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        nameWithOwner
        name
        description
        url
        homepageUrl
        createdAt
        updatedAt
        pushedAt
        stargazerCount
        forkCount
        diskUsage
        primaryLanguage { name }
        languages(first: 10) { nodes { name } }
        repositoryTopics(first: 20) { nodes { topic { name } } }
        licenseInfo { key name }
        isFork
        isArchived
        isPrivate
        isTemplate
        hasWikiEnabled
        hasIssuesEnabled
        watchers { totalCount }
        issues(states: OPEN) { totalCount }
        object(expression: "HEAD:README.md") {
          ... on Blob { text }
        }
      }
    }
  }
  rateLimit {
    remaining
    resetAt
  }
}
"""

# Query to check rate limit status
RATE_LIMIT_QUERY = """
query {
  rateLimit {
    remaining
    resetAt
    limit
    cost
  }
}
"""

# Query to fetch a single repository by owner and name
SINGLE_REPO_QUERY = """
query GetRepo($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    nameWithOwner
    name
    description
    url
    homepageUrl
    createdAt
    updatedAt
    pushedAt
    stargazerCount
    forkCount
    diskUsage
    primaryLanguage {
      name
    }
    languages(first: 10) {
      nodes {
        name
      }
    }
    repositoryTopics(first: 20) {
      nodes {
        topic {
          name
        }
      }
    }
    licenseInfo {
      key
      name
    }
    isFork
    isArchived
    isPrivate
    isTemplate
    hasWikiEnabled
    hasIssuesEnabled
    watchers {
      totalCount
    }
    issues(states: OPEN) {
      totalCount
    }
    owner {
      login
      __typename
      ... on User {
        location
        company
        bio
        email
        followers {
          totalCount
        }
        createdAt
      }
      ... on Organization {
        location
        email
        description
      }
    }
    object(expression: "HEAD:README.md") {
      ... on Blob {
        text
      }
    }
  }
  rateLimit {
    remaining
    resetAt
    limit
    cost
  }
}
"""
