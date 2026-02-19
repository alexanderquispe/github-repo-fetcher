#!/usr/bin/env python3
"""
Fetch Claude Code co-authored commits using Google BigQuery and GH Archive.

This is the FASTEST approach - queries ~12 months of data in minutes.

Prerequisites:
    1. Google Cloud account (free tier includes 1 TB/month)
    2. Install: pip install google-cloud-bigquery pandas pyarrow db-dtypes google-auth
    3. Authenticate (choose one):
       a. Place google-credentials.json in project folder (recommended)
       b. gcloud auth application-default login
       c. Set GOOGLE_APPLICATION_CREDENTIALS environment variable

Usage:
    python scripts/fetch_claude_bigquery.py -o data/output/claude_commits_bq.parquet

    # Dry run (estimate cost without running)
    python scripts/fetch_claude_bigquery.py --dry-run

    # Custom date range
    python scripts/fetch_claude_bigquery.py --start-date 2025-06-01 -o output.parquet

Cost Estimate:
    - Data scanned: ~4 TB for full year
    - Cost: ~$15 ($5/TB after 1TB free tier)
    - New Google Cloud accounts get $300 free credit
"""

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

try:
    from google.cloud import bigquery
    import pandas as pd
except ImportError:
    print("Required packages not installed. Run:")
    print("  pip install google-cloud-bigquery pandas pyarrow db-dtypes")
    sys.exit(1)


def parse_date(date_str: str) -> date:
    """Parse a date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid date format: {date_str}. Use YYYY-MM-DD."
        )


def build_query(start_date: date, end_date: date) -> str:
    """Build the BigQuery SQL query for Claude commits."""

    # Format dates for BigQuery table suffix (YYMMDD)
    start_suffix = start_date.strftime("%y%m%d")
    end_suffix = end_date.strftime("%y%m%d")

    query = f"""
    -- Find all Claude Code co-authored commits from GH Archive
    -- Scans PushEvents and extracts commits with Claude co-author signature

    WITH push_events AS (
      SELECT
        created_at,
        repo.id as repo_id,
        repo.name as repo_name,
        actor.id as actor_id,
        actor.login as actor_login,
        JSON_EXTRACT_ARRAY(payload, '$.commits') as commits
      FROM `githubarchive.day.20*`
      WHERE
        _TABLE_SUFFIX BETWEEN '{start_suffix}' AND '{end_suffix}'
        AND type = 'PushEvent'
    ),

    -- Unnest commits and filter for Claude co-author
    claude_commits AS (
      SELECT
        created_at as event_date,
        repo_id,
        repo_name,
        actor_id,
        actor_login,
        JSON_EXTRACT_SCALAR(commit, '$.sha') as sha,
        JSON_EXTRACT_SCALAR(commit, '$.message') as message,
        JSON_EXTRACT_SCALAR(commit, '$.author.name') as author_name,
        JSON_EXTRACT_SCALAR(commit, '$.author.email') as author_email,
        JSON_EXTRACT_SCALAR(commit, '$.url') as commit_api_url
      FROM push_events, UNNEST(commits) as commit
      WHERE
        JSON_EXTRACT_SCALAR(commit, '$.message') LIKE '%Co-Authored-By%'
        AND JSON_EXTRACT_SCALAR(commit, '$.message') LIKE '%noreply@anthropic.com%'
    )

    SELECT DISTINCT
      sha,
      repo_name,
      repo_id,
      actor_login,
      actor_id,
      author_name,
      author_email,
      message,
      event_date,
      commit_api_url,
      -- Extract Claude model from message
      CASE
        WHEN message LIKE '%Claude Opus%' THEN
          REGEXP_EXTRACT(message, r'Claude (Opus[^<]*)')
        WHEN message LIKE '%Claude Sonnet%' THEN
          REGEXP_EXTRACT(message, r'Claude (Sonnet[^<]*)')
        WHEN message LIKE '%Claude Haiku%' THEN
          REGEXP_EXTRACT(message, r'Claude (Haiku[^<]*)')
        ELSE 'Claude'
      END as claude_model
    FROM claude_commits
    ORDER BY event_date DESC
    """

    return query


def estimate_cost(start_date: date, end_date: date) -> dict:
    """Estimate the cost of running the query."""
    days = (end_date - start_date).days + 1

    # GH Archive is roughly 10-15 GB per day compressed, ~100 GB uncompressed
    # BigQuery scans uncompressed data
    estimated_gb_per_day = 12  # Conservative estimate
    total_gb = days * estimated_gb_per_day
    total_tb = total_gb / 1024

    # BigQuery pricing: $5 per TB, first 1 TB free per month
    free_tier_tb = 1.0
    billable_tb = max(0, total_tb - free_tier_tb)
    cost = billable_tb * 5.0

    return {
        'days': days,
        'estimated_tb': round(total_tb, 2),
        'free_tier_tb': free_tier_tb,
        'billable_tb': round(billable_tb, 2),
        'estimated_cost': round(cost, 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Claude commits from GH Archive using BigQuery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path (.parquet or .csv)"
    )
    parser.add_argument(
        "--start-date",
        type=parse_date,
        default=date(2025, 2, 1),
        help="Start date (default: 2025-02-01)"
    )
    parser.add_argument(
        "--end-date",
        type=parse_date,
        default=None,
        help="End date (default: today)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate cost without running query"
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Google Cloud project ID (optional)"
    )

    args = parser.parse_args()

    end_date = args.end_date or date.today()

    # Estimate cost
    estimate = estimate_cost(args.start_date, end_date)

    print("=" * 60)
    print("BIGQUERY COST ESTIMATE")
    print("=" * 60)
    print(f"Date range: {args.start_date} to {end_date}")
    print(f"Days to scan: {estimate['days']}")
    print(f"Estimated data: ~{estimate['estimated_tb']} TB")
    print(f"Free tier: {estimate['free_tier_tb']} TB/month")
    print(f"Billable: ~{estimate['billable_tb']} TB")
    print(f"Estimated cost: ~${estimate['estimated_cost']:.2f}")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] Query not executed.")
        print("\nTo run the query:")
        print("  1. Set up Google Cloud credentials")
        print("  2. Run without --dry-run flag")
        return

    if not args.output:
        print("\nError: --output is required (unless using --dry-run)")
        sys.exit(1)

    # Build query
    query = build_query(args.start_date, end_date)

    print("\nInitializing BigQuery client...")

    # Look for credentials file
    script_dir = Path(__file__).parent.parent
    credentials_paths = [
        script_dir / "google-credentials.json",
        script_dir / "credentials.json",
        Path.home() / "google-credentials.json",
    ]

    credentials_file = None
    for cred_path in credentials_paths:
        if cred_path.exists():
            credentials_file = cred_path
            break

    try:
        if credentials_file:
            print(f"Using credentials: {credentials_file}")
            from google.oauth2 import service_account
            credentials = service_account.Credentials.from_service_account_file(
                str(credentials_file),
                scopes=["https://www.googleapis.com/auth/bigquery"]
            )
            client = bigquery.Client(credentials=credentials, project=credentials.project_id)
        elif args.project:
            client = bigquery.Client(project=args.project)
        else:
            client = bigquery.Client()
    except Exception as e:
        print(f"\nError: Could not initialize BigQuery client: {e}")
        print("\nTo authenticate, either:")
        print("  1. Place google-credentials.json in the project folder")
        print("  2. Run: gcloud auth application-default login")
        print("  3. Set GOOGLE_APPLICATION_CREDENTIALS environment variable")
        print("\nTo get credentials:")
        print("  1. Go to console.cloud.google.com")
        print("  2. IAM & Admin → Service Accounts")
        print("  3. Create service account with BigQuery User role")
        print("  4. Create JSON key and save as google-credentials.json")
        sys.exit(1)

    print("Running query (this may take 2-5 minutes)...")
    print()

    try:
        # Run query
        query_job = client.query(query)

        # Wait for results
        results = query_job.result()

        # Get job statistics
        bytes_processed = query_job.total_bytes_processed
        gb_processed = bytes_processed / (1024**3)
        tb_processed = gb_processed / 1024
        actual_cost = max(0, tb_processed - 1) * 5  # $5/TB after 1TB free

        print(f"Query complete!")
        print(f"Data processed: {gb_processed:.2f} GB ({tb_processed:.3f} TB)")
        print(f"Actual cost: ~${actual_cost:.2f}")
        print()

        # Convert to DataFrame
        print("Converting to DataFrame...")
        df = results.to_dataframe()

        print(f"Total commits found: {len(df):,}")
        print(f"Unique repos: {df['repo_name'].nunique():,}")
        print(f"Unique authors: {df['actor_login'].nunique():,}")

        # Save output
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == '.csv':
            df.to_csv(output_path, index=False)
        else:
            df.to_parquet(output_path, index=False)

        print(f"\nOutput saved to: {output_path}")

    except Exception as e:
        print(f"\nError running query: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
