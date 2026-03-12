"""
Cross-Tool User Matching for AI Coding Tool Adoption Analysis.

This module provides utilities for matching users across different AI coding tools
(Claude, Codex, Copilot, Jules) to track adoption sequences and identify multi-tool users.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import numpy as np


class CrossToolUserMatcher:
    """
    Matches users across multiple AI coding tool datasets.

    Identifies users who appear in multiple tools and tracks their adoption sequence
    to understand tool switching patterns.
    """

    # Tool configurations
    TOOL_CONFIGS = {
        "claude": {
            "filename": "claude_commits_full.parquet",
            "user_col": "author_login",
            "time_col": "committed_date",
            "activity_col": "sha",  # For counting activity
        },
        "codex": {
            "filename": "codex_prs.parquet",
            "user_col": "user_login",
            "time_col": "created_at",
            "activity_col": "pr_id",
        },
        "copilot": {
            "filename": "copilot_prs.parquet",
            "user_col": "user_login",
            "time_col": "created_at",
            "activity_col": "pr_id",
        },
        "jules": {
            "filename": "jules_commits.parquet",
            "user_col": "committer_login",  # Jules uses bot commits
            "time_col": "committer_date",
            "activity_col": "sha",
        },
    }

    def __init__(self, data_dir: Path):
        """
        Initialize the user matcher.

        Args:
            data_dir: Directory containing parquet files for each tool
        """
        self.data_dir = Path(data_dir)
        self.tool_data: Dict[str, pd.DataFrame] = {}
        self.user_tools: Dict[str, Set[str]] = {}  # user -> set of tools
        self.tool_users: Dict[str, Set[str]] = {}  # tool -> set of users

    def load_data(self, tools: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Load data from all available tool datasets.

        Args:
            tools: Optional list of tools to load. If None, loads all available.

        Returns:
            Dict mapping tool name -> number of records loaded
        """
        if tools is None:
            tools = list(self.TOOL_CONFIGS.keys())

        loaded = {}

        for tool in tools:
            if tool not in self.TOOL_CONFIGS:
                print(f"Unknown tool: {tool}")
                continue

            config = self.TOOL_CONFIGS[tool]
            filepath = self.data_dir / config["filename"]

            if not filepath.exists():
                print(f"  {tool}: File not found - {filepath}")
                continue

            try:
                df = pd.read_parquet(filepath)

                # Parse dates
                time_col = config["time_col"]
                if time_col in df.columns:
                    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

                self.tool_data[tool] = df
                loaded[tool] = len(df)
                print(f"  {tool}: {len(df):,} records")

            except Exception as e:
                print(f"  {tool}: Error loading - {e}")

        return loaded

    def build_user_index(self) -> Dict[str, Set[str]]:
        """
        Build index mapping each user to their tools.

        Returns:
            Dict mapping user_login -> set of tool names
        """
        self.user_tools = {}
        self.tool_users = {}

        for tool, df in self.tool_data.items():
            config = self.TOOL_CONFIGS[tool]
            user_col = config["user_col"]

            if user_col not in df.columns:
                print(f"Warning: {user_col} not found in {tool} data")
                continue

            # Get unique users for this tool
            users = set(df[user_col].dropna().unique())
            self.tool_users[tool] = users

            # Add to user index
            for user in users:
                if user not in self.user_tools:
                    self.user_tools[user] = set()
                self.user_tools[user].add(tool)

        print(f"\nBuilt index for {len(self.user_tools):,} unique users")

        return self.user_tools

    def find_multi_tool_users(self, min_tools: int = 2) -> pd.DataFrame:
        """
        Find users who appear in multiple tools.

        Args:
            min_tools: Minimum number of tools a user must appear in

        Returns:
            DataFrame with user, tools list, and tool count
        """
        if not self.user_tools:
            self.build_user_index()

        multi_tool_users = []

        for user, tools in self.user_tools.items():
            if len(tools) >= min_tools:
                multi_tool_users.append({
                    "user_login": user,
                    "tools": sorted(list(tools)),
                    "tool_count": len(tools),
                    "tools_str": ", ".join(sorted(tools)),
                })

        df = pd.DataFrame(multi_tool_users)
        df = df.sort_values("tool_count", ascending=False)

        print(f"Found {len(df):,} users with {min_tools}+ tools")

        return df

    def get_adoption_sequence(self, user: str) -> List[Tuple[str, datetime, int]]:
        """
        Get the adoption sequence for a user across tools.

        Args:
            user: GitHub username

        Returns:
            List of (tool, first_use_time, activity_count) tuples, sorted by time
        """
        if not self.tool_data:
            raise ValueError("Data not loaded. Call load_data() first.")

        sequence = []

        for tool, df in self.tool_data.items():
            config = self.TOOL_CONFIGS[tool]
            user_col = config["user_col"]
            time_col = config["time_col"]

            if user_col not in df.columns:
                continue

            user_df = df[df[user_col] == user]

            if len(user_df) == 0:
                continue

            # Get first and last use
            first_use = user_df[time_col].min()
            activity_count = len(user_df)

            if pd.notna(first_use):
                sequence.append((tool, first_use, activity_count))

        # Sort by first use time
        sequence.sort(key=lambda x: x[1])

        return sequence

    def build_user_timelines(self) -> pd.DataFrame:
        """
        Build comprehensive timeline data for all users.

        Returns:
            DataFrame with per-user per-tool first/last use and counts
        """
        if not self.user_tools:
            self.build_user_index()

        records = []

        for user in self.user_tools.keys():
            record = {"user_login": user}

            sequence = self.get_adoption_sequence(user)

            # Add per-tool data
            for tool, first_use, count in sequence:
                record[f"{tool}_first_use"] = first_use
                record[f"{tool}_count"] = count

            # Add sequence info
            if sequence:
                record["first_tool"] = sequence[0][0]
                record["first_tool_date"] = sequence[0][1]
                record["last_tool"] = sequence[-1][0]
                record["tool_sequence"] = [s[0] for s in sequence]
                record["num_tools"] = len(sequence)
            else:
                record["num_tools"] = 0

            records.append(record)

        df = pd.DataFrame(records)

        return df

    def identify_switchers(
        self,
        from_tool: str,
        to_tool: str,
        min_gap_days: int = 0
    ) -> pd.DataFrame:
        """
        Identify users who switched from one tool to another.

        A switcher is someone who used from_tool first, then later used to_tool.

        Args:
            from_tool: Original tool (e.g., "copilot")
            to_tool: Target tool (e.g., "claude")
            min_gap_days: Minimum days between first use of each tool

        Returns:
            DataFrame with switcher details
        """
        if not self.user_tools:
            self.build_user_index()

        # Find users who used both tools
        from_users = self.tool_users.get(from_tool, set())
        to_users = self.tool_users.get(to_tool, set())
        both_users = from_users & to_users

        print(f"Users of {from_tool}: {len(from_users):,}")
        print(f"Users of {to_tool}: {len(to_users):,}")
        print(f"Users of both: {len(both_users):,}")

        switchers = []

        for user in both_users:
            sequence = self.get_adoption_sequence(user)
            tool_times = {s[0]: s[1] for s in sequence}

            from_time = tool_times.get(from_tool)
            to_time = tool_times.get(to_tool)

            if from_time is None or to_time is None:
                continue

            # Check if from_tool was used before to_tool
            if from_time < to_time:
                gap_days = (to_time - from_time).days

                if gap_days >= min_gap_days:
                    switchers.append({
                        "user_login": user,
                        "from_tool": from_tool,
                        "to_tool": to_tool,
                        f"{from_tool}_first_use": from_time,
                        f"{to_tool}_first_use": to_time,
                        "gap_days": gap_days,
                        "full_sequence": [s[0] for s in sequence],
                    })

        df = pd.DataFrame(switchers)
        if len(df) > 0:
            df = df.sort_values("gap_days")

        print(f"Switchers ({from_tool} -> {to_tool}): {len(df):,}")

        return df

    def compute_switching_matrix(self) -> pd.DataFrame:
        """
        Compute matrix of switching rates between all tool pairs.

        Returns:
            DataFrame where [i, j] = fraction of tool i users who later used tool j
        """
        tools = list(self.tool_data.keys())
        matrix_data = []

        for from_tool in tools:
            row = {"from_tool": from_tool}
            from_users = self.tool_users.get(from_tool, set())

            for to_tool in tools:
                if from_tool == to_tool:
                    row[to_tool] = 0.0
                    continue

                to_users = self.tool_users.get(to_tool, set())
                both_users = from_users & to_users

                # Count how many switched (used from_tool first)
                switchers = 0
                for user in both_users:
                    sequence = self.get_adoption_sequence(user)
                    tool_order = [s[0] for s in sequence]

                    try:
                        from_idx = tool_order.index(from_tool)
                        to_idx = tool_order.index(to_tool)
                        if from_idx < to_idx:
                            switchers += 1
                    except ValueError:
                        continue

                rate = switchers / len(from_users) if from_users else 0
                row[to_tool] = rate

            matrix_data.append(row)

        return pd.DataFrame(matrix_data).set_index("from_tool")

    def save_results(self, output_dir: Path) -> Dict[str, Path]:
        """
        Save all analysis results to files.

        Args:
            output_dir: Directory to save results

        Returns:
            Dict mapping result name -> file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved = {}

        # Multi-tool users
        multi_tool = self.find_multi_tool_users()
        path = output_dir / "multi_tool_users.parquet"
        multi_tool.to_parquet(path, index=False)
        saved["multi_tool_users"] = path
        print(f"Saved multi-tool users to {path}")

        # User timelines
        timelines = self.build_user_timelines()
        path = output_dir / "user_timelines.parquet"
        # Convert list columns to JSON for parquet
        if "tool_sequence" in timelines.columns:
            timelines["tool_sequence"] = timelines["tool_sequence"].apply(
                lambda x: json.dumps(x) if isinstance(x, list) else x
            )
        timelines.to_parquet(path, index=False)
        saved["user_timelines"] = path
        print(f"Saved user timelines to {path}")

        # Switching matrix
        matrix = self.compute_switching_matrix()
        path = output_dir / "switching_matrix.csv"
        matrix.to_csv(path)
        saved["switching_matrix"] = path
        print(f"Saved switching matrix to {path}")

        return saved


class UserAdoptionAnalyzer:
    """
    Analyzes user adoption patterns and characteristics.
    """

    def __init__(self, matcher: CrossToolUserMatcher):
        """
        Initialize with a user matcher.

        Args:
            matcher: CrossToolUserMatcher with loaded data
        """
        self.matcher = matcher

    def compute_adoption_stats(self) -> Dict[str, Any]:
        """
        Compute comprehensive adoption statistics.

        Returns:
            Dict with various adoption statistics
        """
        stats = {}

        # Per-tool user counts
        tool_counts = {tool: len(users) for tool, users in self.matcher.tool_users.items()}
        stats["users_per_tool"] = tool_counts

        # Multi-tool distribution
        user_tools = self.matcher.user_tools
        tool_count_dist = {}
        for user, tools in user_tools.items():
            n = len(tools)
            tool_count_dist[n] = tool_count_dist.get(n, 0) + 1
        stats["tool_count_distribution"] = tool_count_dist

        # Overlap statistics
        tools = list(self.matcher.tool_data.keys())
        overlaps = {}
        for i, tool1 in enumerate(tools):
            for tool2 in tools[i+1:]:
                users1 = self.matcher.tool_users.get(tool1, set())
                users2 = self.matcher.tool_users.get(tool2, set())
                overlap = len(users1 & users2)
                overlaps[f"{tool1}-{tool2}"] = {
                    "overlap": overlap,
                    "jaccard": overlap / len(users1 | users2) if users1 | users2 else 0,
                    f"{tool1}_exclusive": len(users1 - users2),
                    f"{tool2}_exclusive": len(users2 - users1),
                }
        stats["tool_overlaps"] = overlaps

        return stats

    def get_claude_adoption_paths(self) -> pd.DataFrame:
        """
        Analyze paths to Claude adoption.

        Returns:
            DataFrame showing what tools users used before Claude
        """
        claude_users = self.matcher.tool_users.get("claude", set())

        paths = []
        for user in claude_users:
            sequence = self.matcher.get_adoption_sequence(user)
            tool_order = [s[0] for s in sequence]

            if "claude" not in tool_order:
                continue

            claude_idx = tool_order.index("claude")
            prior_tools = tool_order[:claude_idx]

            paths.append({
                "user_login": user,
                "prior_tools": prior_tools,
                "prior_tools_str": " -> ".join(prior_tools) if prior_tools else "(direct)",
                "num_prior_tools": len(prior_tools),
                "had_copilot": "copilot" in prior_tools,
                "had_codex": "codex" in prior_tools,
            })

        df = pd.DataFrame(paths)

        # Summarize
        print("\nClaude Adoption Paths:")
        print(f"  Total Claude users: {len(df):,}")
        print(f"  Direct adopters (no prior tool): {(df['num_prior_tools'] == 0).sum():,}")
        if len(df) > 0:
            print(f"  From Copilot: {df['had_copilot'].sum():,}")
            print(f"  From Codex: {df['had_codex'].sum():,}")

        return df


if __name__ == "__main__":
    from pathlib import Path

    data_dir = Path("data/output")

    print("Cross-Tool User Matching Analysis")
    print("=" * 50)

    # Initialize matcher
    matcher = CrossToolUserMatcher(data_dir)

    # Load data
    print("\nLoading data...")
    matcher.load_data()

    # Build user index
    print("\nBuilding user index...")
    matcher.build_user_index()

    # Find multi-tool users
    print("\nFinding multi-tool users...")
    multi_tool = matcher.find_multi_tool_users()
    print(multi_tool.head(20))

    # Identify Copilot -> Claude switchers
    print("\n" + "=" * 50)
    print("Copilot -> Claude Switchers")
    print("=" * 50)
    copilot_claude = matcher.identify_switchers("copilot", "claude")
    if len(copilot_claude) > 0:
        print(copilot_claude.head(10))

    # Compute switching matrix
    print("\n" + "=" * 50)
    print("Tool Switching Matrix")
    print("=" * 50)
    matrix = matcher.compute_switching_matrix()
    print(matrix)

    # Save results
    print("\n" + "=" * 50)
    print("Saving Results")
    print("=" * 50)
    saved = matcher.save_results(data_dir)

    # Adoption analysis
    print("\n" + "=" * 50)
    print("Adoption Path Analysis")
    print("=" * 50)
    analyzer = UserAdoptionAnalyzer(matcher)
    stats = analyzer.compute_adoption_stats()
    print("\nUsers per tool:")
    for tool, count in stats["users_per_tool"].items():
        print(f"  {tool}: {count:,}")

    paths = analyzer.get_claude_adoption_paths()
