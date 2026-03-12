"""
Adoption Timeline Builder for AI Tool Adoption Analysis.

This module provides utilities for building per-developer adoption timelines
across multiple AI coding tools.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


@dataclass
class ToolUsage:
    """Usage data for a single tool."""
    first_use: Optional[datetime] = None
    last_use: Optional[datetime] = None
    count: int = 0

    def to_dict(self) -> Dict:
        return {
            "first_use": self.first_use.isoformat() if self.first_use else None,
            "last_use": self.last_use.isoformat() if self.last_use else None,
            "count": self.count,
        }


@dataclass
class UserTimeline:
    """Complete timeline for a single user."""
    user: str
    tools: Dict[str, ToolUsage] = field(default_factory=dict)
    adoption_sequence: List[str] = field(default_factory=list)
    switched_to_claude: bool = False
    switch_date: Optional[datetime] = None
    prior_tools: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "user": self.user,
            "tools": {k: v.to_dict() for k, v in self.tools.items()},
            "adoption_sequence": self.adoption_sequence,
            "switched_to_claude": self.switched_to_claude,
            "switch_date": self.switch_date.isoformat() if self.switch_date else None,
            "prior_tools": self.prior_tools,
            "num_tools": len(self.tools),
        }


class AdoptionTimelineBuilder:
    """
    Builds comprehensive adoption timelines for all developers.

    Extracts first/last use per tool per user and identifies adoption patterns
    like tool switching and multi-tool usage.
    """

    # Tool configurations
    TOOL_CONFIGS = {
        "claude": {
            "filename": "claude_commits_full.parquet",
            "user_col": "author_login",
            "time_col": "committed_date",
        },
        "codex": {
            "filename": "codex_prs.parquet",
            "user_col": "user_login",
            "time_col": "created_at",
        },
        "copilot": {
            "filename": "copilot_prs.parquet",
            "user_col": "user_login",
            "time_col": "created_at",
        },
        "jules": {
            "filename": "jules_commits.parquet",
            "user_col": "committer_login",
            "time_col": "committer_date",
        },
    }

    def __init__(self, data_dir: Path):
        """
        Initialize the timeline builder.

        Args:
            data_dir: Directory containing parquet files for each tool
        """
        self.data_dir = Path(data_dir)
        self.timelines: Dict[str, UserTimeline] = {}
        self.tool_data: Dict[str, pd.DataFrame] = {}

    def load_data(self, tools: Optional[List[str]] = None) -> None:
        """
        Load data from all available tool datasets.

        Args:
            tools: Optional list of tools to load. If None, loads all available.
        """
        if tools is None:
            tools = list(self.TOOL_CONFIGS.keys())

        print("Loading tool data...")

        for tool in tools:
            if tool not in self.TOOL_CONFIGS:
                print(f"  Unknown tool: {tool}")
                continue

            config = self.TOOL_CONFIGS[tool]
            filepath = self.data_dir / config["filename"]

            if not filepath.exists():
                print(f"  {tool}: File not found")
                continue

            try:
                df = pd.read_parquet(filepath)

                # Parse dates
                time_col = config["time_col"]
                if time_col in df.columns:
                    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

                self.tool_data[tool] = df
                print(f"  {tool}: {len(df):,} records")

            except Exception as e:
                print(f"  {tool}: Error - {e}")

    def build_user_timelines(self) -> Dict[str, UserTimeline]:
        """
        Build timelines for all users across all tools.

        Returns:
            Dict mapping user_login -> UserTimeline
        """
        if not self.tool_data:
            self.load_data()

        print("\nBuilding user timelines...")

        # First pass: collect all usage data
        user_tool_data: Dict[str, Dict[str, List[datetime]]] = {}

        for tool, df in self.tool_data.items():
            config = self.TOOL_CONFIGS[tool]
            user_col = config["user_col"]
            time_col = config["time_col"]

            if user_col not in df.columns or time_col not in df.columns:
                print(f"  Warning: Missing columns in {tool}")
                continue

            for _, row in df.iterrows():
                user = row[user_col]
                time = row[time_col]

                if pd.isna(user) or pd.isna(time):
                    continue

                if user not in user_tool_data:
                    user_tool_data[user] = {}
                if tool not in user_tool_data[user]:
                    user_tool_data[user][tool] = []

                user_tool_data[user][tool].append(time)

        print(f"  Found {len(user_tool_data):,} unique users")

        # Second pass: build timelines
        self.timelines = {}

        for user, tools_data in user_tool_data.items():
            timeline = UserTimeline(user=user)

            # Process each tool
            tool_first_use: List[Tuple[str, datetime]] = []

            for tool, times in tools_data.items():
                if times:
                    first_use = min(times)
                    last_use = max(times)
                    count = len(times)

                    timeline.tools[tool] = ToolUsage(
                        first_use=first_use,
                        last_use=last_use,
                        count=count
                    )

                    tool_first_use.append((tool, first_use))

            # Sort by first use to get adoption sequence
            tool_first_use.sort(key=lambda x: x[1])
            timeline.adoption_sequence = [t[0] for t in tool_first_use]

            # Check for Claude switching
            if "claude" in timeline.tools:
                claude_first = timeline.tools["claude"].first_use
                prior = [
                    t for t, usage in timeline.tools.items()
                    if t != "claude" and usage.first_use < claude_first
                ]

                if prior:
                    timeline.switched_to_claude = True
                    timeline.switch_date = claude_first
                    timeline.prior_tools = prior

            self.timelines[user] = timeline

        print(f"  Built {len(self.timelines):,} timelines")

        return self.timelines

    def identify_switchers(
        self,
        to_tool: str = "claude",
        from_tools: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Identify users who switched to a specific tool from other tools.

        Args:
            to_tool: Target tool (default: "claude")
            from_tools: Optional list of source tools to consider

        Returns:
            DataFrame with switcher details
        """
        if not self.timelines:
            self.build_user_timelines()

        switchers = []

        for user, timeline in self.timelines.items():
            if to_tool not in timeline.tools:
                continue

            to_usage = timeline.tools[to_tool]

            for from_tool, from_usage in timeline.tools.items():
                if from_tool == to_tool:
                    continue

                if from_tools and from_tool not in from_tools:
                    continue

                # Check if from_tool was used before to_tool
                if from_usage.first_use < to_usage.first_use:
                    gap_days = (to_usage.first_use - from_usage.first_use).days

                    switchers.append({
                        "user_login": user,
                        "from_tool": from_tool,
                        "to_tool": to_tool,
                        "from_first_use": from_usage.first_use,
                        "to_first_use": to_usage.first_use,
                        "gap_days": gap_days,
                        "from_count": from_usage.count,
                        "to_count": to_usage.count,
                    })

        df = pd.DataFrame(switchers)
        if len(df) > 0:
            df = df.sort_values("gap_days")

        print(f"Found {len(df):,} switchers to {to_tool}")

        return df

    def get_first_adoption_times(
        self,
        tool: str
    ) -> Dict[str, datetime]:
        """
        Get first adoption time for each user of a specific tool.

        Args:
            tool: Tool name

        Returns:
            Dict mapping user_login -> first use datetime
        """
        if not self.timelines:
            self.build_user_timelines()

        adoption_times = {}

        for user, timeline in self.timelines.items():
            if tool in timeline.tools:
                adoption_times[user] = timeline.tools[tool].first_use

        return adoption_times

    def get_all_adoption_times(self) -> Dict[str, Dict[str, datetime]]:
        """
        Get adoption times for all tools.

        Returns:
            Dict mapping tool -> {user -> first use datetime}
        """
        if not self.timelines:
            self.build_user_timelines()

        result = {}

        for tool in self.TOOL_CONFIGS.keys():
            result[tool] = self.get_first_adoption_times(tool)

        return result

    def compute_adoption_statistics(self) -> Dict[str, Any]:
        """
        Compute comprehensive adoption statistics.

        Returns:
            Dict with various statistics
        """
        if not self.timelines:
            self.build_user_timelines()

        stats = {}

        # Users per tool
        tool_users = {tool: 0 for tool in self.TOOL_CONFIGS.keys()}
        for timeline in self.timelines.values():
            for tool in timeline.tools:
                tool_users[tool] = tool_users.get(tool, 0) + 1
        stats["users_per_tool"] = tool_users

        # Multi-tool distribution
        tool_counts = {}
        for timeline in self.timelines.values():
            n = len(timeline.tools)
            tool_counts[n] = tool_counts.get(n, 0) + 1
        stats["tool_count_distribution"] = tool_counts

        # Switchers to Claude
        claude_switchers = sum(1 for t in self.timelines.values() if t.switched_to_claude)
        stats["claude_switchers"] = claude_switchers

        # Adoption by prior tool
        prior_tool_counts = {}
        for timeline in self.timelines.values():
            if timeline.switched_to_claude:
                for prior in timeline.prior_tools:
                    prior_tool_counts[prior] = prior_tool_counts.get(prior, 0) + 1
        stats["claude_switchers_by_prior_tool"] = prior_tool_counts

        # Adoption sequence patterns
        sequence_counts = {}
        for timeline in self.timelines.values():
            seq = " -> ".join(timeline.adoption_sequence)
            if seq:
                sequence_counts[seq] = sequence_counts.get(seq, 0) + 1
        # Get top 10 sequences
        top_sequences = sorted(sequence_counts.items(), key=lambda x: -x[1])[:10]
        stats["top_adoption_sequences"] = dict(top_sequences)

        return stats

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert all timelines to a DataFrame.

        Returns:
            DataFrame with one row per user
        """
        if not self.timelines:
            self.build_user_timelines()

        records = []

        for user, timeline in self.timelines.items():
            record = {
                "user_login": user,
                "num_tools": len(timeline.tools),
                "adoption_sequence": json.dumps(timeline.adoption_sequence),
                "switched_to_claude": timeline.switched_to_claude,
                "switch_date": timeline.switch_date,
                "prior_tools": json.dumps(timeline.prior_tools),
            }

            # Add per-tool columns
            for tool in self.TOOL_CONFIGS.keys():
                if tool in timeline.tools:
                    usage = timeline.tools[tool]
                    record[f"{tool}_first_use"] = usage.first_use
                    record[f"{tool}_last_use"] = usage.last_use
                    record[f"{tool}_count"] = usage.count
                else:
                    record[f"{tool}_first_use"] = None
                    record[f"{tool}_last_use"] = None
                    record[f"{tool}_count"] = 0

            records.append(record)

        return pd.DataFrame(records)

    def save(self, output_path: Path) -> None:
        """
        Save timelines to parquet file.

        Args:
            output_path: Path for output file
        """
        df = self.to_dataframe()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"Saved {len(df):,} timelines to {output_path}")


def analyze_adoption_patterns(data_dir: Path) -> Tuple[pd.DataFrame, Dict]:
    """
    Run complete adoption pattern analysis.

    Args:
        data_dir: Directory containing tool data files

    Returns:
        Tuple of (timelines DataFrame, statistics dict)
    """
    builder = AdoptionTimelineBuilder(data_dir)
    builder.load_data()
    builder.build_user_timelines()

    df = builder.to_dataframe()
    stats = builder.compute_adoption_statistics()

    return df, stats


if __name__ == "__main__":
    data_dir = Path("data/output")

    print("=" * 60)
    print("ADOPTION TIMELINE ANALYSIS")
    print("=" * 60)

    # Build timelines
    builder = AdoptionTimelineBuilder(data_dir)
    builder.load_data()
    timelines = builder.build_user_timelines()

    # Get statistics
    print("\n" + "=" * 60)
    print("ADOPTION STATISTICS")
    print("=" * 60)

    stats = builder.compute_adoption_statistics()

    print("\nUsers per tool:")
    for tool, count in stats["users_per_tool"].items():
        print(f"  {tool}: {count:,}")

    print("\nTool count distribution:")
    for count, users in sorted(stats["tool_count_distribution"].items()):
        print(f"  {count} tools: {users:,} users")

    print(f"\nClaude switchers: {stats['claude_switchers']:,}")

    print("\nClaude switchers by prior tool:")
    for tool, count in stats["claude_switchers_by_prior_tool"].items():
        print(f"  {tool}: {count:,}")

    print("\nTop adoption sequences:")
    for seq, count in stats["top_adoption_sequences"].items():
        print(f"  {seq}: {count:,}")

    # Identify switchers
    print("\n" + "=" * 60)
    print("COPILOT -> CLAUDE SWITCHERS")
    print("=" * 60)

    switchers = builder.identify_switchers("claude", ["copilot"])
    if len(switchers) > 0:
        print(f"\nFound {len(switchers):,} switchers")
        print(f"Average gap: {switchers['gap_days'].mean():.1f} days")
        print(f"Median gap: {switchers['gap_days'].median():.1f} days")

    # Save results
    output_dir = Path("analysis/output")
    output_dir.mkdir(exist_ok=True)

    builder.save(output_dir / "user_timelines.parquet")

    # Save statistics
    with open(output_dir / "adoption_statistics.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"\nSaved statistics to {output_dir / 'adoption_statistics.json'}")
