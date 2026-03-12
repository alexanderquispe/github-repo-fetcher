#!/usr/bin/env python3
"""
Main Analysis Script for AI Tool Adoption Contagion Study.

This script runs the complete pipeline for analyzing contagion/cascade effects
in AI coding tool adoption, focusing on how developers switch from Copilot/Codex
to Claude Code through network influence.

Usage:
    python scripts/run_contagion_analysis.py [--fetch-contributors] [--skip-network]

Research Questions:
    1. Does AI tool adoption spread through developer collaboration networks like a contagion?
    2. Do developers switch from one tool (Copilot, Codex) to Claude Code due to peer influence?
    3. What is the transmission rate? What are the cascade sizes?
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

# Import analysis modules
from analysis.adoption_timeline import AdoptionTimelineBuilder
from analysis.collaboration_network import CollaborationNetworkBuilder, build_full_collaboration_network
from analysis.user_matcher import CrossToolUserMatcher
from analysis.cascade_analysis import analyze_cascades
from analysis.contagion_models import estimate_all_models
from analysis.tool_switching import analyze_tool_switching


def print_header(title: str) -> None:
    """Print a section header."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def fetch_contributors(data_dir: Path) -> None:
    """Fetch contributors for all repositories in the dataset."""
    from src.github_fetcher.contributor_fetcher import ContributorFetcher, get_unique_repos_from_datasets

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("ERROR: GITHUB_TOKEN environment variable not set")
        print("Please set your GitHub Personal Access Token:")
        print("  export GITHUB_TOKEN=ghp_your_token_here")
        sys.exit(1)

    print_header("PHASE 1: FETCHING REPOSITORY CONTRIBUTORS")

    # Get unique repos
    print("\nFinding unique repositories from datasets...")
    repos = get_unique_repos_from_datasets(data_dir)
    print(f"Total unique repositories: {len(repos):,}")

    # Fetch contributors
    fetcher = ContributorFetcher(token)
    df = fetcher.fetch_all_repos(
        repos=repos,
        output_path=data_dir / "contributors.parquet",
        resume_from=data_dir / "contributors.parquet"
    )

    print(f"\nFinal dataset: {len(df):,} contributor records")
    print(f"Unique users: {df['user_login'].nunique():,}")
    print(f"Unique repos: {df['repo_nwo'].nunique():,}")


def build_network_and_timelines(data_dir: Path, output_dir: Path) -> tuple:
    """Build collaboration network and user timelines."""
    print_header("PHASE 2: BUILDING NETWORK AND TIMELINES")

    # Build collaboration network
    print("\n" + "-" * 50)
    print("Building Collaboration Network")
    print("-" * 50)

    G, network_stats = build_full_collaboration_network(data_dir)

    # Save network statistics
    with open(output_dir / "network_statistics.json", "w") as f:
        json.dump(network_stats, f, indent=2, default=str)
    print(f"\nSaved network statistics to {output_dir / 'network_statistics.json'}")

    # Export network to GEXF
    builder = CollaborationNetworkBuilder()
    builder.G = G
    builder.export_to_gexf(output_dir / "collaboration_network.gexf")

    # Build user timelines
    print("\n" + "-" * 50)
    print("Building User Timelines")
    print("-" * 50)

    timeline_builder = AdoptionTimelineBuilder(data_dir)
    timeline_builder.load_data()
    timelines = timeline_builder.build_user_timelines()

    # Get adoption times for Claude
    adoption_times = timeline_builder.get_first_adoption_times("claude")

    # Save timelines
    timeline_builder.save(output_dir / "user_timelines.parquet")

    # Save adoption statistics
    adoption_stats = timeline_builder.compute_adoption_statistics()
    with open(output_dir / "adoption_statistics.json", "w") as f:
        json.dump(adoption_stats, f, indent=2, default=str)
    print(f"Saved adoption statistics to {output_dir / 'adoption_statistics.json'}")

    return G, adoption_times, timeline_builder


def run_cross_tool_analysis(data_dir: Path, output_dir: Path) -> None:
    """Run cross-tool user matching analysis."""
    print_header("PHASE 3: CROSS-TOOL USER MATCHING")

    matcher = CrossToolUserMatcher(data_dir)
    matcher.load_data()
    matcher.build_user_index()

    # Find multi-tool users
    multi_tool = matcher.find_multi_tool_users()
    print(f"\nMulti-tool users: {len(multi_tool):,}")

    # Identify switchers
    print("\n" + "-" * 50)
    print("Identifying Tool Switchers")
    print("-" * 50)

    for from_tool in ["copilot", "codex"]:
        switchers = matcher.identify_switchers(from_tool, "claude")
        if len(switchers) > 0:
            print(f"  {from_tool} -> claude: {len(switchers):,} users")

    # Save results
    matcher.save_results(output_dir)


def run_contagion_models(G, adoption_times: dict, output_dir: Path) -> dict:
    """Run all contagion models."""
    print_header("PHASE 4: CONTAGION MODEL ANALYSIS")

    # Filter adoption times to network users
    network_adoption = {
        u: t for u, t in adoption_times.items()
        if u in G
    }

    print(f"\nAdopters in network: {len(network_adoption):,}")

    if len(network_adoption) < 10:
        print("WARNING: Too few adopters in network for model estimation")
        return {}

    # Run models
    results = estimate_all_models(G, network_adoption, output_dir)

    return results


def run_cascade_analysis(G, adoption_times: dict, output_dir: Path) -> dict:
    """Run cascade detection and analysis."""
    print_header("PHASE 5: CASCADE DETECTION AND ANALYSIS")

    results = analyze_cascades(G, adoption_times, output_dir=output_dir)

    return results


def run_switching_analysis(output_dir: Path, G=None) -> dict:
    """Run tool switching analysis."""
    print_header("PHASE 6: TOOL SWITCHING ANALYSIS")

    # Load timelines
    timelines_path = output_dir / "user_timelines.parquet"
    if not timelines_path.exists():
        print("ERROR: User timelines not found. Run previous phases first.")
        return {}

    timelines_df = pd.read_parquet(timelines_path)
    results = analyze_tool_switching(timelines_df, G=G, output_dir=output_dir)

    return results


def generate_summary_report(output_dir: Path) -> None:
    """Generate a summary report of all analysis results."""
    print_header("PHASE 7: GENERATING SUMMARY REPORT")

    report = []
    report.append("# AI Tool Adoption Contagion Analysis Report")
    report.append(f"\nGenerated: {datetime.now().isoformat()}")
    report.append("\n---\n")

    # Network statistics
    network_stats_path = output_dir / "network_statistics.json"
    if network_stats_path.exists():
        with open(network_stats_path) as f:
            stats = json.load(f)

        report.append("## Network Statistics\n")
        report.append(f"- **Nodes**: {stats.get('nodes', 'N/A'):,}")
        report.append(f"- **Edges**: {stats.get('edges', 'N/A'):,}")
        report.append(f"- **Density**: {stats.get('density', 0):.6f}")
        report.append(f"- **Average Degree**: {stats.get('avg_degree', 0):.2f}")
        report.append(f"- **Largest Component**: {stats.get('largest_component_fraction', 0):.1%} of network")
        report.append("")

    # Adoption statistics
    adoption_stats_path = output_dir / "adoption_statistics.json"
    if adoption_stats_path.exists():
        with open(adoption_stats_path) as f:
            stats = json.load(f)

        report.append("## Adoption Statistics\n")
        report.append("### Users per Tool")
        for tool, count in stats.get("users_per_tool", {}).items():
            report.append(f"- **{tool.capitalize()}**: {count:,}")
        report.append("")
        report.append(f"- **Claude Switchers**: {stats.get('claude_switchers', 0):,}")
        report.append("")

    # Cascade statistics
    cascade_stats_path = output_dir / "cascade_statistics.json"
    if cascade_stats_path.exists():
        with open(cascade_stats_path) as f:
            all_stats = json.load(f)

        stats = all_stats.get("cascade_statistics", {})
        report.append("## Cascade Analysis\n")
        report.append(f"- **Number of Cascades**: {stats.get('num_cascades', 0):,}")
        report.append(f"- **Mean Cascade Size**: {stats.get('mean_cascade_size', 0):.2f}")
        report.append(f"- **Max Cascade Size**: {stats.get('max_cascade_size', 0):,}")
        report.append(f"- **Seed Fraction**: {stats.get('seed_fraction', 0):.1%}")
        report.append(f"- **Influenced Fraction**: {stats.get('influenced_fraction', 0):.1%}")
        report.append("")

        trans = all_stats.get("transmission_metrics", {})
        report.append("### Transmission Metrics")
        report.append(f"- **Transmission Rate**: {trans.get('transmission_rate', 0):.4f}")
        report.append(f"- **Mean Secondary Infections**: {trans.get('mean_secondary_infections', 0):.2f}")
        report.append("")

    # Model parameters
    model_params_path = output_dir / "model_parameters.json"
    if model_params_path.exists():
        with open(model_params_path) as f:
            params = json.load(f)

        report.append("## Contagion Model Results\n")

        ic = params.get("independent_cascade", {})
        report.append("### Independent Cascade Model")
        report.append(f"- **Transmission Probability**: {ic.get('transmission_probability', 0):.4f}")
        report.append(f"- **95% CI**: [{ic.get('ci_95_lower', 0):.4f}, {ic.get('ci_95_upper', 0):.4f}]")
        report.append("")

        lt = params.get("linear_threshold", {})
        report.append("### Linear Threshold Model")
        report.append(f"- **Mean Threshold**: {lt.get('mean', 0):.4f}")
        report.append(f"- **Zero-Threshold (Seed) Fraction**: {lt.get('zero_threshold_fraction', 0):.1%}")
        report.append("")

        sir = params.get("sir", {})
        report.append("### SIR Epidemic Model")
        report.append(f"- **Beta (Infection Rate)**: {sir.get('beta', 0):.6f}")
        report.append(f"- **Gamma (Recovery Rate)**: {sir.get('gamma', 0):.6f}")
        report.append(f"- **R0**: {sir.get('R0', 0):.2f}")
        if "interpretation" in sir:
            report.append(f"- **Interpretation**: {sir['interpretation']}")
        report.append("")

    # Switching analysis
    switching_path = output_dir / "switching_analysis.json"
    if switching_path.exists():
        with open(switching_path) as f:
            switching = json.load(f)

        report.append("## Tool Switching Analysis\n")

        for key, value in switching.items():
            if key.startswith("peer_influence_"):
                tool = key.replace("peer_influence_", "").replace("_to_claude", "")
                report.append(f"### Peer Influence: {tool.capitalize()} -> Claude")
                report.append(f"- **Exposed Switch Rate**: {value.get('exposed_switch_rate', 0):.2%}")
                report.append(f"- **Unexposed Switch Rate**: {value.get('unexposed_switch_rate', 0):.2%}")
                report.append(f"- **Relative Risk**: {value.get('relative_risk', 0):.2f}")
                report.append(f"- **Interpretation**: {value.get('interpretation', 'N/A')}")
                report.append("")

    # Research questions summary
    report.append("## Research Questions Summary\n")
    report.append("### Q1: Does AI tool adoption spread through networks like a contagion?")
    report.append("*See cascade analysis and transmission metrics above.*\n")
    report.append("### Q2: Do developers switch to Claude due to peer influence?")
    report.append("*See peer influence analysis in tool switching section.*\n")
    report.append("### Q3: What is the transmission rate and cascade sizes?")
    report.append("*See Independent Cascade Model and cascade statistics.*\n")

    report.append("\n---\n")
    report.append("*Generated by AI Tool Adoption Contagion Analysis Pipeline*")

    # Save report
    report_path = output_dir / "analysis_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report))

    print(f"Saved summary report to {report_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI Tool Adoption Contagion Analysis Pipeline"
    )
    parser.add_argument(
        "--fetch-contributors",
        action="store_true",
        help="Fetch contributors from GitHub API (requires GITHUB_TOKEN)"
    )
    parser.add_argument(
        "--skip-network",
        action="store_true",
        help="Skip network building (use existing data)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/output"),
        help="Directory containing input data files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/output"),
        help="Directory for output files"
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print_header("AI TOOL ADOPTION CONTAGION ANALYSIS")
    print(f"\nData directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Check for required data files
    required_files = [
        "claude_commits_full.parquet",
        "codex_prs.parquet",
        "copilot_prs.parquet",
    ]

    missing = [f for f in required_files if not (data_dir / f).exists()]
    if missing:
        print(f"\nWARNING: Missing data files: {missing}")
        print("Some analyses may fail or produce incomplete results.")

    # Phase 1: Fetch contributors (optional)
    if args.fetch_contributors:
        fetch_contributors(data_dir)

    # Phase 2: Build network and timelines
    if not args.skip_network:
        G, adoption_times, timeline_builder = build_network_and_timelines(data_dir, output_dir)
    else:
        print("\nSkipping network building, loading existing data...")
        # Load existing network and timelines
        from analysis.adoption_timeline import AdoptionTimelineBuilder
        timeline_builder = AdoptionTimelineBuilder(data_dir)
        timeline_builder.load_data()
        timeline_builder.build_user_timelines()
        adoption_times = timeline_builder.get_first_adoption_times("claude")

        # Rebuild network from saved data
        G, _ = build_full_collaboration_network(data_dir)

    # Phase 3: Cross-tool analysis
    run_cross_tool_analysis(data_dir, output_dir)

    # Phase 4: Contagion models
    model_results = run_contagion_models(G, adoption_times, output_dir)

    # Phase 5: Cascade analysis
    cascade_results = run_cascade_analysis(G, adoption_times, output_dir)

    # Phase 6: Tool switching analysis
    switching_results = run_switching_analysis(output_dir, G=G)

    # Phase 7: Generate summary report
    generate_summary_report(output_dir)

    print_header("ANALYSIS COMPLETE")
    print(f"\nAll results saved to: {output_dir}")
    print("\nKey output files:")
    print(f"  - {output_dir / 'analysis_report.md'} (summary report)")
    print(f"  - {output_dir / 'network_statistics.json'}")
    print(f"  - {output_dir / 'cascade_statistics.json'}")
    print(f"  - {output_dir / 'model_parameters.json'}")
    print(f"  - {output_dir / 'switching_analysis.json'}")
    print(f"  - {output_dir / 'collaboration_network.gexf'}")


if __name__ == "__main__":
    main()
