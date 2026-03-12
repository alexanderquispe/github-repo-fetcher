"""
Tool Switching Model for AI Coding Tool Adoption Analysis.

This module models transitions between AI coding tools as a Markov chain,
analyzing how developers switch from one tool (Copilot, Codex) to another (Claude).
"""

import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


# Tool states
TOOL_STATES = [
    "none",      # No AI tool
    "copilot",   # Copilot only
    "codex",     # Codex only
    "claude",    # Claude only
    "multi"      # Multiple tools
]


class ToolSwitchingModel:
    """
    Models tool transitions as a Markov chain.

    Analyzes:
    - Transition probabilities between tool states
    - Switching rates to Claude from other tools
    - Peer influence on switching behavior
    """

    def __init__(
        self,
        user_timelines: Dict[str, Dict],
        tools: Optional[List[str]] = None
    ):
        """
        Initialize the tool switching model.

        Args:
            user_timelines: Dict mapping user -> timeline data
                           Each timeline has tool first_use times
            tools: List of tool names to consider
        """
        self.timelines = user_timelines
        self.tools = tools or ["copilot", "codex", "claude", "jules"]
        self.transition_matrix: Optional[np.ndarray] = None
        self.state_names = TOOL_STATES

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "ToolSwitchingModel":
        """
        Create model from timeline DataFrame.

        Args:
            df: DataFrame with user timelines

        Returns:
            ToolSwitchingModel instance
        """
        timelines = {}

        for _, row in df.iterrows():
            user = row["user_login"]
            timeline = {}

            for tool in ["copilot", "codex", "claude", "jules"]:
                first_use_col = f"{tool}_first_use"
                if first_use_col in row and pd.notna(row[first_use_col]):
                    timeline[tool] = row[first_use_col]

            timelines[user] = timeline

        return cls(timelines)

    def get_user_state(
        self,
        user: str,
        as_of: datetime
    ) -> str:
        """
        Get user's tool state at a given time.

        Args:
            user: User login
            as_of: Reference time

        Returns:
            State string (e.g., "none", "copilot", "claude", "multi")
        """
        if user not in self.timelines:
            return "none"

        timeline = self.timelines[user]
        active_tools = []

        for tool, first_use in timeline.items():
            if first_use is not None and first_use <= as_of:
                active_tools.append(tool)

        if not active_tools:
            return "none"
        elif len(active_tools) == 1:
            return active_tools[0]
        else:
            return "multi"

    def compute_transition_matrix(
        self,
        time_periods: int = 12,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> np.ndarray:
        """
        Compute state transition matrix from observed data.

        Divides time into periods and counts state transitions.

        Args:
            time_periods: Number of time periods
            start_date: Start of observation window
            end_date: End of observation window

        Returns:
            Transition probability matrix (rows=from, cols=to)
        """
        # Determine time range
        all_times = []
        for timeline in self.timelines.values():
            for first_use in timeline.values():
                if first_use is not None:
                    all_times.append(first_use)

        if not all_times:
            return np.zeros((len(self.state_names), len(self.state_names)))

        if start_date is None:
            start_date = min(all_times) - timedelta(days=30)
        if end_date is None:
            end_date = max(all_times) + timedelta(days=1)

        period_length = (end_date - start_date) / time_periods

        # Count transitions
        transition_counts = defaultdict(lambda: defaultdict(int))

        for user in self.timelines.keys():
            for i in range(time_periods - 1):
                t1 = start_date + i * period_length
                t2 = start_date + (i + 1) * period_length

                state_t1 = self.get_user_state(user, t1)
                state_t2 = self.get_user_state(user, t2)

                transition_counts[state_t1][state_t2] += 1

        # Convert to probability matrix
        n_states = len(self.state_names)
        matrix = np.zeros((n_states, n_states))

        for i, from_state in enumerate(self.state_names):
            row_total = sum(transition_counts[from_state].values())
            if row_total > 0:
                for j, to_state in enumerate(self.state_names):
                    matrix[i, j] = transition_counts[from_state][to_state] / row_total
            else:
                matrix[i, i] = 1.0  # Stay in same state if no data

        self.transition_matrix = matrix
        return matrix

    def get_transition_matrix_df(self) -> pd.DataFrame:
        """
        Get transition matrix as DataFrame.

        Returns:
            DataFrame with labeled rows and columns
        """
        if self.transition_matrix is None:
            self.compute_transition_matrix()

        return pd.DataFrame(
            self.transition_matrix,
            index=self.state_names,
            columns=self.state_names
        )

    def compute_switching_rates(self) -> Dict[str, Dict[str, float]]:
        """
        Compute direct switching rates between tools.

        Returns:
            Dict mapping (from_tool, to_tool) -> rate
        """
        switching_rates = {}

        for from_tool in self.tools:
            from_users = [
                u for u, t in self.timelines.items()
                if from_tool in t and t[from_tool] is not None
            ]

            switching_rates[from_tool] = {}

            for to_tool in self.tools:
                if from_tool == to_tool:
                    continue

                # Count users who switched from from_tool to to_tool
                switchers = 0
                for user in from_users:
                    timeline = self.timelines[user]
                    if to_tool in timeline and timeline[to_tool] is not None:
                        # Check if from_tool was used first
                        if timeline[from_tool] < timeline[to_tool]:
                            switchers += 1

                rate = switchers / len(from_users) if from_users else 0
                switching_rates[from_tool][to_tool] = rate

        return switching_rates

    def analyze_peer_influence_on_switching(
        self,
        G: "nx.Graph",
        from_tool: str = "copilot",
        to_tool: str = "claude"
    ) -> Dict[str, Any]:
        """
        Analyze whether having to_tool-using neighbors increases switch probability.

        Args:
            G: Collaboration network
            from_tool: Source tool
            to_tool: Target tool

        Returns:
            Dict with peer influence analysis
        """
        if not HAS_NETWORKX:
            return {"error": "networkx required"}

        # Get users of from_tool
        from_users = [
            u for u, t in self.timelines.items()
            if from_tool in t and t[from_tool] is not None
            and u in G
        ]

        if not from_users:
            return {"error": "No users of source tool in network"}

        # For each from_tool user, check:
        # 1. Did they switch to to_tool?
        # 2. What fraction of their neighbors used to_tool (before they switched)?

        exposed_switchers = 0
        exposed_non_switchers = 0
        unexposed_switchers = 0
        unexposed_non_switchers = 0

        exposure_data = []

        for user in from_users:
            timeline = self.timelines[user]
            from_time = timeline[from_tool]

            # Did they switch?
            switched = (
                to_tool in timeline
                and timeline[to_tool] is not None
                and timeline[from_tool] < timeline[to_tool]
            )

            # Compute neighbor exposure (before potential switch)
            neighbors = set(G.neighbors(user))
            if not neighbors:
                continue

            # For switchers, check exposure before switch
            # For non-switchers, check exposure at end of observation
            check_time = timeline.get(to_tool) if switched else max(
                t for t in timeline.values() if t is not None
            )

            exposed_neighbors = 0
            for neighbor in neighbors:
                if neighbor in self.timelines:
                    n_timeline = self.timelines[neighbor]
                    if to_tool in n_timeline and n_timeline[to_tool] is not None:
                        if n_timeline[to_tool] < check_time:
                            exposed_neighbors += 1

            exposure_fraction = exposed_neighbors / len(neighbors)
            exposed = exposure_fraction > 0

            exposure_data.append({
                "user": user,
                "switched": switched,
                "exposed": exposed,
                "exposure_fraction": exposure_fraction,
                "n_neighbors": len(neighbors),
            })

            if exposed and switched:
                exposed_switchers += 1
            elif exposed and not switched:
                exposed_non_switchers += 1
            elif not exposed and switched:
                unexposed_switchers += 1
            else:
                unexposed_non_switchers += 1

        # Compute statistics
        total_exposed = exposed_switchers + exposed_non_switchers
        total_unexposed = unexposed_switchers + unexposed_non_switchers

        exposed_switch_rate = exposed_switchers / total_exposed if total_exposed > 0 else 0
        unexposed_switch_rate = unexposed_switchers / total_unexposed if total_unexposed > 0 else 0

        # Relative risk
        relative_risk = exposed_switch_rate / unexposed_switch_rate if unexposed_switch_rate > 0 else float('inf')

        # Chi-square test (simplified)
        contingency = [
            [exposed_switchers, exposed_non_switchers],
            [unexposed_switchers, unexposed_non_switchers]
        ]

        return {
            "from_tool": from_tool,
            "to_tool": to_tool,
            "total_from_tool_users": len(from_users),
            "total_in_network": len([u for u in from_users if u in G]),
            "exposed_switchers": exposed_switchers,
            "exposed_non_switchers": exposed_non_switchers,
            "unexposed_switchers": unexposed_switchers,
            "unexposed_non_switchers": unexposed_non_switchers,
            "exposed_switch_rate": exposed_switch_rate,
            "unexposed_switch_rate": unexposed_switch_rate,
            "relative_risk": relative_risk,
            "absolute_difference": exposed_switch_rate - unexposed_switch_rate,
            "contingency_table": contingency,
            "interpretation": self._interpret_relative_risk(relative_risk),
        }

    def _interpret_relative_risk(self, rr: float) -> str:
        """Interpret relative risk value."""
        if rr == float('inf'):
            return "Cannot compute (no unexposed switchers)"
        elif rr > 2.0:
            return "Strong peer influence effect"
        elif rr > 1.5:
            return "Moderate peer influence effect"
        elif rr > 1.1:
            return "Weak peer influence effect"
        elif rr > 0.9:
            return "No significant effect"
        else:
            return "Negative effect (peer exposure reduces switching)"

    def predict_steady_state(self) -> np.ndarray:
        """
        Compute steady state distribution of the Markov chain.

        Returns:
            Array of steady state probabilities
        """
        if self.transition_matrix is None:
            self.compute_transition_matrix()

        # Find eigenvector for eigenvalue 1
        # Solve: pi * P = pi, sum(pi) = 1
        P = self.transition_matrix.T
        n = len(self.state_names)

        # Add constraint that probabilities sum to 1
        A = P - np.eye(n)
        A = np.vstack([A, np.ones(n)])
        b = np.zeros(n + 1)
        b[-1] = 1

        # Least squares solution
        try:
            steady_state, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            # Normalize to ensure non-negative and sum to 1
            steady_state = np.maximum(steady_state, 0)
            steady_state = steady_state / steady_state.sum()
        except np.linalg.LinAlgError:
            steady_state = np.ones(n) / n

        return steady_state

    def get_steady_state_df(self) -> pd.DataFrame:
        """
        Get steady state as DataFrame.

        Returns:
            DataFrame with state probabilities
        """
        steady_state = self.predict_steady_state()

        return pd.DataFrame({
            "state": self.state_names,
            "probability": steady_state
        })

    def compute_mean_first_passage_time(self, to_state: str) -> Dict[str, float]:
        """
        Compute mean first passage time to a target state from each other state.

        Args:
            to_state: Target state (e.g., "claude")

        Returns:
            Dict mapping from_state -> mean steps to reach to_state
        """
        if self.transition_matrix is None:
            self.compute_transition_matrix()

        to_idx = self.state_names.index(to_state)
        n = len(self.state_names)

        # Remove target state row/col
        indices = [i for i in range(n) if i != to_idx]
        Q = self.transition_matrix[np.ix_(indices, indices)]

        # Mean first passage times: m = (I - Q)^{-1} * 1
        try:
            N = np.linalg.inv(np.eye(len(indices)) - Q)
            mfpt = N @ np.ones(len(indices))
        except np.linalg.LinAlgError:
            mfpt = np.full(len(indices), np.inf)

        # Map back to state names
        result = {}
        for i, idx in enumerate(indices):
            result[self.state_names[idx]] = mfpt[i]

        return result


def analyze_tool_switching(
    timelines_df: pd.DataFrame,
    G: Optional["nx.Graph"] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Run comprehensive tool switching analysis.

    Args:
        timelines_df: DataFrame with user timelines
        G: Optional collaboration network for peer influence analysis
        output_dir: Optional directory to save results

    Returns:
        Dict with all analysis results
    """
    results = {}

    print("\n" + "=" * 60)
    print("TOOL SWITCHING ANALYSIS")
    print("=" * 60)

    # Create model
    model = ToolSwitchingModel.from_dataframe(timelines_df)

    # 1. Transition matrix
    print("\n1. Computing transition matrix...")
    matrix = model.compute_transition_matrix()
    matrix_df = model.get_transition_matrix_df()
    results["transition_matrix"] = matrix_df.to_dict()
    print("\nTransition Matrix:")
    print(matrix_df.round(4))

    # 2. Switching rates
    print("\n2. Computing switching rates...")
    switching_rates = model.compute_switching_rates()
    results["switching_rates"] = switching_rates
    print("\nSwitching rates to Claude:")
    for from_tool, rates in switching_rates.items():
        if "claude" in rates:
            print(f"  {from_tool} -> claude: {rates['claude']:.2%}")

    # 3. Steady state
    print("\n3. Computing steady state distribution...")
    steady_state_df = model.get_steady_state_df()
    results["steady_state"] = steady_state_df.to_dict()
    print("\nSteady State:")
    for _, row in steady_state_df.iterrows():
        print(f"  {row['state']}: {row['probability']:.2%}")

    # 4. Mean first passage time to Claude
    print("\n4. Computing mean first passage time to Claude...")
    mfpt = model.compute_mean_first_passage_time("claude")
    results["mfpt_to_claude"] = mfpt
    print("\nMean steps to reach Claude:")
    for state, time in sorted(mfpt.items(), key=lambda x: x[1]):
        if time < float('inf'):
            print(f"  {state}: {time:.2f} periods")
        else:
            print(f"  {state}: unreachable")

    # 5. Peer influence analysis
    if G is not None and HAS_NETWORKX:
        print("\n5. Analyzing peer influence on switching...")

        for from_tool in ["copilot", "codex"]:
            influence = model.analyze_peer_influence_on_switching(
                G, from_tool=from_tool, to_tool="claude"
            )
            results[f"peer_influence_{from_tool}_to_claude"] = influence

            if "error" not in influence:
                print(f"\n  {from_tool.capitalize()} -> Claude:")
                print(f"    Exposed switch rate: {influence['exposed_switch_rate']:.2%}")
                print(f"    Unexposed switch rate: {influence['unexposed_switch_rate']:.2%}")
                print(f"    Relative risk: {influence['relative_risk']:.2f}")
                print(f"    Interpretation: {influence['interpretation']}")

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save transition matrix
        matrix_df.to_csv(output_dir / "transition_matrix.csv")

        # Save all results as JSON
        with open(output_dir / "switching_analysis.json", "w") as f:
            # Convert numpy types
            def convert(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict()
                return str(obj)

            json.dump(results, f, indent=2, default=convert)

        print(f"\nSaved results to {output_dir}")

    return results


if __name__ == "__main__":
    from pathlib import Path

    data_dir = Path("data/output")
    output_dir = Path("analysis/output")

    # Load timelines
    timelines_path = output_dir / "user_timelines.parquet"

    if timelines_path.exists():
        print(f"Loading timelines from {timelines_path}...")
        timelines_df = pd.read_parquet(timelines_path)

        # Run analysis
        results = analyze_tool_switching(timelines_df, output_dir=output_dir)
    else:
        print(f"Timelines not found at {timelines_path}")
        print("Please run adoption_timeline.py first.")
