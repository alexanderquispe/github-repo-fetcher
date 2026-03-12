"""
Influence vs. Homophily Analysis for AI Coding Tool Adoption.

This module provides methods to disentangle peer influence from homophily
in the adoption of AI coding tools, using temporal analysis and matched
pair designs.

References:
    - Aral, Muchnik, & Sundararajan (2009). "Distinguishing influence-based
      contagion from homophily-driven diffusion." PNAS.
    - Shalizi & Thomas (2011). "Homophily and Contagion Are Generically
      Confounded in Observational Social Network Studies." Sociological
      Methods & Research.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class InfluenceAnalyzer:
    """
    Analyzes peer influence in AI tool adoption using temporal methods.
    """

    def __init__(self, G: nx.Graph, adoption_times: Dict[str, datetime]):
        """
        Initialize with network and adoption data.

        Args:
            G: Collaboration network (nodes = developers)
            adoption_times: Dict mapping developer -> first adoption datetime
        """
        self.G = G
        self.adoption_times = adoption_times
        self.network_users = set(G.nodes())
        self.adopters = set(adoption_times.keys()) & self.network_users

    def compute_neighbor_exposure(
        self,
        user: str,
        before_time: datetime,
        window_days: int = 30
    ) -> Dict:
        """
        Compute a user's exposure to adopting neighbors before a given time.

        Args:
            user: Developer username
            before_time: Cutoff time
            window_days: Time window to consider

        Returns:
            Dict with exposure metrics
        """
        if user not in self.G:
            return {"exposed": False, "num_adopting_neighbors": 0, "fraction_adopting": 0}

        neighbors = set(self.G.neighbors(user))
        if not neighbors:
            return {"exposed": False, "num_adopting_neighbors": 0, "fraction_adopting": 0}

        window_start = before_time - timedelta(days=window_days)

        # Count neighbors who adopted in the window
        adopting_neighbors = 0
        for neighbor in neighbors:
            if neighbor in self.adoption_times:
                neighbor_time = self.adoption_times[neighbor]
                if window_start <= neighbor_time < before_time:
                    adopting_neighbors += 1

        fraction = adopting_neighbors / len(neighbors)

        return {
            "exposed": adopting_neighbors > 0,
            "num_adopting_neighbors": adopting_neighbors,
            "fraction_adopting": fraction,
            "total_neighbors": len(neighbors)
        }

    def temporal_influence_regression(
        self,
        user_features: Optional[Dict[str, Dict]] = None,
        time_periods: int = 10
    ) -> Dict:
        """
        Run temporal regression to estimate peer influence effects.

        Tests whether neighbor adoption at t-1 predicts own adoption at t,
        controlling for user characteristics.

        Args:
            user_features: Dict mapping user -> feature dict (e.g., language, topics)
            time_periods: Number of time periods to divide data into

        Returns:
            Regression results including coefficients and significance
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required. Install with: pip install scikit-learn")

        # Get time range
        times = [t for u, t in self.adoption_times.items() if u in self.network_users]
        if not times:
            return {"error": "No valid adoption times found"}

        min_time = min(times)
        max_time = max(times)
        period_length = (max_time - min_time) / time_periods

        # Build panel dataset
        records = []

        for period in range(1, time_periods):  # Start at 1 to have lag
            period_start = min_time + period * period_length
            period_end = period_start + period_length
            lag_start = period_start - period_length

            for user in self.network_users:
                # Outcome: Did user adopt in this period?
                adopted_this_period = False
                if user in self.adoption_times:
                    adopt_time = self.adoption_times[user]
                    if period_start <= adopt_time < period_end:
                        adopted_this_period = True
                    elif adopt_time < period_start:
                        # Already adopted, skip
                        continue

                # Exposure: Fraction of neighbors who adopted in previous period
                neighbors = set(self.G.neighbors(user))
                if not neighbors:
                    continue

                lag_adopters = 0
                for neighbor in neighbors:
                    if neighbor in self.adoption_times:
                        n_time = self.adoption_times[neighbor]
                        if lag_start <= n_time < period_start:
                            lag_adopters += 1

                exposure = lag_adopters / len(neighbors)

                record = {
                    "user": user,
                    "period": period,
                    "adopted": int(adopted_this_period),
                    "neighbor_exposure": exposure,
                    "degree": len(neighbors)
                }

                # Add user features if provided
                if user_features and user in user_features:
                    record.update(user_features[user])

                records.append(record)

        df = pd.DataFrame(records)

        if len(df) < 100:
            return {"error": "Insufficient data for regression", "n_obs": len(df)}

        # Prepare features
        feature_cols = ["neighbor_exposure", "degree"]
        if user_features:
            # Add numeric features from user_features
            numeric_cols = [c for c in df.columns
                          if c not in ["user", "period", "adopted", "neighbor_exposure", "degree"]
                          and df[c].dtype in [np.float64, np.int64]]
            feature_cols.extend(numeric_cols)

        X = df[feature_cols].fillna(0)
        y = df["adopted"]

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit logistic regression
        model = LogisticRegression(max_iter=1000)
        model.fit(X_scaled, y)

        # Results
        results = {
            "n_observations": len(df),
            "n_adoptions": y.sum(),
            "adoption_rate": y.mean(),
            "coefficients": dict(zip(feature_cols, model.coef_[0])),
            "intercept": model.intercept_[0],
        }

        # Compute odds ratios
        results["odds_ratios"] = {
            col: np.exp(coef) for col, coef in results["coefficients"].items()
        }

        # Key metric: neighbor exposure effect
        exposure_coef = results["coefficients"]["neighbor_exposure"]
        results["influence_effect"] = {
            "coefficient": exposure_coef,
            "odds_ratio": np.exp(exposure_coef),
            "interpretation": f"1 SD increase in neighbor adoption associated with "
                            f"{(np.exp(exposure_coef) - 1) * 100:.1f}% change in adoption odds"
        }

        return results


class MatchedPairAnalyzer:
    """
    Matched pair analysis to isolate influence effects.

    Compares adoption rates between developers exposed to adopting neighbors
    vs. matched controls without such exposure.
    """

    def __init__(
        self,
        G: nx.Graph,
        adoption_times: Dict[str, datetime],
        user_features: Dict[str, Dict]
    ):
        """
        Initialize with network, adoption data, and user characteristics.

        Args:
            G: Collaboration network
            adoption_times: Developer -> adoption time
            user_features: Developer -> feature dict for matching
        """
        self.G = G
        self.adoption_times = adoption_times
        self.user_features = user_features

    def find_matches(
        self,
        treated_users: Set[str],
        control_pool: Set[str],
        match_on: List[str],
        max_matches: int = 1
    ) -> Dict[str, List[str]]:
        """
        Find matched controls for treated users based on characteristics.

        Args:
            treated_users: Users exposed to adopting neighbors
            control_pool: Potential control users (not exposed)
            match_on: Feature names to match on
            max_matches: Maximum controls per treated user

        Returns:
            Dict mapping treated user -> list of matched controls
        """
        matches = {}

        for treated in treated_users:
            if treated not in self.user_features:
                continue

            treated_features = self.user_features[treated]
            best_matches = []

            for control in control_pool:
                if control not in self.user_features:
                    continue

                control_features = self.user_features[control]

                # Compute similarity (simple exact matching on categorical)
                match_score = sum(
                    1 for feat in match_on
                    if treated_features.get(feat) == control_features.get(feat)
                )

                if match_score == len(match_on):
                    best_matches.append(control)
                    if len(best_matches) >= max_matches:
                        break

            if best_matches:
                matches[treated] = best_matches

        return matches

    def analyze_matched_pairs(
        self,
        exposure_window_days: int = 30,
        outcome_window_days: int = 60
    ) -> Dict:
        """
        Analyze adoption rates in matched exposed vs. control pairs.

        Args:
            exposure_window_days: Window to define exposure
            outcome_window_days: Window to measure adoption outcome

        Returns:
            Analysis results
        """
        # Get time range
        times = list(self.adoption_times.values())
        mid_time = min(times) + (max(times) - min(times)) / 2

        # Identify exposed and unexposed users as of mid_time
        exposed = set()
        unexposed = set()

        for user in self.G.nodes():
            if user in self.adoption_times:
                if self.adoption_times[user] < mid_time:
                    continue  # Already adopted

            exposure = self._compute_exposure(user, mid_time, exposure_window_days)
            if exposure > 0:
                exposed.add(user)
            else:
                unexposed.add(user)

        # Find matches (simplified - in practice use propensity scoring)
        match_features = list(next(iter(self.user_features.values())).keys()) if self.user_features else []

        matches = self.find_matches(exposed, unexposed, match_features)

        # Compute adoption rates in outcome window
        outcome_start = mid_time
        outcome_end = mid_time + timedelta(days=outcome_window_days)

        exposed_adopted = 0
        exposed_total = 0
        control_adopted = 0
        control_total = 0

        for treated, controls in matches.items():
            exposed_total += 1
            if treated in self.adoption_times:
                if outcome_start <= self.adoption_times[treated] < outcome_end:
                    exposed_adopted += 1

            for control in controls:
                control_total += 1
                if control in self.adoption_times:
                    if outcome_start <= self.adoption_times[control] < outcome_end:
                        control_adopted += 1

        results = {
            "n_exposed": exposed_total,
            "n_control": control_total,
            "exposed_adoption_rate": exposed_adopted / exposed_total if exposed_total > 0 else 0,
            "control_adoption_rate": control_adopted / control_total if control_total > 0 else 0,
        }

        if results["exposed_adoption_rate"] > 0 and results["control_adoption_rate"] > 0:
            results["relative_risk"] = results["exposed_adoption_rate"] / results["control_adoption_rate"]
            results["absolute_difference"] = results["exposed_adoption_rate"] - results["control_adoption_rate"]

        return results

    def _compute_exposure(self, user: str, as_of: datetime, window_days: int) -> float:
        """Compute fraction of neighbors who adopted in window."""
        if user not in self.G:
            return 0

        neighbors = set(self.G.neighbors(user))
        if not neighbors:
            return 0

        window_start = as_of - timedelta(days=window_days)
        adopters = 0

        for neighbor in neighbors:
            if neighbor in self.adoption_times:
                n_time = self.adoption_times[neighbor]
                if window_start <= n_time < as_of:
                    adopters += 1

        return adopters / len(neighbors)


class NetworkRandomizationTest:
    """
    Permutation tests to assess whether adoption clustering exceeds chance.

    Shuffles adoption times while preserving network structure to generate
    null distribution.
    """

    def __init__(self, G: nx.Graph, adoption_times: Dict[str, datetime]):
        self.G = G
        self.adoption_times = adoption_times

    def compute_adoption_clustering(self, times: Dict[str, datetime]) -> float:
        """
        Compute clustering metric: average fraction of neighbors who are co-adopters.
        """
        adopters = set(times.keys()) & set(self.G.nodes())

        if len(adopters) < 2:
            return 0

        total_clustering = 0
        count = 0

        for adopter in adopters:
            neighbors = set(self.G.neighbors(adopter))
            if neighbors:
                adopting_neighbors = neighbors & adopters
                total_clustering += len(adopting_neighbors) / len(neighbors)
                count += 1

        return total_clustering / count if count > 0 else 0

    def run_permutation_test(self, n_permutations: int = 1000) -> Dict:
        """
        Run permutation test for adoption clustering.

        Shuffles which users adopted (preserving total number) to generate
        null distribution.

        Returns:
            Test results including p-value
        """
        # Observed clustering
        observed = self.compute_adoption_clustering(self.adoption_times)

        # Network users
        network_users = list(set(self.G.nodes()))
        n_adopters = len(set(self.adoption_times.keys()) & set(network_users))

        # Null distribution
        null_values = []

        for _ in range(n_permutations):
            # Random sample of adopters (same size)
            random_adopters = np.random.choice(network_users, size=n_adopters, replace=False)

            # Assign random times (actual times shuffled)
            times = list(self.adoption_times.values())
            np.random.shuffle(times)
            shuffled_times = dict(zip(random_adopters, times[:n_adopters]))

            null_value = self.compute_adoption_clustering(shuffled_times)
            null_values.append(null_value)

        # P-value
        p_value = np.mean([v >= observed for v in null_values])

        return {
            "observed_clustering": observed,
            "null_mean": np.mean(null_values),
            "null_std": np.std(null_values),
            "p_value": p_value,
            "z_score": (observed - np.mean(null_values)) / np.std(null_values) if np.std(null_values) > 0 else 0,
            "significant": p_value < 0.05,
            "interpretation": "Adoption is significantly clustered in the network"
                             if p_value < 0.05 else "Adoption clustering not significant"
        }


# =============================================================================
# Example usage
# =============================================================================

def example_influence_analysis():
    """Example: Analyze influence effects in Claude Code adoption."""
    from network_builder import DeveloperNetworkBuilder, AdoptionTimeExtractor

    data_dir = Path("data/output")

    # Build network
    print("Building collaboration network...")
    builder = DeveloperNetworkBuilder()
    G = builder.build_from_commits(str(data_dir / "claude_commits.parquet"))

    # Get adoption times
    print("Extracting adoption times...")
    adoption_times = AdoptionTimeExtractor.from_commits(
        str(data_dir / "claude_commits.parquet")
    )

    # 1. Temporal regression analysis
    print("\n" + "=" * 60)
    print("TEMPORAL INFLUENCE REGRESSION")
    print("=" * 60)

    analyzer = InfluenceAnalyzer(G, adoption_times)
    regression_results = analyzer.temporal_influence_regression()

    print(f"\nObservations: {regression_results.get('n_observations', 'N/A')}")
    print(f"Adoption rate: {regression_results.get('adoption_rate', 0):.2%}")

    if "influence_effect" in regression_results:
        effect = regression_results["influence_effect"]
        print(f"\nPeer Influence Effect:")
        print(f"  Coefficient: {effect['coefficient']:.4f}")
        print(f"  Odds Ratio: {effect['odds_ratio']:.4f}")
        print(f"  {effect['interpretation']}")

    # 2. Randomization test
    print("\n" + "=" * 60)
    print("NETWORK RANDOMIZATION TEST")
    print("=" * 60)

    randomization = NetworkRandomizationTest(G, adoption_times)
    test_results = randomization.run_permutation_test(n_permutations=500)

    print(f"\nObserved clustering: {test_results['observed_clustering']:.4f}")
    print(f"Null mean: {test_results['null_mean']:.4f} (+/- {test_results['null_std']:.4f})")
    print(f"Z-score: {test_results['z_score']:.2f}")
    print(f"P-value: {test_results['p_value']:.4f}")
    print(f"\nConclusion: {test_results['interpretation']}")

    return regression_results, test_results


if __name__ == "__main__":
    print("Influence vs. Homophily Analysis")
    print("=================================\n")

    try:
        example_influence_analysis()
    except FileNotFoundError as e:
        print(f"Data file not found: {e}")
        print("Please ensure you have collected data using the fetcher scripts.")
    except ImportError as e:
        print(f"Missing dependency: {e}")
