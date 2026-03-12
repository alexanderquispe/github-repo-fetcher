"""
Contagion Models for AI Tool Adoption Analysis.

This module implements various epidemic/diffusion models to analyze how
AI coding tool adoption spreads through developer collaboration networks.

Models:
1. Independent Cascade Model (IC) - Probabilistic edge-based spreading
2. Linear Threshold Model (LT) - Threshold-based adoption
3. SIR Epidemic Model - Susceptible-Infected-Recovered dynamics
"""

import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import optimize

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class IndependentCascadeModel:
    """
    Independent Cascade Model for adoption spreading.

    Theory: Each "infected" (adopted) developer has probability p of
    "infecting" each susceptible neighbor in the next time step.
    """

    def __init__(
        self,
        G: "nx.Graph",
        adoption_times: Dict[str, datetime]
    ):
        """
        Initialize the Independent Cascade Model.

        Args:
            G: NetworkX graph representing collaboration network
            adoption_times: Dict mapping user -> first adoption datetime
        """
        if not HAS_NETWORKX:
            raise ImportError("networkx required")

        self.G = G
        self.adoption_times = adoption_times
        self.network_users = set(G.nodes())
        self.adopters = set(adoption_times.keys()) & self.network_users

        # Sort adopters by time
        self.sorted_adopters = sorted(
            [(u, t) for u, t in adoption_times.items() if u in self.network_users],
            key=lambda x: x[1]
        )

    def estimate_transmission_probability(
        self,
        time_window_days: int = 30
    ) -> Dict[str, float]:
        """
        Estimate transmission probability using Maximum Likelihood.

        For each new adopter, we look at their neighbors:
        - Infected neighbors (adopted before, within window)
        - Susceptible status of the adopter

        The MLE estimate is:
        p = number of successful transmissions / number of transmission opportunities

        Args:
            time_window_days: Max days between neighbor adoption and own adoption

        Returns:
            Dict with estimated parameters and confidence intervals
        """
        successful_transmissions = 0
        transmission_opportunities = 0

        already_adopted = set()

        for user, adopt_time in self.sorted_adopters:
            neighbors = set(self.G.neighbors(user))
            window_start = adopt_time - timedelta(days=time_window_days)

            # Count infected neighbors (adopted before within window)
            infected_neighbors = 0
            for neighbor in neighbors:
                if neighbor in already_adopted:
                    neighbor_time = self.adoption_times[neighbor]
                    if window_start <= neighbor_time < adopt_time:
                        infected_neighbors += 1
                        transmission_opportunities += 1

            # If any infected neighbor, one of them "caused" the adoption
            if infected_neighbors > 0:
                successful_transmissions += 1

            already_adopted.add(user)

        # MLE estimate
        if transmission_opportunities > 0:
            p_hat = successful_transmissions / transmission_opportunities
        else:
            p_hat = 0.0

        # Standard error (binomial)
        if transmission_opportunities > 0:
            se = np.sqrt(p_hat * (1 - p_hat) / transmission_opportunities)
        else:
            se = 0.0

        # 95% confidence interval
        ci_lower = max(0, p_hat - 1.96 * se)
        ci_upper = min(1, p_hat + 1.96 * se)

        return {
            "transmission_probability": p_hat,
            "standard_error": se,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "successful_transmissions": successful_transmissions,
            "transmission_opportunities": transmission_opportunities,
            "time_window_days": time_window_days,
        }

    def simulate_cascade(
        self,
        seeds: Set[str],
        p: float,
        max_steps: int = 100
    ) -> Tuple[Set[str], List[Set[str]]]:
        """
        Simulate cascade from seed set.

        Args:
            seeds: Initial adopters
            p: Transmission probability per edge
            max_steps: Maximum simulation steps

        Returns:
            Tuple of (final adopters, list of new adopters per step)
        """
        adopted = set(seeds)
        newly_adopted_history = [set(seeds)]

        for step in range(max_steps):
            newly_adopted = set()

            # Each current adopter tries to infect neighbors
            for adopter in adopted:
                for neighbor in self.G.neighbors(adopter):
                    if neighbor not in adopted:
                        # Each edge has probability p of transmission
                        if np.random.random() < p:
                            newly_adopted.add(neighbor)

            if not newly_adopted:
                break

            adopted |= newly_adopted
            newly_adopted_history.append(newly_adopted)

        return adopted, newly_adopted_history

    def validate_model(
        self,
        train_fraction: float = 0.7,
        n_simulations: int = 100
    ) -> Dict[str, Any]:
        """
        Validate model by training on early adopters and predicting later ones.

        Args:
            train_fraction: Fraction of adopters to use for training
            n_simulations: Number of Monte Carlo simulations

        Returns:
            Validation metrics
        """
        n_train = int(len(self.sorted_adopters) * train_fraction)

        # Training set
        train_adopters = dict(self.sorted_adopters[:n_train])
        test_adopters = dict(self.sorted_adopters[n_train:])

        # Estimate p from training data
        train_model = IndependentCascadeModel(self.G, train_adopters)
        params = train_model.estimate_transmission_probability()
        p_hat = params["transmission_probability"]

        # Simulate from training set seeds
        seeds = set(train_adopters.keys())
        cascade_sizes = []
        test_overlap = []

        for _ in range(n_simulations):
            final_adopters, _ = self.simulate_cascade(seeds, p_hat)
            cascade_sizes.append(len(final_adopters))

            # Check overlap with actual test adopters
            predicted_new = final_adopters - seeds
            actual_new = set(test_adopters.keys())
            overlap = len(predicted_new & actual_new)
            test_overlap.append(overlap / len(actual_new) if actual_new else 0)

        return {
            "train_size": n_train,
            "test_size": len(test_adopters),
            "estimated_p": p_hat,
            "mean_cascade_size": np.mean(cascade_sizes),
            "std_cascade_size": np.std(cascade_sizes),
            "actual_total_size": len(self.adopters),
            "mean_test_recall": np.mean(test_overlap),
            "std_test_recall": np.std(test_overlap),
        }


class LinearThresholdModel:
    """
    Linear Threshold Model for adoption spreading.

    Theory: Each developer has a threshold theta_i. They adopt when the
    fraction of their adopted neighbors exceeds their threshold.
    """

    def __init__(
        self,
        G: "nx.Graph",
        adoption_times: Dict[str, datetime]
    ):
        """
        Initialize the Linear Threshold Model.

        Args:
            G: NetworkX graph
            adoption_times: Dict mapping user -> first adoption datetime
        """
        if not HAS_NETWORKX:
            raise ImportError("networkx required")

        self.G = G
        self.adoption_times = adoption_times
        self.network_users = set(G.nodes())
        self.adopters = set(adoption_times.keys()) & self.network_users
        self.thresholds: Dict[str, float] = {}

    def estimate_thresholds(self) -> Dict[str, float]:
        """
        Infer threshold for each adopter based on when they adopted.

        For each adopter, threshold = fraction of neighbors who adopted
        before them.

        Returns:
            Dict mapping user -> estimated threshold
        """
        sorted_adopters = sorted(
            [(u, t) for u, t in self.adoption_times.items() if u in self.network_users],
            key=lambda x: x[1]
        )

        already_adopted = set()
        self.thresholds = {}

        for user, adopt_time in sorted_adopters:
            neighbors = set(self.G.neighbors(user))

            if neighbors:
                adopted_neighbors = neighbors & already_adopted
                threshold = len(adopted_neighbors) / len(neighbors)
            else:
                threshold = 0.0  # No neighbors, spontaneous adoption

            self.thresholds[user] = threshold
            already_adopted.add(user)

        return self.thresholds

    def get_threshold_distribution(self) -> Dict[str, float]:
        """
        Get statistics about the threshold distribution.

        Returns:
            Dict with distribution statistics
        """
        if not self.thresholds:
            self.estimate_thresholds()

        values = list(self.thresholds.values())

        return {
            "mean": np.mean(values),
            "median": np.median(values),
            "std": np.std(values),
            "min": min(values),
            "max": max(values),
            "zero_threshold_count": sum(1 for v in values if v == 0),
            "zero_threshold_fraction": sum(1 for v in values if v == 0) / len(values),
        }

    def simulate_spread(
        self,
        seeds: Set[str],
        threshold_dist: str = "uniform",
        max_steps: int = 100
    ) -> Tuple[Set[str], List[Set[str]]]:
        """
        Simulate spread from seed set.

        Args:
            seeds: Initial adopters
            threshold_dist: "uniform" for random, "inferred" to use estimated
            max_steps: Maximum steps

        Returns:
            Tuple of (final adopters, new adopters per step)
        """
        adopted = set(seeds)
        history = [set(seeds)]

        # Assign thresholds to non-seed nodes
        node_thresholds = {}
        for node in self.G.nodes():
            if node in seeds:
                node_thresholds[node] = 0.0  # Already adopted
            elif threshold_dist == "uniform":
                node_thresholds[node] = np.random.random()
            elif threshold_dist == "inferred" and node in self.thresholds:
                node_thresholds[node] = self.thresholds[node]
            else:
                node_thresholds[node] = np.random.random()

        for step in range(max_steps):
            newly_adopted = set()

            for node in self.G.nodes():
                if node in adopted:
                    continue

                neighbors = set(self.G.neighbors(node))
                if not neighbors:
                    continue

                # Check if threshold exceeded
                adopted_neighbors = neighbors & adopted
                fraction = len(adopted_neighbors) / len(neighbors)

                if fraction >= node_thresholds[node]:
                    newly_adopted.add(node)

            if not newly_adopted:
                break

            adopted |= newly_adopted
            history.append(newly_adopted)

        return adopted, history


class SIRModel:
    """
    SIR (Susceptible-Infected-Recovered) Epidemic Model.

    States:
    - S: Never used the AI tool
    - I: Currently using (active in last N days)
    - R: Stopped using (no activity for N days)
    """

    def __init__(
        self,
        G: "nx.Graph",
        adoption_times: Dict[str, datetime],
        activity_data: Optional[Dict[str, List[datetime]]] = None,
        activity_window_days: int = 90
    ):
        """
        Initialize SIR model.

        Args:
            G: NetworkX graph
            adoption_times: Dict mapping user -> first adoption time
            activity_data: Optional dict mapping user -> list of activity times
            activity_window_days: Days of inactivity before considered "recovered"
        """
        if not HAS_NETWORKX:
            raise ImportError("networkx required")

        self.G = G
        self.adoption_times = adoption_times
        self.activity_data = activity_data or {}
        self.activity_window = activity_window_days
        self.network_users = set(G.nodes())

    def classify_states(
        self,
        as_of: datetime
    ) -> Dict[str, str]:
        """
        Classify each developer as S, I, or R.

        Args:
            as_of: Reference datetime

        Returns:
            Dict mapping user -> state ("S", "I", or "R")
        """
        states = {}
        window_start = as_of - timedelta(days=self.activity_window)

        for user in self.network_users:
            if user not in self.adoption_times:
                states[user] = "S"  # Never adopted
            elif self.adoption_times[user] > as_of:
                states[user] = "S"  # Not yet adopted
            else:
                # Check if still active
                if user in self.activity_data:
                    recent_activity = [
                        t for t in self.activity_data[user]
                        if window_start <= t <= as_of
                    ]
                    if recent_activity:
                        states[user] = "I"  # Active
                    else:
                        states[user] = "R"  # Recovered/inactive
                else:
                    # Assume infected if adopted within activity window
                    if self.adoption_times[user] >= window_start:
                        states[user] = "I"
                    else:
                        states[user] = "R"

        return states

    def estimate_parameters(
        self,
        time_periods: int = 10
    ) -> Dict[str, float]:
        """
        Estimate beta (infection rate) and gamma (recovery rate).

        beta = new infections / (S * I contacts)
        gamma = recoveries / I

        Args:
            time_periods: Number of time periods for estimation

        Returns:
            Dict with estimated parameters
        """
        # Get time range
        times = list(self.adoption_times.values())
        if not times:
            return {"error": "No adoption data"}

        min_time = min(times)
        max_time = max(times)
        period_length = (max_time - min_time) / time_periods

        infections_list = []
        si_contacts_list = []
        recoveries_list = []
        infected_list = []

        for i in range(time_periods - 1):
            t1 = min_time + i * period_length
            t2 = min_time + (i + 1) * period_length

            states_t1 = self.classify_states(t1)
            states_t2 = self.classify_states(t2)

            # Count state transitions
            S_t1 = [u for u, s in states_t1.items() if s == "S"]
            I_t1 = [u for u, s in states_t1.items() if s == "I"]
            I_t2 = [u for u, s in states_t2.items() if s == "I"]
            R_t2 = [u for u, s in states_t2.items() if s == "R"]

            # New infections: S at t1 -> I at t2
            new_infections = sum(1 for u in S_t1 if states_t2.get(u) == "I")
            infections_list.append(new_infections)

            # SI contacts at t1
            si_contacts = 0
            for s_user in S_t1:
                for neighbor in self.G.neighbors(s_user):
                    if neighbor in I_t1:
                        si_contacts += 1
            si_contacts_list.append(si_contacts)

            # Recoveries: I at t1 -> R at t2
            recoveries = sum(1 for u in I_t1 if states_t2.get(u) == "R")
            recoveries_list.append(recoveries)
            infected_list.append(len(I_t1))

        # Estimate beta
        total_infections = sum(infections_list)
        total_si_contacts = sum(si_contacts_list)
        beta = total_infections / total_si_contacts if total_si_contacts > 0 else 0

        # Estimate gamma
        total_recoveries = sum(recoveries_list)
        total_infected = sum(infected_list)
        gamma = total_recoveries / total_infected if total_infected > 0 else 0

        return {
            "beta": beta,
            "gamma": gamma,
            "total_infections": total_infections,
            "total_si_contacts": total_si_contacts,
            "total_recoveries": total_recoveries,
            "total_infected_periods": total_infected,
        }

    def compute_R0(self) -> Dict[str, float]:
        """
        Compute basic reproduction number R0.

        R0 = beta * <k> / gamma

        where <k> is the average degree.

        Returns:
            Dict with R0 and components
        """
        params = self.estimate_parameters()

        if "error" in params:
            return params

        beta = params["beta"]
        gamma = params["gamma"]

        # Average degree
        avg_degree = sum(dict(self.G.degree()).values()) / self.G.number_of_nodes()

        # R0 calculation
        if gamma > 0:
            R0 = beta * avg_degree / gamma
        else:
            R0 = float("inf") if beta > 0 else 0

        return {
            "R0": R0,
            "beta": beta,
            "gamma": gamma,
            "avg_degree": avg_degree,
            "interpretation": self._interpret_R0(R0),
        }

    def _interpret_R0(self, R0: float) -> str:
        """Interpret R0 value."""
        if R0 < 1:
            return "Sub-critical: Adoption will die out"
        elif R0 == 1:
            return "Critical: Adoption at equilibrium"
        elif R0 < 2:
            return "Super-critical (mild): Slow spread"
        elif R0 < 5:
            return "Super-critical (moderate): Significant spread"
        else:
            return "Highly contagious: Rapid spread"


def estimate_all_models(
    G: "nx.Graph",
    adoption_times: Dict[str, datetime],
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Run all contagion models and compile results.

    Args:
        G: Collaboration network
        adoption_times: User adoption times
        output_dir: Optional directory to save results

    Returns:
        Dict with all model results
    """
    results = {}

    print("\n" + "=" * 60)
    print("CONTAGION MODEL ANALYSIS")
    print("=" * 60)

    # 1. Independent Cascade Model
    print("\n1. Independent Cascade Model")
    print("-" * 40)
    ic_model = IndependentCascadeModel(G, adoption_times)
    ic_params = ic_model.estimate_transmission_probability()
    results["independent_cascade"] = ic_params
    print(f"  Transmission probability: {ic_params['transmission_probability']:.4f}")
    print(f"  95% CI: [{ic_params['ci_95_lower']:.4f}, {ic_params['ci_95_upper']:.4f}]")

    # Validation
    print("\n  Validating model...")
    ic_validation = ic_model.validate_model()
    results["ic_validation"] = ic_validation
    print(f"  Mean cascade size: {ic_validation['mean_cascade_size']:.0f}")
    print(f"  Actual total: {ic_validation['actual_total_size']}")
    print(f"  Test recall: {ic_validation['mean_test_recall']:.2%}")

    # 2. Linear Threshold Model
    print("\n2. Linear Threshold Model")
    print("-" * 40)
    lt_model = LinearThresholdModel(G, adoption_times)
    lt_thresholds = lt_model.estimate_thresholds()
    lt_dist = lt_model.get_threshold_distribution()
    results["linear_threshold"] = lt_dist
    print(f"  Mean threshold: {lt_dist['mean']:.4f}")
    print(f"  Median threshold: {lt_dist['median']:.4f}")
    print(f"  Zero-threshold (seed) fraction: {lt_dist['zero_threshold_fraction']:.2%}")

    # 3. SIR Model
    print("\n3. SIR Epidemic Model")
    print("-" * 40)
    sir_model = SIRModel(G, adoption_times)
    sir_R0 = sir_model.compute_R0()
    results["sir"] = sir_R0
    if "error" not in sir_R0:
        print(f"  Beta (infection rate): {sir_R0['beta']:.6f}")
        print(f"  Gamma (recovery rate): {sir_R0['gamma']:.6f}")
        print(f"  R0: {sir_R0['R0']:.2f}")
        print(f"  Interpretation: {sir_R0['interpretation']}")
    else:
        print(f"  Error: {sir_R0['error']}")

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "model_parameters.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved results to {output_dir / 'model_parameters.json'}")

    return results


if __name__ == "__main__":
    from pathlib import Path
    from .collaboration_network import build_full_collaboration_network
    from .adoption_timeline import AdoptionTimelineBuilder

    data_dir = Path("data/output")
    output_dir = Path("analysis/output")

    # Build network
    G, stats = build_full_collaboration_network(data_dir)

    # Get adoption times
    builder = AdoptionTimelineBuilder(data_dir)
    builder.load_data()
    builder.build_user_timelines()
    adoption_times = builder.get_first_adoption_times("claude")

    print(f"\nNetwork: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f"Claude adopters: {len(adoption_times):,}")

    # Run models
    results = estimate_all_models(G, adoption_times, output_dir)
