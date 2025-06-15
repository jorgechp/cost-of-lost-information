"""
Discrete Innovation Loss Model
============================

This module implements a discrete class-based innovation loss model
with fixed slowdown values for different node categories.

Features:
---------
- Class-based slowdown factors
- Citation threshold analysis
- Customizable slowdown values
- Node classification
- Detailed metrics tracking
- Graph annotation

Required Dependencies:
--------------------
- numpy
- pandas
- networkx (through base class)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from economics.base_innovation_loss_model import BaseInnovationLossModel


class InnovationLossModelDiscrete(BaseInnovationLossModel):
    """
    Discrete innovation loss model with fixed class-based slowdown values.
    """

    DEFAULT_SLOWDOWN = {
        "healthy": 1.0,    # Normal progress
        "affected": 0.3,   # Reduced progress
        "disruptive": 1.2  # Enhanced progress
    }

    def __init__(self,
                 data: Dict[str, Any],
                 innovation_probability: float,
                 slowdown_values: Optional[Dict[str, float]] = None,
                 top_percentile: float = 0.95):
        """
        Initialize discrete innovation loss model.

        Args:
            data: Dictionary containing:
                - graph: Citation network (NetworkX DiGraph)
                - db: Database interface
                - non_reproducible_refs: Set of non-reproducible references
            innovation_probability: Probability of innovation (0-1)
            slowdown_values: Optional dictionary with values for
                           'healthy', 'affected', 'disruptive' states
            top_percentile: Threshold for identifying disruptive nodes (0-1)
        """
        super().__init__(data, innovation_probability)

        self._validate_parameters(innovation_probability, top_percentile)

        self.top_percentile = top_percentile
        self.slowdown_values = self._validate_slowdown(
            slowdown_values or self.DEFAULT_SLOWDOWN
        )

        # Analysis results
        self.average_innovation: Optional[float] = None
        self.node_data: Optional[pd.DataFrame] = None

    def _validate_parameters(self,
                             probability: float,
                             percentile: float) -> None:
        """
        Validate input parameters.

        Args:
            probability: Innovation probability
            percentile: Top percentile threshold

        Raises:
            ValueError: If parameters are invalid
        """
        if not 0 <= probability <= 1:
            raise ValueError("Innovation probability must be between 0 and 1")

        if not 0 < percentile < 1:
            raise ValueError("Top percentile must be between 0 and 1")

    def _validate_slowdown(self,
                           slowdown: Dict[str, float]) -> Dict[str, float]:
        """
        Validate slowdown values.

        Args:
            slowdown: Dictionary of slowdown values

        Returns:
            Validated slowdown dictionary

        Raises:
            ValueError: If slowdown values are invalid
        """
        required_keys = {"healthy", "affected", "disruptive"}

        if not all(key in slowdown for key in required_keys):
            raise ValueError(
                f"Slowdown must contain all keys: {required_keys}"
            )

        if not all(isinstance(v, (int, float)) for v in slowdown.values()):
            raise ValueError("All slowdown values must be numeric")

        if not all(v >= 0 for v in slowdown.values()):
            raise ValueError("All slowdown values must be non-negative")

        return slowdown

    def compute_loss(self) -> float:
        """
        Compute innovation loss using discrete slowdown values.

        Returns:
            Calculated loss value
        """
        nodes = list(self.graph.nodes())

        # Calculate citation counts
        citation_counts = {
            node: len(list(self.graph.successors(node)))
            for node in nodes
        }

        # Calculate threshold
        threshold = np.percentile(
            list(citation_counts.values()),
            self.top_percentile * 100
        )

        # Process nodes
        node_slowdowns: List[float] = []
        for i, node in enumerate(nodes, 1):
            print(f"Processing node {i}/{len(nodes)}: {node}")

            citations = citation_counts[node]
            is_affected = self.db.is_paper_affected(node)

            # Determine node class and slowdown
            if is_affected:
                slowdown = self.slowdown_values["affected"]
            elif citations >= threshold:
                slowdown = self.slowdown_values["disruptive"]
            else:
                slowdown = self.slowdown_values["healthy"]

            node_slowdowns.append(slowdown)

        # Calculate average innovation
        self.average_innovation = np.mean(node_slowdowns)

        # Store node data
        self.node_data = pd.DataFrame({
            "node": nodes,
            "citations": [citation_counts[n] for n in nodes],
            "slowdown": node_slowdowns
        })

        # Annotate graph
        self._annotate_graph()

        # Calculate total loss
        return (
                len(self.non_reproducible_refs) *
                self.average_innovation *
                self.innovation_probability
        )

    def _annotate_graph(self) -> None:
        """
        Annotate graph with node metrics.
        """
        if self.node_data is None:
            raise ValueError("Must compute loss before annotation")

        for _, row in self.node_data.iterrows():
            node = row["node"]

            if "innovation" not in self.graph.nodes[node]:
                self.graph.nodes[node]["innovation"] = {}

            self.graph.nodes[node]["innovation"]["discrete"] = {
                "citations": row["citations"],
                "slowdown": row["slowdown"]
            }

    def description(self) -> str:
        """
        Get model description.

        Returns:
            Model description string
        """
        return (
            f"Discrete Innovation Loss Model\n"
            f"----------------------------\n"
            f"Formula: Loss = N_R * I_avg * P\n"
            f"Where:\n"
            f"- N_R: {len(self.non_reproducible_refs)} non-reproducible references\n"
            f"- I_avg: {self.average_innovation:.4f} (mean slowdown)\n"
            f"- P: {self.innovation_probability} (innovation probability)\n"
            f"\nSlowdown Values:\n"
            f"- Healthy: {self.slowdown_values['healthy']}\n"
            f"- Affected: {self.slowdown_values['affected']}\n"
            f"- Disruptive: {self.slowdown_values['disruptive']}\n"
            f"\nParameters:\n"
            f"- Top percentile: {self.top_percentile}"
        )

    def get_node_class(self, node: str) -> str:
        """
        Get classification of specific node.

        Args:
            node: Node identifier

        Returns:
            Node class ('healthy', 'affected', or 'disruptive')
        """
        if self.node_data is None:
            raise ValueError("Must compute loss first")

        node_data = self.node_data[
            self.node_data["node"] == node
            ].iloc[0]

        slowdown = node_data["slowdown"]

        if slowdown == self.slowdown_values["affected"]:
            return "affected"
        elif slowdown == self.slowdown_values["disruptive"]:
            return "disruptive"
        else:
            return "healthy"


def main():
    """Example usage of InnovationLossModelDiscrete."""
    import networkx as nx

    # Create sample network
    G = nx.DiGraph()
    G.add_edges_from([
        ('A', 'B'), ('B', 'C'), ('C', 'D'),
        ('D', 'E'), ('B', 'E'), ('C', 'E')
    ])

    # Mock database class
    class MockDB:
        def is_paper_affected(self, node):
            return node in {'B', 'C'}

    # Prepare input data
    data = {
        "graph": G,
        "db": MockDB(),
        "non_reproducible_refs": {'B', 'C'}
    }

    # Custom slowdown values
    slowdown_values = {
        "healthy": 1.0,
        "affected": 0.2,
        "disruptive": 1.5
    }

    # Create and run model
    model = InnovationLossModelDiscrete(
        data=data,
        innovation_probability=0.3,
        slowdown_values=slowdown_values,
        top_percentile=0.95
    )

    loss = model.compute_loss()
    print(f"\nComputed Loss: {loss:.4f}")
    print("\nModel Description:")
    print(model.description())

    print("\nNode Classifications:")
    for node in G.nodes():
        node_class = model.get_node_class(node)
        print(f"{node}: {node_class}")


if __name__ == "__main__":
    main()