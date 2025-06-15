"""
Innovation Loss Model
===================

This module implements an innovation loss calculation model using
weighted softmax and dynamic slowdown factors.

Features:
---------
- Softmax-based innovation scoring
- Dynamic slowdown factors
- Citation-based analysis
- Configurable thresholds
- Detailed node annotation
- Comprehensive loss calculation

Required Dependencies:
--------------------
- numpy
- pandas
- networkx (through base class)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from base_innovation_loss_model import BaseInnovationLossModel


class InnovationLossModel(BaseInnovationLossModel):
    """
    Innovation loss model using weighted softmax and dynamic slowdown.
    """

    def __init__(self,
                 data: Dict[str, Any],
                 innovation_probability: float,
                 top_percentile: float = 0.95):
        """
        Initialize innovation loss model.

        Args:
            data: Dictionary containing:
                - graph: Citation network (NetworkX DiGraph)
                - db: Database interface
                - non_reproducible_refs: Set of non-reproducible references
            innovation_probability: Probability of innovation (0-1)
            top_percentile: Threshold for identifying disruptive nodes (0-1)
        """
        super().__init__(data, innovation_probability)

        self._validate_percentile(top_percentile)

        self.top_percentile = top_percentile
        self.average_innovation: Optional[float] = None
        self.analysis_data: Optional[pd.DataFrame] = None

    def _validate_percentile(self, percentile: float) -> None:
        """
        Validate percentile value.

        Args:
            percentile: Percentile threshold

        Raises:
            ValueError: If percentile is invalid
        """
        if not 0 < percentile < 1:
            raise ValueError("Percentile must be between 0 and 1")

    def _softmax(self, values: np.ndarray) -> np.ndarray:
        """
        Compute softmax of input values.

        Args:
            values: Input values array

        Returns:
            Softmax probabilities array
        """
        exp_values = np.exp(values - np.max(values))
        return exp_values / exp_values.sum()

    def _compute_average_innovation(self) -> None:
        """
        Compute average innovation using weighted softmax.
        """
        nodes = list(self.graph.nodes())

        # Calculate citation counts
        citation_counts = {
            node: len(list(self.graph.successors(node)))
            for node in nodes
        }

        # Calculate citation threshold
        threshold = np.percentile(
            list(citation_counts.values()),
            self.top_percentile * 100
        )

        scores: List[float] = []
        slowdown_factors: List[float] = []

        # Process each node
        for i, node in enumerate(nodes, 1):
            print(f"Processing node {i}/{len(nodes)}: {node}")

            citations = citation_counts[node]
            is_affected = self.db.is_paper_affected(node)

            # Determine slowdown factor
            if is_affected:
                slowdown = 0.3  # Significant slowdown for affected nodes
            elif citations >= threshold:
                slowdown = 1.2  # Boost for highly cited papers
            else:
                slowdown = 1.0  # Normal progress

            slowdown_factors.append(slowdown)
            scores.append(citations * slowdown)

        # Calculate weights and total innovation
        weights = self._softmax(np.array(scores))
        total_innovation = sum(weights * len(nodes))

        # Store results
        self.average_innovation = (
            total_innovation / len(self.non_reproducible_refs)
            if self.non_reproducible_refs
            else 0
        )

        # Create analysis dataframe
        self.analysis_data = pd.DataFrame({
            "node": nodes,
            "citations": [citation_counts[n] for n in nodes],
            "slowdown": slowdown_factors,
            "score": scores,
            "weight": weights
        })

    def _annotate_graph(self) -> None:
        """
        Annotate graph with innovation metrics.
        """
        if self.analysis_data is None:
            raise ValueError("Must compute innovation before annotation")

        for _, row in self.analysis_data.iterrows():
            node = row["node"]
            self.graph.nodes[node]["innovation"] = {
                "metrics": {
                    "citations": row["citations"],
                    "slowdown": row["slowdown"],
                    "weight": row["weight"]
                }
            }

    def compute_loss(self) -> float:
        """
        Compute total innovation loss.

        Returns:
            Calculated loss value
        """
        self._compute_average_innovation()
        self._annotate_graph()

        return (
                len(self.non_reproducible_refs) *
                self.average_innovation *
                self.innovation_probability
        )

    def description(self) -> str:
        """
        Get model description.

        Returns:
            Model description string
        """
        return (
            f"Innovation Loss Model\n"
            f"-------------------\n"
            f"Formula: Loss = N_R * I_avg * P\n"
            f"Where:\n"
            f"- N_R: {len(self.non_reproducible_refs)} non-reproducible references\n"
            f"- I_avg: {self.average_innovation:.4f} (weighted softmax)\n"
            f"- P: {self.innovation_probability} (innovation probability)\n"
            f"Additional Parameters:\n"
            f"- Top percentile: {self.top_percentile}\n"
            f"- Total nodes: {len(self.graph.nodes())}"
        )

    def get_node_metrics(self, node: str) -> Dict[str, float]:
        """
        Get innovation metrics for specific node.

        Args:
            node: Node identifier

        Returns:
            Dictionary of node metrics
        """
        if self.analysis_data is None:
            raise ValueError("Must compute innovation first")

        node_data = self.analysis_data[
            self.analysis_data["node"] == node
            ].iloc[0]

        return {
            "citations": node_data["citations"],
            "slowdown": node_data["slowdown"],
            "score": node_data["score"],
            "weight": node_data["weight"]
        }


def main():
    """Example usage of InnovationLossModel."""
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

    # Create and run model
    model = InnovationLossModel(
        data=data,
        innovation_probability=0.3,
        top_percentile=0.95
    )

    loss = model.compute_loss()
    print(f"\nComputed Loss: {loss:.4f}")
    print("\nModel Description:")
    print(model.description())

    print("\nNode Metrics:")
    for node in G.nodes():
        metrics = model.get_node_metrics(node)
        print(f"\n{node}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()