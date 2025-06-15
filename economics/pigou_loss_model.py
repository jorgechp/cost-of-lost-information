"""
Pigou Knowledge Loss Model
========================

This module implements a Pigouvian model for calculating knowledge loss
due to negative externalities in research citations.

The model quantifies the propagation of negative effects through the citation
network when non-reproducible research is cited by other papers.

Features:
---------
- Citation network analysis
- Social cost assessment
- Impact propagation calculation
- Average citation metrics
- Network-based externality evaluation
"""

import numpy as np
from typing import Dict, Any, List, Optional

from economics.economic_loss_model import EconomicLossModel


class PigouKnowledgeLossModel(EconomicLossModel):
    """
    Pigouvian model for calculating knowledge loss from citation externalities.

    The model calculates the total loss based on:
    - Number of non-reproducible references
    - Average citation count for affected papers
    - Social cost per citation
    """

    def __init__(self,
                 data: Dict[str, Any],
                 social_cost: float = 1.0):
        """
        Initialize the Pigou knowledge loss model.

        Args:
            data: Dictionary containing:
                - graph: Citation network (NetworkX DiGraph)
                - db: Database interface for affected papers
                - non_reproducible_refs: Count of non-reproducible references
            social_cost: Social cost coefficient per citation (default: 1.0)

        Raises:
            ValueError: If required data is missing or invalid
        """
        super().__init__(data)
        self._validate_input(data, social_cost)

        # Store input parameters
        self.graph = data["graph"]
        self.db = data["db"]
        self.non_reproducible_refs = data["non_reproducible_refs"]
        self.social_cost = social_cost

        # Initialize analysis results
        self.average_citations: Optional[float] = None
        self.total_loss: Optional[float] = None
        self.citation_counts: Optional[List[int]] = None

    def _validate_input(self,
                        data: Dict[str, Any],
                        social_cost: float) -> None:
        """
        Validate input parameters.

        Args:
            data: Input data dictionary
            social_cost: Social cost coefficient

        Raises:
            ValueError: If parameters are invalid
        """
        required_keys = {'graph', 'db', 'non_reproducible_refs'}
        if not all(key in data for key in required_keys):
            missing = required_keys - set(data.keys())
            raise ValueError(
                f"Missing required data keys: {', '.join(missing)}"
            )

        if not isinstance(social_cost, (int, float)):
            raise ValueError("Social cost must be numeric")

        if social_cost < 0:
            raise ValueError("Social cost cannot be negative")

    def compute_loss(self) -> float:
        """
        Compute the total knowledge loss using the Pigouvian model.

        Returns:
            float: Total calculated loss

        Formula:
            Loss = N_NR * avg_citations * C_s
            where:
            - N_NR: Number of non-reproducible references
            - avg_citations: Average citations per affected paper
            - C_s: Social cost per citation
        """
        # Get affected papers from database
        affected_nodes = self.db.get_all_affected_papers()

        # Calculate citation counts for affected papers
        self.citation_counts = [
            len(list(self.graph.successors(node)))
            for node in affected_nodes
            if node in self.graph
        ]

        # Calculate average citations
        self.average_citations = (
            np.mean(self.citation_counts)
            if self.citation_counts
            else 0.0
        )

        # Calculate total loss
        self.total_loss = (
                self.non_reproducible_refs *
                self.average_citations *
                self.social_cost
        )

        return self.total_loss

    def description(self) -> str:
        """
        Get detailed model description with calculations.

        Returns:
            str: Formatted description string
        """
        if self.average_citations is None:
            self.compute_loss()

        return (
            f"Pigou Knowledge Loss Model\n"
            f"========================\n\n"
            f"Formula:\n"
            f"Loss = N_NR * avg_citations * C_s\n\n"
            f"Parameters:\n"
            f"- Non-reproducible refs (N_NR): {self.non_reproducible_refs}\n"
            f"- Average citations: {self.average_citations:.2f}\n"
            f"- Social cost (C_s): {self.social_cost}\n\n"
            f"Calculation:\n"
            f"Loss = {self.non_reproducible_refs} * "
            f"{self.average_citations:.2f} * {self.social_cost}\n"
            f"Total Loss = {self.total_loss:.2f}"
        )

    def get_metrics(self) -> Dict[str, float]:
        """
        Get comprehensive model metrics.

        Returns:
            Dictionary containing:
            - Total loss
            - Average citations
            - Citation variance
            - Total citations
            - Number of affected papers
        """
        if self.citation_counts is None:
            self.compute_loss()

        citation_array = np.array(self.citation_counts or [0])

        return {
            "total_loss": self.total_loss,
            "average_citations": self.average_citations,
            "citation_variance": np.var(citation_array),
            "total_citations": np.sum(citation_array),
            "affected_papers": len(self.citation_counts or [])
        }

    def get_citation_distribution(self) -> Dict[str, Any]:
        """
        Get detailed citation distribution statistics.

        Returns:
            Dictionary containing citation statistics and distribution data
        """
        if self.citation_counts is None:
            self.compute_loss()

        if not self.citation_counts:
            return {"error": "No citation data available"}

        citations = np.array(self.citation_counts)

        return {
            "min": np.min(citations),
            "max": np.max(citations),
            "median": np.median(citations),
            "mean": np.mean(citations),
            "std": np.std(citations),
            "quartiles": np.percentile(citations, [25, 50, 75]),
            "histogram_data": np.histogram(citations, bins='auto')
        }


def main():
    """Example usage of PigouKnowledgeLossModel."""
    import networkx as nx

    # Create sample citation network
    G = nx.DiGraph()
    G.add_edges_from([
        ('A', 'B'), ('B', 'C'), ('C', 'D'),
        ('D', 'E'), ('B', 'E'), ('C', 'E')
    ])

    # Mock database interface
    class MockDB:
        def get_all_affected_papers(self):
            return ['B', 'C', 'D']

    # Prepare input data
    data = {
        "graph": G,
        "db": MockDB(),
        "non_reproducible_refs": 3
    }

    # Create and run model
    model = PigouKnowledgeLossModel(
        data=data,
        social_cost=1.5
    )

    # Display results
    print("\nModel Description:")
    print(model.description())

    print("\nDetailed Metrics:")
    metrics = model.get_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nCitation Distribution:")
    dist = model.get_citation_distribution()
    for key, value in dist.items():
        if key != "histogram_data":
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()