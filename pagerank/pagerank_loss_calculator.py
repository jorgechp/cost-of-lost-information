"""
PageRank Loss Calculator
=======================

This module implements various PageRank-based methods for calculating
the impact of missing or compromised references in citation networks.

Features:
---------
- Standard PageRank computation
- Weighted PageRank for penalized edges
- Personalized PageRank for affected nodes
- Loss calculation based on PageRank differences
- Configurable parameters for different scenarios

Methods:
--------
1. Standard PageRank: Basic PageRank algorithm
2. Weighted PageRank: Edge-weight based penalization
3. Personalized PageRank: Node-based penalization

Required Dependencies:
--------------------
- networkx
"""

import networkx as nx
from typing import Dict, Set, Optional, Union
import numpy as np


class PageRankLossCalculator:
    """
    Calculates citation network impact using PageRank variations.
    """

    def __init__(self,
                 graph: nx.DiGraph,
                 default_alpha: float = 0.85):
        """
        Initialize calculator.

        Args:
            graph: Citation network
            default_alpha: Damping factor (default: 0.85)
        """
        if not isinstance(graph, nx.DiGraph):
            raise ValueError("Graph must be a directed graph (DiGraph)")

        self.graph = graph
        self.default_alpha = default_alpha
        self._validate_graph()

    def _validate_graph(self) -> None:
        """Validate graph properties."""
        if not self.graph.nodes():
            raise ValueError("Graph is empty")
        if not nx.is_directed_acyclic_graph(self.graph):
            print("Warning: Graph contains cycles, which may affect PageRank convergence")

    def compute_standard_pagerank(self,
                                  alpha: Optional[float] = None) -> Dict[str, float]:
        """
        Compute standard PageRank scores.

        Args:
            alpha: Optional damping factor

        Returns:
            Dictionary of node PageRank scores
        """
        return nx.pagerank(
            self.graph,
            alpha=alpha or self.default_alpha
        )

    def compute_weighted_pagerank(self,
                                  missing_refs: Set[str],
                                  penalized_weight: float = 0.1,
                                  alpha: Optional[float] = None) -> Dict[str, float]:
        """
        Compute PageRank with penalized edges to missing references.

        Args:
            missing_refs: Set of nodes with missing references
            penalized_weight: Weight for penalized edges (default: 0.1)
            alpha: Optional damping factor

        Returns:
            Dictionary of node PageRank scores
        """
        if not 0 < penalized_weight <= 1:
            raise ValueError("Penalized weight must be between 0 and 1")

        # Create weighted graph
        G_weighted = self.graph.copy()

        for u, v in G_weighted.edges():
            weight = penalized_weight if v in missing_refs else 1.0
            G_weighted[u][v]['weight'] = weight

        return nx.pagerank(
            G_weighted,
            alpha=alpha or self.default_alpha,
            weight='weight'
        )

    def compute_personalized_pagerank(self,
                                      missing_refs: Set[str],
                                      penalization_value: float = 0.01,
                                      alpha: Optional[float] = None) -> Dict[str, float]:
        """
        Compute PageRank with personalized node penalties.

        Args:
            missing_refs: Set of nodes with missing references
            penalization_value: Penalty factor (default: 0.01)
            alpha: Optional damping factor

        Returns:
            Dictionary of node PageRank scores
        """
        if not 0 <= penalization_value <= 1:
            raise ValueError("Penalization value must be between 0 and 1")

        # Create personalization dict
        personalization = {
            node: penalization_value if node in missing_refs else 1.0
            for node in self.graph.nodes()
        }

        # Normalize
        total = sum(personalization.values())
        personalization = {
            k: v / total
            for k, v in personalization.items()
        }

        return nx.pagerank(
            self.graph,
            alpha=alpha or self.default_alpha,
            personalization=personalization
        )

    def calculate_loss(self,
                       original_scores: Dict[str, float],
                       modified_scores: Dict[str, float],
                       method: str = 'relative') -> float:
        """
        Calculate loss between two PageRank score sets.

        Args:
            original_scores: Original PageRank scores
            modified_scores: Modified PageRank scores
            method: Loss calculation method ('relative' or 'absolute')

        Returns:
            Calculated loss value
        """
        if method not in ['relative', 'absolute']:
            raise ValueError("Method must be 'relative' or 'absolute'")

        if method == 'relative':
            losses = [
                abs((modified_scores[node] - original_scores[node]) / original_scores[node])
                for node in original_scores
                if original_scores[node] > 0
            ]
        else:
            losses = [
                abs(modified_scores[node] - original_scores[node])
                for node in original_scores
            ]

        return np.mean(losses)

    def analyze_impact(self,
                       missing_refs: Set[str],
                       methods: Optional[Set[str]] = None) -> Dict[str, Dict[str, Union[float, Dict]]]:
        """
        Comprehensive impact analysis using multiple methods.

        Args:
            missing_refs: Set of nodes with missing references
            methods: Set of methods to use (default: all)

        Returns:
            Dictionary with analysis results
        """
        available_methods = {
            'weighted': (self.compute_weighted_pagerank, {'missing_refs': missing_refs}),
            'personalized': (self.compute_personalized_pagerank, {'missing_refs': missing_refs})
        }

        if methods:
            invalid_methods = methods - available_methods.keys()
            if invalid_methods:
                raise ValueError(f"Invalid methods: {invalid_methods}")
        else:
            methods = available_methods.keys()

        # Compute baseline
        baseline_scores = self.compute_standard_pagerank()

        results = {}
        for method in methods:
            func, kwargs = available_methods[method]
            modified_scores = func(**kwargs)

            results[method] = {
                'relative_loss': self.calculate_loss(
                    baseline_scores,
                    modified_scores,
                    'relative'
                ),
                'absolute_loss': self.calculate_loss(
                    baseline_scores,
                    modified_scores,
                    'absolute'
                ),
                'scores': modified_scores
            }

        return results


def main():
    """Example usage of PageRank Loss Calculator."""
    # Create sample citation network
    G = nx.DiGraph()
    G.add_edges_from([
        ('A', 'B'), ('B', 'C'), ('C', 'D'),
        ('D', 'A'), ('B', 'D'), ('C', 'A')
    ])

    # Define missing references
    missing_refs = {'B', 'C'}

    # Initialize calculator
    calculator = PageRankLossCalculator(G)

    # Analyze impact
    results = calculator.analyze_impact(missing_refs)

    # Print results
    print("\nPageRank Loss Analysis:")
    print("=" * 50)
    for method, data in results.items():
        print(f"\nMethod: {method}")
        print(f"Relative Loss: {data['relative_loss']:.4f}")
        print(f"Absolute Loss: {data['absolute_loss']:.4f}")


if __name__ == "__main__":
    main()