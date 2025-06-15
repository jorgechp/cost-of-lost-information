"""
Specificity-Sensitivity Model for Citation Analysis
================================================

This module implements a probabilistic model for analyzing citation patterns
based on specificity and sensitivity measures in academic networks.

Features:
---------
- Probability estimation for citation patterns
- Analysis of reference status (alive/dead)
- Citation impact assessment
- Network-based probability calculations
- Conditional probability analysis

The model considers four main states:
1. Cited papers with all references alive
2. Non-cited papers with all references alive
3. Cited papers with dead references
4. Non-cited papers with dead references

Required Dependencies:
--------------------
- networkx
"""

from typing import Tuple, Dict, Optional
import networkx as nx


class SpecificitySensitivityModel:
    """
    Implements specificity and sensitivity analysis for citation networks.

    This model analyzes citation patterns and reference status to estimate
    various probabilities related to knowledge generation and information loss.
    """

    def __init__(self, graph: nx.DiGraph, database: any):
        """
        Initialize the model.

        Args:
            graph: Citation network as NetworkX directed graph
            database: Database interface for paper status queries
        """
        self.graph = graph
        self.database = database

        # Probability of knowledge generation
        self.P_X1: Optional[float] = None  # New knowledge generated
        self.P_X0: Optional[float] = None  # No new knowledge

        # Conditional probabilities
        self.P_X0_S1: Optional[float] = None  # No citations, all refs alive
        self.P_S1_X1: Optional[float] = None  # All refs alive given citations
        self.P_S0_X1: Optional[float] = None  # Dead refs given citations
        self.P_S1_X0: Optional[float] = None  # All refs alive given no citations
        self.P_X1_S0: Optional[float] = None  # Citations given dead refs
        self.P_S0_X0: Optional[float] = None  # Dead refs given no citations
        self.P_X1_S1: Optional[float] = None  # Citations given all refs alive
        self.P_X0_S0: Optional[float] = None  # No citations given dead refs

    def count_paper_categories(self) -> Dict[str, int]:
        """
        Count papers in different categories.

        Returns:
            Dictionary with counts for different paper categories
        """
        counts = {
            "cited_alive_refs": 0,
            "non_cited_alive_refs": 0,
            "cited_dead_refs": 0,
            "non_cited_dead_refs": 0,
            "total_citations": 0
        }

        # Count citations
        counts["total_citations"] = sum(1 for _ in self.graph.edges)

        # Count papers in each category
        for node in self.graph.nodes:
            is_cited = self.graph.in_degree(node) > 0
            has_dead_refs = self.database.is_paper_affected(node)

            if is_cited and not has_dead_refs:
                counts["cited_alive_refs"] += 1
            elif not is_cited and not has_dead_refs:
                counts["non_cited_alive_refs"] += 1
            elif not is_cited and has_dead_refs:
                counts["non_cited_dead_refs"] += 1
            else:  # is_cited and has_dead_refs
                counts["cited_dead_refs"] += 1

        return counts

    def estimate_probabilities(self) -> None:
        """
        Estimate all model probabilities from the network.

        This method calculates:
        - Basic probabilities of knowledge generation
        - Conditional probabilities related to citation patterns
        - Joint probabilities of citation and reference status
        """
        # Get paper counts
        counts = self.count_paper_categories()
        total_papers = self.database.count_total_articles()

        # Basic probabilities
        self.P_X1 = counts["total_citations"] / total_papers
        self.P_X0 = 1.0 - self.P_X1

        # Joint probabilities
        self.P_X1_S0 = counts["cited_dead_refs"] / total_papers
        self.P_X0_S1 = counts["non_cited_alive_refs"] / total_papers
        self.P_X1_S1 = counts["cited_alive_refs"] / total_papers
        self.P_X0_S0 = counts["non_cited_dead_refs"] / total_papers

        # Calculate conditional probabilities
        S1_total = self.P_X1_S1 + self.P_X0_S1  # Total probability of alive refs
        S0_total = self.P_X1_S0 + self.P_X0_S0  # Total probability of dead refs

        if S1_total > 0:
            self.P_S1_X1 = self.P_X1_S1 / S1_total
            self.P_S1_X0 = self.P_X0_S1 / S1_total

        if S0_total > 0:
            self.P_S0_X1 = self.P_X1_S0 / S0_total
            self.P_S0_X0 = self.P_X0_S0 / S0_total

    def get_probabilities(self) -> Tuple[float, float, float, float, float, float]:
        """
        Get computed probabilities.

        Returns:
            Tuple of probabilities:
            (P_X1, P_X0, P_X0_S1, P_X1_S1, P_X1_S0, P_X0_S0)
        """
        return (
            self.P_X1,
            self.P_X0,
            self.P_X0_S1,
            self.P_X1_S1,
            self.P_X1_S0,
            self.P_X0_S0
        )

    def get_metrics(self) -> Dict[str, float]:
        """
        Calculate model metrics.

        Returns:
            Dictionary containing:
            - Sensitivity: P(S1|X1)
            - Specificity: P(S0|X0)
            - Precision: P(X1|S1)
            - Accuracy: Overall correct predictions
        """
        if not all([self.P_S1_X1, self.P_S0_X0, self.P_X1_S1]):
            raise ValueError("Probabilities not estimated. Call estimate_probabilities() first.")

        metrics = {
            "sensitivity": self.P_S1_X1,
            "specificity": self.P_S0_X0,
            "precision": self.P_X1_S1,
            "accuracy": (self.P_X1_S1 + self.P_X0_S0) if self.P_X1_S1 and self.P_X0_S0 else None
        }

        return metrics

    def summarize(self) -> str:
        """
        Generate model summary.

        Returns:
            Formatted string with model statistics and metrics
        """
        metrics = self.get_metrics()

        summary = [
            "Specificity-Sensitivity Model Summary",
            "====================================",
            f"Total papers analyzed: {self.database.count_total_articles()}",
            f"Total citations: {sum(1 for _ in self.graph.edges)}",
            "",
            "Basic Probabilities:",
            f"P(X1) [Knowledge generated]: {self.P_X1:.4f}",
            f"P(X0) [No knowledge generated]: {self.P_X0:.4f}",
            "",
            "Model Metrics:",
            f"Sensitivity [P(S1|X1)]: {metrics['sensitivity']:.4f}",
            f"Specificity [P(S0|X0)]: {metrics['specificity']:.4f}",
            f"Precision [P(X1|S1)]: {metrics['precision']:.4f}",
            f"Accuracy: {metrics['accuracy']:.4f}",
        ]

        return "\n".join(summary)


def main():
    """Example usage of the model."""
    # This would be your actual implementation
    pass


if __name__ == "__main__":
    main()