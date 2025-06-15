"""
Economic Impact Analysis Tool
===========================

This module analyzes economic impacts in academic citation networks,
implementing various economic models for loss estimation.

Features:
---------
- Multiple economic models
- Parameter estimation
- Author analysis
- Innovation loss calculation
- Knowledge productivity assessment
- Network-based metrics

Required Dependencies:
--------------------
- networkx
- arxiv
- numpy
- pickle
"""

import argparse
import pickle
from typing import Set, List, Dict, Any
import networkx as nx
import arxiv
from dataclasses import dataclass


@dataclass
class NetworkMetrics:
    """Container for network-based metrics."""
    total_papers: int
    non_reproducible: int
    probability_of_citation: float
    knowledge_impact: float
    unique_authors: int


class AuthorAnalyzer:
    """Handles author-related analysis."""

    @staticmethod
    def get_authors_from_arxiv(arxiv_id: str) -> List[str]:
        """
        Retrieve author information from arXiv.

        Args:
            arxiv_id: arXiv paper ID

        Returns:
            List of author names
        """
        client = arxiv.Client()
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            result = next(client.results(search))
            return [author.name for author in result.authors]
        except Exception as e:
            print(f"Error fetching arXiv data: {str(e)}")
            return []

    @staticmethod
    def extract_unique_authors(graph: nx.DiGraph) -> Set[str]:
        """
        Extract unique authors from graph.

        Args:
            graph: Citation network

        Returns:
            Set of unique author names
        """
        unique_authors = set()

        for _, data in graph.nodes(data=True):
            for author_parts in data.get("authors", []):
                if isinstance(author_parts, list):
                    # Join non-empty parts
                    author_name = " ".join(
                        part for part in author_parts if part
                    )
                    if author_name:
                        unique_authors.add(author_name)
                else:
                    unique_authors.add(str(author_parts))

        return unique_authors


class NetworkAnalyzer:
    """Analyzes network properties and metrics."""

    @staticmethod
    def estimate_citation_probability(graph: nx.DiGraph,
                                      reproducible_nodes: Set[str]) -> float:
        """
        Estimate probability of citation for reproducible papers.

        Args:
            graph: Citation network
            reproducible_nodes: Set of reproducible paper IDs

        Returns:
            Estimated probability
        """
        if not reproducible_nodes:
            return 0.0

        cited_reproducibles = [
            node for node in reproducible_nodes
            if list(graph.successors(node))
        ]

        return len(cited_reproducibles) / len(reproducible_nodes)

    @staticmethod
    def estimate_knowledge_impact(graph: nx.DiGraph,
                                  reproducible_nodes: Set[str],
                                  max_depth: int = 2) -> float:
        """
        Estimate knowledge impact of reproducible papers.

        Args:
            graph: Citation network
            reproducible_nodes: Set of reproducible paper IDs
            max_depth: Maximum depth for impact calculation

        Returns:
            Average impact score
        """
        def count_descendants(node: str) -> int:
            """Count descendants up to max_depth."""
            visited = set()
            frontier = {node}

            for _ in range(max_depth):
                next_frontier = set()
                for n in frontier:
                    children = set(graph.successors(n)) - visited
                    next_frontier |= children
                    visited |= children
                frontier = next_frontier

            return len(visited)

        if not reproducible_nodes:
            return 0.0

        impact_values = [
            count_descendants(node)
            for node in reproducible_nodes
        ]

        return sum(impact_values) / len(impact_values)


class EconomicModel:
    """Base class for economic impact models."""

    def __init__(self, data: Dict[str, Any]):
        """
        Initialize model.

        Args:
            data: Model parameters and data
        """
        self.data = data

    def compute_loss(self) -> float:
        """
        Compute economic loss.

        Returns:
            Computed loss value
        """
        raise NotImplementedError

    def description(self) -> str:
        """
        Get model description.

        Returns:
            Model description string
        """
        raise NotImplementedError


class OpportunityCostModel(EconomicModel):
    """Models opportunity cost of non-reproducible papers."""

    def __init__(self, data: Dict[str, Any], po: float, ik: float):
        """
        Initialize model.

        Args:
            data: Model data
            po: Probability of citation
            ik: Knowledge impact factor
        """
        super().__init__(data)
        self.po = po
        self.ik = ik

    def compute_loss(self) -> float:
        """Compute opportunity cost loss."""
        return (
                self.data['non_reproducible_papers'] *
                self.po *
                self.ik
        )

    def description(self) -> str:
        """Get model description."""
        return (
            "Opportunity Cost Model\n"
            f"P_O (citation probability): {self.po:.4f}\n"
            f"I_K (knowledge impact): {self.ik:.4f}"
        )


class KnowledgeProductivityModel(EconomicModel):
    """Models knowledge productivity loss."""

    def __init__(self, data: Dict[str, Any], a: float, l: int):
        """
        Initialize model.

        Args:
            data: Model data
            a: Productivity factor
            l: Number of researchers
        """
        super().__init__(data)
        self.a = a
        self.l = l

    def compute_loss(self) -> float:
        """Compute productivity loss."""
        return (
                self.a *
                self.l *
                self.data['non_reproducible_papers'] /
                self.data['total_papers']
        )

    def description(self) -> str:
        """Get model description."""
        return (
            "Knowledge Productivity Model\n"
            f"A (productivity factor): {self.a:.4f}\n"
            f"L (researchers): {self.l}"
        )


def analyze_network(graph: nx.DiGraph,
                    affected_nodes: Set[str]) -> NetworkMetrics:
    """
    Perform complete network analysis.

    Args:
        graph: Citation network
        affected_nodes: Set of affected paper IDs

    Returns:
        Network metrics
    """
    analyzer = NetworkAnalyzer()
    reproducible = set(graph.nodes())

    return NetworkMetrics(
        total_papers=len(graph.nodes()),
        non_reproducible=len(affected_nodes),
        probability_of_citation=analyzer.estimate_citation_probability(
            graph, reproducible
        ),
        knowledge_impact=analyzer.estimate_knowledge_impact(
            graph, reproducible
        ),
        unique_authors=len(AuthorAnalyzer.extract_unique_authors(graph))
    )


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Economic impact analysis for citation networks"
    )
    parser.add_argument(
        'network_path',
        type=str,
        help="Path to network pickle file"
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=2,
        help="Maximum depth for impact calculation"
    )

    args = parser.parse_args()

    try:
        # Load network
        with open(args.network_path, 'rb') as f:
            graph = pickle.load(f)

        # Get affected nodes (implementation depends on your data structure)
        affected_nodes = set()  # Define your affected nodes

        # Analyze network
        metrics = analyze_network(graph, affected_nodes)

        # Initialize models
        models = [
            OpportunityCostModel(
                data={
                    'non_reproducible_papers': metrics.non_reproducible,
                    'total_papers': metrics.total_papers
                },
                po=metrics.probability_of_citation,
                ik=metrics.knowledge_impact
            ),
            KnowledgeProductivityModel(
                data={
                    'non_reproducible_papers': metrics.non_reproducible,
                    'total_papers': metrics.total_papers
                },
                a=metrics.knowledge_impact,
                l=metrics.unique_authors
            )
        ]

        # Print results
        print("\nNetwork Statistics:")
        print(f"Total papers: {metrics.total_papers}")
        print(f"Non-reproducible papers: {metrics.non_reproducible}")
        print(f"Unique authors: {metrics.unique_authors}")
        print(f"Citation probability: {metrics.probability_of_citation:.4f}")
        print(f"Knowledge impact: {metrics.knowledge_impact:.4f}\n")

        print("Economic Impact Analysis:")
        for model in models:
            print("\n" + "="*40)
            print(model.description())
            print(f"Estimated loss: {model.compute_loss():.2f}")

        return 0

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return 1


if __name__ == "__main__":
    main()