"""
Base Innovation Loss Model
=========================

This module provides an abstract base class for implementing
innovation loss models in academic citation networks.

Features:
---------
- Abstract interface for loss calculation
- Graph annotation capabilities
- Economic loss modeling integration
- Configurable innovation probability
- Data validation

Required Dependencies:
--------------------
- abc (Abstract Base Classes)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import networkx as nx

from economics.economic_loss_model import EconomicLossModel


class BaseInnovationLossModel(EconomicLossModel, ABC):
    """
    Abstract base class for innovation loss models.

    This class provides a framework for implementing specific
    innovation loss models in citation networks.
    """

    def __init__(self,
                 data: Dict[str, Any],
                 innovation_probability: float):
        """
        Initialize innovation loss model.

        Args:
            data: Dictionary containing:
                - graph: Citation network (NetworkX DiGraph)
                - db: Database interface
                - non_reproducible_refs: Set of non-reproducible references
            innovation_probability: Probability of innovation (0-1)
        """
        super().__init__(data)

        # Validate input data
        self._validate_input_data(data)

        # Initialize attributes
        self.graph: nx.DiGraph = data["graph"]
        self.db = data["db"]
        self.non_reproducible_refs = data["non_reproducible_refs"]
        self.innovation_probability = self._validate_probability(innovation_probability)

        # Analysis results
        self.average_innovation: Optional[float] = None
        self.innovation_factors: Optional[Dict[str, float]] = None

    def _validate_input_data(self, data: Dict[str, Any]) -> None:
        """
        Validate input data structure.

        Args:
            data: Input data dictionary

        Raises:
            ValueError: If required data is missing or invalid
        """
        required_keys = {"graph", "db", "non_reproducible_refs"}
        missing_keys = required_keys - set(data.keys())

        if missing_keys:
            raise ValueError(
                f"Missing required data: {', '.join(missing_keys)}"
            )

        if not isinstance(data["graph"], nx.DiGraph):
            raise ValueError("Graph must be a NetworkX DiGraph")

        if not isinstance(data["non_reproducible_refs"], (set, list, tuple)):
            raise ValueError(
                "non_reproducible_refs must be a collection (set, list, or tuple)"
            )

    def _validate_probability(self, p: float) -> float:
        """
        Validate probability value.

        Args:
            p: Probability value

        Returns:
            Validated probability value

        Raises:
            ValueError: If probability is invalid
        """
        if not isinstance(p, (int, float)):
            raise ValueError("Probability must be a number")

        if not 0 <= p <= 1:
            raise ValueError("Probability must be between 0 and 1")

        return float(p)

    @abstractmethod
    def compute_loss(self) -> float:
        """
        Compute innovation loss.

        Returns:
            Calculated loss value

        This method must be implemented by subclasses to define
        specific loss calculation logic.
        """
        pass

    @abstractmethod
    def _annotate_graph(self) -> None:
        """
        Annotate graph with model-specific attributes.

        This method must be implemented by subclasses to define
        how the graph should be annotated with additional information
        needed for loss calculation.
        """
        pass

    @abstractmethod
    def description(self) -> str:
        """
        Get model description.

        Returns:
            Model description string

        This method must be implemented by subclasses to provide
        a detailed description of the specific model implementation.
        """
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get model statistics.

        Returns:
            Dictionary containing:
            - Number of nodes
            - Number of edges
            - Number of non-reproducible references
            - Innovation probability
            - Average innovation (if computed)
        """
        stats = {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "non_reproducible_refs": len(self.non_reproducible_refs),
            "innovation_probability": self.innovation_probability,
            "average_innovation": self.average_innovation
        }
        return stats

    def validate_graph(self) -> bool:
        """
        Validate graph properties.

        Returns:
            True if graph is valid, False otherwise
        """
        is_valid = True
        issues = []

        if not self.graph.nodes():
            issues.append("Graph is empty")
            is_valid = False

        if not nx.is_directed_acyclic_graph(self.graph):
            issues.append("Graph contains cycles")
            is_valid = False

        invalid_refs = set(self.non_reproducible_refs) - set(self.graph.nodes())
        if invalid_refs:
            issues.append(
                f"Invalid references: {', '.join(map(str, invalid_refs))}"
            )
            is_valid = False

        if issues:
            print("Graph validation issues:")
            for issue in issues:
                print(f"- {issue}")

        return is_valid


def main():
    """
    Example implementation of BaseInnovationLossModel.

    This shows how to create a concrete implementation
    of the abstract base class.
    """
    class SimpleInnovationLossModel(BaseInnovationLossModel):
        def compute_loss(self) -> float:
            self._annotate_graph()
            # Simple implementation: proportion of affected nodes
            return len(self.non_reproducible_refs) / self.graph.number_of_nodes()

        def _annotate_graph(self) -> None:
            for node in self.graph.nodes():
                self.graph.nodes[node]['affected'] = node in self.non_reproducible_refs

        def description(self) -> str:
            return "Simple innovation loss model based on proportion of affected nodes"

    # Example usage
    G = nx.DiGraph()
    G.add_edges_from([
        ('A', 'B'), ('B', 'C'), ('C', 'D'),
        ('D', 'E'), ('B', 'E'), ('C', 'E')
    ])

    data = {
        "graph": G,
        "db": None,  # Would be your database interface
        "non_reproducible_refs": {'B', 'C'}
    }

    model = SimpleInnovationLossModel(data, innovation_probability=0.3)

    if model.validate_graph():
        loss = model.compute_loss()
        print(f"\nComputed Loss: {loss:.4f}")
        print("\nModel Statistics:")
        for key, value in model.get_statistics().items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()