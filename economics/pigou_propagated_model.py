"""
Pigou Propagated Loss Model
==========================

This module implements a Pigouvian model that calculates knowledge loss
propagation through citation networks, considering depth decay and citation weights.

Features:
---------
- Network propagation analysis
- Configurable depth limits
- Customizable decay functions
- Citation-weighted impact
- Frontier-based propagation
"""

from typing import Dict, Any, Set, Union, Callable, Optional

from economics.economic_loss_model import EconomicLossModel


class PigouPropagatedModel(EconomicLossModel):
    """
    Pigouvian model with loss propagation through citation networks.
    Calculates economic impact considering network depth and citation weights.
    """

    def __init__(self,
                 data: Dict[str, Any],
                 max_depth: int = 3,
                 decay: Union[str, Callable[[int], float]] = "linear",
                 citation_weight: bool = False):
        """
        Initialize propagated loss model.

        Args:
            data: Dictionary containing:
                - graph: Citation network (NetworkX DiGraph)
                - db: Database interface for affected papers
                - non_reproducible_refs: Count of non-reproducible references
            max_depth: Maximum propagation depth (default: 3)
            decay: Decay function type:
                - "linear": 1/depth
                - "none": No decay
                - callable: Custom decay function
            citation_weight: Whether to weight by citation count (default: False)

        Raises:
            ValueError: If input parameters are invalid
        """
        super().__init__(data)
        self._validate_input(data, max_depth, decay)

        self.graph = data["graph"]
        self.db = data["db"]
        self.non_reproducible_refs = data["non_reproducible_refs"]
        self.max_depth = max_depth
        self.decay = decay
        self.citation_weight = citation_weight

        # Analysis results
        self.total_loss: Optional[float] = None
        self.propagation_stats: Dict[int, float] = {}
        self.affected_nodes_count: int = 0

    def _validate_input(self,
                        data: Dict[str, Any],
                        max_depth: int,
                        decay: Union[str, Callable]) -> None:
        """
        Validate input parameters.

        Args:
            data: Input data dictionary
            max_depth: Maximum propagation depth
            decay: Decay function or type

        Raises:
            ValueError: If parameters are invalid
        """
        required_keys = {'graph', 'db', 'non_reproducible_refs'}
        if not all(key in data for key in required_keys):
            missing = required_keys - set(data.keys())
            raise ValueError(
                f"Missing required data keys: {', '.join(missing)}"
            )

        if not isinstance(max_depth, int) or max_depth < 1:
            raise ValueError(
                "max_depth must be a positive integer"
            )

        if isinstance(decay, str) and decay not in ["linear", "none"]:
            raise ValueError(
                'decay must be "linear", "none", or a callable'
            )

        if not isinstance(decay, str) and not callable(decay):
            raise ValueError(
                "decay must be either a string or callable"
            )

    def _get_decay_factor(self, depth: int) -> float:
        """
        Calculate decay factor for given depth.

        Args:
            depth: Current propagation depth

        Returns:
            Decay factor value
        """
        if self.decay == "linear":
            return 1.0 / depth
        elif self.decay == "none":
            return 1.0
        else:
            return self.decay(depth)

    def compute_loss(self) -> float:
        """
        Compute propagated loss through citation network.

        Returns:
            Total calculated loss

        Algorithm:
            1. Start from affected nodes
            2. Propagate through citation network up to max_depth
            3. Apply decay and citation weights at each level
            4. Accumulate total loss
        """
        graph = self.graph
        affected_nodes = self.db.get_all_affected_papers()
        self.affected_nodes_count = len(affected_nodes)
        total_loss = 0.0

        for node in affected_nodes:
            if node not in graph:
                continue

            visited: Set[str] = set()
            frontier = {node}

            for depth in range(1, self.max_depth + 1):
                next_frontier = set()
                level_loss = 0.0

                for current_node in frontier:
                    # Get unvisited successors
                    children = set(graph.successors(current_node)) - visited

                    for child in children:
                        # Calculate weights
                        depth_weight = self._get_decay_factor(depth)
                        citation_factor = (
                            len(list(graph.successors(child)))
                            if self.citation_weight
                            else 1
                        )

                        # Accumulate loss
                        node_loss = depth_weight * citation_factor
                        level_loss += node_loss

                    # Update tracking sets
                    visited |= children
                    next_frontier |= children

                # Store statistics and update frontier
                self.propagation_stats[depth] = level_loss
                total_loss += level_loss
                frontier = next_frontier

                if not frontier:
                    break

        self.total_loss = total_loss
        return total_loss

    def description(self) -> str:
        """
        Get detailed model description.

        Returns:
            Formatted description string
        """
        if self.total_loss is None:
            self.compute_loss()

        decay_type = (
            "1/depth" if self.decay == "linear"
            else "none" if self.decay == "none"
            else "custom"
        )

        citation_desc = (
            "weighted by citations"
            if self.citation_weight
            else "equal weight"
        )

        description = [
            "Pigou Propagated Loss Model",
            "========================\n",
            f"Configuration:",
            f"- Maximum depth: {self.max_depth}",
            f"- Decay function: {decay_type}",
            f"- Citation weighting: {citation_desc}",
            f"\nResults:",
            f"- Affected nodes: {self.affected_nodes_count}",
            f"- Total propagated loss: {self.total_loss:.2f}",
            f"\nLoss by depth:"
        ]

        for depth, loss in self.propagation_stats.items():
            description.append(f"- Depth {depth}: {loss:.2f}")

        return "\n".join(description)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive model metrics.

        Returns:
            Dictionary containing:
            - Total loss
            - Loss by depth
            - Affected nodes count
            - Configuration parameters
        """
        if self.total_loss is None:
            self.compute_loss()

        return {
            "total_loss": self.total_loss,
            "loss_by_depth": self.propagation_stats,
            "affected_nodes": self.affected_nodes_count,
            "max_depth": self.max_depth,
            "citation_weighted": self.citation_weight,
            "decay_type": (
                "linear" if self.decay == "linear"
                else "none" if self.decay == "none"
                else "custom"
            )
        }


def main():
    """Example usage of PigouPropagatedModel."""
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
            return ['A', 'B']

    # Custom decay function example
    def custom_decay(depth: int) -> float:
        return 1.0 / (depth * depth)

    # Prepare input data
    data = {
        "graph": G,
        "db": MockDB(),
        "non_reproducible_refs": 2
    }

    # Create and run models with different configurations
    models = [
        PigouPropagatedModel(data),  # Default
        PigouPropagatedModel(data, decay="none"),  # No decay
        PigouPropagatedModel(data, decay=custom_decay),  # Custom decay
        PigouPropagatedModel(data, citation_weight=True)  # With citation weights
    ]

    # Display results
    for i, model in enumerate(models, 1):
        print(f"\nModel {i}:")
        print(model.description())
        print("\nMetrics:")
        metrics = model.get_metrics()
        for key, value in metrics.items():
            if key != "loss_by_depth":
                print(f"{key}: {value}")


if __name__ == "__main__":
    main()
