"""
Economic Loss Model Base Class
============================

This module provides an abstract base class for economic loss models
in academic research networks.

Features:
---------
- Abstract interface for loss calculation
- Flexible data structure support
- Model description capabilities
- Standardized loss computation interface

Required Dependencies:
--------------------
- abc (Abstract Base Classes)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class EconomicLossModel(ABC):
    """
    Abstract base class for economic loss models.

    This class provides a framework for implementing specific
    economic loss calculation models in research networks.
    """

    def __init__(self, data: Dict[str, Any]):
        """
        Initialize economic loss model.

        Args:
            data: Dictionary containing base information for the model,
                 such as affected nodes, total papers, network structure, etc.
        """
        self._validate_data(data)
        self.data = data
        self.computed_loss: Optional[float] = None

    def _validate_data(self, data: Dict[str, Any]) -> None:
        """
        Validate input data structure.

        Args:
            data: Input data dictionary

        Raises:
            ValueError: If data is invalid or missing required elements
        """
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")

        if not data:
            raise ValueError("Data dictionary cannot be empty")

    @abstractmethod
    def compute_loss(self) -> float:
        """
        Compute estimated knowledge loss.

        Returns:
            Calculated loss value

        This method must be implemented by subclasses to define
        specific loss calculation logic.
        """
        pass

    @abstractmethod
    def description(self) -> str:
        """
        Get model description.

        Returns:
            String containing model description and parameters

        This method must be implemented by subclasses to provide
        a detailed description of the specific model implementation.
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata.

        Returns:
            Dictionary containing:
            - Model description
            - Data summary
            - Computed loss (if available)
        """
        return {
            "description": self.description(),
            "data_summary": self._get_data_summary(),
            "computed_loss": self.computed_loss
        }

    def _get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary of input data.

        Returns:
            Dictionary containing data statistics
        """
        return {
            "data_keys": list(self.data.keys()),
            "data_types": {
                key: type(value).__name__
                for key, value in self.data.items()
            }
        }

    def validate(self) -> bool:
        """
        Validate model state.

        Returns:
            True if model is valid, False otherwise
        """
        try:
            self._validate_data(self.data)
            return True
        except ValueError as e:
            print(f"Validation error: {str(e)}")
            return False


def main():
    """
    Example implementation of EconomicLossModel.

    This shows how to create a concrete implementation
    of the abstract base class.
    """

    class SimpleEconomicLossModel(EconomicLossModel):
        """Simple implementation for demonstration."""

        def compute_loss(self) -> float:
            """
            Simple loss calculation based on affected ratio.
            """
            total_nodes = self.data.get('total_nodes', 0)
            affected_nodes = self.data.get('affected_nodes', 0)

            if total_nodes == 0:
                return 0.0

            self.computed_loss = affected_nodes / total_nodes
            return self.computed_loss

        def description(self) -> str:
            """
            Model description.
            """
            return (
                "Simple economic loss model based on "
                "ratio of affected nodes to total nodes"
            )

    # Example usage
    sample_data = {
        'total_nodes': 100,
        'affected_nodes': 25,
        'impact_factor': 0.5
    }

    model = SimpleEconomicLossModel(sample_data)

    if model.validate():
        loss = model.compute_loss()
        print(f"\nComputed Economic Loss: {loss:.4f}")

        print("\nModel Metadata:")
        for key, value in model.get_metadata().items():
            print(f"\n{key}:")
            if isinstance(value, dict):
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"  {value}")


if __name__ == "__main__":
    main()