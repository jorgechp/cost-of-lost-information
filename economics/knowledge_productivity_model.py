"""
Knowledge Productivity Model
==========================

This module implements a model for calculating knowledge productivity
based on research efficiency and workforce parameters.

Features:
---------
- Research efficiency modeling
- Workforce impact analysis
- Non-reproducible research impact
- Productivity calculation
- Clear metrics reporting

Key Concepts:
------------
- A: Production efficiency coefficient
- L: Number of researchers
- N_NR: Number of non-reproducible papers
- N_T: Total number of papers
"""

from typing import Dict, Any

from economics.economic_loss_model import EconomicLossModel


class KnowledgeProductivityModel(EconomicLossModel):
    """
    Model for calculating knowledge productivity considering
    research efficiency and workforce factors.
    """

    def __init__(self,
                 data: Dict[str, Any],
                 production_efficiency: float,
                 researcher_count: int):
        """
        Initialize knowledge productivity model.

        Args:
            data: Dictionary containing:
                - non_reproducible_papers: Number of non-reproducible papers
                - total_papers: Total number of papers
            production_efficiency: Research production efficiency coefficient
            researcher_count: Number of active researchers
        """
        super().__init__(data)

        self._validate_input(data, production_efficiency, researcher_count)

        self.production_efficiency = production_efficiency
        self.researcher_count = researcher_count
        self.non_reproducible_count = data['non_reproducible_papers']
        self.total_papers = data['total_papers']

    def _validate_input(self,
                        data: Dict[str, Any],
                        efficiency: float,
                        researchers: int) -> None:
        """
        Validate input parameters.

        Args:
            data: Input data dictionary
            efficiency: Production efficiency value
            researchers: Number of researchers

        Raises:
            ValueError: If parameters are invalid
        """
        required_keys = {'non_reproducible_papers', 'total_papers'}
        missing_keys = required_keys - set(data.keys())

        if missing_keys:
            raise ValueError(
                f"Missing required data keys: {', '.join(missing_keys)}"
            )

        if not isinstance(efficiency, (int, float)) or efficiency < 0:
            raise ValueError(
                "Production efficiency must be a non-negative number"
            )

        if not isinstance(researchers, int) or researchers < 0:
            raise ValueError(
                "Researcher count must be a non-negative integer"
            )

        if not isinstance(data['non_reproducible_papers'], (int, float)):
            raise ValueError(
                "Non-reproducible papers count must be numeric"
            )

        if not isinstance(data['total_papers'], (int, float)):
            raise ValueError(
                "Total papers count must be numeric"
            )

        if data['total_papers'] <= 0:
            raise ValueError(
                "Total papers count must be positive"
            )

        if data['non_reproducible_papers'] > data['total_papers']:
            raise ValueError(
                "Non-reproducible papers cannot exceed total papers"
            )

    def compute_loss(self) -> float:
        """
        Compute knowledge productivity loss.

        Returns:
            Calculated productivity change (ΔK)

        Formula:
            ΔK = A * L * (1 - N_NR / N_T)
            where:
            - A: Production efficiency
            - L: Number of researchers
            - N_NR: Number of non-reproducible papers
            - N_T: Total number of papers
        """
        reproducibility_factor = 1 - (
                self.non_reproducible_count / self.total_papers
        )

        return (
                self.production_efficiency *
                self.researcher_count *
                reproducibility_factor
        )

    def description(self) -> str:
        """
        Get model description with calculations.

        Returns:
            Formatted description string
        """
        return (
            f"Knowledge Productivity Model\n"
            f"---------------------------\n"
            f"Formula: ΔK = A * L * (1 - N_NR / N_T)\n"
            f"Where:\n"
            f"- A (Production Efficiency): {self.production_efficiency}\n"
            f"- L (Researchers): {self.researcher_count}\n"
            f"- N_NR (Non-reproducible): {self.non_reproducible_count}\n"
            f"- N_T (Total Papers): {self.total_papers}\n"
            f"Calculation:\n"
            f"ΔK = {self.production_efficiency} * {self.researcher_count} * "
            f"(1 - {self.non_reproducible_count}/{self.total_papers})"
        )

    def get_metrics(self) -> Dict[str, float]:
        """
        Get model metrics.

        Returns:
            Dictionary containing:
            - Productivity change
            - Reproducibility ratio
            - Per-researcher productivity
        """
        productivity_change = self.compute_loss()
        reproducibility_ratio = 1 - (
                self.non_reproducible_count / self.total_papers
        )
        per_researcher = productivity_change / self.researcher_count

        return {
            "productivity_change": productivity_change,
            "reproducibility_ratio": reproducibility_ratio,
            "per_researcher_productivity": per_researcher
        }


def main():
    """Example usage of KnowledgeProductivityModel."""

    # Sample data
    data = {
        "non_reproducible_papers": 50,
        "total_papers": 200
    }

    # Create model instance
    model = KnowledgeProductivityModel(
        data=data,
        production_efficiency=0.8,
        researcher_count=100
    )

    # Calculate and display results
    productivity_change = model.compute_loss()
    print("\nModel Description:")
    print(model.description())

    print("\nDetailed Metrics:")
    metrics = model.get_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()