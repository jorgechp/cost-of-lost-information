"""
Opportunity Cost Model
====================

This module implements a model for calculating opportunity costs
in research based on non-reproducible references and potential knowledge impact.

Features:
---------
- Opportunity cost calculation
- Knowledge impact assessment
- Probability-based modeling
- Impact evaluation
- Clear metrics reporting

Key Concepts:
------------
- N_R: Number of non-reproducible references
- P_O: Probability of generating new knowledge
- I_K: Average impact of new knowledge
"""

from typing import Dict, Any

from economics.economic_loss_model import EconomicLossModel


class OpportunityCostModel(EconomicLossModel):
    """
    Model for calculating opportunity costs in research
    considering potential knowledge generation and impact.
    """

    def __init__(self,
                 data: Dict[str, Any],
                 knowledge_probability: float,
                 knowledge_impact: float):
        """
        Initialize opportunity cost model.

        Args:
            data: Dictionary containing:
                - non_reproducible_refs: Number of non-reproducible references
            knowledge_probability: Probability of generating new knowledge (0-1)
            knowledge_impact: Average impact factor of new knowledge
        """
        super().__init__(data)

        self._validate_input(data, knowledge_probability, knowledge_impact)

        self.knowledge_probability = knowledge_probability
        self.knowledge_impact = knowledge_impact

    def _validate_input(self,
                        data: Dict[str, Any],
                        probability: float,
                        impact: float) -> None:
        """
        Validate input parameters.

        Args:
            data: Input data dictionary
            probability: Knowledge generation probability
            impact: Knowledge impact factor

        Raises:
            ValueError: If parameters are invalid
        """
        if 'non_reproducible_refs' not in data:
            raise ValueError(
                "Data must contain 'non_reproducible_refs' key"
            )

        if not isinstance(data['non_reproducible_refs'], (int, float)):
            raise ValueError(
                "Number of non-reproducible references must be numeric"
            )

        if not 0 <= probability <= 1:
            raise ValueError(
                "Knowledge probability must be between 0 and 1"
            )

        if not isinstance(impact, (int, float)) or impact < 0:
            raise ValueError(
                "Knowledge impact must be a non-negative number"
            )

    def compute_loss(self) -> float:
        """
        Compute opportunity cost loss.

        Returns:
            Calculated opportunity cost

        Formula:
            Loss = N_R * P_O * I_K
            where:
            - N_R: Number of non-reproducible references
            - P_O: Probability of generating new knowledge
            - I_K: Average impact of new knowledge
        """
        return (
                self.data['non_reproducible_refs'] *
                self.knowledge_probability *
                self.knowledge_impact
        )

    def description(self) -> str:
        """
        Get model description with calculations.

        Returns:
            Formatted description string
        """
        return (
            f"Opportunity Cost Model\n"
            f"--------------------\n"
            f"Formula: Loss = N_R * P_O * I_K\n"
            f"Where:\n"
            f"- N_R (Non-reproducible Refs): "
            f"{self.data['non_reproducible_refs']}\n"
            f"- P_O (Knowledge Probability): "
            f"{self.knowledge_probability}\n"
            f"- I_K (Knowledge Impact): {self.knowledge_impact}\n"
            f"\nCalculation:\n"
            f"Loss = {self.data['non_reproducible_refs']} * "
            f"{self.knowledge_probability} * {self.knowledge_impact}"
        )

    def get_metrics(self) -> Dict[str, float]:
        """
        Get model metrics.

        Returns:
            Dictionary containing:
            - Total opportunity cost
            - Per-reference cost
            - Expected knowledge generation
        """
        total_cost = self.compute_loss()
        per_reference = total_cost / self.data['non_reproducible_refs']
        expected_knowledge = (
                self.data['non_reproducible_refs'] *
                self.knowledge_probability
        )

        return {
            "total_opportunity_cost": total_cost,
            "per_reference_cost": per_reference,
            "expected_knowledge_loss": expected_knowledge
        }

    def get_sensitivity_analysis(self,
                                 probability_delta: float = 0.1,
                                 impact_delta: float = 0.1) -> Dict[str, float]:
        """
        Perform sensitivity analysis.

        Args:
            probability_delta: Change in probability for analysis
            impact_delta: Change in impact for analysis

        Returns:
            Dictionary containing impact of parameter changes
        """
        base_loss = self.compute_loss()

        # Probability sensitivity
        prob_up = OpportunityCostModel(
            self.data,
            min(1.0, self.knowledge_probability + probability_delta),
            self.knowledge_impact
        ).compute_loss()

        prob_down = OpportunityCostModel(
            self.data,
            max(0.0, self.knowledge_probability - probability_delta),
            self.knowledge_impact
        ).compute_loss()

        # Impact sensitivity
        impact_up = OpportunityCostModel(
            self.data,
            self.knowledge_probability,
            self.knowledge_impact + impact_delta
        ).compute_loss()

        impact_down = OpportunityCostModel(
            self.data,
            self.knowledge_probability,
            max(0.0, self.knowledge_impact - impact_delta)
        ).compute_loss()

        return {
            "probability_increase": prob_up - base_loss,
            "probability_decrease": base_loss - prob_down,
            "impact_increase": impact_up - base_loss,
            "impact_decrease": base_loss - impact_down
        }


def main():
    """Example usage of OpportunityCostModel."""

    # Sample data
    data = {
        "non_reproducible_refs": 100
    }

    # Create model instance
    model = OpportunityCostModel(
        data=data,
        knowledge_probability=0.3,
        knowledge_impact=2.5
    )

    # Calculate and display results
    print("\nModel Description:")
    print(model.description())

    print("\nDetailed Metrics:")
    metrics = model.get_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nSensitivity Analysis:")
    sensitivity = model.get_sensitivity_analysis()
    for key, value in sensitivity.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()