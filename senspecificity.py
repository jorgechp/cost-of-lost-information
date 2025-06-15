"""
Citation Network Sensitivity and Specificity Analysis Tool
=======================================

This tool analyzes citation networks to evaluate knowledge generation and information preservation
patterns using sensitivity and specificity metrics. It calculates various conditional probabilities
to understand how effectively scientific knowledge is transmitted through citations.

Key Metrics Calculated:
----------------------
1. Knowledge Generation Probabilities:
   - P(X=1): Probability of generating new knowledge
   - P(X=0): Probability of not generating new knowledge

2. Conditional Probabilities:
   - P(X=0|S=1): Probability of no new knowledge given preserved information
   - P(X=1|S=1): Probability of new knowledge given preserved information
   - P(X=1|S=0): Probability of new knowledge given lost information
   - P(X=0|S=0): Probability of no new knowledge given lost information

Usage:
------
Command line:
    python senspecificity.py path/to/graph.pkl
"""

import argparse
import networkx as nx

from transmission.transmission_model import TransmissionModel
from relevance.database import Database
from senspecificity.senspecificy_model import SpecificitySensitivityModel


def fill_nodes_information(graph, history, beta, gamma, delta):
    """
    Populate nodes with SAIS model historical information.

    Parameters:
        graph (nx.Graph): Citation network graph
        history (list): Time series of node states
        beta (float): Transmission rate parameter
        gamma (float): Recovery rate parameter
        delta (float): External influence parameter

    The function adds a 'sais' dictionary to each node containing:
        - Parameter-specific history indexed by 'beta_gamma_delta'
        - Time series of node states for each step
    """
    for node in graph.nodes:
        # Initialize SAIS dictionary for each node
        graph.nodes[node]['sais'] = {}

        # Create unique parameter identifier
        parameter_id = f"{beta}_{gamma}_{delta}"

        # Store state history for each time step
        graph.nodes[node]['sais'][parameter_id] = {
            step: history[step][node] for step in range(len(history))
        }


def main():
    """
    Execute sensitivity analysis on citation network.

    Process:
    1. Loads citation network from pickle file
    2. Initializes database connection
    3. Creates transmission model
    4. Performs sensitivity/specificity analysis
    5. Calculates and displays probability metrics
    """
    # Set up command line argument parser
    parser = argparse.ArgumentParser(
        description="Analyze transmission patterns in citation networks using sensitivity/specificity metrics."
    )
    parser.add_argument(
        'pickle_path',
        type=str,
        help="Path to the pickle file containing the citation network graph"
    )

    # Parse command line arguments
    args = parser.parse_args()

    # Initialize database and transmission model
    db = Database()
    tr_model = TransmissionModel(db)
    tr_model.load_graph(args.pickle_path)

    # Create and run sensitivity/specificity analysis
    spmodel = SpecificitySensitivityModel(tr_model.get_graph(), db)
    spmodel.estimate_probabilities()

    # Get probability results
    P_X1, P_X0, P_X0_S1, P_X1_S1, P_X1_S0, P_X0_S0 = spmodel.get_probabilities()

    # Display results with descriptions
    print("\nCitation Network Analysis Results:")
    print("=================================")
    print(f"Knowledge Generation Probabilities:")
    print(f"- P_X1: {P_X1:.4f} - Probability of generating new knowledge")
    print(f"- P_X0: {P_X0:.4f} - Probability of not generating new knowledge")
    print(f"\nConditional Probabilities:")
    print(f"- P_X0_S1: {P_X0_S1:.4f} - Probability of not generating new knowledge given preserved information")
    print(f"- P_X1_S1: {P_X1_S1:.4f} - Probability of generating new knowledge given preserved information")
    print(f"- P_X1_S0: {P_X1_S0:.4f} - Probability of generating new knowledge given lost information")
    print(f"- P_X0_S0: {P_X0_S0:.4f} - Probability of not generating new knowledge given lost information")


if __name__ == "__main__":
    main()