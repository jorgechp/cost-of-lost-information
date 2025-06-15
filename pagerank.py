"""
PageRank Analysis Tool for Citation Networks
=========================================

This module implements PageRank-based analysis for citation networks, including
standard PageRank, weighted PageRank, and personalized PageRank calculations.
It provides visualization and analysis tools for understanding citation influence
and knowledge propagation.

Features:
---------
- Multiple PageRank implementations
- Impact analysis
- Visualization tools
- Superpropagator identification
- Depth-based impact analysis

Required Dependencies:
--------------------
- networkx
- numpy
- pandas
- seaborn
- matplotlib
- scipy
"""

import argparse

import matplotlib
import networkx as nx
from matplotlib import pyplot as plt
from typing import Dict, Set, Tuple

from transmission.transmission_model import TransmissionModel
from pagerank.pagerank_loss_calculator import PageRankLossCalculator
from relevance.database import Database

matplotlib.use('tkAGG')
def fill_nodes_information(graph: nx.Graph, values: Tuple,
                           penalized_weight: float = 0.1,
                           alpha: float = 0.85) -> None:
    """
    Store PageRank values in graph nodes.

    Args:
        graph: NetworkX graph
        values: Tuple of (standard, weighted, personalized) PageRank values
        penalized_weight: Weight for penalized nodes
        alpha: Damping factor for PageRank
    """
    pagerank_standard, pagerank_weighted, pagerank_personalized = values
    for node in graph.nodes:
        if 'pagerank' not in graph.nodes[node]:
            graph.nodes[node]['pagerank'] = {
                'penalty': {
                    'weighted': {},
                    'personalization': {}
                }
            }
        parameter_id = f"{penalized_weight}_{alpha}"
        graph.nodes[node]['pagerank']['original'] = pagerank_standard[node]
        graph.nodes[node]['pagerank']['penalty']['weighted'][parameter_id] = pagerank_weighted[node]
        graph.nodes[node]['pagerank']['penalty']['personalization'][parameter_id] = pagerank_personalized[node]


def print_top_pagerank(pagerank_values: Dict, N: int = 10,
                       title: str = "Top PageRank Nodes") -> None:
    """
    Print top N nodes by PageRank value.

    Args:
        pagerank_values: Dictionary of node:pagerank pairs
        N: Number of top nodes to display
        title: Title for the output
    """
    print(f"\n{title}")
    sorted_pr = sorted(pagerank_values.items(), key=lambda x: x[1], reverse=True)
    for i, (node, value) in enumerate(sorted_pr[:N], 1):
        print(f"{i}. Node {node}: PageRank = {value:.8f}")


def compute_pagerank_impact(graph: nx.Graph, pr_standard: Dict,
                            pr_penalized: Dict, affected_nodes: Set) -> Dict:
    """
    Calculate PageRank impact as difference between standard and penalized values.

    Args:
        graph: NetworkX graph
        pr_standard: Standard PageRank values
        pr_penalized: Penalized PageRank values
        affected_nodes: Set of affected nodes

    Returns:
        Dictionary of node:impact pairs
    """
    return {
        node: pr_penalized.get(node, 0.0) - pr_standard.get(node, 0.0)
        for node in graph.nodes
    }


def get_node_depths(graph: nx.Graph, sources: Set, max_depth: int = 5) -> Dict:
    """
    Calculate minimum distances from source nodes.

    Args:
        graph: NetworkX graph
        sources: Set of source nodes
        max_depth: Maximum depth to consider

    Returns:
        Dictionary of node:depth pairs
    """
    depths = {}
    for source in sources:
        if source not in graph:
            continue
        for target, length in nx.single_source_shortest_path_length(
                graph, source, cutoff=max_depth).items():
            if target not in depths or length < depths[target]:
                depths[target] = length
    return depths


def identify_superpropagators(graph: nx.Graph, delta_pr: Dict,
                              affected_nodes: Set, top_n: int = 10) -> Dict:
    """
    Identify nodes that most effectively propagate PageRank changes.

    Args:
        graph: NetworkX graph
        delta_pr: Dictionary of PageRank changes
        affected_nodes: Set of affected nodes
        top_n: Number of top propagators to identify

    Returns:
        Dictionary of top propagator nodes and their impact scores
    """
    transmitted = {}
    for node in affected_nodes:
        if node not in graph:
            continue
        impacted = sum(
            abs(delta_pr.get(target, 0.0))
            for target in graph.nodes
            if target != node and nx.has_path(graph, node, target)
        )
        transmitted[node] = impacted

    sorted_transmitters = sorted(transmitted.items(), key=lambda x: -x[1])
    print(f"\nTop {top_n} Superpropagators:")
    for i, (node, score) in enumerate(sorted_transmitters[:top_n], 1):
        print(f"{i}. Node {node} → Propagated impact = {score:.8f}")

    return dict(sorted_transmitters[:top_n])


def plot_superpropagators(superpropagators: Dict,
                          output_path: str = "superpropagators_pagerank.svg") -> None:
    """
    Create bar plot of top superpropagators.

    Args:
        superpropagators: Dictionary of node:impact pairs
        output_path: Path to save the plot
    """
    nodes = list(superpropagators.keys())
    scores = list(superpropagators.values())

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(nodes)), scores, color="darkorange")
    plt.yticks(range(len(nodes)), nodes)
    plt.xlabel("Propagated |Δ PageRank|")
    plt.title("Top Superpropagators (PageRank)")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()
    print(f"Superpropagator chart saved to {output_path}")


def analyze_network(graph: nx.Graph, affected_nodes: Set,
                    pagerank_calculator: PageRankLossCalculator) -> Tuple:
    """
    Perform comprehensive PageRank analysis of the network.

    Args:
        graph: NetworkX graph
        affected_nodes: Set of affected nodes
        pagerank_calculator: PageRank calculator instance

    Returns:
        Tuple of (standard, weighted, personalized) PageRank dictionaries
    """
    print("\nCalculating PageRank values...")
    pr_standard = pagerank_calculator.compute_standard_pagerank()
    pr_weighted = pagerank_calculator.compute_weighted_pagerank(affected_nodes)
    pr_personalized = pagerank_calculator.compute_personalized_pagerank(affected_nodes)

    print("\nComparing PageRank distributions...")
    print_top_pagerank(pr_standard, title="Top nodes (Standard PageRank)")
    print_top_pagerank(pr_weighted, title="Top nodes (Weighted PageRank)")
    print_top_pagerank(pr_personalized, title="Top nodes (Personalized PageRank)")

    return pr_standard, pr_weighted, pr_personalized


def main():
    """
    Main execution function for PageRank analysis.
    """
    parser = argparse.ArgumentParser(
        description="PageRank analysis for citation networks."
    )
    parser.add_argument(
        'pickle_path',
        type=str,
        help="Path to the pickle file containing the graph"
    )
    args = parser.parse_args()

    # Initialize models
    db = Database()
    tr_model = TransmissionModel(db)
    tr_model.load_graph(args.pickle_path)
    graph = tr_model.get_graph()
    affected_nodes = set(tr_model.get_affected_nodes())

    # Calculate PageRank values
    pr_calculator = PageRankLossCalculator(graph)
    pr_standard, pr_weighted, pr_personalized = analyze_network(
        graph, affected_nodes, pr_calculator
    )

    # Store results in graph
    fill_nodes_information(
        graph,
        [pr_standard, pr_weighted, pr_personalized],
        penalized_weight=0.1,
        alpha=0.85
    )

    # Calculate and analyze impact
    delta_pr = compute_pagerank_impact(
        graph, pr_standard, pr_weighted, affected_nodes
    )
    valid_affected = {node for node in affected_nodes if node in graph}

    # Identify and visualize superpropagators
    superpropagators = identify_superpropagators(
        graph, delta_pr, valid_affected, top_n=15
    )
    plot_superpropagators(superpropagators)

    # Save results
    tr_model.save_graph(args.pickle_path)
    print("\nAnalysis complete. Results saved to graph file.")


if __name__ == "__main__":
    main()