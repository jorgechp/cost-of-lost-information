"""
OpenAlex Citation Network Viewer
==============================

This module provides functionality to visualize citation networks from OpenAlex data.
It creates hierarchical visualizations of citation trees with customizable depth
and root nodes.

Features:
---------
- Load citation networks from pickle files
- Extract subgraphs with specified depth
- Create hierarchical visualizations
- Save visualizations in SVG format
- Customizable visualization parameters

Required Dependencies:
--------------------
- networkx
- matplotlib
- pickle
"""

import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import pickle
from typing import Dict, Tuple


def load_graph(pickle_path: str) -> nx.DiGraph:
    """
    Load a NetworkX graph from a pickle file.

    Args:
        pickle_path (str): Path to the pickle file containing the graph

    Returns:
        nx.DiGraph: Loaded graph

    Raises:
        FileNotFoundError: If the pickle file doesn't exist
        pickle.UnpicklingError: If the file contains invalid data
    """
    try:
        with open(pickle_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Graph file not found: {pickle_path}")
    except pickle.UnpicklingError:
        raise ValueError(f"Invalid pickle file: {pickle_path}")


def get_subgraph_by_depth(graph: nx.DiGraph, root: str,
                          max_depth: int) -> Tuple[nx.DiGraph, Dict[str, int]]:
    """
    Extract a subgraph from the main graph up to a specified depth.

    Args:
        graph (nx.DiGraph): Source graph
        root (str): Root node ID
        max_depth (int): Maximum depth for traversal

    Returns:
        Tuple containing:
        - nx.DiGraph: Extracted subgraph
        - Dict[str, int]: Mapping of nodes to their depths

    Raises:
        ValueError: If root node is not in graph
    """
    if root not in graph:
        raise ValueError(f"Root node '{root}' not found in graph")

    visited = set([root])
    tree_nodes = set([root])
    depth_map = {root: 0}
    queue = deque([(root, 0)])

    while queue:
        current, depth = queue.popleft()
        if depth < max_depth:
            for neighbor in graph.successors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    tree_nodes.add(neighbor)
                    depth_map[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1))

    return graph.subgraph(tree_nodes).copy(), depth_map


def hierarchy_pos(depth_map: Dict[str, int]) -> Dict[str, Tuple[float, float]]:
    """
    Calculate hierarchical layout positions for nodes.

    Args:
        depth_map (Dict[str, int]): Mapping of nodes to their depths

    Returns:
        Dict[str, Tuple[float, float]]: Node positions in 2D space
    """
    pos = {}
    layers = {}

    # Group nodes by depth
    for node, depth in depth_map.items():
        layers.setdefault(depth, []).append(node)

    # Calculate positions
    for depth in layers:
        width = len(layers[depth])
        for i, node in enumerate(layers[depth]):
            pos[node] = (i - width / 2, -depth)

    return pos


def draw_and_save_svg(subgraph: nx.DiGraph, depth_map: Dict[str, int],
                      output_file: str, title: str) -> None:
    """
    Create and save visualization of the citation network.

    Args:
        subgraph (nx.DiGraph): Graph to visualize
        depth_map (Dict[str, int]): Mapping of nodes to their depths
        output_file (str): Path for output SVG file
        title (str): Title for the visualization

    Raises:
        ValueError: If visualization parameters are invalid
    """
    if not subgraph.nodes:
        raise ValueError("Empty graph - nothing to visualize")

    try:
        plt.figure(figsize=(12, 8))
        pos = hierarchy_pos(depth_map)

        # Draw nodes
        nx.draw_networkx_nodes(
            subgraph,
            pos,
            node_size=20,
            node_color='lightblue'
        )

        # Draw edges
        nx.draw_networkx_edges(
            subgraph,
            pos,
            arrows=True,
            width=0.3,
            alpha=0.5,
            edge_color='gray'
        )

        plt.title(title)
        plt.axis("off")
        plt.savefig(output_file, format="svg", bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to: {output_file}")

    except Exception as e:
        raise ValueError(f"Error creating visualization: {str(e)}")


def visualize_citation_network(pickle_path: str, root_node: str,
                               max_depth: int, output_path: str) -> None:
    """
    Create complete citation network visualization.

    Args:
        pickle_path (str): Path to graph pickle file
        root_node (str): Starting node for visualization
        max_depth (int): Maximum depth to visualize
        output_path (str): Path for output SVG file

    Raises:
        ValueError: If input parameters are invalid
    """
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")

    # Load and process graph
    print(f"Loading graph from {pickle_path}...")
    graph = load_graph(pickle_path)

    print(f"Extracting subgraph from root {root_node} with depth {max_depth}...")
    subgraph, depth_map = get_subgraph_by_depth(graph, root_node, max_depth)

    print("Creating visualization...")
    title = f"Citation Network (Depth ≤ {max_depth}) from {root_node}"
    draw_and_save_svg(subgraph, depth_map, output_path, title)


def main():
    """
    Main execution function for citation network visualization.
    """
    # Configuration parameters
    PICKLE_PATH = "output/openalex_citation_network.pkl"
    ROOT_NODE = "W2031816812"  # Example root node
    MAX_DEPTH = 2
    OUTPUT_SVG = f"citation_tree_{ROOT_NODE}_depth{MAX_DEPTH}.svg"

    try:
        visualize_citation_network(
            PICKLE_PATH,
            ROOT_NODE,
            MAX_DEPTH,
            OUTPUT_SVG
        )
        print("Visualization complete!")

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    main()