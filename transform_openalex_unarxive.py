"""
OpenAlex Graph Transformer
========================

This script transforms citation network graphs from OpenAlex format into a standardized format
for citation analysis. It performs necessary data structure modifications and ensures
consistency in node identifiers and attributes.

Features:
- Converts graph to DirectedGraph if needed
- Normalizes OpenAlex URIs to simple IDs
- Ensures required node attributes are present
- Preserves existing graph data while adding missing fields

Required Node Attributes:
- sections: List of paper sections with citation information
- transmission_value: Dictionary for transmission analysis values
- fspc: Dictionary for section-specific metrics

Usage:
------
Command line:
    python transform_openalex_unarxive.py input_graph.pkl output_graph.pkl

As module:
    from transform_openalex_unarxive import transform_openalex_graph
    transform_openalex_graph(input_path, output_path)
"""

import pickle
import networkx as nx


def transform_openalex_graph(input_path, output_path):
    """
    Transform OpenAlex citation network graph to standardized format.

    This function performs the following transformations:
    1. Ensures the graph is a directed graph (DiGraph)
    2. Normalizes node IDs by removing OpenAlex URI prefixes
    3. Adds required node attributes if missing
    4. Preserves existing node data
    5. Transforms edges to use normalized IDs

    Parameters:
        input_path (str): Path to input pickle file containing OpenAlex graph
        output_path (str): Path where transformed graph will be saved

    Example:
        >>> transform_openalex_graph("raw_graph.pkl", "processed_graph.pkl")
        "Transformed graph saved in: processed_graph.pkl"
    """
    # Load the input graph
    with open(input_path, "rb") as f:
        G = pickle.load(f)

    # Convert to DiGraph if necessary
    if not isinstance(G, nx.DiGraph):
        G = nx.DiGraph(G)

    # Initialize new transformed graph
    G_transformed = nx.DiGraph()

    # Process nodes
    for node, data in G.nodes(data=True):
        # Normalize node ID if it's an OpenAlex URI
        if isinstance(node, str) and node.startswith("https://openalex.org/W"):
            new_id = node.split("/")[-1]  # Extract "W2031816812" from URI
        else:
            new_id = node

        # Copy existing node data or initialize empty dict
        new_data = data.copy() if data else {}

        # Add required fields if missing
        if 'sections' not in new_data:
            new_data['sections'] = [
                {'section_name': 'Introduction', 'external_uris': [], 'cited_references': []},
                {'section_name': 'Methodology', 'external_uris': [], 'cited_references': []},
                {'section_name': 'Results', 'external_uris': [], 'cited_references': []},
                {'section_name': 'Discussions/Conclusions', 'external_uris': [], 'cited_references': []}
            ]

        if 'transmission_value' not in new_data:
            new_data['transmission_value'] = {}

        if 'fspc' not in new_data:
            new_data['fspc'] = {}

        # Add node to transformed graph
        G_transformed.add_node(new_id, **new_data)

    # Process edges
    for source, target in G.edges:
        # Normalize source node ID
        if isinstance(source, str) and source.startswith("https://openalex.org/W"):
            source = source.split("/")[-1]

        # Normalize target node ID
        if isinstance(target, str) and target.startswith("https://openalex.org/W"):
            target = target.split("/")[-1]

        # Add edge to transformed graph
        G_transformed.add_edge(source, target)

    # Save transformed graph
    with open(output_path, "wb") as f:
        pickle.dump(G_transformed, f)

    print(f"Transformed graph saved in: {output_path}")


if __name__ == "__main__":
    import argparse

    # Set up command line argument parser
    parser = argparse.ArgumentParser(
        description="Transform an OpenAlex graph to the format required by analysis models"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to .pkl file containing OpenAlex graph"
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output path for transformed .pkl file"
    )

    # Parse arguments and execute transformation
    args = parser.parse_args()
    transform_openalex_graph(args.input, args.output)