"""
Network Analysis Tool
===================

This module provides tools for analyzing and summarizing network/graph structures.
It includes functionality to load, inspect, and sanitize NetworkX graphs, with
detailed reporting of graph properties and node attributes.

Features:
---------
- Load graphs from pickle files
- Analyze graph structure and connectivity
- Inspect node attributes
- Generate detailed network statistics
- Sanitize graph attributes for export

Required Dependencies:
--------------------
- networkx
- pickle
"""

import pickle
import networkx as nx
from collections import Counter
import os
from typing import Any, Dict, List, Optional


class NetworkAnalyzer:
    """Network analysis and summary generation tool."""

    def __init__(self, filepath: Optional[str] = None):
        """
        Initialize the analyzer.

        Args:
            filepath: Optional path to pickle file containing graph
        """
        self.graph = None
        if filepath:
            self.load_graph(filepath)

    def load_graph(self, filepath: str) -> nx.Graph:
        """
        Load a NetworkX graph from a pickle file.

        Args:
            filepath: Path to pickle file

        Returns:
            Loaded NetworkX graph

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file contains invalid data
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Graph file not found: {filepath}")

        try:
            with open(filepath, 'rb') as f:
                self.graph = pickle.load(f)
            return self.graph
        except Exception as e:
            raise ValueError(f"Error loading graph: {str(e)}")

    def get_graph_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive graph statistics.

        Returns:
            Dictionary containing graph metrics
        """
        if not self.graph:
            raise ValueError("No graph loaded")

        degrees = dict(self.graph.degree())
        avg_degree = sum(degrees.values()) / len(degrees)
        max_degree_node = max(degrees, key=degrees.get)
        min_degree_node = min(degrees, key=degrees.get)

        summary = {
            "type": type(self.graph).__name__,
            "directed": self.graph.is_directed(),
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "avg_degree": avg_degree,
            "max_degree": {
                "node": max_degree_node,
                "value": degrees[max_degree_node]
            },
            "min_degree": {
                "node": min_degree_node,
                "value": degrees[min_degree_node]
            }
        }

        # Connectivity analysis
        if self.graph.is_directed():
            summary.update({
                "strongly_connected_components":
                    nx.number_strongly_connected_components(self.graph),
                "weakly_connected_components":
                    nx.number_weakly_connected_components(self.graph)
            })
        else:
            summary["connected_components"] = nx.number_connected_components(self.graph)

        return summary

    def print_graph_summary(self) -> None:
        """Print formatted graph summary to console."""
        summary = self.get_graph_summary()

        print("\n=== Graph Summary ===")
        print(f"Type: {summary['type']}")
        print(f"Directed: {summary['directed']}")
        print(f"Nodes: {summary['nodes']}")
        print(f"Edges: {summary['edges']}")
        print(f"Density: {summary['density']:.6f}")

        print("\n--- Degree Statistics ---")
        print(f"Average degree: {summary['avg_degree']:.2f}")
        print(f"Maximum degree: {summary['max_degree']['value']} "
              f"(Node: {summary['max_degree']['node']})")
        print(f"Minimum degree: {summary['min_degree']['value']} "
              f"(Node: {summary['min_degree']['node']})")

        print("\n--- Connectivity ---")
        if summary.get('strongly_connected_components'):
            print(f"Strongly connected components: "
                  f"{summary['strongly_connected_components']}")
            print(f"Weakly connected components: "
                  f"{summary['weakly_connected_components']}")
        else:
            print(f"Connected components: {summary['connected_components']}")

    def inspect_node_attributes(self, num_samples: int = 5) -> Dict[str, Counter]:
        """
        Analyze node attribute structure.

        Args:
            num_samples: Number of example nodes to inspect

        Returns:
            Dictionary with attribute statistics
        """
        if not self.graph:
            raise ValueError("No graph loaded")

        # Count attribute occurrences
        attr_counter = Counter()
        for _, attrs in self.graph.nodes(data=True):
            attr_counter.update(attrs.keys())

        # Get sample nodes
        sample_nodes = {}
        for i, (node, attrs) in enumerate(self.graph.nodes(data=True)):
            if i >= num_samples:
                break
            sample_nodes[node] = attrs

        return {
            "attribute_frequencies": attr_counter,
            "sample_nodes": sample_nodes
        }

    def print_node_inspection(self, num_samples: int = 5) -> None:
        """
        Print formatted node attribute analysis.

        Args:
            num_samples: Number of example nodes to show
        """
        inspection = self.inspect_node_attributes(num_samples)

        print("\n=== Node Attribute Analysis ===")
        print("Attribute frequencies:")
        for attr, count in inspection["attribute_frequencies"].most_common():
            print(f"  {attr}: {count}")

        print(f"\nSample of {num_samples} nodes:")
        for node, attrs in inspection["sample_nodes"].items():
            print(f"\nNode: {node}")
            for key, value in attrs.items():
                print(f"  {key}: {value}")

    def sanitize_attributes(self) -> None:
        """
        Sanitize graph attributes for export compatibility.

        Converts complex attributes to strings and removes invalid values.
        """
        if not self.graph:
            raise ValueError("No graph loaded")

        def sanitize(value: Any) -> Optional[str]:
            """Convert value to compatible format or None if invalid."""
            if isinstance(value, (str, int, float, bool)):
                return value
            try:
                return str(value)
            except Exception:
                return None

        # Sanitize node attributes
        for node, attrs in self.graph.nodes(data=True):
            for key, value in list(attrs.items()):
                self.graph.nodes[node][key] = sanitize(value)

        # Sanitize edge attributes
        for u, v, attrs in self.graph.edges(data=True):
            for key, value in list(attrs.items()):
                self.graph.edges[u, v][key] = sanitize(value)


def main():
    """Main execution function."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python network_summary.py <path_to_pickle_file>")
        return 1

    try:
        analyzer = NetworkAnalyzer()
        analyzer.load_graph(sys.argv[1])
        analyzer.print_graph_summary()
        analyzer.print_node_inspection()
        return 0

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    main()