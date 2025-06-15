"""
Gozinto Impact Analysis
======================

This module provides tools for analyzing and visualizing impact propagation
in citation networks using the Gozinto model. It includes functions for
various types of impact analysis and network visualization.

Features:
---------
- Impact score calculation
- Network visualization
- Propagation analysis
- Super-propagator detection
- Citation tree visualization
- Impact depth analysis

Required Dependencies:
--------------------
- networkx
- numpy
- pandas
- seaborn
- matplotlib
"""

import argparse
import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import Dict, List, Set, Tuple, Any, Optional


class NetworkVisualizer:
    """Handles visualization of network structures and impacts."""

    @staticmethod
    def plot_impact_scores(impact_scores: Dict[str, float],
                           top_n: int = 20,
                           title: str = "Most relevant nodes") -> None:
        """
        Plot impact scores as a bar chart.

        Args:
            impact_scores: Dictionary of node IDs and their impact scores
            top_n: Number of top nodes to show
            title: Plot title
        """
        sorted_items = sorted(impact_scores.items(), key=lambda x: -x[1])[:top_n]
        nodes, scores = zip(*sorted_items)

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(nodes)), scores, color='steelblue')
        plt.xticks(range(len(nodes)), nodes, rotation=45, ha='right')
        plt.title(title)
        plt.xlabel("Node")
        plt.ylabel("Impact score")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_impact_comparison(combined_impact: Dict[str, Dict[str, float]],
                               top_n: int = 15) -> None:
        """
        Compare received vs structural impact.

        Args:
            combined_impact: Dictionary with received and structural impacts
            top_n: Number of top nodes to show
        """
        sorted_items = sorted(
            combined_impact.items(),
            key=lambda x: -x[1]["received"]
        )[:top_n]

        nodes = [item[0] for item in sorted_items]
        received = [item[1]["received"] for item in sorted_items]
        structural = [item[1]["structural"] for item in sorted_items]

        x = range(len(nodes))
        width = 0.35

        plt.figure(figsize=(12, 6))
        plt.bar(x, received, width=width, label='Received', color='indianred')
        plt.bar([i + width for i in x], structural, width=width,
                label='Structural', color='steelblue')
        plt.xticks([i + width/2 for i in x], nodes, rotation=45, ha='right')
        plt.ylabel("Impact score")
        plt.title("Received vs Structural Impact")
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_citation_tree(graph: nx.DiGraph,
                           root_node: str,
                           impact_info: Optional[Dict] = None,
                           max_depth: int = 5) -> None:
        """
        Visualize citation tree from root node.

        Args:
            graph: NetworkX directed graph
            root_node: Starting node for tree
            impact_info: Optional impact information
            max_depth: Maximum depth to visualize
        """
        def hierarchy_pos(G: nx.DiGraph, root: str, width: float = 2.0,
                          vert_gap: float = 0.4, vert_loc: float = 0,
                          xcenter: float = 0.5) -> Dict[str, Tuple[float, float]]:
            """Calculate hierarchical layout positions."""
            pos = {root: (xcenter, vert_loc)}
            children = list(G.successors(root))

            if children:
                dx = width / len(children)
                nextx = xcenter - width/2
                for child in children:
                    nextx += dx
                    pos[child] = (nextx, vert_loc - vert_gap)
                    pos.update(hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                             vert_loc=vert_loc-vert_gap, xcenter=nextx))
            return pos

        # Build subgraph
        subgraph = nx.DiGraph()
        queue = [(root_node, 0)]
        visited = {root_node}

        while queue:
            node, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            for neighbor in graph.successors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    subgraph.add_edge(node, neighbor)
                    queue.append((neighbor, depth + 1))

        # Calculate layout
        pos = hierarchy_pos(subgraph, root_node)

        # Visualization
        plt.figure(figsize=(15, 10))
        nx.draw(subgraph, pos, with_labels=True, node_color='lightblue',
                node_size=1000, arrowsize=20, font_size=8)

        if impact_info:
            nx.draw_networkx_edge_labels(
                subgraph, pos,
                edge_labels={(u,v): f"{impact_info.get((u,v), 0):.2f}"
                             for u,v in subgraph.edges()}
            )

        plt.title(f"Citation tree from {root_node}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


class ImpactAnalyzer:
    """Analyzes impact propagation in citation networks."""

    @staticmethod
    def compute_propagation_metrics(model: Any,
                                    affected_nodes: Set[str],
                                    max_depth_range: range = range(1, 11)) -> Dict:
        """
        Analyze impact propagation at different depths.

        Args:
            model: Impact propagation model
            affected_nodes: Set of initially affected nodes
            max_depth_range: Range of depths to analyze

        Returns:
            Dictionary with propagation metrics
        """
        results = {
            "depth": [],
            "total_impacted": [],
            "average_impact": [],
        }

        for depth in max_depth_range:
            impact = model.compute_impact_from_nodes(affected_nodes, max_depth=depth)
            impacted_nodes = set()
            impact_values = []

            for source in impact:
                for target, info in impact[source].items():
                    impacted_nodes.add(target)
                    impact_values.append(info["impact"])

            results["depth"].append(depth)
            results["total_impacted"].append(len(impacted_nodes))
            results["average_impact"].append(
                np.mean(impact_values) if impact_values else 0
            )

        return results

    @staticmethod
    def detect_superpropagators(model: Any,
                                centrality_dict: Dict[str, float],
                                impact_counts: Dict[str, int],
                                top_n: int = 20) -> List[Tuple]:
        """
        Identify nodes with unexpectedly high impact.

        Args:
            model: Impact propagation model
            centrality_dict: Node centrality measures
            impact_counts: Impact propagation counts
            top_n: Number of top propagators to return

        Returns:
            List of (node, centrality, impact) tuples
        """
        candidates = []
        for node in impact_counts:
            centrality = centrality_dict.get(node, 0)
            impacted = impact_counts[node]
            if centrality < 1e-4 and impacted > 0:
                candidates.append((node, centrality, impacted))

        return sorted(candidates, key=lambda x: -x[2])[:top_n]

    @staticmethod
    def analyze_impact_depth(graph: nx.DiGraph,
                             impact_scores: Dict[str, float],
                             affected_nodes: Set[str],
                             output_path: str = "impact_depth.svg") -> None:
        """
        Analyze and visualize impact propagation by depth.

        Args:
            graph: Citation network
            impact_scores: Impact scores for nodes
            affected_nodes: Initially affected nodes
            output_path: Path for output visualization
        """
        def get_node_depths(g: nx.DiGraph,
                            sources: Set[str],
                            max_depth: int = 5) -> Dict[str, int]:
            """Calculate minimum depths from source nodes."""
            depths = {}
            for source in sources:
                if source not in g:
                    continue
                for target, length in nx.single_source_shortest_path_length(
                        g, source, cutoff=max_depth).items():
                    if target not in depths or length < depths[target]:
                        depths[target] = length
            return depths

        depths = get_node_depths(graph, affected_nodes)

        data = []
        for node in graph.nodes:
            impact = abs(impact_scores.get(node, 0.0))
            depth = depths.get(node)
            if depth is not None:
                data.append({
                    "node": node,
                    "abs_impact": impact,
                    "depth": depth
                })

        df = pd.DataFrame(data)

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x="depth", y="abs_impact")
        plt.yscale("log")
        plt.xlabel("Depth from affected nodes")
        plt.ylabel("|Impact score|")
        plt.title("Impact Propagation by Depth")
        plt.tight_layout()
        plt.savefig(output_path, format="svg")
        plt.close()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Analyze citation network impact propagation."
    )
    parser.add_argument(
        'network_path',
        type=str,
        help="Path to network file"
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=5,
        help="Maximum propagation depth"
    )
    parser.add_argument(
        '--output',
        type=str,
        default="impact_analysis.svg",
        help="Output path for visualizations"
    )

    args = parser.parse_args()

    # Implementation of main analysis workflow
    # (Specific implementation depends on your impact model and data structure)


if __name__ == "__main__":
    main()