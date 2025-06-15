"""
Citation Network Analysis Tool
============================

This tool analyzes citation networks to understand influence propagation patterns and identify key articles
in academic literature. It processes a NetworkX graph containing citation relationships and transmission
values to generate three types of analyses:

1. Propagation Analysis: Shows how influence spreads through citation levels
2. Threshold Sensitivity: Examines how different threshold values affect top influential papers
3. Super-spreader Detection: Identifies papers with unusual influence relative to their citations

Input Requirements:
- NetworkX graph stored in pickle format
- Nodes must have 'transmission_value' attribute containing a dictionary of threshold values
- Graph edges should represent citation relationships

Output:
- Multiple visualizations showing different aspects of influence propagation
- Console output with detailed statistics
- Interactive plots for data exploration

Dependencies:
- NetworkX: Graph processing
- Matplotlib: Visualization
- NumPy: Numerical operations
"""

import pickle
import matplotlib
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Set matplotlib backend for GUI integration
matplotlib.use('TkAgg')

# Load the citation network from pickle file
with open("./output/unarXive_230324_open_subset__text_arxiv_only.pickle", "rb") as f:
    G = pickle.load(f)

# Extract threshold values from the first node's transmission values
# These thresholds represent different levels of influence strength
thresholds = sorted(next(iter(nx.get_node_attributes(G, 'transmission_value').values())).keys())


# -----------------------------------------------
# 1. Propagation Level Analysis
# -----------------------------------------------
def affected_by_level(graph, threshold):
    """
    Calculate the number of affected articles at each propagation level.

    Args:
        graph (nx.Graph): Citation network graph where:
            - Nodes represent academic articles
            - Edges represent citations between articles
        threshold (float): Minimum transmission value to consider an article affected

    Returns:
        dict: Maps propagation levels to count of affected articles
            - Keys: Level numbers (int, starting from 0)
            - Values: Number of articles affected at that level

    Example:
        >>> counts = affected_by_level(citation_graph, 0.5)
        >>> print(counts)
        {0: 10, 1: 25, 2: 15}  # 10 articles at level 0, 25 at level 1, etc.
    """
    # Find articles affected above the threshold
    affected_nodes = {n for n, d in graph.nodes(data=True)
                      if d.get('transmission_value', {}).get(threshold, 0) > 0}

    # Initialize tracking variables
    level_counts = {}  # Stores count of affected articles per level
    visited = set()  # Tracks processed articles
    current_level = affected_nodes  # Articles at current propagation level

    # Process each level until no more affected articles are found
    level = 0
    while current_level:
        # Record count of articles at current level
        level_counts[level] = len(current_level)

        # Mark current articles as visited
        visited.update(current_level)

        # Find affected articles in the next level
        next_level = set()
        for node in current_level:
            # Check all articles that cite the current article
            for desc in nx.descendants(graph, node):
                # Include only unvisited articles above threshold
                if desc not in visited and graph.nodes[desc].get('transmission_value', {}).get(threshold, 0) > 0:
                    next_level.add(desc)

        current_level = next_level
        level += 1

    return level_counts


# Generate and display propagation statistics for each threshold
for t in thresholds:
    counts = affected_by_level(G, t)
    print(f"Threshold {t}: {counts}")

# Visualize propagation patterns
plt.figure()
for t in thresholds:
    counts = affected_by_level(G, t)
    levels = list(counts.keys())
    values = list(counts.values())
    plt.plot(levels, values, label=f"Threshold {t:.1f}")
plt.xlabel("Propagation Level (Steps from Source)")
plt.ylabel("Number of Affected Articles")
plt.title("Influence Propagation Through Citation Network")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------------------------
# 2. Threshold Sensitivity Analysis
# -----------------------------------------------
# Number of top articles to analyze
top_n = 10

# Calculate average transmission values for all articles
transmission_by_node = {
    node: data.get('transmission_value', {})
    for node, data in G.nodes(data=True)
}
avg_values = {
    node: np.mean(list(v.values()))
    for node, v in transmission_by_node.items()
}

# Identify the most influential articles
top_nodes = sorted(avg_values, key=avg_values.get, reverse=True)[:top_n]

# Visualize threshold sensitivity for top articles
plt.figure()
for node in top_nodes:
    values = [transmission_by_node[node].get(t, 0) for t in thresholds]
    plt.plot(thresholds, values, label=f"Article {node}")
plt.xlabel("Influence Threshold")
plt.ylabel("Transmission Value")
plt.title("Influence Stability Analysis of Top Articles")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------------------------
# 3. Super-spreader Analysis
# -----------------------------------------------
# Calculate average transmission value for each article
avg_transmission = {
    node: np.mean(list(data.get('transmission_value', {}).values()))
    for node, data in G.nodes(data=True)
}

# Get citation counts (out-degree) for each article
out_degrees = dict(G.out_degree())

# Prepare data for scatter plot
x = [out_degrees[n] for n in G.nodes()]
y = [avg_transmission[n] for n in G.nodes()]

# Create visualization of super-spreader characteristics
plt.figure()
plt.scatter(x, y, alpha=0.6)
plt.xlabel("Citations Made (Out-degree)")
plt.ylabel("Average Influence Value")
plt.title("Super-spreader Detection: Citation Count vs Influence")
plt.grid(True)
plt.tight_layout()
plt.show()