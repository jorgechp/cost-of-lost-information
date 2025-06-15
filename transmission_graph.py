import pickle
import matplotlib
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('TkAgg')

# Load graph from pickle file
with open("./output/unarXive_230324_open_subset__text_arxiv_only.pickle", "rb") as f:
    G = pickle.load(f)

# Get transmission thresholds used in the analysis
thresholds = sorted(next(iter(nx.get_node_attributes(G, 'transmission_value').values())).keys())


# -----------------------------------------------
# 1. Evolution of affected articles by level
# -----------------------------------------------
def affected_by_level(graph, threshold):
    """
    Calculate the number of affected articles at each propagation level for a given threshold.

    Args:
        graph: NetworkX graph representing citation network
        threshold: Transmission threshold value

    Returns:
        dict: Dictionary mapping propagation levels to number of affected articles
    """
    # Find nodes with transmission value above 0 for given threshold
    affected_nodes = {n for n, d in graph.nodes(data=True)
                      if d.get('transmission_value', {}).get(threshold, 0) > 0}
    level_counts = {}
    visited = set()
    current_level = affected_nodes

    level = 0
    while current_level:
        level_counts[level] = len(current_level)
        visited.update(current_level)
        next_level = set()
        for node in current_level:
            for desc in nx.descendants(graph, node):
                if desc not in visited and graph.nodes[desc].get('transmission_value', {}).get(threshold, 0) > 0:
                    next_level.add(desc)
        current_level = next_level
        level += 1
    return level_counts


# Print and plot affected articles by level for each threshold
for t in thresholds:
    counts = affected_by_level(G, t)
    print(f"Threshold {t}: {counts}")

plt.figure()
for t in thresholds:
    counts = affected_by_level(G, t)
    levels = list(counts.keys())
    values = list(counts.values())
    plt.plot(levels, values, label=f"Threshold {t:.1f}")
plt.xlabel("Propagation level")
plt.ylabel("Affected articles")
plt.title("Affected articles per level")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------------------------
# 2. Threshold sensitivity analysis for top-n nodes
# -----------------------------------------------
top_n = 10
# Calculate average transmission value for each node
transmission_by_node = {
    node: data.get('transmission_value', {}) for node, data in G.nodes(data=True)
}
avg_values = {node: np.mean(list(v.values())) for node, v in transmission_by_node.items()}
# Get top-n nodes by average transmission value
top_nodes = sorted(avg_values, key=avg_values.get, reverse=True)[:top_n]

# Plot threshold sensitivity
plt.figure()
for node in top_nodes:
    values = [transmission_by_node[node].get(t, 0) for t in thresholds]
    plt.plot(thresholds, values, label=f"Node {node}")
plt.xlabel("Threshold")
plt.ylabel("Transmission value")
plt.title("Threshold sensitivity for top affected nodes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------------------------
# 3. Super-spreader detection analysis
# -----------------------------------------------
# Calculate average transmission value across all thresholds for each node
avg_transmission = {
    node: np.mean(list(data.get('transmission_value', {}).values()))
    for node, data in G.nodes(data=True)
}
# Get out-degrees (number of citations made by each paper)
out_degrees = dict(G.out_degree())

# Plot relationship between out-degree and transmission value
x = [out_degrees[n] for n in G.nodes()]
y = [avg_transmission[n] for n in G.nodes()]

plt.figure()
plt.scatter(x, y, alpha=0.6)
plt.xlabel("Out-degree")
plt.ylabel("Avg. transmission value")
plt.title("Super-spreaders detection")
plt.grid(True)
plt.tight_layout()
plt.show()