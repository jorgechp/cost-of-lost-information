import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import pickle

def load_graph(pickle_path):
    with open(pickle_path, "rb") as f:
        return pickle.load(f)

def get_subgraph_by_depth(graph, root, max_depth):
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

def hierarchy_pos(depth_map):
    pos = {}
    layers = {}
    for node, depth in depth_map.items():
        layers.setdefault(depth, []).append(node)
    for depth in layers:
        width = len(layers[depth])
        for i, node in enumerate(layers[depth]):
            pos[node] = (i - width / 2, -depth)
    return pos

def draw_and_save_svg(subgraph, depth_map, output_file, title):
    plt.figure(figsize=(12, 8))
    pos = hierarchy_pos(depth_map)
    nx.draw_networkx_nodes(subgraph, pos, node_size=20)
    nx.draw_networkx_edges(subgraph, pos, arrows=True, width=0.3, alpha=0.5)
    plt.title(title)
    plt.axis("off")
    plt.savefig(output_file, format="svg")
    plt.close()
    print(f"SVG saved to {output_file}")

if __name__ == "__main__":
    # Configura aquí tus parámetros
    PICKLE_PATH = "output/openalex_citation_network.pkl"
    ROOT_NODE = "W2031816812"
    MAX_DEPTH = 2
    OUTPUT_SVG = f"citation_tree_{ROOT_NODE}_depth{MAX_DEPTH}.svg"

    G = load_graph(PICKLE_PATH)
    subgraph, depth_map = get_subgraph_by_depth(G, ROOT_NODE, MAX_DEPTH)
    draw_and_save_svg(subgraph, depth_map, OUTPUT_SVG,
                      f"Citation Tree (Depth ≤ {MAX_DEPTH}) from {ROOT_NODE}")
