import argparse
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from models.abstract_tr import TrModel
from relevance.database import Database
from transmission.affectation_transmission_model import AffectationTransmissionModel

def plot_affected_by_level(graph, threshold):
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

    levels = list(level_counts.keys())
    values = list(level_counts.values())
    plt.figure()
    plt.plot(levels, values, marker='o')
    plt.xlabel("Propagation level")
    plt.ylabel("Affected articles")
    plt.title(f"Affected articles per level (threshold {threshold})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"affected_by_level_t{threshold}.png")
    plt.close()

def plot_threshold_sensitivity(graph, thresholds, top_n=10):
    transmission_by_node = {
        node: data.get('transmission_value', {}) for node, data in graph.nodes(data=True)
    }
    avg_values = {node: np.mean(list(v.values())) for node, v in transmission_by_node.items() if v}
    top_nodes = sorted(avg_values, key=avg_values.get, reverse=True)[:top_n]

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
    plt.savefig("threshold_sensitivity.png")
    plt.close()

def plot_super_spreaders(graph):
    avg_transmission = {
        node: np.mean(list(data.get('transmission_value', {}).values()))
        for node, data in graph.nodes(data=True) if data.get('transmission_value', {})
    }
    out_degrees = dict(graph.out_degree())

    x = [out_degrees[n] for n in avg_transmission.keys()]
    y = [avg_transmission[n] for n in avg_transmission.keys()]

    plt.figure()
    plt.scatter(x, y, alpha=0.6)
    plt.xlabel("Out-degree")
    plt.ylabel("Avg. transmission value")
    plt.title("Super-spreaders detection")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("super_spreaders.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Transmission models for citation network.")
    parser.add_argument('pickle_path', type=str, help="The path to the pickle file containing the graph.")
    parser.add_argument('-s', action='store_true', dest='by_section', help="Analyze by sections.")
    parser.add_argument('-t', dest='thresholds', type=float, nargs='+',
                        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        help="List of thresholds to process.")

    args = parser.parse_args()

    db = Database()
    tr_model = TrModel(db)
    tr_model.load_graph(args.pickle_path)
    graph = tr_model.get_graph()
    if not graph.is_directed():
        graph = nx.DiGraph(graph)

    affected_nodes = tr_model.get_affected_nodes()
    affected_levels = tr_model.count_affected_levels(affected_nodes)
    print("Number of affected nodes at each level:")
    for level, node_count in affected_levels.items():
        print(f"Level {level}: {node_count}")

    section_weights = {
        "Introduction": 0.15,
        "Methodology": 0.30,
        "Results": 0.30,
        "Discussions/Conclusions": 0.25
    }

    for t in args.thresholds:
        model = AffectationTransmissionModel(graph, db)
        model.remove_back_edges()

        pending_nodes = []
        for node in graph.nodes:
            node_value = model.compute_affectation_transmitted(node, section_weights, t, by_section=args.by_section)
            if node_value > 0:
                pending_nodes.append(node)

        affectation_transmission = model.get_affectation_transmission_dict()
        print(f"Threshold: {t}")
        print(f"Affected articles: {len(affectation_transmission)}")

        level = 0
        while pending_nodes:
            target_nodes = pending_nodes
            print(f"Level {level}: {len(target_nodes)}")
            pending_nodes = []
            for node in target_nodes:
                descendants = list(nx.descendants(graph, node))
                affected_descendants = [desc for desc in descendants if affectation_transmission.get(desc, False)]
                pending_nodes.extend(affected_descendants)
            level += 1

        for node in graph.nodes:
            graph.nodes[node].setdefault('transmission_value', {})[t] = affectation_transmission.get(node, 0)

        # Visualización por nivel para cada umbral
        plot_affected_by_level(graph, t)

    # Visualización general
    plot_threshold_sensitivity(graph, args.thresholds)
    plot_super_spreaders(graph)

    tr_model.save_graph(args.pickle_path)

if __name__ == "__main__":
    main()
