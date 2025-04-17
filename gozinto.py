import argparse
import seaborn as sns
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize

from gozinto.gozinto_loss import GozintoModel
from models.abstract_tr import TrModel
from relevance.database import Database
from sais.sais_model import SAISModel
from transmission.affectation_transmission_model import AffectationTransmissionModel


def fill_nodes_information(graph, history, beta, gamma, delta):
    for node in graph.nodes:
        if 'sais' not in graph.nodes[node]:
            graph.nodes[node]['sais'] = {}
        parameter_id = "{}_{}_{}".format(beta, gamma, delta)
        graph.nodes[node]['sais'][parameter_id] = {
            step: history[step][node] for step in range(len(history))
        }

def apply_horizontal_jitter_by_level(pos, jitter_strength=0.1):
    level_nodes = defaultdict(list)

    # Agrupar nodos por nivel Y
    for node, (x, y) in pos.items():
        level_nodes[round(y, 3)].append((node, round(x, 3)))

    # Aplicar jitter donde hay duplicados
    for y, nodes in level_nodes.items():
        x_counts = defaultdict(list)
        for node, x in nodes:
            x_counts[x].append(node)

        for x, node_list in x_counts.items():
            if len(node_list) > 1:
                total = len(node_list)
                offsets = np.linspace(-jitter_strength, jitter_strength, total)
                for i, node in enumerate(sorted(node_list)):
                    old_x, old_y = pos[node]
                    pos[node] = (old_x + offsets[i], old_y)

    return pos
def rank_affected_nodes_by_gozinto_impact(model, affected_nodes, max_depth=10, print_top_n=10):
    """
    Calcula y ordena el impacto total Gozinto de cada nodo afectado.

    Retorna:
        Diccionario {nodo: impacto_total} ordenado de mayor a menor impacto.
    """
    scores = {
        node: model.compute_node_impact_score(node, max_depth=max_depth)
        for node in affected_nodes
        if node in model.index_map
    }

    sorted_scores = dict(sorted(scores.items(), key=lambda x: -x[1]))

    if print_top_n:
        print(f"\nTop {print_top_n} nodos afectados por impacto Gozinto:")
        for i, (node, score) in enumerate(list(sorted_scores.items())[:print_top_n]):
            print(f"{i+1}. Nodo {node}: impacto = {score:.6f}")

    return sorted_scores


def plot_gozinto_impact_scores(impact_scores, top_n=20, title="Most relevant nodes (Gozinto"):
    # Ordenar por impacto
    sorted_items = sorted(impact_scores.items(), key=lambda x: -x[1])[:top_n]
    nodes, scores = zip(*sorted_items)

    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(nodes)), scores, color='steelblue')
    plt.xticks(range(len(nodes)), nodes, rotation=45, ha='right')
    plt.title(title)
    plt.xlabel("Node")
    plt.ylabel("Total impact (Gozinto)")
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()

def plot_received_impact(received_impact, top_n=20, title="Most affected nodes (Gozinto)"):
    sorted_items = sorted(received_impact.items(), key=lambda x: -x[1])[:top_n]
    nodes, scores = zip(*sorted_items)

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(nodes)), scores, color='indianred')
    plt.yticks(range(len(nodes)), nodes)
    plt.xlabel("Total impact (Gozinto)")
    plt.title(title)
    plt.gca().invert_yaxis()  # El más afectado arriba
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def compare_received_vs_structural_impact(model, affected_nodes, max_depth=10):
        received = model.compute_received_impact_from(affected_nodes, max_depth=max_depth)
        structural = {
            node: model.compute_node_impact_score(node, max_depth=max_depth)
            for node in received  # Solo los que recibieron impacto
        }

        combined = {
            node: {
                "received": received[node],
                "structural": structural[node]
            }
            for node in received
        }

        return combined

def plot_received_vs_structural(combined_impact, top_n=15):
    sorted_items = sorted(combined_impact.items(), key=lambda x: -x[1]["received"])[:top_n]
    nodes = [item[0] for item in sorted_items]
    received = [item[1]["received"] for item in sorted_items]
    structural = [item[1]["structural"] for item in sorted_items]

    x = range(len(nodes))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x, received, width=width, label='Received impact', color='indianred')
    plt.bar([i + width for i in x], structural, width=width, label='Estructural impact', color='steelblue')
    plt.xticks([i + width/2 for i in x], nodes, rotation=45, ha='right')
    plt.ylabel("Gozinto score")
    plt.title("Comparing received vs structural impact")
    plt.legend()
    plt.tight_layout()
    plt.show()

def rank_nodes_by_transmitted_impact(model, affected_nodes, max_depth=10):
    T = model.compute_total_influence_matrix(max_depth=max_depth)
    transmitted_scores = {}

    for node in affected_nodes:
        if node not in model.index_map:
            continue
        idx = model.index_map[node]
        col = T[:, idx].toarray().flatten()
        # Puedes excluir el nodo en sí mismo si quieres:
        col[idx] = 0
        transmitted_scores[node] = col.sum()

    return dict(sorted(transmitted_scores.items(), key=lambda x: -x[1]))

def plot_transmitted_impact(transmitted_scores, top_n=20, title="Moret affected nodes (Gozinto)"):
    sorted_items = sorted(transmitted_scores.items(), key=lambda x: -x[1])[:top_n]
    nodes, scores = zip(*sorted_items)

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(nodes)), scores, color='darkorange')
    plt.yticks(range(len(nodes)), nodes)
    plt.xlabel("Transmitted impact (Gozinto)")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def hierarchy_pos(G, root=None, width=2.0, vert_gap=0.4, vert_loc=0, xcenter=0.5, pos=None, parent=None, min_dx=0.1):
    """
    Posiciones jerárquicas en forma de árbol con espacio mínimo horizontal garantizado.
    """
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)

    children = list(G.successors(root))
    if len(children) != 0:
        dx = max(width / len(children), min_dx)
        total_width = dx * len(children)
        nextx = xcenter - total_width / 2
        for child in children:
            pos = hierarchy_pos(G, root=child, width=dx, vert_gap=vert_gap,
                                vert_loc=vert_loc - vert_gap, xcenter=nextx, pos=pos, parent=root, min_dx=min_dx)
            nextx += dx
    print(pos)
    return pos
def plot_num_nodos_impactados(impact_dict, top_n=20, title="Superpropagators (Gozinto)"):
    node_counts = {
        nodo: len(info) for nodo, info in impact_dict.items()
    }

    sorted_items = sorted(node_counts.items(), key=lambda x: -x[1])[:top_n]
    nodos, counts = zip(*sorted_items)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(nodos)), counts, color='seagreen')
    plt.yticks(range(len(nodos)), nodos)
    plt.xlabel("Number of impacted nodes")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_subgraph_from_impact(graph, impact_info, source_node):
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    if source_node not in impact_info:
        print(f"No impact information found for node {source_node}")
        return

    subgraph = nx.DiGraph()
    path_lengths = {}
    node_impact = {}

    for target, data in impact_info[source_node].items():
        node_impact[target] = data["impact"]
        for path in data["paths"]:
            nx.add_path(subgraph, path)
            for i, node in enumerate(path):
                if node not in path_lengths or path_lengths[node] > i:
                    path_lengths[node] = i

    path_lengths[source_node] = 0
    node_impact[source_node] = 0.0

    pos = hierarchy_pos(subgraph, root=source_node, width=7.0, vert_gap=0.8, min_dx=9.8)
    pos = apply_horizontal_jitter_by_level(pos, jitter_strength=3.6)

    max_depth = max(path_lengths.values()) if path_lengths else 1
    color_map = cm.viridis
    norm_depth = mcolors.Normalize(vmin=0, vmax=max_depth)
    node_colors = [color_map(norm_depth(path_lengths[node])) for node in subgraph.nodes]

    max_impact = max(node_impact.values()) if node_impact else 1
    edge_cmap = plt.cm.magma  # buena visibilidad sobre fondo blanco
    edge_colors = [edge_cmap(node_impact.get(v, 0.0) / max_impact) for u, v in subgraph.edges()]

    fig, ax = plt.subplots(figsize=(18, 10))  # Espacio amplio

    nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=500, cmap=color_map, ax=ax)
    nx.draw_networkx_edges(subgraph, pos, edge_color=edge_colors, arrows=True, ax=ax)

    # Alternancia de nombre e impacto por nivel
    levels = {}
    for node, (x, y) in pos.items():
        level = round(y, 3)
        levels.setdefault(level, []).append((node, x))

    for level_nodes in levels.values():
        sorted_nodes = sorted(level_nodes, key=lambda t: t[1])  # Ordenar por x
        for i, (node, x) in enumerate(sorted_nodes):
            _, y = pos[node]
            label_offset = 0.07 if i % 2 == 0 else -0.07
            impact_offset = -0.12 if label_offset > 0 else 0.12

            # Nombre del nodo
            ax.text(x, y + label_offset, node, fontsize=9, ha='center', va='center', color='black')

            # Valor del impacto (si hay)
            impact_val = node_impact.get(node, 0.0)
            if impact_val > 0:
                ax.text(x, y + impact_offset, f"{impact_val:.2f}", fontsize=8, ha='center', color='black')

    # Barra de color
    sm_nodes = plt.cm.ScalarMappable(cmap=color_map, norm=norm_depth)
    sm_nodes.set_array([])
    cbar_nodes = plt.colorbar(sm_nodes, ax=ax)
    cbar_nodes.set_label("Depth from source node")

    ax.set_xlim(ax.get_xlim()[0] - 1, ax.get_xlim()[1] + 1)
    ax.set_ylim(ax.get_ylim()[0] - 0.5, ax.get_ylim()[1] + 0.5)

    plt.title(f"Citation impact network from node {source_node}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def plot_incoming_citation_tree(graph, target_node, max_depth=5):
        import networkx as nx
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        from collections import deque

        if target_node not in graph:
            print(f"Node {target_node} not found in the graph.")
            return

        print(f"Predecesores de {target_node}:", list(graph.predecessors(target_node)))

        # Construir el subgrafo de citas entrantes hasta `max_depth`
        subgraph = nx.DiGraph()
        subgraph.add_node(target_node)
        depths = {target_node: 0}
        queue = deque([(target_node, 0)])


        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for pred in graph.predecessors(current):
                subgraph.add_edge(pred, current)
                if pred not in depths or depths[pred] > depth + 1:
                    depths[pred] = depth + 1
                    queue.append((pred, depth + 1))

        # Layout jerárquico hacia atrás
        pos = hierarchy_pos(subgraph, root=target_node, width=7.0, vert_gap=0.8, min_dx=9.8)
        pos = apply_horizontal_jitter_by_level(pos, jitter_strength=3.6)

        # Colores por profundidad (ahora descendiendo hacia raíces)
        max_depth_found = max(depths.values()) if depths else 1
        color_map = cm.viridis
        norm_depth = mcolors.Normalize(vmin=0, vmax=max_depth_found)
        node_colors = [color_map(norm_depth(depths[n])) for n in subgraph.nodes]

        fig, ax = plt.subplots(figsize=(14, 9))
        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=500, cmap=color_map, ax=ax)
        nx.draw_networkx_edges(subgraph, pos, edge_color='gray', arrows=True, ax=ax)

        # Etiquetas alternas en distintos niveles
        label_positions = {}
        level_counts = defaultdict(int)
        for node, (x, y) in pos.items():
            count = level_counts[round(y, 2)]
            offset = 0.06 if count % 2 == 0 else -0.08
            label_positions[node] = (x, y + offset)
            level_counts[round(y, 2)] += 1

        nx.draw_networkx_labels(subgraph, label_positions, font_size=9, font_color='black', ax=ax)

        # Barra de color
        sm = cm.ScalarMappable(cmap=color_map, norm=norm_depth)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Depth toward target node")

        plt.title(f"Incoming citation network to node {target_node}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

def analyze_propagation_vs_depth(model, affected_nodes, max_depth_range=range(1, 11)):
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
        results["average_impact"].append(np.mean(impact_values) if impact_values else 0)

    return results

def plot_propagation_vs_depth(results):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(results["depth"], results["total_impacted"], marker='o')
    plt.title("Impacted nodes vs Depth")
    plt.xlabel("max_depth")
    plt.ylabel("# Impacted nodes")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(results["depth"], results["average_impact"], marker='s', color='darkgreen')
    plt.title("Average impact vs Depth")
    plt.xlabel("max_depth")
    plt.ylabel("Avg impact per node")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def detect_superpropagators(gozinto_model, centrality_dict, impacted_counts, top_n=20):
    candidates = []
    for node in impacted_counts:
        centrality = centrality_dict.get(node, 0)
        impacted = impacted_counts[node]
        if centrality < 1e-4 and impacted > 0:
            candidates.append((node, centrality, impacted))

    candidates.sort(key=lambda x: -x[2])
    return candidates[:top_n]


def plot_superpropagator_scatter(combined_impact):
    import matplotlib.pyplot as plt

    received = [v["received"] for v in combined_impact.values()]
    structural = [v["structural"] for v in combined_impact.values()]
    nodes = list(combined_impact.keys())

    plt.figure(figsize=(8, 8))
    plt.scatter(structural, received, alpha=0.6, edgecolors='black')

    # Línea diagonal de referencia y = x
    max_val = max(max(structural), max(received)) * 1.05
    plt.plot([0, max_val], [0, max_val], color='gray', linestyle='--', label='y = x')

    plt.xlabel("Structural impact (Gozinto)")
    plt.ylabel("Received impact (Gozinto)")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Structural vs Received Impact")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

def audit_node_impact(impact_dict, received_impact, target_node="2211.97990"):
    print(f"Auditoría del nodo: {target_node}")
    print("-" * 50)

    # 1. ¿Está el nodo marcado como afectado directamente?
    is_directly_affected = target_node in impact_dict
    print(f"¿Nodo directamente afectado? {is_directly_affected}")

    # 2. ¿Tiene algún impacto recibido?
    if target_node in received_impact:
        print(f"Impacto total recibido: {received_impact[target_node]:.6f}")
    else:
        print("❌ El nodo no aparece en el diccionario de impacto recibido.")

    # 3. ¿Qué nodo lo impactó? Buscarlo en los caminos de impacto
    found = False
    for source, data in impact_dict.items():
        if target_node in data:
            print(f"\n✅ Nodo impactado por: {source}")
            print(f"  Impacto: {data[target_node]['impact']:.6f}")
            print("  Caminos:")
            for path in data[target_node]['paths']:
                print("   → " + " → ".join(path))
            found = True
            break

    if not found:
        print("❌ No se encontró ningún nodo que haya impactado a este nodo.")

    print("-" * 50)

def plot_gozinto_impact_by_depth(graph, impact_gozinto, affected_nodes, output_path="gozinto_impact_by_depth.svg"):
    def get_node_depths(graph, sources, max_depth=5):
        depths = {}
        for source in sources:
            if source not in graph:
                continue
            for target, length in nx.single_source_shortest_path_length(graph, source, cutoff=max_depth).items():
                if target not in depths or length < depths[target]:
                    depths[target] = length
        return depths

    depths = get_node_depths(graph, affected_nodes, max_depth=5)

    data = []
    for node in graph.nodes:
        impact = abs(impact_gozinto.get(node, 0.0))
        depth = depths.get(node, None)
        if depth is not None:
            data.append({"node": node, "abs_impact": impact, "depth_from_affected": depth})

    df = pd.DataFrame(data)

    # Plot en escala logarítmica
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="depth_from_affected", y="abs_impact")
    plt.yscale("log")
    plt.xlabel("Depth from affected nodes")
    plt.ylabel("|Gozinto impact|")
    plt.title("Propagation of Gozinto Loss Impact")
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()
    print(f"Saved Gozinto impact plot to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Transmission models for citation network.")
    parser.add_argument('pickle_path', type=str, help="The path to the pickle file containing the graph.")
    parser.add_argument('-s', type=int, dest='steps', help="Number of steps")

    args = parser.parse_args()

    # Initialize the database
    db = Database()
    tr_model = TrModel(db)
    tr_model.load_graph(args.pickle_path)


    # Identify affected nodes
    affected_nodes = tr_model.get_affected_nodes()

    gozinto = GozintoModel(tr_model.graph)
    # impact = gozinto.compute_impact_from_nodes(affected_nodes, max_depth=5)
    #
    # score = gozinto.compute_node_impact_score("2106.13393", max_depth=10)
    # ranked = rank_affected_nodes_by_gozinto_impact(gozinto, affected_nodes, max_depth=10)
    # print(ranked)
    #
    # for source, impacted in impact.items():
    #     print(f"\nNodo afectado: {source}")
    #     print(f"Nodos impactados: {len(impacted)}")
    #     top = sorted(impacted.items(), key=lambda x: -x[1]['impact'])[:10]
    #     for node, data in top:
    #         print(f" - {node}: impacto = {data['impact']:.6f}")
    #         for path in data['paths']:
    #             print(f"    camino: {' -> '.join(path)}")

    # plot_gozinto_impact_scores(ranked, top_n=20)

    received = gozinto.compute_received_impact_from(affected_nodes)

    # top = sorted(received.items(), key=lambda x: -x[1])[:10]
    # print("Top nodos más afectados por la pérdida:")
    # for i, (node, score) in enumerate(top, 1):
    #     print(f"{i}. Nodo {node} → impacto recibido = {score:.6f}")

    # plot_received_impact(received, top_n=20)
    combined = compare_received_vs_structural_impact(gozinto, affected_nodes, max_depth=10)
    # plot_received_vs_structural(combined, top_n=20)

    # transmitted = rank_nodes_by_transmitted_impact(gozinto, affected_nodes, max_depth=10)
    #
    # print("\nTop nodos más afectantes:")
    # for i, (node, score) in enumerate(list(transmitted.items())[:10], 1):
    #     print(f"{i}. Nodo {node}: pérdida transmitida = {score:.6f}")

    # plot_transmitted_impact(transmitted, top_n=20)

    # plot_num_nodos_impactados(impact, top_n=20)
    # plot_subgraph_from_impact(tr_model.graph, impact, "2111.13923")
    #
    # # Puedes llamarlos desde main si lo deseas:
    # propagation_results = analyze_propagation_vs_depth(gozinto, affected_nodes, max_depth_range=range(1, 11))
    # plot_propagation_vs_depth(propagation_results)

    # eigen_centrality = nx.eigenvector_centrality(tr_model.graph, max_iter=500)
    # impacted_counts = {k: len(v) for k, v in impact.items()}
    # combined = compare_received_vs_structural_impact(gozinto, affected_nodes, max_depth=10)
    # plot_superpropagator_scatter(combined)
    # plot_superpropagator_scatter(combined)
    # top_unexpected = detect_superpropagators(gozinto, eigen_centrality, impacted_counts)
    # print("Top unexpected superpropagators:", top_unexpected)

    # plot_incoming_citation_tree(tr_model.graph, target_node="2211.97990", max_depth=5)

    impact_gozinto = gozinto.compute_received_impact_from(affected_nodes)
    plot_gozinto_impact_by_depth(tr_model.get_graph(), impact_gozinto, affected_nodes)



if __name__ == "__main__":
    main()
