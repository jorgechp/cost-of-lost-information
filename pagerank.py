import argparse

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from models.abstract_tr import TrModel
from pagerank.pagerank_loss_calculator import PageRankLossCalculator
from relevance.database import Database

matplotlib.use('TkAgg')
def fill_nodes_information(graph, values, penalized_weight=0.1, alpha=0.85):
    pagerank_standard, pagerank_weighted, pagerank_personalized = values
    for node in graph.nodes:
        if 'pagerank' not in graph.nodes[node]:
            graph.nodes[node]['pagerank'] = {}
            graph.nodes[node]['pagerank']['penalty'] = {}
            graph.nodes[node]['pagerank']['penalty']['weighted'] = {}
            graph.nodes[node]['pagerank']['penalty']['personalization'] = {}
        parameter_id = "{}_{}".format(penalized_weight, alpha)
        graph.nodes[node]['pagerank']['original'] = pagerank_standard[node]
        graph.nodes[node]['pagerank']['penalty']['weighted'][parameter_id] = pagerank_weighted[node]
        graph.nodes[node]['pagerank']['penalty']['personalization'][parameter_id] = pagerank_weighted[node]

def print_top_pagerank(pagerank_values, N=10, title="Top N nodos por PageRank"):
    print(f"\n{title}")
    sorted_pr = sorted(pagerank_values.items(), key=lambda x: x[1], reverse=True)
    for i, (node, value) in enumerate(sorted_pr[:N]):
        print(f"{i+1}. Nodo {node}: PageRank = {value:.8f}")

def compute_pagerank_impact(graph, pr_standard, pr_penalized, affected_nodes):
    delta_pr = {}
    for node in graph.nodes:
        std = pr_standard.get(node, 0.0)
        pen = pr_penalized.get(node, 0.0)
        delta_pr[node] = pen - std
    return delta_pr

def get_node_depths(graph, sources, max_depth=5):
    depths = {}
    for source in sources:
        if source not in graph:
            continue
        for target, length in nx.single_source_shortest_path_length(graph, source, cutoff=max_depth).items():
            if target not in depths or length < depths[target]:
                depths[target] = length
    return depths

def plot_impact_by_depth(graph, delta_pr, depths):
    # Agrupamos por profundidad
    depth_impact = {}
    for node, depth in depths.items():
        impact = abs(delta_pr.get(node, 0.0))
        depth_impact.setdefault(depth, []).append(impact)

    # Promediamos
    depth_levels = sorted(depth_impact.keys())
    avg_impact = [np.mean(depth_impact[d]) for d in depth_levels]

    plt.figure(figsize=(8, 5))
    plt.plot(depth_levels, avg_impact, marker='o')
    plt.title("Average |Δ PageRank| vs Distance from Affected Nodes")
    plt.xlabel("Depth from affected nodes")
    plt.ylabel("Average |Δ PageRank|")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_network_impact(graph, delta_pr, affected_nodes):
    pos = nx.spring_layout(graph, seed=42)
    node_colors = [delta_pr.get(n, 0.0) for n in graph.nodes]
    cmap = plt.cm.RdYlBu
    vmin = min(node_colors)
    vmax = max(node_colors)

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, cmap=cmap, node_size=100,
                           vmin=vmin, vmax=vmax)
    nx.draw_networkx_edges(graph, pos, alpha=0.2)
    nx.draw_networkx_nodes(graph, pos, nodelist=affected_nodes, node_color='black', node_size=80, label='Affected')
    plt.title("Change in PageRank due to Affected Nodes")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    plt.colorbar(sm, label="Δ PageRank (penalized - standard)")
    plt.axis('off')
    plt.legend()
    plt.tight_layout()
    plt.show()

def analyze_top_impact_nodes(graph, delta_pr, affected_nodes, max_depth=5, top_n=20):
        """
        Identifica los nodos más afectados por el cambio en PageRank y los clasifica según su cercanía a nodos afectados.
        """

        def get_node_depths(graph, sources, max_depth=5):
            depths = {}
            for source in sources:
                if source not in graph:
                    continue
                for target, length in nx.single_source_shortest_path_length(graph, source, cutoff=max_depth).items():
                    if target not in depths or length < depths[target]:
                        depths[target] = length
            return depths

        depths = get_node_depths(graph, affected_nodes, max_depth=max_depth)

        data = []
        for node in graph.nodes:
            delta = delta_pr.get(node, 0.0)
            abs_delta = abs(delta)
            depth = depths.get(node, None)
            is_affected = node in affected_nodes
            data.append({
                "node": node,
                "delta_pagerank": delta,
                "abs_delta_pagerank": abs_delta,
                "depth_from_affected": depth,
                "directly_affected": is_affected
            })

        df = pd.DataFrame(data)
        top_df = df.sort_values(by="abs_delta_pagerank", ascending=False).head(top_n)

        print("\nTop nodos más afectados por pérdida de PageRank:")
        print(top_df.to_string(index=False))

        return df, top_df

def plot_impact_vs_depth(df, output_path=None):
    plt.figure(figsize=(8, 6))
    affected = df["directly_affected"]
    plt.scatter(df["depth_from_affected"][~affected],
                df["abs_delta_pagerank"][~affected],
                label="Indirectamente afectados", alpha=0.7)
    plt.scatter(df["depth_from_affected"][affected],
                df["abs_delta_pagerank"][affected],
                label="Directamente afectados", color='red', alpha=0.7)
    plt.xlabel("Depth from affected nodes")
    plt.ylabel("|Δ PageRank|")
    plt.title("Propagation of PageRank Loss Impact")
    plt.legend()
    plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, format="svg")
    else:
        plt.show()

def plot_boxplot_impact_by_depth(df, output_path=None):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[df["depth_from_affected"].notnull()],
                x="depth_from_affected",
                y="abs_delta_pagerank")
    plt.xlabel("Depth from affected nodes")
    plt.ylabel("|Δ PageRank|")
    plt.title("Propagation of PageRank Loss Impact")
    plt.yscale("log")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, format="svg")
    else:
        plt.show()

def generate_impact_plots(graph, delta_pr, affected_nodes):
    # Calcular profundidad desde nodos afectados
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

    # Compilar DataFrame
    data = []
    for node in graph.nodes:
        delta = delta_pr.get(node, 0.0)
        abs_delta = abs(delta)
        depth = depths.get(node, None)
        is_affected = node in affected_nodes
        data.append({
            "node": node,
            "delta_pagerank": delta,
            "abs_delta_pagerank": abs_delta,
            "depth_from_affected": depth,
            "directly_affected": is_affected
        })

    df = pd.DataFrame(data)

    # Generar gráficos
    plot_impact_vs_depth(df, output_path="impacto_vs_profundidad_dispersion.svg")
    plot_boxplot_impact_by_depth(df, output_path="impacto_por_profundidad_boxplot.svg")
    print("Gráficos guardados como 'impacto_vs_profundidad_dispersion.svg' y 'impacto_por_profundidad_boxplot.svg'.")

def identify_superpropagators(graph, delta_pr, affected_nodes, top_n=10):
    transmitted = {}
    for node in affected_nodes:
        impacted = 0.0
        for target in graph.nodes:
            if target == node or target not in graph.nodes:
                continue
            if nx.has_path(graph, node, target):
                impacted += abs(delta_pr.get(target, 0.0))
            transmitted[node] = impacted

    sorted_transmitters = sorted(transmitted.items(), key=lambda x: -x[1])
    print(f"\nTop {top_n} superpropagators (PageRank):")
    for i, (node, score) in enumerate(sorted_transmitters[:top_n], 1):
        print(f"{i}. Node {node} → propagated loss = {score:.8f}")

    return dict(sorted_transmitters[:top_n])

def plot_superpropagators(superpropagators, output_path="superpropagators_pagerank.svg"):
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
    print(f"Saved superpropagator chart to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Transmission models for citation network.")
    parser.add_argument('pickle_path', type=str, help="The path to the pickle file containing the graph.")

    args = parser.parse_args()

    # Initialize the database
    db = Database()
    tr_model = TrModel(db)
    tr_model.load_graph(args.pickle_path)

    affected_nodes = set(tr_model.get_affected_nodes())

    pr_model = PageRankLossCalculator(tr_model.get_graph())
    pagerank_standard = pr_model.compute_standard_pagerank()
    pagerank_weighted = pr_model.compute_weighted_pagerank(affected_nodes)
    pagerank_personalized = pr_model.compute_personalized_pagerank(affected_nodes)
    penalized_weight = 0.1
    alpha = 0.85

    # print(f"PageRank standard: {pagerank_standard}")
    # print(f"PageRank weighted: {pagerank_weighted}")
    # print(f"PageRank personalized: {pagerank_personalized}")

    fill_nodes_information(tr_model.get_graph(),
                           [pagerank_standard, pagerank_weighted, pagerank_personalized],
                           penalized_weight=penalized_weight, alpha=alpha)
    tr_model.save_graph(args.pickle_path)

    print_top_pagerank(pagerank_standard, title="Top 10 nodos por PageRank estándar")
    print_top_pagerank(pagerank_weighted, title="Top 10 nodos por PageRank penalizado (pesos)")
    print_top_pagerank(pagerank_personalized, title="Top 10 nodos por PageRank penalizado (personalización)")

    # std_rank = list(sorted(pagerank_standard, key=pagerank_standard.get, reverse=True))
    # weighted_rank = list(sorted(pagerank_weighted, key=pagerank_weighted.get, reverse=True))
    # personalizaed_rank = list(sorted(pagerank_personalized, key=pagerank_personalized.get, reverse=True))
    # tau, _ = kendalltau(std_rank, weighted_rank)
    # print(f"Kendall Tau entre rankings, std vs weigthed: {tau:.4f}")
    #
    # tau, _ = kendalltau(std_rank, personalizaed_rank)
    # print(f"Kendall Tau entre rankings, std vs personalized: {tau:.4f}")
    #
    # nodes = sorted(pagerank_standard.keys())
    # pr_std_array = np.array([pagerank_standard[n] for n in nodes])
    # pr_weighted_array = np.array([pagerank_weighted[n] for n in nodes])
    # pr_personalized_array = np.array([pagerank_personalized[n] for n in nodes])
    #
    # data = np.vstack((pr_std_array, pr_weighted_array, pr_personalized_array)).T
    # stat, p = friedmanchisquare(data[:, 0], data[:, 1], data[:, 2])
    # print(f"Friedman chi2 = {stat:.4f}, p = {p:.12f}")
    #
    #
    # if p < 0.05:
    #     print("Hay diferencias significativas entre los modelos. Ejecutando Nemenyi...")
    #     nemenyi = sp.posthoc_nemenyi_friedman(data)
    #     print(nemenyi)
    # else:
    #     print("No hay diferencias significativas entre los modelos según Friedman.")


    delta_pr = compute_pagerank_impact(tr_model.get_graph(), pagerank_standard, pagerank_weighted, affected_nodes)
    depths = get_node_depths(tr_model.get_graph(), affected_nodes, max_depth=5)
    # plot_impact_by_depth(tr_model.get_graph(), delta_pr, depths)
    # plot_network_impact(tr_model.get_graph(), delta_pr, affected_nodes)

    # delta_pr = compute_pagerank_impact(tr_model.get_graph(), pagerank_standard, pagerank_weighted, affected_nodes)
    # analyze_top_impact_nodes(tr_model.get_graph(), delta_pr, affected_nodes, max_depth=5, top_n=20)
    # generate_impact_plots(tr_model.get_graph(), delta_pr, affected_nodes)
    valid_affected_nodes = [node for node in affected_nodes if node in tr_model.get_graph()]
    superpropagators = identify_superpropagators(tr_model.get_graph(), delta_pr, valid_affected_nodes, top_n=15)
    print(superpropagators)
    plot_superpropagators(superpropagators)

if __name__ == "__main__":
    main()
