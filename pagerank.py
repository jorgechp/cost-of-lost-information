import argparse

import matplotlib
import numpy as np
import scikit_posthocs as sp
from matplotlib import pyplot as plt
from scipy.stats import kendalltau, friedmanchisquare

from fspc.fspc_model import FSPCModel
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

    print(f"PageRank standard: {pagerank_standard}")
    print(f"PageRank weighted: {pagerank_weighted}")
    print(f"PageRank personalized: {pagerank_personalized}")

    fill_nodes_information(tr_model.get_graph(),
                           [pagerank_standard, pagerank_weighted, pagerank_personalized],
                           penalized_weight=penalized_weight, alpha=alpha)
    tr_model.save_graph(args.pickle_path)

    print_top_pagerank(pagerank_standard, title="Top 10 nodos por PageRank estándar")
    print_top_pagerank(pagerank_weighted, title="Top 10 nodos por PageRank penalizado (pesos)")
    print_top_pagerank(pagerank_personalized, title="Top 10 nodos por PageRank penalizado (personalización)")

    std_rank = list(sorted(pagerank_standard, key=pagerank_standard.get, reverse=True))
    weighted_rank = list(sorted(pagerank_weighted, key=pagerank_weighted.get, reverse=True))
    tau, _ = kendalltau(std_rank, weighted_rank)
    print(f"Kendall Tau entre rankings: {tau:.4f}")

    nodes = sorted(pagerank_standard.keys())
    pr_std_array = np.array([pagerank_standard[n] for n in nodes])
    pr_weighted_array = np.array([pagerank_weighted[n] for n in nodes])
    pr_personalized_array = np.array([pagerank_personalized[n] for n in nodes])

    data = np.vstack((pr_std_array, pr_weighted_array, pr_personalized_array)).T
    stat, p = friedmanchisquare(data[:, 0], data[:, 1], data[:, 2])
    print(f"Friedman chi2 = {stat:.4f}, p = {p:.4f}")

    if p < 0.05:
        print("Hay diferencias significativas entre los modelos. Ejecutando Nemenyi...")
        nemenyi = sp.posthoc_nemenyi_friedman(data)
        print(nemenyi)
    else:
        print("No hay diferencias significativas entre los modelos según Friedman.")


if __name__ == "__main__":
    main()
