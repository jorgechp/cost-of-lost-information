import argparse

import numpy as np

from fspc.fspc_model import FSPCModel
from models.abstract_tr import TrModel
from relevance.database import Database


def fill_nodes_information(graph, values, K, decay, tol, penalty_factor):
    fspc_values, fspc_values_orig = values
    for node in graph.nodes:
        if 'fspc' not in graph.nodes[node]:
            graph.nodes[node]['fspc'] = {}
        parameter_id = "{}_{}_{}_{}".format(K, decay, tol, penalty_factor)
        graph.nodes[node]['fspc']['original'] = fspc_values[node]
        graph.nodes[node]['fspc'][parameter_id] = fspc_values_orig[node]


def main():
    parser = argparse.ArgumentParser(description="Transmission models for citation network.")
    parser.add_argument('pickle_path', type=str, help="The path to the pickle file containing the graph.")

    args = parser.parse_args()

    # Initialize the database
    db = Database()
    tr_model = TrModel(db)
    tr_model.load_graph(args.pickle_path)

    # Identify affected nodes
    affected_nodes = set(tr_model.get_affected_nodes())

    penalty_factor = 0.5
    K = 5
    tol = 1e-5
    decay = -0.5

    fspc_model = FSPCModel(tr_model.get_graph())
    fspc_values, fspc_values_orig, diff = fspc_model.propagate_information_loss(missing_refs=affected_nodes,
                                                                                K=K,
                                                                                decay=lambda k: np.exp(decay * (k - 1)),
                                                                                tol=tol,
                                                                                penalty_factor=penalty_factor
                                                                                )
    print(f"FSPC con pérdida propagada (diff): {diff}")
    fill_nodes_information(tr_model.get_graph(), [fspc_values, fspc_values_orig], K=K,
                                                                                decay=decay,
                                                                                tol=tol,
                                                                                penalty_factor=penalty_factor)
    tr_model.save_graph(args.pickle_path)


if __name__ == "__main__":
    main()
