import argparse

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix

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
    # fill_nodes_information(tr_model.get_graph(), [fspc_values, fspc_values_orig], K=K,
    #                                                                             decay=decay,
    #                                                                             tol=tol,
    #                                                                             penalty_factor=penalty_factor)
    # tr_model.save_graph(args.pickle_path)


    graph = tr_model.get_graph()

    # Extraer valores FSPC
    node_data = nx.get_node_attributes(graph, "fspc")
    param_key = "{}_{}_{}_{}".format(K, decay, tol, penalty_factor)
    fspc_original = {n: d.get("original", 0) for n, d in node_data.items()}
    fspc_propagated = {n: d.get(param_key, 0) for n, d in node_data.items()}

    # Calcular impacto relativo
    impact_relative = {
        n: (fspc_original[n] - fspc_propagated[n]) / fspc_original[n]
        if fspc_original[n] > 0 else 0
        for n in graph.nodes
    }

    # Añadir como atributo al grafo
    nx.set_node_attributes(graph, impact_relative, "impact_relative")

    # Calcular score de superpropagadores
    print("Calculando superpropagadores...")
    # Mapeo de nodos a índices
    node_list = list(graph.nodes)
    node_idx = {n: i for i, n in enumerate(node_list)}

    # Vector de impacto relativo
    impact_vec = np.array([impact_relative[n] for n in node_list])

    # Matriz de adyacencia dispersa (matriz de sucesores)
    A = nx.to_scipy_sparse_array(graph, nodelist=node_list, format='csr')

    # Cálculo vectorizado del impacto en vecinos
    neighbor_impact = A.dot(impact_vec)

    # Superpropagador = impacto propio + impacto a vecinos
    score_vec = impact_vec + neighbor_impact
    superpropagator_score = {node_list[i]: score_vec[i] for i in range(len(node_list))}
    nx.set_node_attributes(graph, superpropagator_score, "superpropagator_score")

    # Crear DataFrame de análisis
    df = pd.DataFrame({
        "fspc_original": fspc_original,
        "fspc_propagated": fspc_propagated,
        "impact_relative": impact_relative,
        "superpropagator_score": superpropagator_score
    })
    df["fspc_diff"] = df["fspc_original"] - df["fspc_propagated"]

    # Guardar CSV opcional
    df.to_csv("fspc_analysis.csv", index_label="node")

    # --- Visualizaciones ---

    # Histograma del impacto relativo
    plt.figure(figsize=(8, 4))
    plt.hist(df["impact_relative"], bins=30, color='skyblue')
    plt.title("Relative Impact of FSPC Propagation")
    plt.xlabel("Impact Relative")
    plt.ylabel("Number of Nodes")
    plt.tight_layout()
    plt.savefig("fspc_relative_impact_histogram.png")
    plt.close()

    # Gráfico de dispersión: FSPC original vs propagado
    plt.figure(figsize=(6, 6))
    plt.scatter(df["fspc_original"], df["fspc_propagated"], alpha=0.6)
    plt.plot([0, max(df["fspc_original"])], [0, max(df["fspc_original"])], 'r--')
    plt.title("FSPC: Original vs Propagated")
    plt.xlabel("Original FSPC")
    plt.ylabel("Propagated FSPC")
    plt.tight_layout()
    plt.savefig("fspc_original_vs_propagated.png")
    plt.close()

    # Visualización de la red coloreada por impacto relativo
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph, seed=42)
    node_color = [impact_relative[n] for n in graph.nodes]
    nx.draw(graph, pos, node_color=node_color, cmap=plt.cm.viridis, node_size=50, with_labels=False)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(node_color), vmax=max(node_color)))
    plt.colorbar(sm, label="Impact Relative")
    plt.title("FSPC Impact Propagation on Graph")
    plt.tight_layout()
    plt.savefig("fspc_graph_impact.png")
    plt.close()


if __name__ == "__main__":
    main()
