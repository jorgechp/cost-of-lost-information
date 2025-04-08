import os
import pickle

import networkx as nx
import numpy as np


class FSPCModel:
    def __init__(self, graph):
        self.graph = graph


    def __compute_fspc(self, K=5, decay=lambda k: np.exp(-0.5 * (k - 1))):
        """
        Calcula el FSPC inicial para todos los nodos en la red.
        """
        fspc_values = {node: 0 for node in self.graph.nodes}
        len_graph_nodes = len(self.graph.nodes)
        for index, target_node in enumerate(self.graph.nodes):
            print(f"Calculando FSPC para el nodo {index + 1} de {len_graph_nodes}")
            for source_node in self.graph.nodes:
                if source_node == target_node:
                    continue

                all_paths = list(nx.all_simple_paths(self.graph, source=source_node, target=target_node, cutoff=K))
                for path in all_paths:
                    length = len(path) - 1
                    if length > 0 and length <= K:
                        fspc_values[target_node] += decay(length)

        return fspc_values


    def propagate_information_loss(self, missing_refs, penalty_factor=0.5, max_iter=10, tol=1e-5, K=5, decay=lambda k: np.exp(-0.5 * (k - 1))):
        """
        Propaga la pérdida de información por la red de citaciones.

        Parámetros:
        - graph: networkx.DiGraph, grafo de citaciones
        - missing_refs: conjunto de nodos con referencias perdidas
        - penalty_factor: cuánto reduce el FSPC de un nodo con referencias perdidas
        - max_iter: máximo número de iteraciones
        - tol: tolerancia para la convergencia

        Retorna:
        - Diccionario con los valores de FSPC propagados
        """
        # Paso 1: Calcular FSPC inicial

        pickle_path = 'tempdata/fspc_values_orig.pkl'

        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                fspc_values_orig = pickle.load(f)
        else:
            # Calculate initial FSPC values
            fspc_values_orig = self.__compute_fspc(K=K, decay=decay)
            # Save the FSPC values to a pickle file
            os.makedirs('tempdata', exist_ok=True)
            with open(pickle_path, 'wb') as f:
                pickle.dump(fspc_values_orig, f)

        fspc_values = fspc_values_orig.copy()

        # Paso 2: Propagar la pérdida de información
        for _ in range(max_iter):
            new_fspc_values = fspc_values.copy()

            for node in missing_refs:
                if node not in new_fspc_values:
                    print(node, "no está en el grafo")
                    continue

                new_fspc_values[node] *= penalty_factor  # Penalización directa

                # Ajustar el valor de los nodos que citan a este nodo
                for successor in self.graph.successors(node):
                    influence = fspc_values[node] / max(1, self.graph.in_degree(successor))
                    new_fspc_values[successor] -= influence * (1 - penalty_factor)

            # Verificar convergencia
            diff = sum(abs(new_fspc_values[n] - fspc_values[n]) for n in self.graph.nodes)
            fspc_values = new_fspc_values
            if diff < tol:
                break

        diff = sum(abs(fspc_values_orig[n] - fspc_values[n]) for n in self.graph.nodes)
        return fspc_values, fspc_values_orig, diff