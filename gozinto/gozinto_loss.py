import networkx as nx
from collections import defaultdict

import numpy as np
from scipy.sparse import dok_matrix, identity, csr_matrix


class GozintoModel:
    def __init__(self, graph):
        self.graph = graph
        self.node_list = list(graph.nodes)
        self.index_map = {node: idx for idx, node in enumerate(self.node_list)}
        self.reverse_index_map = {i: node for node, i in self.index_map.items()}
        self.adjacency_matrix = self._build_sparse_adjacency_matrix()

    def _build_sparse_adjacency_matrix(self):
        n = len(self.node_list)
        matrix = dok_matrix((n, n), dtype=np.float32)
        for source in self.graph.nodes:
            for target in self.graph.successors(source):
                i = self.index_map[source]
                j = self.index_map[target]
                matrix[i, j] = 1.0
        return matrix.tocsr()

    def compute_total_influence_matrix(self, max_depth=10):
        n = len(self.node_list)
        total_influence = csr_matrix((n, n), dtype=np.float32)
        power = identity(n, format="csr", dtype=np.float32)
        for _ in range(1, max_depth + 1):
            power = power @ self.adjacency_matrix
            total_influence += power
        return total_influence

    def compute_node_impact_score(self, node, max_depth=10):
        T = self.compute_total_influence_matrix(max_depth=max_depth)
        if node not in self.index_map:
            return 0.0
        idx = self.index_map[node]
        return T[:, idx].sum()

    def compute_impact_from_nodes(self, source_nodes, threshold=1e-6, max_depth=10):
        T = self.compute_total_influence_matrix(max_depth=max_depth)
        results = {}
        for node in source_nodes:
            if node not in self.index_map:
                continue
            idx = self.index_map[node]
            row = T[idx].tocoo()
            impacted_nodes = {}
            for j, val in zip(row.col, row.data):
                if val > threshold:
                    target_node = self.reverse_index_map[j]
                    try:
                        paths = list(nx.all_simple_paths(self.graph, source=node, target=target_node, cutoff=max_depth))
                    except nx.NetworkXNoPath:
                        paths = []
                    impacted_nodes[target_node] = {
                        "impact": float(val),
                        "paths": paths
                    }
            results[node] = impacted_nodes
        return results

    def compute_received_impact_from(self, affected_nodes, max_depth=10, threshold=1e-6):
        """
        Devuelve para cada nodo cuánto impacto ha recibido desde los nodos afectados.

        Returns:
            Dict {node: impacto acumulado recibido}
        """
        T = self.compute_total_influence_matrix(max_depth=max_depth)
        received_impact = np.zeros(len(self.node_list))

        for node in affected_nodes:
            if node in self.index_map:
                idx = self.index_map[node]
                received_impact += T[:, idx].toarray().flatten()

        return {
            self.node_list[i]: received_impact[i]
            for i in range(len(self.node_list))
            if received_impact[i] > threshold
        }