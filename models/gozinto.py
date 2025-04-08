import networkx as nx
import numpy as np
import pickle
import os

import scipy as sp

from models.abstract_transmission import AbstractTransmission

class Gozinto(AbstractTransmission):
    """
    Gozinto relevance model.
    """

    def __init__(self, graph: nx.DiGraph, affected_nodes: list):
        """
        Initialize the Gozinto relevance model.

        Parameters:
        graph (DiGraph): The directed graph.
        """
        super().__init__(affected_nodes)
        self.graph = graph
        self.affected_nodes = affected_nodes

    def remove_back_edges(self):
        """
        Detect cycles in the graph and remove back edges to eliminate cycles.

        Returns:
        nx.DiGraph: The graph with back edges removed.
        """
        cycles = list(nx.simple_cycles(self.graph))
        while len(cycles) > 0:
            for cycle in cycles:
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    if self.graph.has_edge(v, u):
                        self.graph.remove_edge(v, u)
                        break
            cycles = list(nx.simple_cycles(self.graph))

    def compute_relevance_values(self):
        """
        Calculate the Gozinto values for all nodes in the graph.

        Returns:
        dict: A dictionary with nodes as keys and their Gozinto values as values.
        """
        self.remove_back_edges()

        # Define the file path for the serialized Gozinto matrix
        file_path = 'tmpdata/gozinto_matrix.pkl'

        # Check if the serialized file exists
        if os.path.exists(file_path):
            # Load the Gozinto matrix from the file
            with open(file_path, 'rb') as f:
                Gozinto_matrix = pickle.load(f)
        else:
            # Create the adjacency matrix A as a sparse matrix
            A = nx.to_scipy_sparse_matrix(self.graph, nodelist=sorted(self.graph.nodes), dtype='uint8', format='csc')

            # Calculate the Gozinto matrix (I - A)^-1
            I = sp.eye(A.shape[0], format='csc')
            Gozinto_matrix = spla.inv(I - A)

            # Serialize the Gozinto matrix to a file
            with open(file_path, 'wb') as f:
                pickle.dump(Gozinto_matrix, f)

        # Sum the columns of Gozinto_matrix to get the total influence for each node
        self.transmission_values = {node: Gozinto_matrix[:, i].sum() for i, node in enumerate(sorted(self.graph.nodes))}

        min_value = np.min(Gozinto_matrix)
        max_value = np.max(Gozinto_matrix)

        # Normalizar los valores
        self.transmission_values = {node: (value - min_value) / (max_value - min_value) for node, value in
                                    self.transmission_values.items()}



