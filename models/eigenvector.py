import networkx as nx
from models.abstract_transmission import AbstractTransmission


class EigenvectorCentrality(AbstractTransmission):
    """
    Eigenvector Centrality relevance model.
    """

    def __init__(self, graph: nx.DiGraph, affected_nodes: list):
        """
        Initialize the Eigenvector Centrality relevance model.

        Parameters:
        graph (DiGraph): The directed graph.
        """
        super().__init__(affected_nodes)
        self.graph = graph



    def compute_relevance_values(self):
        """
        Calculate the eigenvector centrality values for all nodes in the graph.

        Returns:
        dict: A dictionary with nodes as keys and their eigenvector centrality values as values.
        """
        self.transmission_values = nx.eigenvector_centrality(self.graph)
