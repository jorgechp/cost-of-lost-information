import networkx as nx
from models.abstract_transmission import AbstractTransmission

class EigenvectorCentrality(AbstractTransmission):
    """
    Eigenvector Centrality transmission model.
    """

    def __init__(self, graph: nx.DiGraph):
        """
        Initialize the Eigenvector Centrality transmission model.

        Parameters:
        graph (DiGraph): The directed graph.
        """
        self.graph = graph
        self.transmission_values = nx.eigenvector_centrality(self.graph)

    def get_transmission_value(self, node):
        """
        Get the eigenvector centrality value of a node.

        Parameters:
        node (str): The node ID.

        Returns:
        value (float): The eigenvector centrality value of the node.
        """
        return self.transmission_values[node]