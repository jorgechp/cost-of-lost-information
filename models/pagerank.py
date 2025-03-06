import networkx as nx

from models.abstract_transmission import AbstractTransmission


class PageRank(AbstractTransmission):
    """
    PageRank transmission model.
    """

    def __init__(self, graph: nx.Graph, damping_factor=0.85, max_iterations=100, tolerance=1e-6):
        """
        Initialize the PageRank transmission model.

        Parameters:
        graph (Graph): The graph.
        damping_factor (float): The damping factor.
        max_iterations (int): The maximum number of iterations.
        tolerance (float): The tolerance for convergence.
        """
        self.graph = graph
        self.transmission_values = nx.pagerank(self.graph, alpha=damping_factor, max_iter=max_iterations, tol=tolerance)

    def get_transmission_value(self, node):
        """
        Get the transmission value of a node.

        Parameters:
        node (str): The node ID.

        Returns:
        value (float): The transmission value of the node.
        """

        return self.transmission_values[node]
