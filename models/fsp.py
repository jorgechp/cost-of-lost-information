import networkx as nx
from models.abstract_transmission import AbstractTransmission


class FSPC(AbstractTransmission):
    """
    Forward Search Path Count (FSPC) relevance model.
    """

    def __init__(self, graph: nx.DiGraph, affected_nodes: list, phi=lambda k: 1.0 / (k + 1), K=3):
        """
        Initialize the FSPC relevance model.

        Parameters:
        graph (DiGraph): The directed graph.
        phi (function): The decay factor function.
        K (int): The maximum path length to consider.
        """
        super().__init__(affected_nodes)
        self.graph = graph
        self.phi = phi
        self.K = K


    def calculate_fspc_values(self):
        """
        Calculate the FSPC values for all nodes in the graph.

        Returns:
        dict: A dictionary with nodes as keys and their FSPC values as values.
        """
        fspc_values = {node: 0 for node in self.graph.nodes}

        for node in self.graph.nodes:
            fspc_values[node] = self.calculate_node_fspc(node)

        self.transmission_values = fspc_values

    def calculate_node_fspc(self, node):
        """
        Calculate the FSPC value for a specific node.

        Parameters:
        node (str): The node ID.

        Returns:
        value (float): The FSPC value of the node.
        """

        def dfs(v, depth):
            if depth > self.K:
                return 0
            count = 0
            for neighbor in self.graph.predecessors(v):
                count += self.phi(depth) + dfs(neighbor, depth + 1)
            return count

        return dfs(node, 1)


    def compute_relevance_values(self):
        self.calculate_fspc_values()




