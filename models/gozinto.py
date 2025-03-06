import networkx as nx
import numpy as np
from models.abstract_transmission import AbstractTransmission

class Gozinto(AbstractTransmission):
    """
    Gozinto transmission model.
    """

    def __init__(self, graph: nx.DiGraph):
        """
        Initialize the Gozinto transmission model.

        Parameters:
        graph (DiGraph): The directed graph.
        """
        self.graph = graph
        self.transmission_values = self.calculate_gozinto_values()

    def calculate_gozinto_values(self):
        """
        Calculate the Gozinto values for all nodes in the graph.

        Returns:
        dict: A dictionary with nodes as keys and their Gozinto values as values.
        """
        # Create the adjacency matrix A
        A = nx.to_numpy_array(self.graph, nodelist=sorted(self.graph.nodes))

        # Identity matrix I
        I = np.eye(A.shape[0])

        # Calculate the influence matrix C = (I - A)^-1
        try:
            C = np.linalg.inv(I - A)
        except np.linalg.LinAlgError:
            raise ValueError("The matrix (I - A) is not invertible.")

        # Sum the columns of C to get the total influence for each node
        gozinto_values = {node: C[:, i].sum() for i, node in enumerate(sorted(self.graph.nodes))}

        return gozinto_values

    def get_transmission_value(self, node):
        """
        Get the Gozinto value of a node.

        Parameters:
        node (str): The node ID.

        Returns:
        value (float): The Gozinto value of the node.
        """
        return self.transmission_values[node]