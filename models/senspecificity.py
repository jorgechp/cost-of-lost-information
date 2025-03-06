import networkx as nx
from models.abstract_transmission import AbstractTransmission

class SpecificitySensitivity(AbstractTransmission):
    """
    Specificity and Sensitivity transmission model.
    """

    def __init__(self, graph: nx.DiGraph, P_X1, P_X0, P_S1_given_X1, P_S0_given_X0):
        """
        Initialize the Specificity and Sensitivity transmission model.

        Parameters:
        graph (DiGraph): The directed graph.
        P_X1 (float): The probability of generating new knowledge.
        P_X0 (float): The probability of not generating new knowledge.
        P_S1_given_X1 (float): The probability of no lost information given new knowledge is generated.
        P_S0_given_X0 (float): The probability of lost information given no new knowledge is generated.
        """
        self.graph = graph
        self.P_X1 = P_X1
        self.P_X0 = P_X0
        self.P_S1_given_X1 = P_S1_given_X1
        self.P_S0_given_X0 = P_S0_given_X0
        self.transmission_values = self.calculate_specificity_sensitivity_values()

    def calculate_specificity_sensitivity_values(self):
        """
        Calculate the Specificity and Sensitivity values for all nodes in the graph.

        Returns:
        dict: A dictionary with nodes as keys and their Specificity and Sensitivity values as values.
        """
        values = {}
        for node in self.graph.nodes:
            values[node] = {
                'sensitivity': self.calculate_sensitivity(node),
                'specificity': self.calculate_specificity(node)
            }
        return values

    def calculate_sensitivity(self, node):
        """
        Calculate the Sensitivity value for a specific node.

        Parameters:
        node (str): The node ID.

        Returns:
        value (float): The Sensitivity value of the node.
        """
        P_S1 = self.P_S1_given_X1 * self.P_X1 + (1 - self.P_S0_given_X0) * self.P_X0
        sensitivity = (self.P_S1_given_X1 * self.P_X1) / P_S1
        return sensitivity

    def calculate_specificity(self, node):
        """
        Calculate the Specificity value for a specific node.

        Parameters:
        node (str): The node ID.

        Returns:
        value (float): The Specificity value of the node.
        """
        P_S0 = (1 - self.P_S1_given_X1) * self.P_X1 + self.P_S0_given_X0 * self.P_X0
        specificity = (self.P_S0_given_X0 * self.P_X0) / P_S0
        return specificity

    def get_transmission_value(self, node):
        """
        Get the Sensitivity and Specificity values of a node.

        Parameters:
        node (str): The node ID.

        Returns:
        value (dict): The Sensitivity and Specificity values of the node.
        """
        return self.transmission_values[node]