import networkx as nx
from models.abstract_transmission import AbstractTransmission

class AffectationTransmission(AbstractTransmission):
    """
    Affectation relevance model.
    """

    def __init__(self, graph: nx.DiGraph, section_weights, threshold=0.5):
        """
        Initialize the Affectation relevance model.

        Parameters:
        graph (DiGraph): The directed graph.
        section_weights (dict): Weights of the sections.
        threshold (float): The threshold for affectation.
        """
        self.graph = graph
        self.section_weights = section_weights
        self.threshold = threshold
        self.transmission_values = self.calculate_affectation_values()

    def calculate_affectation_values(self):
        """
        Calculate the affectation values for all nodes in the graph.

        Returns:
        dict: A dictionary with nodes as keys and their affectation values as values.
        """
        values = {}
        for node in self.graph.nodes:
            values[node] = self.calculate_article_affectation(node)
        return values

    def calculate_article_affectation(self, article):
        """
        Calculate the affectation value for a specific article.

        Parameters:
        article (str): The article ID.

        Returns:
        value (float): The affectation value of the article.
        """
        sections = self.graph.nodes[article]['sections']
        affectation = 0
        for section, weight in self.section_weights.items():
            if section in sections:
                affectation += weight * self.calculate_section_affectation(sections[section])
        return affectation

    def calculate_section_affectation(self, section):
        """
        Calculate the affectation value for a specific section.

        Parameters:
        section (dict): The section data.

        Returns:
        value (float): The affectation value of the section.
        """
        external_references = section['external_references']
        internal_references = section['internal_references']

        # Check external references
        for ref in external_references:
            if ref['state'] == 'Infected':
                return 1

        # Apply Linear Threshold Model for internal references
        influence = sum(ref['weight'] * self.transmission_values[ref['article']] for ref in internal_references)
        if influence >= self.threshold:
            return 1

        return 0

    def get_transmission_value(self, node):
        """
        Get the affectation value of a node.

        Parameters:
        node (str): The node ID.

        Returns:
        value (float): The affectation value of the node.
        """
        return self.transmission_values[node]