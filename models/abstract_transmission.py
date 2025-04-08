import re
from abc import ABC, abstractmethod
from urllib.parse import urlparse

import requests

URI_ALIVE_TIMEOUT = 18

RSTRIP_URL_CHARACTERS = '),].'  # Characters to strip from the end of a URL


class AbstractTransmission(ABC):

    def __init__(self, affected_nodes: list):
        """
        Initialize the relevance model.

        """
        self._nodes_value = dict()
        self.affected_nodes = affected_nodes
        self.transmission_values = {}

    @abstractmethod
    def compute_relevance_values(self):
        """
        Compute the relevance values for all nodes in the graph.

        """
        raise NotImplementedError

    @abstractmethod
    def get_transmission_value(self, node):
        """
        Get the relevance value of a node.

        Parameters:
        node (str): The node ID.

        Returns:
        value (float): The relevance value of the node.
        """
        raise NotImplementedError

    @property
    def nodes_value(self):
        return self._nodes_value

    @nodes_value.setter
    def nodes_value(self, value):
        self._nodes_value = value.upper()

    def fix_uri(self, uri):
        try:
            match = re.match(r'([a-zA-Z]+://[^\]]+)\](\1)', uri)
            parsed_uri = match.group(1) if match else uri
            parsed_uri = urlparse(parsed_uri.rstrip(RSTRIP_URL_CHARACTERS))
            fixed_uri = parsed_uri._replace(path=parsed_uri.path.rstrip(RSTRIP_URL_CHARACTERS)).geturl()
            return fixed_uri
        except ValueError:
            print("Error in fix_uri: ", uri)

    def is_uri_alive(self, uri):
        try:
            # Validate the URI
            result = urlparse(uri)
            if not all([result.scheme, result.netloc]):
                return False

            response = requests.head(uri, allow_redirects=True, timeout=15)
            return response.status_code in [200, 301, 302, 418]
        except requests.RequestException:
            return False

    def compute_affectation(self):
        """
        Compute the affectation levels for all nodes in the graph.

        Parameters:
        affected_nodes (list): List of nodes that are directly affected.

        Returns:
        dict: A dictionary with nodes as keys and their affectation levels as values.
        """
        affectation = {node: 0 for node in self.graph.nodes}

        # Initialize affectation for directly affected nodes
        for node in self.affected_nodes:
            affectation[node] = 1

        # Propagate affectation through the graph
        for node in sorted(self.graph.nodes, key=lambda n: self.transmission_values[n], reverse=True):
            for neighbor in self.graph.predecessors(node):
                affectation[neighbor] += affectation[node] * self.transmission_values[neighbor]

        return affectation

    def get_transmission_value(self, node):
        """
        Get the Sensitivity and Specificity values of a node.

        Parameters:
        node (str): The node ID.

        Returns:
        value (dict): The Sensitivity and Specificity values of the node.
        """
        return self.transmission_values[node]

    def get_transmission_values(self):
        """
        Get the Sensitivity and Specificity values of all nodes.
        """
        return self.transmission_values
