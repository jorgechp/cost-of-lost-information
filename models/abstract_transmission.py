import re
from abc import ABC, abstractmethod
from urllib.parse import urlparse

import requests

URI_ALIVE_TIMEOUT = 18

RSTRIP_URL_CHARACTERS = '),].'  # Characters to strip from the end of a URL


class AbstractTransmission(ABC):

    def __init__(self):
        """
        Initialize the transmission model.

        """
        self._nodes_value = dict()

    @abstractmethod
    def compute_transmission_values(self):
        """
        Compute the transmission values for all nodes in the graph.

        """
        raise NotImplementedError

    @abstractmethod
    def get_transmission_value(self, node):
        """
        Get the transmission value of a node.

        Parameters:
        node (str): The node ID.

        Returns:
        value (float): The transmission value of the node.
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
