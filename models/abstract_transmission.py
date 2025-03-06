from abc import ABC, abstractmethod


class AbstractTransmission(ABC):

    def __init__(self):
        """
        Initialize the transmission model.

        """
        self._nodes_value = dict()

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
