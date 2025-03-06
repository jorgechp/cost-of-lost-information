import networkx as nx
import random
from models.abstract_transmission import AbstractTransmission

class SIS(AbstractTransmission):
    """
    Susceptible, Infection and Susceptible (SIS) transmission model.
    """

    def __init__(self, graph: nx.DiGraph, beta=0.1, iterations=100):
        """
        Initialize the SIS transmission model.

        Parameters:
        graph (DiGraph): The directed graph.
        beta (float): The transmission rate.
        iterations (int): The number of iterations to simulate.
        """
        self.graph = graph
        self.beta = beta
        self.iterations = iterations
        self.transmission_values = self.simulate_sis()

    def simulate_sis(self):
        """
        Simulate the SIS model on the graph.

        Returns:
        dict: A dictionary with nodes as keys and their infection status as values.
        """
        # Initialize all nodes as susceptible
        status = {node: 'S' for node in self.graph.nodes}

        # Randomly infect some initial nodes
        initial_infected = random.sample(self.graph.nodes, k=int(0.1 * len(self.graph.nodes)))
        for node in initial_infected:
            status[node] = 'I'

        for _ in range(self.iterations):
            new_status = status.copy()
            for node in self.graph.nodes:
                if status[node] == 'S':
                    # Check if the node gets infected
                    infected_neighbors = [n for n in self.graph.predecessors(node) if status[n] == 'I']
                    if infected_neighbors and random.random() < self.beta * len(infected_neighbors):
                        new_status[node] = 'I'
                elif status[node] == 'I':
                    # Infected nodes remain infected (gamma = 0)
                    new_status[node] = 'I'
            status = new_status

        return {node: 1 if status[node] == 'I' else 0 for node in self.graph.nodes}

    def get_transmission_value(self, node):
        """
        Get the infection status of a node.

        Parameters:
        node (str): The node ID.

        Returns:
        value (int): The infection status of the node (1 for infected, 0 for susceptible).
        """
        return self.transmission_values[node]