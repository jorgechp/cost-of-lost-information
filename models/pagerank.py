import concurrent
import os
import pickle
import sqlite3

import networkx as nx
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from models.abstract_transmission import AbstractTransmission
from relevance.database import Database

CONCURRENT_WAIT_TIMEOUT = 20


class PageRank(AbstractTransmission):
    """
    PageRank relevance model.
    """

    def __init__(self, affected_nodes: list, db: Database, graph: nx.Graph, damping_factor=0.85, max_iterations=100, tolerance=1e-6):
        """
        Initialize the PageRank relevance model.

        Parameters:
        graph (Graph): The graph.
        damping_factor (float): The damping factor.
        max_iterations (int): The maximum number of iterations.
        tolerance (float): The tolerance for convergence.
        """
        super().__init__(affected_nodes)
        self.average_pagerank = None
        self.average_pagerank_no_lost_references = None
        self.transmission_values = None
        self.db = db
        self.graph = graph
        self.damping_factor = damping_factor
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.transmission_values_lost_references = None
        self.grap_no_lost_references = None
        self.tmpdata_dir = 'tmpdata'
        self.pickle_file = os.path.join(self.tmpdata_dir, 'pagerank_no_lost_references.pkl')
        self.invalid_nodes_file = os.path.join(self.tmpdata_dir, 'invalid_nodes.pkl')
        self.valid_nodes_file = os.path.join(self.tmpdata_dir, 'valid_nodes.pkl')
        self.lock = Lock()

    def compute_relevance_values(self):
        """
        Compute the relevance values for all nodes in the graph.
        """
        self.compute_pagerank()
        self.compute_pagerank_without_invalid_references()

    def get_transmission_value(self, node):
        """
        Get the relevance value of a node.

        Parameters:
        node (str): The node ID.

        Returns:
        value (float): The relevance value of the node.
        """
        return self.transmission_values[node]

    def get_all_relevance_values(self):
        node_pagerank_dict = {
            node: {
                'pagerank': self.transmission_values[node],
                'pagerank_affected': self.transmission_values_lost_references[node] if node in self.transmission_values_lost_references else None
            }
            for node in self.graph.nodes
        }

        return node_pagerank_dict

    def compute_pagerank(self):
        self.transmission_values = nx.pagerank(self.graph,
                                               alpha=self.damping_factor,
                                               max_iter=self.max_iterations,
                                               tol=self.tolerance)
        self.average_pagerank = sum(self.transmission_values.values()) / len(self.transmission_values)

    def is_uri_alive_concurrent(self, node, external_uris):
        for uri in external_uris:
            if not self.is_uri_alive(uri):
                return node, uri, False
        return node, None, True

    def __save_temporary_files(self, invalid_nodes, valid_nodes):
        os.makedirs(self.tmpdata_dir, exist_ok=True)
        with open(self.invalid_nodes_file, 'wb') as f:
            pickle.dump(invalid_nodes, f)
        with open(self.valid_nodes_file, 'wb') as f:
            pickle.dump(valid_nodes, f)

    def compute_pagerank_without_invalid_references(self):
        """
        Compute the PageRank of the graph excluding nodes with invalid external references.
        """

        invalid_nodes = self.db.get_all_affected_papers()

        # Create a grap_no_lost_references excluding invalid nodes
        self.grap_no_lost_references = self.graph.copy()
        self.grap_no_lost_references.remove_nodes_from(invalid_nodes)

        # Compute PageRank for the grap_no_lost_references
        self.transmission_values_lost_references = nx.pagerank(self.grap_no_lost_references, alpha=self.damping_factor, max_iter=self.max_iterations,
                                                  tol=self.tolerance)
        self.average_pagerank_no_lost_references = sum(self.transmission_values_lost_references.values()) / len(
            self.transmission_values_lost_references)

    def get_results(self):
        return self.average_pagerank, self.average_pagerank_no_lost_references

