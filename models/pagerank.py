import concurrent
import os
import pickle
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from models.abstract_transmission import AbstractTransmission

CONCURRENT_WAIT_TIMEOUT = 20


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
        super().__init__()
        self.average_pagerank = None
        self.average_pagerank_no_lost_references = None
        self.transmission_values = None
        self.graph = graph
        self.damping_factor = damping_factor
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.tmpdata_dir = 'tmpdata'
        self.pickle_file = os.path.join(self.tmpdata_dir, 'pagerank_no_lost_references.pkl')
        self.invalid_nodes_file = os.path.join(self.tmpdata_dir, 'invalid_nodes.pkl')
        self.valid_nodes_file = os.path.join(self.tmpdata_dir, 'valid_nodes.pkl')
        self.lock = Lock()

    def compute_transmission_values(self):
        """
        Compute the transmission values for all nodes in the graph.
        """
        self.compute_pagerank()
        self.compute_pagerank_without_invalid_references()

    def get_transmission_value(self, node):
        """
        Get the transmission value of a node.

        Parameters:
        node (str): The node ID.

        Returns:
        value (float): The transmission value of the node.
        """
        return self.transmission_values[node]

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
        if os.path.exists(self.pickle_file):
            with open(self.pickle_file, 'rb') as f:
                pagerank_no_lost_references = pickle.load(f)
        else:
            if os.path.exists(self.invalid_nodes_file) and os.path.exists(self.valid_nodes_file):
                with open(self.invalid_nodes_file, 'rb') as f:
                    invalid_nodes = pickle.load(f)
                with open(self.valid_nodes_file, 'rb') as f:
                    valid_nodes = pickle.load(f)
            else:
                invalid_nodes = set()
                valid_nodes = set()
            paper_count = 0
            batch_size = 100  # Process nodes in batches of 100
            nodes = list(self.graph.nodes(data=True))
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]
                with ThreadPoolExecutor(max_workers=10) as executor:  # Limit the number of threads
                    futures = []
                    for node, data in batch:
                        with self.lock:
                            if node in valid_nodes or node in invalid_nodes:
                                continue
                        sections = data.get('sections', [])
                        external_uris_complete = [uri for section in sections for uri in section.get('external_uris', [])]
                        if len(external_uris_complete) > 0:
                            for section in sections:
                                external_uris = section.get('external_uris', [])
                                futures.append(executor.submit(self.is_uri_alive_concurrent, node, external_uris))
                        else:
                            valid_nodes.add(node)
                        paper_count += 1
                        if paper_count % 200 == 0:
                            print(f"Processed {paper_count} papers. Invalid nodes: {len(valid_nodes)}. Invalid nodes: {len(invalid_nodes)}")
                            self.__save_temporary_files(invalid_nodes, valid_nodes)
                    try:
                        for future in concurrent.futures.as_completed(futures, timeout=CONCURRENT_WAIT_TIMEOUT):  # Timeout of 60 seconds
                            try:
                                node, uri, is_valid = future.result()
                                with self.lock:
                                    if is_valid:
                                        valid_nodes.add(node)
                                    else:
                                        invalid_nodes.add(node)
                            except Exception as e:
                                print(f"Error processing future: {e}")
                            finally:
                                futures.remove(future)
                    except concurrent.futures.TimeoutError:
                        print("Timeout occurred while waiting for futures to complete")

                self.__save_temporary_files(invalid_nodes, valid_nodes)

            subgraph = self.graph.copy()
            subgraph.remove_nodes_from(invalid_nodes)
            pagerank_no_lost_references = nx.pagerank(subgraph, alpha=self.damping_factor, max_iter=self.max_iterations,
                                                      tol=self.tolerance)
            with open(self.pickle_file, 'wb') as f:
                pickle.dump(pagerank_no_lost_references, f)

        self.average_pagerank_no_lost_references = sum(pagerank_no_lost_references.values()) / len(
            pagerank_no_lost_references)
