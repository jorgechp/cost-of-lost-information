import pickle


class TransmissionModel:

    def __init__(self, db):
        """
        Initialize the transmission model.

        Parameters:
        graph (networkx.Graph): The graph.
        db (Database): The database instance.
        """
        self.graph = None
        self.db = db

    def load_graph(self, pickle_path):
        """
        Load a NetworkX graph from a pickle file.

        Parameters:
        pickle_path (str): The path to the pickle file.

        Returns:
        graph (networkx.Graph): The loaded graph.
        """
        with open(pickle_path, 'rb') as f:
            self.graph = pickle.load(f)

    def get_affected_nodes(self):
        """
        Identify affected nodes by querying the database.

        Parameters:
        db (Database): The database instance.

        Returns:
        affected_nodes (list): List of affected nodes.
        """
        return list(self.db.get_all_affected_papers())

    def get_citation_tree(self, node):
        """
        Get the citation tree for a given node using DFS.

        Parameters:
        graph (networkx.Graph): The graph.
        node (str): The node ID.

        Returns:
        list: List of nodes in the citation tree.
        """
        citation_tree = []
        stack = [node]
        visited = set()

        while stack:
            current_node = stack.pop()
            if current_node not in visited:
                visited.add(current_node)
                citation_tree.append(current_node)
                stack.extend(self.graph.successors(current_node))

        return citation_tree

    def count_affected_levels(self, affected_nodes):
        """
        Count the number of nodes affected at each level.

        Parameters:
        graph (networkx.Graph): The graph.
        affected_nodes (list): List of directly affected nodes.

        Returns:
        dict: A dictionary with levels as keys and the number of affected nodes at each level as values.
        """
        levels = {}
        visited = set(affected_nodes)
        current_level_nodes = set(affected_nodes)
        level = 0

        while current_level_nodes:
            levels[level] = len(current_level_nodes)
            next_level_nodes = set()

            for node in current_level_nodes:
                node_str = str(node)
                alt_node = node_str.split('.')
                if len(alt_node) == 2:
                    while alt_node[1][0] == '0':
                        alt_node[1] = alt_node[1][1:]
                    alt_node[0] = alt_node[0].zfill(4)
                    alt_node[1] = alt_node[1].ljust(5, '0')
                    alt_node_str = f"{alt_node[0]}.{alt_node[1]}"
                if node in self.graph:
                    def_node = node
                elif node_str in self.graph:
                    def_node = node_str
                    affected_nodes.remove(node)
                    affected_nodes.append(def_node)
                else:
                    print(f"Node {node} not found in the graph.")
                    continue
                for successor in self.graph.successors(def_node):
                    if successor not in visited:
                        visited.add(successor)
                        next_level_nodes.add(successor)

            current_level_nodes = next_level_nodes
            level += 1

        return levels

    def save_graph(self, pickle_path):
        """
        Save a NetworkX graph to a pickle file.

        Parameters:
        graph (networkx.Graph): The graph to save.
        pickle_path (str): The path to the pickle file.
        """
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.graph, f)

    def get_graph(self):
        """
        Get the graph.

        Returns:
        graph (networkx.Graph): The graph.
        """
        return self.graph