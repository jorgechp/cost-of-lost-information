import argparse
import pickle
import networkx as nx

from models.gozinto import Gozinto
from models.pagerank import PageRank
from transmission.database import Database


def load_graph(pickle_path):
    """
    Loads a NetworkX graph from a pickle file.

    Parameters:
    pickle_path (str): The path to the pickle file.

    Returns:
    G (networkx.Graph): The loaded graph.
    """
    with open(pickle_path, 'rb') as f:
        G = pickle.load(f)
    return G

def main():
    parser = argparse.ArgumentParser(description="Load a NetworkX graph from a pickle file.")
    parser.add_argument('pickle_path', type=str, help="The path to the pickle file containing the graph.")

    args = parser.parse_args()

    # Load the graph from the pickle file
    G = load_graph(args.pickle_path)

    # Print some information about the graph
    print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    print(f"Populating database with graph data...")
    db = Database()
    db.create_schema()
    db.populate_database(G)

    # models = {
    #     'pagerank': PageRank(G, damping_factor=0.85, max_iterations=100, tolerance=1e-6),
    #     # 'gozinto': Gozinto(G),
    #     # Agregar otros modelos aquí
    # }
    #
    # for model in models:
    #     print(f"Calculating transmission values using {model}...")
    #     selected_model = models[model]
    #     selected_model.compute_transmission_values()
    #
    #     transmission_values = {
    #         node: selected_model.get_transmission_value(node)
    #         for node in G.nodes(data=True)
    #     }
    #     print(f"Transmission values calculated using {model}.")
    #     print(transmission_values)


if __name__ == "__main__":
    main()