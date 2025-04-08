import argparse
import os
import pickle

from models.eigenvector import EigenvectorCentrality
from models.fsp import FSPC
from models.gozinto import Gozinto
from models.pagerank import PageRank
from models.senspecificity import SpecificitySensitivity
from models.sis import SIS
from relevance.database import Database

LOCK_FILE_PATH = 'tmpdata/db_populated.lock'

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

def save_graph(graph, pickle_path):
    """
    Save a NetworkX graph to a pickle file.

    Parameters:
    graph (networkx.Graph): The graph to save.
    pickle_path (str): The path to the pickle file.
    """
    with open(pickle_path, 'wb') as f:
        pickle.dump(graph, f)

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
    # Check if the lock file exists
    if not os.path.exists(LOCK_FILE_PATH):
        print(f"Populating database with graph data...")
        db.populate_database(G)
        # Create the lock file to indicate that the database has been populated
        with open(LOCK_FILE_PATH, 'w') as lock_file:
            lock_file.write('Database populated.\n')
    else:
        print(f"Database already populated. Skipping population step.")

    affected_nodes = db.get_all_affected_papers()
    models = {
        'gozinto': Gozinto(G, affected_nodes),
        'fsp': FSPC(G, affected_nodes),
        'eigenvector': EigenvectorCentrality(G, affected_nodes),
        'sais': SIS(G, affected_nodes),
        'senspecificity': SpecificitySensitivity(G, affected_nodes, P_S0_given_X0=0.1, P_S1_given_X1=0.9, P_X0=0.5, P_X1=0.5),
        # Agregar otros modelos aquí
    }

    pagerank = PageRank(affected_nodes, db, G, damping_factor=0.85, max_iterations=100, tolerance=1e-6)
    pagerank.compute_relevance_values()
    average_pagerank, average_pagerank_no_lost_references = pagerank.get_results()
    print(f"Average PageRank: {average_pagerank}")
    print(f"Average PageRank no lost references: {average_pagerank_no_lost_references}")
    pagerank_results = pagerank.get_all_relevance_values()
    extrinsic_importance = {}

    for model in models:
        print(f"Calculating transmission values using {model}...")
        selected_model = models[model]
        selected_model.compute_relevance_values()
        importance_values = selected_model.get_transmission_values()
        extrinsic_importance[model] = importance_values

    extrinsic_importance['pagerank'] = {node: value['pagerank'] for node, value in pagerank_results.items()}
    # Add the extrinsic importance values to the graph
    for model, values in extrinsic_importance.items():
        for node, value in values.items():
            if 'extrinsic_importance' not in G.nodes[node]:
                G.nodes[node]['extrinsic_importance'] = {}
            G.nodes[node]['extrinsic_importance'][model] = value
    del extrinsic_importance

    save_graph(G, args.pickle_path)


if __name__ == "__main__":
    main()