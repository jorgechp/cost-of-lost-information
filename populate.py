import argparse
import os
import pickle


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


if __name__ == "__main__":
    main()