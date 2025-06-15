"""
Database Population Tool for Graph Data
=====================================

This module provides functionality to load graph data from pickle files and populate
a database with the graph information. It includes safeguards to prevent duplicate
population attempts.

Features:
---------
- Load NetworkX graphs from pickle files
- Save graphs to pickle files
- Database population with graph data
- Lock mechanism to prevent duplicate population
- Basic graph statistics reporting

Required Dependencies:
--------------------
- pickle: For graph serialization
- os: For file operations
- argparse: For command line interface
- NetworkX: For graph operations

Usage:
------
Command line:
    python populate.py path/to/graph.pkl
"""

import argparse
import os
import pickle
from typing import Any
from relevance.database import Database

# Path for the lock file to prevent duplicate database population
LOCK_FILE_PATH = 'tmpdata/db_populated.lock'


def load_graph(pickle_path: str) -> Any:
    """
    Load a NetworkX graph from a pickle file.

    This function deserializes a graph object stored in a pickle file.
    The graph should be a NetworkX graph object.

    Args:
        pickle_path (str): Path to the pickle file containing the graph

    Returns:
        Any: NetworkX graph object

    Raises:
        FileNotFoundError: If the pickle file doesn't exist
        pickle.UnpicklingError: If the file contains invalid pickle data
    """
    try:
        with open(pickle_path, 'rb') as f:
            graph = pickle.load(f)
        return graph
    except FileNotFoundError:
        raise FileNotFoundError(f"Graph file not found at: {pickle_path}")
    except pickle.UnpicklingError:
        raise ValueError(f"Invalid pickle file format at: {pickle_path}")


def save_graph(graph: Any, pickle_path: str) -> None:
    """
    Save a NetworkX graph to a pickle file.

    This function serializes a graph object to a pickle file for later use.
    Creates parent directories if they don't exist.

    Args:
        graph (Any): NetworkX graph object to save
        pickle_path (str): Destination path for the pickle file

    Raises:
        OSError: If there's an error creating directories or writing the file
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(pickle_path), exist_ok=True)

        # Save the graph
        with open(pickle_path, 'wb') as f:
            pickle.dump(graph, f)
    except OSError as e:
        raise OSError(f"Error saving graph to {pickle_path}: {str(e)}")


def ensure_lock_directory() -> None:
    """
    Ensure the directory for the lock file exists.

    Creates the directory structure for the lock file if it doesn't exist.

    Raises:
        OSError: If directory creation fails
    """
    try:
        os.makedirs(os.path.dirname(LOCK_FILE_PATH), exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create lock directory: {str(e)}")


def is_database_populated() -> bool:
    """
    Check if the database has already been populated.

    Returns:
        bool: True if the database has been populated (lock file exists),
              False otherwise
    """
    return os.path.exists(LOCK_FILE_PATH)


def create_lock_file() -> None:
    """
    Create a lock file to indicate database population is complete.

    Raises:
        OSError: If creating the lock file fails
    """
    try:
        ensure_lock_directory()
        with open(LOCK_FILE_PATH, 'w') as lock_file:
            lock_file.write('Database populated at: ' +
                            os.path.basename(__file__) + '\n')
    except OSError as e:
        raise OSError(f"Failed to create lock file: {str(e)}")


def print_graph_stats(graph: Any) -> None:
    """
    Print basic statistics about the graph.

    Args:
        graph (Any): NetworkX graph object
    """
    print("\nGraph Statistics:")
    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")
    if graph.number_of_nodes() > 0:
        print(f"Average degree: {2 * graph.number_of_edges() / graph.number_of_nodes():.2f}")


def main() -> None:
    """
    Main execution function for database population.

    Process:
    1. Parse command line arguments
    2. Load graph from pickle file
    3. Print graph statistics
    4. Populate database if not already done
    5. Create lock file after successful population
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Load graph data and populate database.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'pickle_path',
        type=str,
        help="Path to the pickle file containing the graph"
    )

    # Parse arguments
    args = parser.parse_args()

    try:
        # Load the graph
        print(f"Loading graph from {args.pickle_path}...")
        graph = load_graph(args.pickle_path)
        print("Graph loaded successfully.")

        # Display graph statistics
        print_graph_stats(graph)

        # Initialize database
        db = Database()
        db.create_schema()

        # Check if database is already populated
        if not is_database_populated():
            print("\nPopulating database with graph data...")
            try:
                db.populate_database(graph)
                create_lock_file()
                print("Database population completed successfully.")
            except Exception as e:
                print(f"Error during database population: {str(e)}")
                return
        else:
            print("\nDatabase already populated. Skipping population step.")

    except Exception as e:
        print(f"Error: {str(e)}")
        return

if __name__ == "__main__":
    main()