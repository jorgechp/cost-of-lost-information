import os
import pickle
from datetime import datetime, timedelta

import networkx as nx
import requests
import time
from tqdm import tqdm

PICKLE_PATH = "citation_network.pkl"

# Function to save the graph to a pickle file
def save_graph(graph, path):
    with open(path, "wb") as f:
        pickle.dump(graph, f)
    print(f"Graph saved to '{path}'.")

# Function to load the graph from a pickle file
def load_graph(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

# Function to get OpenAlex ID from DOI
def get_openalex_id(doi):
    url = f"https://api.openalex.org/works/https://doi.org/{doi}"
    while True:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data['id'].split('/')[-1]
        elif response.status_code == 429:
            print("Rate limit exceeded. Retrying after a delay...")
            time.sleep(10)  # Wait before retrying
        else:
            print(f"Error {response.status_code} for DOI: {doi}")
            return None

# Function to retrieve all works citing a given OpenAlex ID
def get_citing_works(openalex_id):
    citing_works = []
    base_url = f"https://api.openalex.org/works?filter=cites:{openalex_id}&per-page=200"
    cursor = "*"

    while cursor:
        url = f"{base_url}&cursor={cursor}"
        while True:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                for item in data['results']:
                    citing_works.append({
                        "id": item['id'].split('/')[-1],
                        "title": item.get('title', 'No title'),
                        "publication_year": item.get('publication_year', 'Unknown')
                    })
                cursor = data['meta'].get('next_cursor')
                break
            elif response.status_code == 429:
                next_attempt_time = datetime.now() + timedelta(hours=24)
                print(
                    f"Rate limit exceeded. Retrying after 24"
                    f" hours at {next_attempt_time.strftime('%Y-%m-%d %H:%M:%S')}...")
                time.sleep(86400)  # Wait for 24 hours (24 * 60 * 60 seconds)
            else:
                print(f"Error {response.status_code} while fetching citations for ID: {openalex_id}")
                cursor = None
                break

    return citing_works

# Build the citation network
def build_citation_network(start_dois, graph, max_depth=None):
    visited = set(graph.nodes)
    start_ids = []

    for doi in start_dois:
        openalex_id = get_openalex_id(doi)
        if openalex_id:
            start_ids.append(openalex_id)
        time.sleep(0.15)

    current_level = start_ids
    depth = 0

    pbar = tqdm(total=len(current_level), desc=f"Depth {depth}")

    while current_level:
        next_level = []

        for node_id in current_level:
            if node_id in visited:
                continue
            visited.add(node_id)

            try:
                citing_works = get_citing_works(node_id)
                for work in citing_works:
                    if work['id'] not in graph:
                        graph.add_edge(node_id, work['id'])
                        graph.nodes[work['id']]['title'] = work['title']
                        graph.nodes[work['id']]['year'] = work['publication_year']
                        next_level.append(work['id'])

                pbar.update(1)
            except Exception as e:
                print(f"Exception occurred: {e}")
                save_graph(graph, PICKLE_PATH)
                print("Graph saved before exiting.")
                raise

        depth += 1
        if max_depth is not None and depth > max_depth:
            break

        current_level = next_level
        pbar.total = pbar.n + len(current_level)
        pbar.set_description(f"Depth {depth}")

    pbar.close()
    return graph

# Main execution
if __name__ == "__main__":
    # Load or initialize the graph
    graph = load_graph(PICKLE_PATH)
    if graph is None:
        print("No existing graph found. Creating a new one.")
        graph = nx.DiGraph()

    # List of DOIs for lost datasets
    dois = [
        "10.22605/RRH3250",
        "10.1146/annurev-ento-031616-035444",
        "10.1111/gcb.12776",
        "10.1038/s42003-020-01324-2",
        "10.1371/journal.pone.0050950",
        "10.1155/2022/1588638",
        "10.1186/s13104-019-4594-4",
        "10.1371/journal.pone.0166403",
        "10.1111/gcb.14768",
        "10.1109/JSTARS.2022.3164771"
    ]

    try:
        graph = build_citation_network(dois, graph, max_depth=None)
        save_graph(graph, PICKLE_PATH)
        print("Graph successfully built and saved.")
    except Exception as e:
        print(f"Program terminated due to an exception: {e}")