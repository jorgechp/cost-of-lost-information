"""
OpenAlex Citation Network Builder
===============================

This module builds citation networks using the OpenAlex API. It fetches citation
information for academic papers and constructs a directed graph representing
citation relationships.

Features:
---------
- Fetch citation data from OpenAlex API
- Build citation networks with configurable depth
- Handle rate limiting and API errors
- Save/load networks from pickle files
- Progress tracking with tqdm

Required Dependencies:
--------------------
- networkx
- requests
- tqdm
"""

import os
import pickle
from datetime import datetime, timedelta
import networkx as nx
import requests
import time
from tqdm import tqdm
from typing import List, Dict, Optional


class OpenAlexNetwork:
    """Citation network builder using OpenAlex API."""

    def __init__(self, pickle_path: str = "citation_network.pkl"):
        """
        Initialize the network builder.

        Args:
            pickle_path: Path for saving/loading the network
        """
        self.pickle_path = pickle_path
        self.rate_limit_delay = 0.15  # seconds between requests
        self.retry_delay = 10  # seconds after rate limit
        self.max_retries = 3

    def save_graph(self, graph: nx.DiGraph) -> None:
        """
        Save the citation network to a pickle file.

        Args:
            graph: NetworkX directed graph
        """
        try:
            with open(self.pickle_path, "wb") as f:
                pickle.dump(graph, f)
            print(f"Network saved to '{self.pickle_path}'")
        except Exception as e:
            print(f"Error saving network: {str(e)}")

    def load_graph(self) -> Optional[nx.DiGraph]:
        """
        Load the citation network from a pickle file.

        Returns:
            NetworkX directed graph or None if file doesn't exist
        """
        if os.path.exists(self.pickle_path):
            try:
                with open(self.pickle_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading network: {str(e)}")
        return None

    def get_openalex_id(self, doi: str) -> Optional[str]:
        """
        Convert DOI to OpenAlex ID.

        Args:
            doi: Digital Object Identifier

        Returns:
            OpenAlex ID or None if not found
        """
        url = f"https://api.openalex.org/works/https://doi.org/{doi}"
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    return response.json()['id'].split('/')[-1]
                elif response.status_code == 429:
                    print(f"Rate limit hit (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(self.retry_delay)
                else:
                    print(f"Error {response.status_code} for DOI: {doi}")
                    return None
            except Exception as e:
                print(f"Request error: {str(e)}")
                time.sleep(self.retry_delay)
        return None

    def get_citing_works(self, openalex_id: str) -> List[Dict]:
        """
        Get all works citing a given paper.

        Args:
            openalex_id: OpenAlex ID of the paper

        Returns:
            List of citing works with their metadata
        """
        citing_works = []
        base_url = (f"https://api.openalex.org/works"
                    f"?filter=cites:{openalex_id}&per-page=200")
        cursor = "*"

        while cursor:
            url = f"{base_url}&cursor={cursor}"
            try:
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
                    time.sleep(self.rate_limit_delay)
                elif response.status_code == 429:
                    next_attempt = datetime.now() + timedelta(hours=24)
                    print(f"Rate limit exceeded. Next attempt: {next_attempt}")
                    time.sleep(86400)  # 24 hours
                else:
                    print(f"Error {response.status_code} for ID: {openalex_id}")
                    break
            except Exception as e:
                print(f"Request error: {str(e)}")
                break

        return citing_works

    def build_network(self, start_dois: List[str],
                      max_depth: Optional[int] = None) -> nx.DiGraph:
        """
        Build citation network starting from given DOIs.

        Args:
            start_dois: List of starting DOIs
            max_depth: Maximum citation depth (None for unlimited)

        Returns:
            NetworkX directed graph of citations
        """
        graph = self.load_graph() or nx.DiGraph()
        visited = set(graph.nodes)
        start_ids = []

        # Convert DOIs to OpenAlex IDs
        print("Converting DOIs to OpenAlex IDs...")
        for doi in start_dois:
            if openalex_id := self.get_openalex_id(doi):
                start_ids.append(openalex_id)
            time.sleep(self.rate_limit_delay)

        current_level = start_ids
        depth = 0
        pbar = tqdm(total=len(current_level), desc=f"Depth {depth}")

        try:
            while current_level:
                next_level = []

                for node_id in current_level:
                    if node_id in visited:
                        continue
                    visited.add(node_id)

                    citing_works = self.get_citing_works(node_id)
                    for work in citing_works:
                        if work['id'] not in graph:
                            graph.add_edge(node_id, work['id'])
                            graph.nodes[work['id']].update({
                                'title': work['title'],
                                'year': work['publication_year']
                            })
                            next_level.append(work['id'])

                    pbar.update(1)
                    self.save_graph(graph)  # Periodic save

                depth += 1
                if max_depth is not None and depth > max_depth:
                    break

                current_level = next_level
                pbar.total = pbar.n + len(current_level)
                pbar.set_description(f"Depth {depth}")

        except Exception as e:
            print(f"Error during network building: {str(e)}")
            self.save_graph(graph)  # Emergency save
            raise

        finally:
            pbar.close()

        return graph


def main():
    """Main execution function."""
    # Example DOIs
    dois = [
        "10.22605/RRH3250",
        "10.1146/annurev-ento-031616-035444",
        "10.1111/gcb.12776"
    ]

    network_builder = OpenAlexNetwork("citation_network.pkl")

    try:
        print("Building citation network...")
        graph = network_builder.build_network(dois, max_depth=2)
        network_builder.save_graph(graph)
        print("\nNetwork statistics:")
        print(f"Nodes: {graph.number_of_nodes()}")
        print(f"Edges: {graph.number_of_edges()}")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()