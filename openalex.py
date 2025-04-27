import requests
import time
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

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

# Function to get OpenAlex ID from DOI
def get_openalex_id(doi):
    url = f"https://api.openalex.org/works/https://doi.org/{doi}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['id'].split('/')[-1]
    else:
        print(f"Not found in OpenAlex: {doi}")
        return None

# Function to retrieve all works citing a given OpenAlex ID (with pagination)
def get_citing_works(openalex_id):
    citing_works = []
    base_url = f"https://api.openalex.org/works?filter=cites:{openalex_id}&per-page=200"
    cursor = "*"

    while cursor:
        url = f"{base_url}&cursor={cursor}"
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
            time.sleep(0.15)
        else:
            print(f"Error fetching citations for ID: {openalex_id}")
            break

    return citing_works

# Build the citation network
def build_citation_network(start_dois, max_depth=None):
    G = nx.DiGraph()
    visited = set()

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

            citing_works = get_citing_works(node_id)
            for work in citing_works:
                G.add_edge(node_id, work['id'])
                G.nodes[work['id']]['title'] = work['title']
                G.nodes[work['id']]['year'] = work['publication_year']
                next_level.append(work['id'])

            pbar.update(1)

        depth += 1
        if max_depth is not None and depth > max_depth:
            break

        current_level = next_level
        pbar.total = pbar.n + len(current_level)
        pbar.set_description(f"Depth {depth}")

    pbar.close()
    return G

# Function to quickly plot the graph
def plot_graph(G, max_nodes=200):
    plt.figure(figsize=(12, 8))
    if len(G) > max_nodes:
        H = G.subgraph(list(G.nodes)[:max_nodes])
    else:
        H = G
    pos = nx.spring_layout(H)
    nx.draw(H, pos, node_size=20, arrows=True, with_labels=False)
    plt.title("Quick visualization of the citation network")
    plt.show()

# Main execution
if __name__ == "__main__":
    graph = build_citation_network(dois, max_depth=None)

    # Save the graph in GraphML format
    nx.write_graphml(graph, "citation_network.graphml")
    print("\nCitation network saved as 'citation_network.graphml'.")

    # Save the graph in Pickle format
    with open("citation_network.pkl", "wb") as f:
        pickle.dump(graph, f)
    print("Citation network saved as 'citation_network.pkl'.")

    # Display a quick visualization
    plot_graph(graph)
