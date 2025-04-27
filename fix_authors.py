import copy
import os
import json
import pickle
import arxiv
import networkx as nx
from tqdm import tqdm

UNARXIVE_PATH = "unarXive_230324_open_subset"  # Cambiar por tu ruta
GRAPH_PATH = "output/unarXive_230324_open_subset__text_arxiv_only.pickle"  # Cambiar por tu ruta

TMP_AUTHOR_DICT_PATH = "tmp/author_dict.pkl"  # Ruta para guardar el diccionario temporal

def save_author_dict(author_dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(author_dict, f)

def load_author_dict(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None
def parse_authors(authors_parsed):
    authors = []
    for entry in authors_parsed:
        # ["LastName", "FirstName", "MiddleName"] → "FirstName MiddleName LastName"
        parts = [entry[1], entry[2] if len(entry) > 2 else '', entry[0]]
        author_str = ' '.join(p for p in parts if p)
        authors.append(author_str)
    return authors

def build_author_dict_from_jsonl(dataset_path):
    author_dict = {}
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.jsonl'):
                jsonl_path = os.path.join(root, file)
                with open(jsonl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            arxiv_id = data.get("paper_id")
                            metadata = data.get("metadata", [])
                            if len(metadata) > 0:
                                parsed_authors = metadata.get("authors_parsed")
                                if arxiv_id and parsed_authors:
                                    author_dict[arxiv_id] = parse_authors(parsed_authors)
                        except json.JSONDecodeError as e:
                            print(f"Error parsing line in {jsonl_path}: {e}")
    return author_dict

def fetch_authors_from_arxiv(arxiv_id):
    try:
        search = arxiv.Search(id_list=[arxiv_id])
        for result in search.results():
            return [author.name for author in result.authors]
    except Exception as e:
        print(f"Error consultando {arxiv_id} en arxiv: {e}")
    return []

def update_graph_with_authors(graph_path, author_dict):
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)

    updated = False
    i = 0
    for node in tqdm(G.nodes, desc="Procesando nodos del grafo"):
        if 'authors' not in G.nodes[node] or not G.nodes[node]['authors']:

            authors = author_dict.get(node)
            base_node = copy.copy(node)
            if not authors:
                node = node if node[-1] != '0' else node[:-1]
                authors = author_dict.get(node)
                if not authors:
                    authors = fetch_authors_from_arxiv(node)
            if authors:
                G.nodes[base_node]['authors'] = authors
                updated = True
        i += 1
        if i % 100 == 0 and updated:
            print(f"Procesados {i} nodos...")
            with open(GRAPH_PATH, 'wb') as f:
                pickle.dump(G, f)
                updated = False
    return G, updated

if __name__ == "__main__":
    print("Construyendo diccionario de autores desde UnarXive JSONL...")
    author_dict = load_author_dict(TMP_AUTHOR_DICT_PATH)
    if author_dict is None:
        print("No se encontró el diccionario temporal. Creando uno nuevo...")
        author_dict = build_author_dict_from_jsonl(UNARXIVE_PATH)
        save_author_dict(author_dict, TMP_AUTHOR_DICT_PATH)

    print("Cargando y actualizando grafo...")
    G, updated = update_graph_with_authors(GRAPH_PATH, author_dict)

    if updated:
        print("Guardando grafo actualizado...")
        with open(GRAPH_PATH, 'wb') as f:
            pickle.dump(G, f)
        print("Grafo actualizado y guardado.")
    else:
        print("No se realizaron actualizaciones en el grafo.")
