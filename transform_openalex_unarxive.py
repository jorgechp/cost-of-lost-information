import pickle
import networkx as nx


def transformar_grafo_openalex(path_entrada, path_salida):
    with open(path_entrada, "rb") as f:
        G = pickle.load(f)

    # 1. Asegurarse de que sea un DiGraph
    if not isinstance(G, nx.DiGraph):
        G = nx.DiGraph(G)

    G_transformado = nx.DiGraph()

    for nodo, datos in G.nodes(data=True):
        # Normalizar el ID si es URI
        if isinstance(nodo, str) and nodo.startswith("https://openalex.org/W"):
            nuevo_id = nodo.split("/")[-1]  # "W2031816812"
        else:
            nuevo_id = nodo

        # Inicializar datos si están vacíos
        nuevos_datos = datos.copy() if datos else {}

        # Añadir campos requeridos
        if 'sections' not in nuevos_datos:
            nuevos_datos['sections'] = [
                {'section_name': 'Introduction', 'external_uris': [], 'cited_references': []},
                {'section_name': 'Methodology', 'external_uris': [], 'cited_references': []},
                {'section_name': 'Results', 'external_uris': [], 'cited_references': []},
                {'section_name': 'Discussions/Conclusions', 'external_uris': [], 'cited_references': []}
            ]

        if 'transmission_value' not in nuevos_datos:
            nuevos_datos['transmission_value'] = {}

        if 'fspc' not in nuevos_datos:
            nuevos_datos['fspc'] = {}

        G_transformado.add_node(nuevo_id, **nuevos_datos)

    for origen, destino in G.edges:
        if isinstance(origen, str) and origen.startswith("https://openalex.org/W"):
            origen = origen.split("/")[-1]
        if isinstance(destino, str) and destino.startswith("https://openalex.org/W"):
            destino = destino.split("/")[-1]

        G_transformado.add_edge(origen, destino)

    with open(path_salida, "wb") as f:
        pickle.dump(G_transformado, f)

    print(f"Grafo transformado guardado en: {path_salida}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Transforma un grafo OpenAlex al formato esperado por los modelos")
    parser.add_argument("input", type=str, help="Ruta al .pkl con el grafo de OpenAlex")
    parser.add_argument("output", type=str, help="Ruta de salida del .pkl transformado")
    args = parser.parse_args()

    transformar_grafo_openalex(args.input, args.output)
