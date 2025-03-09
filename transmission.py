import argparse
import os
import json
import concurrent.futures
import threading
import networkdisk as nd

lock = threading.Lock()
node_count = 0

def process_file(file_path, graph):
    global node_count
    with open(file_path, 'r') as f:
        for line in f:
            article = json.loads(line)
            paper_id = article.get("paper_id")

            with lock:
                # Comprobar si el nodo del paper ya existe y tiene los atributos necesarios
                if not graph.has_node(paper_id) or not all(
                        attr in graph.nodes[paper_id] for attr in ['title', 'year', 'authors', 'doi']):
                    title = article.get("title")
                    year = article.get("year")
                    authors = article.get("authors")
                    doi = article.get("doi")

                    # Añadir nodo del artículo con atributos
                    graph.add_node(paper_id, title=title, year=year, authors=authors, doi=doi)
                    node_count += 1

                    # Imprimir marca cada 10,000 nodos
                    if node_count % 10000 == 0:
                        print(f"Se han procesado {node_count} nodos.")

            sections = article.get("sections", [])
            for section in sections:
                section_name = section.get("section_name")
                section_id = f"{paper_id}_{section_name}"

                with lock:
                    # Añadir nodo de la sección con atributos
                    graph.add_node(section_id, paper_id=paper_id, section_name=section_name)
                    node_count += 1

                    # Imprimir marca cada 10,000 nodos
                    if node_count % 10000 == 0:
                        print(f"Se han procesado {node_count} nodos.")

                # Aquí puedes calcular y guardar el peso de la sección
                weight = calculate_section_weight(section)
                graph.nodes[section_id]['weight'] = weight

                paragraphs = section.get("paragraphs", [])
                for paragraph in paragraphs:
                    cited_references = paragraph.get("cited_references", [])
                    for ref in cited_references:
                        ref_id = ref.get("arxiv_id")
                        if ref_id:
                            with lock:
                                graph.add_edge(paper_id, ref_id)

                    external_uris = paragraph.get("external_uris", [])
                    for uri in external_uris:
                        with lock:
                            graph.add_node(uri, section_id=section_id)
                            node_count += 1

                            # Imprimir marca cada 10,000 nodos
                            if node_count % 10000 == 0:
                                print(f"Se han procesado {node_count} nodos.")

            # Liberar memoria eliminando referencias a objetos grandes
            del article
            del sections
            del paragraphs

def process_subset(subset_dir, graph_db_path):
    marker_file = ".dataset_finished"

    # Verificar si el archivo de marca existe
    if os.path.exists(marker_file):
        print("El dataset ya ha sido procesado anteriormente.")
        return

    # Crear un grafo en disco
    graph = nd.sqlite.Graph(db=graph_db_path)

    files = [os.path.join(root, file)
             for root, _, files in os.walk(subset_dir)
             for file in files if file.endswith('.jsonl')]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, file, graph) for file in files]
        for future in concurrent.futures.as_completed(futures):
            future.result()  # Para capturar excepciones si las hay

    # Crear el archivo de marca
    with open(marker_file, 'w') as f:
        f.write("Dataset procesado")

    # Cerrar el grafo para liberar recursos
    graph.close()

def calculate_section_weight(section):
    # Implementa la lógica para calcular el peso de la sección
    return 1.0  # Ejemplo de peso fijo

def calculate_and_save_model_values(graph, models):
    for paper_id in graph.nodes:
        if 'title' in graph.nodes[paper_id]:  # Verificar si es un nodo de artículo
            for model_name, model in models.items():
                value = model.get_transmission_value(paper_id)
                graph.nodes[paper_id][model_name] = value

def main():
    parser = argparse.ArgumentParser(description="Process a subset of papers and calculate model values.")
    parser.add_argument('subset_directory', type=str, help="The directory containing the subset JSONL files.")
    parser.add_argument('graph_db_path', type=str, help="The path to the graph database file.")

    args = parser.parse_args()

    # Procesar el subdirectorio y guardar artículos en el grafo
    process_subset(args.subset_directory, args.graph_db_path)

    # Inicializar modelos
    graph = nd.DiGraph(args.graph_db_path)
    models = {
        'pagerank': PageRank(graph),
        'gozinto': Gozinto(graph),
        # Agregar otros modelos aquí
    }

    # Calcular y guardar valores de modelos
    calculate_and_save_model_values(graph, models)

    # Cerrar el grafo para liberar recursos
    graph.close()

if __name__ == "__main__":
    main()