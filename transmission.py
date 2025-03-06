import sqlite3
import os
import json
import argparse
import networkx as nx
from models.pagerank import PageRank
from models.gozinto import Gozinto

def initialize_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Crear tabla de artículos
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            paper_id TEXT PRIMARY KEY,
            article TEXT
        )
    ''')

    # Crear tabla de valores de modelos
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_values (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT,
            model_name TEXT,
            value REAL,
            FOREIGN KEY (paper_id) REFERENCES articles (paper_id)
        )
    ''')

    # Crear tabla de metadatos
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')

    conn.commit()
    return conn

def save_article_to_db(conn, paper_id, article):
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO articles (paper_id, article)
        VALUES (?, ?)
    ''', (paper_id, json.dumps(article)))
    conn.commit()

def process_subset(subset_dir, db_path):
    conn = initialize_db(db_path)
    cursor = conn.cursor()

    # Verificar si los artículos ya han sido insertados
    cursor.execute('SELECT value FROM metadata WHERE key = "initialized"')
    result = cursor.fetchone()
    if result and result[0] == 'true':
        print("La base de datos ya ha sido inicializada.")
        conn.close()
        return

    paper_count = 0
    for root, _, files in os.walk(subset_dir):
        for file in files:
            if file.endswith('.jsonl'):
                with open(os.path.join(root, file), 'r') as f:
                    for line in f:
                        article = json.loads(line)
                        paper_id = article.get("paper_id")
                        save_article_to_db(conn, paper_id, article)
                        paper_count += 1

    # Marcar la base de datos como inicializada
    cursor.execute('''
        INSERT OR REPLACE INTO metadata (key, value)
        VALUES ("initialized", "true")
    ''')
    conn.commit()
    conn.close()

def save_model_value_to_db(conn, paper_id, model_name, value):
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO model_values (paper_id, model_name, value)
        VALUES (?, ?, ?)
    ''', (paper_id, model_name, value))
    conn.commit()

def calculate_and_save_model_values(conn, models):
    cursor = conn.cursor()
    cursor.execute('SELECT paper_id, article FROM articles')
    while True:
        rows = cursor.fetchmany(1000)  # Procesar en lotes de 1000 filas
        if not rows:
            break

        for paper_id, article_json in rows:
            article = json.loads(article_json)
            for model_name, model in models.items():
                value = model.get_transmission_value(paper_id)
                save_model_value_to_db(conn, paper_id, model_name, value)

def build_graph_from_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT paper_id, article FROM articles')
    graph = nx.DiGraph()

    while True:
        rows = cursor.fetchmany(1000)  # Procesar en lotes de 1000 filas
        if not rows:
            break

        for paper_id, article_json in rows:
            article = json.loads(article_json)
            graph.add_node(paper_id)
            sections = article.get("sections", [])
            for section in sections:
                paragraphs = section.get("paragraphs", [])
                for paragraph in paragraphs:
                    cited_references = paragraph.get("cited_references", [])
                    for ref in cited_references:
                        ref_id = ref.get("arxiv_id")
                        if ref_id:  # Ensure the reference is not empty
                            graph.add_edge(paper_id, ref_id)

    conn.close()
    return graph

def main():
    parser = argparse.ArgumentParser(description="Process a subset of papers and calculate model values.")
    parser.add_argument('subset_directory', type=str, help="The directory containing the subset JSONL files.")
    parser.add_argument('db_path', type=str, help="The path to the SQLite database file.")

    args = parser.parse_args()

    # Procesar el subdirectorio y guardar artículos en la base de datos
    process_subset(args.subset_directory, args.db_path)

    # Inicializar modelos
    graph = build_graph_from_db(args.db_path)
    models = {
        'pagerank': PageRank(graph),
        'gozinto': Gozinto(graph),
        # Agregar otros modelos aquí
    }

    # Calcular y guardar valores de modelos
    conn = sqlite3.connect(args.db_path)
    calculate_and_save_model_values(conn, models)
    conn.close()

if __name__ == "__main__":
    main()