import argparse
import json
import os
import re
import sqlite3
from collections import defaultdict
from urllib.parse import urlparse

import requests

EPSILON = 1e-80
RSTRIP_URL_CHARACTERS = '),].'


def initialize_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            paper_id TEXT PRIMARY KEY,
            article TEXT,
            infection_level REAL
        )
    ''')
    conn.commit()
    return conn


def save_article_to_db(conn, paper_id, article, infection_level=-1):
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO articles (paper_id, article, infection_level)
        VALUES (?, ?, ?)
    ''', (paper_id, json.dumps(article), infection_level))
    conn.commit()


def get_article_from_db(conn, paper_id):
    cursor = conn.cursor()
    cursor.execute('SELECT article, infection_level FROM articles WHERE paper_id = ?', (paper_id,))
    row = cursor.fetchone()
    if row:
        return json.loads(row[0]), row[1]
    return None, None


def update_infection_level_in_db(conn, paper_id, infection_level):
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE articles
        SET infection_level = ?
        WHERE paper_id = ?
    ''', (infection_level, paper_id))
    conn.commit()


def fix_uri(uri):
    global RSTRIP_URL_CHARACTERS
    try:
        match = re.match(r'([a-zA-Z]+://[^\]]+)\](\1)', uri)
        parsed_uri = match.group(1) if match else uri
        parsed_uri = urlparse(parsed_uri.rstrip(RSTRIP_URL_CHARACTERS))
        fixed_uri = parsed_uri._replace(path=parsed_uri.path.rstrip(RSTRIP_URL_CHARACTERS)).geturl()
        return fixed_uri
    except ValueError:
        print("Error in fix_uri: ", uri)


def is_uri_alive(uri):
    try:
        response = requests.get(uri, allow_redirects=True)
        return response.status_code in [200, 418]
    except requests.RequestException:
        return False


def check_reference_status(uri, cache):
    if uri in cache:
        return cache[uri]
    else:
        status = is_uri_alive(uri)
        cache[uri] = status
        return status


def extract_internal_references(article):
    references = defaultdict(list)
    paper_id = article.get("paper_id")
    for section in article.get("sections", []):
        section_number = section.get("section_number")
        for paragraph in section.get("paragraphs", []):
            references[section_number].extend([r['arxiv_id'] for r in paragraph.get("cited_references", [])
                                               if r['arxiv_id'] != paper_id])
    return references


def extract_external_references(article):
    uris = defaultdict(list)
    for section in article.get("sections", []):
        section_number = section.get("section_number")
        for paragraph in section.get("paragraphs", []):
            uris[section_number].extend(paragraph.get("external_uris", []))
    return uris


def ltm(internal_references, cache, conn):
    global EPSILON
    value = 0
    theta = 0.5
    weight = 1 / (len(internal_references) + EPSILON)

    for reference in internal_references:
        article, infection_level = get_article_from_db(conn, reference)

        value = value + weight * get_infection_state(article, cache, conn)
    return 1 if value >= theta else 0


def check_section_status(external_references, internal_references, cache, conn):
    value_external = 0
    for reference in external_references:
        fixed_ref = fix_uri(reference)
        if not check_reference_status(fixed_ref, cache):
            value_external = 1

    value_internal = ltm(internal_references, cache, conn)
    return max(value_external, value_internal)


def get_infection_state(article, cache, conn):
    if article is None:
        return 0

    paper_id = article.get("paper_id")
    _, infection_level = get_article_from_db(conn, paper_id)

    if infection_level > -1:
        return infection_level
    else:
        internal_references = extract_internal_references(article)
        external_references = extract_external_references(article)
        value = 0
        for section in article.get("sections", []):
            weight = 1
            section_number = section.get("section_number")
            value += weight * check_section_status(external_references[section_number],
                                                   internal_references[section_number], cache, conn)
        update_infection_level_in_db(conn, paper_id, value)
        return value


def process_subset(subset_dir, output_path, db_path):
    cache = {}
    results = []
    conn = initialize_db(db_path)
    paper_count = 0
    max_papers = 100

    for root, _, files in os.walk(subset_dir):
        for file in files:
            if file.endswith('.jsonl'):
                with open(os.path.join(root, file), 'r') as f:
                    for line in f:
                        if paper_count >= max_papers:
                            break
                        article = json.loads(line)
                        paper_id = article.get("paper_id")
                        save_article_to_db(conn, paper_id, article)
                        paper_count += 1
            if paper_count >= max_papers:
                break
        if paper_count >= max_papers:
            break

    cursor = conn.cursor()
    cursor.execute('SELECT paper_id FROM articles')
    paper_ids = cursor.fetchall()
    for paper_id in paper_ids:
        paper_id = paper_id[0]
        article, _ = get_article_from_db(conn, paper_id)
        infection_level = get_infection_state(article, cache, conn)
        results.append({"paper_id": paper_id, "infection_level": infection_level})

    with open(output_path, 'w') as out_f:
        json.dump(results, out_f, indent=4)

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Apply the infection model to a subset of papers.")
    parser.add_argument('subset_directory', type=str, help="The directory containing the subset JSONL files.")
    parser.add_argument('output_path', type=str, help="The path to save the results.")
    parser.add_argument('db_path', type=str, help="The path to the SQLite database file.")

    args = parser.parse_args()

    process_subset(args.subset_directory, args.output_path, args.db_path)


if __name__ == "__main__":
    main()
