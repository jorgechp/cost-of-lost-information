from urllib.parse import urlparse

import requests
import sqlite3
import os
import networkx as nx
import urllib3


class Database:
    def __init__(self):
        """
        Initialize the Database class.
        """
        self.tmpdata_dir = 'tmpdata'
        self.db_path = os.path.join(self.tmpdata_dir, 'database_transmission.sqlite')
        os.makedirs(self.tmpdata_dir, exist_ok=True)

    def transform_id(self, paper_id):
        paper_split = paper_id.split('.')
        if len(paper_split) == 2:
            while len(paper_split[0]) < 3:
                paper_split[0] = '0' + paper_split[1]
            while len(paper_split[1]) < 5:
                paper_split[1] = paper_split[1] + '0'
            return '.'.join(paper_split)
        return paper_id


    def create_schema(self):
        """
        Create the database schema with tables 'sections' and 'external_uris'.
        """
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        # Create the 'sections' table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER NOT NULL,
                section_title TEXT NOT NULL
            )
        ''')

        # Create the 'external_uris' table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS external_uris (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                section_id INTEGER NOT NULL,
                uri TEXT NOT NULL,
                is_alive BOOLEAN NOT NULL,
                FOREIGN KEY (section_id) REFERENCES sections (id)
            )
        ''')

        connection.commit()
        connection.close()

    def get_existing_paper_ids(self, remove_last=False):
        """
        Get a set of existing paper_ids from the database.

        Returns:
        set: A set of existing paper_ids.
        """
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        cursor.execute('SELECT DISTINCT paper_id FROM sections ORDER BY id ASC')
        if remove_last:
            existing_paper_ids = {str(row[0]) for row in cursor.fetchall()[:-1]}
        else:
            existing_paper_ids = {str(row[0]) for row in cursor.fetchall()}

        connection.close()
        return existing_paper_ids

    def count_articles(self):
        """
        Count the number of articles in the database.

        Returns:
        int: The number of articles.
        """
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        cursor.execute('SELECT COUNT(DISTINCT paper_id) FROM sections')
        count = cursor.fetchone()[0]

        connection.close()
        return count

    def is_uri_alive(self, uri):
        try:
            # Validate the URI
            result = urlparse(uri)
            if not all([result.scheme, result.netloc]):
                return False

            response = requests.head(uri, allow_redirects=True, timeout=15)
            return response.status_code in [200, 301, 302, 418]
        except requests.RequestException:
            return False
        except urllib3.exceptions.LocationParseError:
            return False

    def populate_database(self, graph):
        """
        Populate the database with data from the graph.

        Parameters:
        graph (networkx.Graph): The graph containing the data.
        """


        existing_paper_ids = self.get_existing_paper_ids(remove_last=True)

        article_count = 0

        for index, (node, data) in enumerate(graph.nodes(data=True)):
            if node in existing_paper_ids:
                article_count += 1
                continue

            connection = sqlite3.connect(self.db_path)
            cursor = connection.cursor()
            paper_id = node
            sections = data.get('sections', [])

            for section in sections:
                section_title = section.get('section_name', '')

                # Assign a default title if section_title is empty
                if not section_title:
                    section_title = "unnamed"

                # Check if the section already exists
                cursor.execute('''
                    SELECT id FROM sections WHERE paper_id = ? AND section_title = ?
                ''', (paper_id, section_title))
                is_exists = cursor.fetchone()

                if not is_exists:
                    cursor.execute('''
                        INSERT INTO sections (paper_id, section_title)
                        VALUES (?, ?)
                    ''', (paper_id, section_title))
                    section_id = cursor.lastrowid

                    external_uris = section.get('external_uris', [])
                    for uri in external_uris:
                        # Skip URIs that are NULL or empty
                        if not uri:
                            continue

                        is_alive = self.is_uri_alive(uri)

                        # Check if the URI already exists
                        cursor.execute('''
                            SELECT id FROM external_uris WHERE section_id = ? AND uri = ?
                        ''', (section_id, uri))
                        uri_row = cursor.fetchone()

                        if not uri_row:
                            cursor.execute('''
                                INSERT INTO external_uris (section_id, uri, is_alive)
                                VALUES (?, ?, ?)
                            ''', (section_id, uri, is_alive))
            connection.commit()
            connection.close()

            article_count += 1
            if article_count % 1000 == 0:
                print(f"Processed {article_count} articles.")

