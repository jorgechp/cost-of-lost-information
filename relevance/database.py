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

    def transform_all_paper_ids(self):
        """
        Transforms all paper_ids in the database to the specified format.
        """
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        cursor.execute('SELECT DISTINCT paper_id FROM sections')
        paper_ids = cursor.fetchall()

        for (paper_id,) in paper_ids:
            if '.' in paper_id:
                left, right = paper_id.split('.')
                while right[0] == '0':
                    right = right[1:]
                if len(left) < 4:
                    left = left.zfill(4)
                if len(right) < 5:
                    right = right.ljust(5, '0')
                new_name = f"{left}.{right}"

                cursor.execute('''
                    UPDATE sections
                    SET paper_id = ?
                    WHERE paper_id = ?
                ''', (new_name, paper_id))

        connection.commit()
        connection.close()

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

    def count_total_articles(self):
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

    def is_paper_affected(self, paper_id):
        """
        Check if any of the references in any section of the paper are not alive.

        Parameters:
        paper_id (str): The ID of the paper.

        Returns:
        bool: True if any reference is not alive, False otherwise.
        """
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        cursor.execute('''
            SELECT external_uris.is_alive
            FROM sections
            JOIN external_uris ON sections.id = external_uris.section_id
            WHERE sections.paper_id = ?
        ''', (paper_id,))
        rows = cursor.fetchall()

        connection.close()

        return any(not is_alive for (is_alive,) in rows)

    def get_paper_affectation(self, paper_id):
        """
        Get a dictionary of sections of the paper, and for each section, a dictionary of references and their is_alive status.

        Parameters:
        paper_id (str): The ID of the paper.

        Returns:
        dict: A dictionary of sections and their references with is_alive status.
        """
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        cursor.execute('''
            SELECT sections.section_title, external_uris.uri, external_uris.is_alive
            FROM sections
            JOIN external_uris ON sections.id = external_uris.section_id
            WHERE sections.paper_id = ?
        ''', (paper_id,))
        rows = cursor.fetchall()

        connection.close()

        paper_affectation = {}
        for section_title, uri, is_alive in rows:
            if section_title not in paper_affectation:
                paper_affectation[section_title] = {}
            paper_affectation[section_title][uri] = is_alive

        return paper_affectation

    def get_all_affected_papers(self):
        """
        Get a set of IDs of affected nodes (where is_alive is False or null).

        Returns:
        set: A set of IDs of affected nodes.
        """
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        cursor.execute('''
            SELECT DISTINCT sections.paper_id
            FROM sections
            JOIN external_uris ON sections.id = external_uris.section_id
            WHERE external_uris.is_alive = 0 OR external_uris.is_alive IS NULL
        ''')
        rows = cursor.fetchall()

        connection.close()

        return {row[0] for row in rows}

    def get_all_references(self, paper_id):
        """
        Get all references for a given paper_id.

        Parameters:
        paper_id (str): The ID of the paper.

        Returns:
        list: A list of references for the given paper_id.
        """
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        cursor.execute('''
            SELECT external_uris.uri
            FROM sections
            JOIN external_uris ON sections.id = external_uris.section_id
            WHERE sections.paper_id = ?
        ''', (paper_id,))
        rows = cursor.fetchall()

        connection.close()

        return [row[0] for row in rows]

    def _count_citing_papers(self, query):
        """
        Cuenta el número total de papers que citan (tienen una external_uri).

        Returns:
        int: El número total de papers que citan.
        """
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        # Contar el número de papers que tienen una external_uri enlazando con sections y papers
        cursor.execute('''
            SELECT COUNT(DISTINCT s.paper_id)
            FROM SECTIONS s JOIN EXTERNAL_URIS e ON s.id = e.section_id
        ''')
        citing_papers_count = cursor.fetchone()[0]

        connection.close()
        return citing_papers_count

    def count_citing_papers(self, count_only_all_alive_references=False, count_only_all_dead_references=False):
        if count_only_all_alive_references:
            query = '''
                SELECT COUNT(DISTINCT s.paper_id)
                FROM sections s
                JOIN external_uris e ON s.id = e.section_id
                GROUP BY s.paper_id
                HAVING SUM(CASE WHEN e.is_alive = 0 OR e.is_alive IS NULL THEN 1 ELSE 0 END) = 0
            '''
        elif count_only_all_dead_references:
            query = '''
                SELECT COUNT(DISTINCT s.paper_id)
                FROM sections s
                JOIN external_uris e ON s.id = e.section_id
                GROUP BY s.paper_id
                HAVING SUM(CASE WHEN e.is_alive = 0 OR e.is_alive IS NULL THEN 1 ELSE 0 END) > 0
            '''
        else:
            query = '''
            SELECT COUNT(DISTINCT s.paper_id)
            FROM SECTIONS s 
            JOIN EXTERNAL_URIS e ON s.id = e.section_id
            '''

        return self._count_citing_papers(query)

    def get_db_path(self):
        return self.db_path

