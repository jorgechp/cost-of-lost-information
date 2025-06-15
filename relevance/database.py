"""
Academic Paper Database Management System
======================================

This module implements a SQLite-based database system for managing academic papers,
their sections, and external references.

Features:
---------
- Paper metadata storage
- Section management
- External URI tracking
- Reference status verification
- Paper ID normalization
- Batch processing capabilities

Schema:
-------
- sections: Stores paper sections
- external_uris: Stores and tracks external references

Required Dependencies:
--------------------
- sqlite3
- requests
- urllib3
- networkx
"""

import os
import sqlite3
from typing import Set, Dict, Any
import requests
import urllib3
from urllib.parse import urlparse
import networkx as nx


class Database:
    """
    Manages academic paper data storage and retrieval.
    """

    def __init__(self, base_dir: str = 'tmpdata'):
        """
        Initialize database.

        Args:
            base_dir: Base directory for database storage
        """
        self.tmpdata_dir = base_dir
        self.db_path = os.path.join(self.tmpdata_dir, 'database_transmission.sqlite')
        os.makedirs(self.tmpdata_dir, exist_ok=True)
        self.create_schema()

    def create_schema(self) -> None:
        """Create database tables and schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Sections table
            cursor.execute('''
                           CREATE TABLE IF NOT EXISTS sections (
                                                                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                                   paper_id TEXT NOT NULL,
                                                                   section_title TEXT NOT NULL
                           )
                           ''')

            # External URIs table
            cursor.execute('''
                           CREATE TABLE IF NOT EXISTS external_uris (
                                                                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                                        section_id INTEGER NOT NULL,
                                                                        uri TEXT NOT NULL,
                                                                        is_alive BOOLEAN NOT NULL,
                                                                        FOREIGN KEY (section_id) REFERENCES sections (id)
                               )
                           ''')

    def transform_id(self, paper_id: str) -> str:
        """
        Normalize paper ID format.

        Args:
            paper_id: Original paper ID

        Returns:
            Normalized paper ID
        """
        parts = paper_id.split('.')
        if len(parts) != 2:
            return paper_id

        left, right = parts
        # Pad numbers
        left = left.zfill(4)
        right = right.rjust(5, '0')

        return f"{left}.{right}"

    def transform_all_paper_ids(self) -> None:
        """Transform all paper IDs in database to normalized format."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get all paper IDs
            cursor.execute('SELECT DISTINCT paper_id FROM sections')
            paper_ids = cursor.fetchall()

            # Transform each ID
            for (paper_id,) in paper_ids:
                new_id = self.transform_id(paper_id)
                if new_id != paper_id:
                    cursor.execute('''
                                   UPDATE sections
                                   SET paper_id = ?
                                   WHERE paper_id = ?
                                   ''', (new_id, paper_id))

    def is_uri_alive(self, uri: str) -> bool:
        """
        Check if URI is accessible.

        Args:
            uri: URI to check

        Returns:
            True if URI is accessible, False otherwise
        """
        try:
            # Validate URI format
            result = urlparse(uri)
            if not all([result.scheme, result.netloc]):
                return False

            # Check URI accessibility
            response = requests.head(
                uri,
                allow_redirects=True,
                timeout=15
            )
            return response.status_code in [200, 301, 302, 418]

        except (requests.RequestException, urllib3.exceptions.LocationParseError):
            return False

    def populate_database(self, graph: nx.Graph) -> None:
        """
        Populate database from graph data.

        Args:
            graph: NetworkX graph containing paper data
        """
        existing_ids = self.get_existing_paper_ids(remove_last=True)
        processed = 0

        for node, data in graph.nodes(data=True):
            if node in existing_ids:
                processed += 1
                continue

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                for section in data.get('sections', []):
                    # Insert section
                    section_title = section.get('section_name') or "unnamed"
                    cursor.execute('''
                                   INSERT INTO sections (paper_id, section_title)
                                   VALUES (?, ?)
                                   ''', (node, section_title))
                    section_id = cursor.lastrowid

                    # Insert URIs
                    for uri in section.get('external_uris', []):
                        if not uri:
                            continue

                        is_alive = self.is_uri_alive(uri)
                        cursor.execute('''
                                       INSERT INTO external_uris (section_id, uri, is_alive)
                                       VALUES (?, ?, ?)
                                       ''', (section_id, uri, is_alive))

            processed += 1
            if processed % 1000 == 0:
                print(f"Processed {processed} articles")

    def get_paper_affectation(self, paper_id: str) -> Dict[str, Dict[str, bool]]:
        """
        Get paper's sections and their reference status.

        Args:
            paper_id: Paper ID

        Returns:
            Dictionary of sections and their references
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                           SELECT
                               sections.section_title,
                               external_uris.uri,
                               external_uris.is_alive
                           FROM sections
                                    JOIN external_uris ON sections.id = external_uris.section_id
                           WHERE sections.paper_id = ?
                           ''', (paper_id,))

            results = {}
            for title, uri, status in cursor.fetchall():
                if title not in results:
                    results[title] = {}
                results[title][uri] = status

            return results

    def get_all_affected_papers(self) -> Set[str]:
        """
        Get papers with dead references.

        Returns:
            Set of affected paper IDs
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                           SELECT DISTINCT sections.paper_id
                           FROM sections
                                    JOIN external_uris ON sections.id = external_uris.section_id
                           WHERE external_uris.is_alive = 0
                              OR external_uris.is_alive IS NULL
                           ''')

            return {row[0] for row in cursor.fetchall()}

    def count_citing_papers(self,
                            count_only_all_alive_references: bool = False,
                            count_only_all_dead_references: bool = False) -> int:
        """
        Count papers based on reference status.

        Args:
            count_only_all_alive_references: Count papers with all references alive
            count_only_all_dead_references: Count papers with any dead references

        Returns:
            Number of papers matching criteria
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if count_only_all_alive_references:
                query = '''
                        SELECT COUNT(DISTINCT s.paper_id)
                        FROM sections s
                                 JOIN external_uris e ON s.id = e.section_id
                        GROUP BY s.paper_id
                        HAVING SUM(CASE WHEN e.is_alive = 0
                            OR e.is_alive IS NULL
                                            THEN 1 ELSE 0 END) = 0 \
                        '''
            elif count_only_all_dead_references:
                query = '''
                        SELECT COUNT(DISTINCT s.paper_id)
                        FROM sections s
                                 JOIN external_uris e ON s.id = e.section_id
                        GROUP BY s.paper_id
                        HAVING SUM(CASE WHEN e.is_alive = 0
                            OR e.is_alive IS NULL
                                            THEN 1 ELSE 0 END) > 0 \
                        '''
            else:
                query = '''
                        SELECT COUNT(DISTINCT s.paper_id)
                        FROM sections s
                                 JOIN external_uris e ON s.id = e.section_id \
                        '''

            cursor.execute(query)
            return cursor.fetchone()[0] or 0

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            'total_papers': self.count_total_articles(),
            'affected_papers': len(self.get_all_affected_papers()),
            'citing_papers': self.count_citing_papers(),
            'papers_all_alive': self.count_citing_papers(
                count_only_all_alive_references=True
            ),
            'papers_any_dead': self.count_citing_papers(
                count_only_all_dead_references=True
            )
        }


def main():
    """Example usage of the database system."""
    # Initialize database
    db = Database()

    # Print statistics
    stats = db.get_statistics()
    print("\nDatabase Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()