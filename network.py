"""
Citation Network Builder
======================

This module builds and exports citation networks from academic paper data stored
in JSONL format. It processes paper metadata and citation relationships,
creating a directed graph representation of the citation network.

Features:
---------
- Build citation networks from JSONL files
- Process paper metadata and section information
- Support multiple export formats
- Handle external references and URIs
- Optional removal of incomplete entries

Required Dependencies:
--------------------
- networkx
- json
"""

import argparse
import json
import os
import networkx as nx
import pickle
from typing import Dict, Optional


class CitationNetworkBuilder:
    """Builder for citation networks from academic paper data."""

    def __init__(self, remove_empty_nodes: bool = False):
        """
        Initialize the network builder.

        Args:
            remove_empty_nodes: Whether to remove incomplete entries
        """
        self.graph = nx.DiGraph()
        self.remove_empty_nodes = remove_empty_nodes

    def process_paper(self, paper: Dict) -> None:
        """
        Process a single paper and add it to the network.

        Args:
            paper: Dictionary containing paper metadata
        """
        paper_id = paper.get("paper_id")
        if not paper_id:
            return

        # Add basic paper metadata
        self.graph.add_node(
            paper_id,
            title=paper.get("title"),
            year=paper.get("year"),
            authors=paper.get("authors")
        )

        # Process sections and citations
        sections = []
        for section in paper.get("sections", []):
            section_data = self._process_section(paper_id, section)
            if section_data:
                sections.append(section_data)

        self.graph.nodes[paper_id]['sections'] = sections

    def _process_section(self, paper_id: str, section: Dict) -> Optional[Dict]:
        """
        Process a paper section and its citations.

        Args:
            paper_id: ID of the paper
            section: Section metadata

        Returns:
            Processed section data or None if invalid
        """
        section_name = section.get("section_name")
        if not section_name:
            return None

        external_uris = set()
        cited_references = set()

        for paragraph in section.get("paragraphs", []):
            external_uris.update(paragraph.get("external_uris", []))

            for ref in paragraph.get("cited_references", []):
                if arxiv_id := ref.get("arxiv_id"):
                    cited_references.add(arxiv_id)
                    self.graph.add_edge(paper_id, arxiv_id)

        return {
            "section_name": section_name,
            "external_uris": list(external_uris),
            "cited_references": list(cited_references)
        }

    def build_from_directory(self, directory: str) -> nx.DiGraph:
        """
        Build citation network from all JSONL files in a directory.

        Args:
            directory: Path to directory containing JSONL files

        Returns:
            Built citation network as NetworkX DiGraph

        Raises:
            FileNotFoundError: If directory doesn't exist
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")

        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.jsonl'):
                    self._process_file(os.path.join(root, file))

        if self.remove_empty_nodes:
            self._clean_network()

        return self.graph

    def _process_file(self, filepath: str) -> None:
        """
        Process a single JSONL file.

        Args:
            filepath: Path to JSONL file
        """
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    try:
                        paper = json.loads(line)
                        self.process_paper(paper)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error processing file {filepath}: {str(e)}")

    def _clean_network(self) -> None:
        """Remove nodes with missing essential data."""
        nodes_to_remove = [
            node for node, data in self.graph.nodes(data=True)
            if not data.get("title") or not data.get("year")
        ]
        self.graph.remove_nodes_from(nodes_to_remove)


class NetworkExporter:
    """Handles export of citation networks to various formats."""

    SUPPORTED_FORMATS = {
        'gexf': nx.write_gexf,
        'graphml': nx.write_graphml,
        'edgelist': nx.write_edgelist,
        'adjlist': nx.write_adjlist,
    }

    @classmethod
    def export(cls, graph: nx.DiGraph, output_path: str,
               format: str = 'gexf') -> None:
        """
        Export network to specified format.

        Args:
            graph: NetworkX graph to export
            output_path: Path for output file
            format: Export format

        Raises:
            ValueError: If format is not supported
        """
        if format == 'pickle':
            with open(output_path, 'wb') as f:
                pickle.dump(graph, f)
        elif format in cls.SUPPORTED_FORMATS:
            cls.SUPPORTED_FORMATS[format](graph, output_path)
        else:
            raise ValueError(
                f"Unsupported format. Choose from: {', '.join(cls.SUPPORTED_FORMATS)}"
                f", pickle"
            )


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Build and export citation networks from paper data."
    )
    parser.add_argument(
        'subset_directory',
        type=str,
        help="Directory containing JSONL files"
    )
    parser.add_argument(
        'output_path',
        type=str,
        help="Path for exported network"
    )
    parser.add_argument(
        '--format',
        type=str,
        default='gexf',
        choices=['gexf', 'graphml', 'edgelist', 'adjlist', 'pickle'],
        help="Export format (default: gexf)"
    )
    parser.add_argument(
        '--remove_empty_nodes',
        action='store_true',
        help="Remove incomplete entries"
    )

    args = parser.parse_args()

    try:
        # Build network
        builder = CitationNetworkBuilder(args.remove_empty_nodes)
        network = builder.build_from_directory(args.subset_directory)

        # Export network
        NetworkExporter.export(network, args.output_path, args.format)

        # Print statistics
        print("\nNetwork Statistics:")
        print(f"Nodes: {network.number_of_nodes()}")
        print(f"Edges: {network.number_of_edges()}")
        print(f"Network saved to: {args.output_path}")

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    main()