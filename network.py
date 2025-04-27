import argparse
import json
import os
import networkx as nx
import pickle

def build_citation_network(subset_dir, remove_empty_nodes=False):
    """
    Builds a citation network from the subset of papers and adds section information.

    Parameters:
    subset_dir (str): The directory containing the subset JSONL files.
    remove_empty_nodes (bool): Whether to remove nodes without title, year, or references.

    Returns:
    G (networkx.DiGraph): The citation network as a directed graph.
    """
    G = nx.DiGraph()

    for root, _, files in os.walk(subset_dir):
        for file in files:
            if file.endswith('.jsonl'):
                with open(os.path.join(root, file), 'r') as f:
                    for line in f:
                        paper = json.loads(line)
                        paper_id = paper.get("paper_id")
                        title = paper.get("title")
                        authors = paper.get("authors")
                        year = paper.get("year")
                        G.add_node(paper_id, title=title, year=year, authors=authors)

                        sections = paper.get("sections", [])
                        section_info = []
                        for section in sections:
                            section_name = section.get("section_name")
                            external_uris = set()
                            cited_references = set()
                            for paragraph in section.get("paragraphs", []):
                                external_uris.update(paragraph.get("external_uris", []))
                                for ref in paragraph.get("cited_references", []):
                                    arxiv_id = ref.get("arxiv_id")
                                    if arxiv_id:
                                        cited_references.add(arxiv_id)
                                        G.add_edge(paper_id, arxiv_id)
                            section_info.append({
                                "section_name": section_name,
                                "external_uris": list(external_uris),
                                "cited_references": list(cited_references)
                            })
                        G.nodes[paper_id]['sections'] = section_info

    if remove_empty_nodes:
        nodes_to_remove = [node for node, data in G.nodes(data=True) if not data.get("title") or not data.get("year")]
        G.remove_nodes_from(nodes_to_remove)

    return G

def export_network(G, output_path, output_format='gexf'):
    """
    Exports the citation network to a specified format.

    Parameters:
    G (networkx.DiGraph): The citation network as a directed graph.
    output_path (str): The path to save the exported network.
    format (str): The format to export the network ('gexf', 'graphml', 'edgelist', 'adjlist', 'pickle').
    """
    if output_format == 'gexf':
        nx.write_gexf(G, output_path)
    elif output_format == 'graphml':
        nx.write_graphml(G, output_path)
    elif output_format == 'edgelist':
        nx.write_edgelist(G, output_path)
    elif output_format == 'adjlist':
        nx.write_adjlist(G, output_path)
    elif output_format == 'pickle':
        with open(output_path, 'wb') as f:
            pickle.dump(G, f)
    else:
        raise ValueError("Unsupported format. Choose from 'gexf', 'graphml', 'edgelist', 'adjlist', 'pickle'.")

def main():
    parser = argparse.ArgumentParser(
        description="Build a citation network from a subset of papers and export it to a specified format.")
    parser.add_argument('subset_directory', type=str, help="The directory containing the subset JSONL files.")
    parser.add_argument('output_path', type=str, help="The path to save the exported network.")
    parser.add_argument('--format', type=str, default='gexf',
                        choices=['gexf', 'graphml', 'edgelist', 'adjlist', 'pickle'],
                        help="The format to export the network ('gexf', 'graphml', 'edgelist', 'adjlist', 'pickle'). Default is 'gexf'.")
    parser.add_argument('--remove_empty_nodes', action='store_true', help="Remove nodes without title, year, or references.")

    args = parser.parse_args()

    G = build_citation_network(args.subset_directory, args.remove_empty_nodes)
    export_network(G, args.output_path, args.format)

if __name__ == "__main__":
    main()